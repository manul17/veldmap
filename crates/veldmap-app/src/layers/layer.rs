//! Single georeferenced raster overlay (`OverlayLayer`), georeference/culling
//! binding (`GeoBinding`), per-stream tile state (`LayerStream`), and reload
//! lifecycle transitions.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use veldmap_core::geo::{CullFrustum, FootprintBounds, Frame};
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{
    BandKey, OverlayCullTree, PixelFormat, RASTER_TILE_SIZE, SourceId, StreamKey, TileKey,
    TileStreamSpec,
};
use veldmap_net::HttpRangeSource;
use veldmap_raster::{BandStats, PyramidMeta, RasterSource};

use crate::gpu_tiles::tile_format_bytes;
use crate::render::DrawOverlayTrack;
use crate::{clamp_opacity, display_name_from_path};

use super::style::{LayerStyle, LayerStyleSpec};
use super::{CrsOverride, LayerId, LayerInitOptions, OverlayPersistEntry};

/// Layer-domain state transition record returned by [`OverlayLayer::begin_reload`].
pub(super) struct ReloadPrep {
    pub(super) reload_seq: u64,
    pub(super) old_source: SourceId,
    pub(super) old_cancel: Option<CancelToken>,
}

/// Cohesive per-layer reload lifecycle state.
#[derive(Default)]
struct ReloadState {
    in_progress: bool,
    seq: u64,
    cancel: Option<CancelToken>,
}

/// Cohesive georeferencing, CRS override, and frustum-culling state for an
/// [`OverlayLayer`].
pub(super) struct GeoBinding {
    frame: Arc<Frame>,
    crs_override: Option<CrsOverride>,
    crs_override_epoch: u64,
    relaid: u64,
    cull: OverlayCullTree,
    source_pixel_size_m: Option<(f64, f64)>,
    footprint: Option<FootprintBounds>,
}

impl GeoBinding {
    fn new(
        frame: Frame,
        width: u32,
        height: u32,
        max_level: u8,
        crs_override: Option<CrsOverride>,
        relaid: u64,
    ) -> Self {
        let source_pixel_size_m = frame.pixel_size_m(width, height);
        let footprint = frame.footprint_bounds();
        let cull = OverlayCullTree::new(width, height, max_level);
        Self {
            frame: Arc::new(frame),
            crs_override,
            crs_override_epoch: 0,
            relaid,
            cull,
            source_pixel_size_m,
            footprint,
        }
    }

    pub(super) fn frame(&self) -> &Arc<Frame> {
        &self.frame
    }

    pub(super) fn crs_override(&self) -> Option<&CrsOverride> {
        self.crs_override.as_ref()
    }

    pub(super) fn crs_override_epoch(&self) -> u64 {
        self.crs_override_epoch
    }

    pub(super) fn relaid(&self) -> u64 {
        self.relaid
    }

    pub(super) fn source_pixel_size_m(&self) -> Option<(f64, f64)> {
        self.source_pixel_size_m
    }

    pub(super) fn footprint(&self) -> Option<FootprintBounds> {
        self.footprint
    }

    pub(super) fn visible_cells(&mut self, level: u8, frustum: &CullFrustum) -> Vec<(u32, u32)> {
        self.cull.visible_cells(&self.frame, level, frustum)
    }

    fn rebuild_cull(&mut self, width: u32, height: u32, max_level: u8) {
        self.cull = OverlayCullTree::new(width, height, max_level);
    }

    fn apply_rebound(
        &mut self,
        frame: Frame,
        width: u32,
        height: u32,
        max_level: u8,
        crs_override: Option<CrsOverride>,
        relaid: u64,
    ) {
        self.source_pixel_size_m = frame.pixel_size_m(width, height);
        self.footprint = frame.footprint_bounds();
        self.frame = Arc::new(frame);
        self.crs_override = crs_override;
        self.crs_override_epoch = self.crs_override_epoch.saturating_add(1);
        self.relaid = relaid;
        self.rebuild_cull(width, height, max_level);
    }
}

/// One active tile stream inside an [`OverlayLayer`].
pub(super) struct LayerStream {
    meta: PyramidMeta,
    wanted_shared: Arc<Mutex<HashSet<TileKey>>>,
    wanted: HashSet<TileKey>,
    observed_tile_bytes: usize,
}

impl LayerStream {
    fn new(meta: PyramidMeta) -> Self {
        let initial_bytes = tile_format_bytes(RASTER_TILE_SIZE, meta.spec.format);
        Self {
            meta,
            wanted_shared: Arc::new(Mutex::new(HashSet::new())),
            wanted: HashSet::new(),
            observed_tile_bytes: initial_bytes,
        }
    }

    pub(super) fn meta(&self) -> &PyramidMeta {
        &self.meta
    }

    pub(super) fn stream_key(&self) -> StreamKey {
        self.meta.stream_key()
    }

    pub(super) fn observed_tile_bytes(&self) -> usize {
        self.observed_tile_bytes
    }

    pub(super) fn observe_tile_bytes(&mut self, bytes: usize) {
        self.observed_tile_bytes = self.observed_tile_bytes.max(bytes);
    }

    pub(super) fn wanted(&self) -> &HashSet<TileKey> {
        &self.wanted
    }

    pub(super) fn wanted_shared(&self) -> Arc<Mutex<HashSet<TileKey>>> {
        Arc::clone(&self.wanted_shared)
    }

    pub(super) fn replace_wanted(&mut self, tiles: impl IntoIterator<Item = TileKey>) {
        self.wanted.clear();
        self.wanted.extend(tiles);
        lock_unpoisoned(&self.wanted_shared).clone_from(&self.wanted);
    }

    pub(super) fn clear_wanted(&mut self) {
        self.wanted.clear();
        lock_unpoisoned(&self.wanted_shared).clear();
    }
}

/// Origin of an [`OverlayLayer`]: either a local file on disk or a remote
/// Cloud-Optimized GeoTIFF accessed via HTTP Range requests.
enum LayerOrigin {
    Local {
        path: PathBuf,
    },
    Remote {
        display_url: String,
        range: HttpRangeSource,
    },
}

/// Input parameters for constructing a remote Cloud-Optimized GeoTIFF
/// [`OverlayLayer`] via [`OverlayLayer::from_remote_source`].
pub(super) struct RemoteLayerSpec {
    pub(super) id: LayerId,
    pub(super) title: String,
    pub(super) display_url: String,
    pub(super) range_source: HttpRangeSource,
    pub(super) source: Arc<dyn RasterSource>,
    pub(super) frame: Frame,
    pub(super) options: LayerInitOptions,
    pub(super) relaid: u64,
}

/// Single georeferenced raster overlay inside [`super::stack::LayerStack`].
///
/// Invariant: `self.streams` is always non-empty (`self.streams[0]` is the
/// primary active stream).
pub(super) struct OverlayLayer {
    id: LayerId,
    name: String,
    origin: LayerOrigin,
    source: Arc<dyn RasterSource>,
    streams: Vec<LayerStream>,
    style: LayerStyle,
    geo: GeoBinding,
    /// Opt-in flag controlling whether this layer's root tiles stay pinned in
    /// `PendingUploads` (defaults `true` for local files; remote COGs and future
    /// DEM mosaics / timelines opt out).
    pin_root: bool,
    fingerprint_hex: String,
    last_error: Option<String>,
    visible: bool,
    opacity: f32,
    reload: ReloadState,
}

impl OverlayLayer {
    fn from_parts(
        id: LayerId,
        name: String,
        origin: LayerOrigin,
        source: Arc<dyn RasterSource>,
        frame: Frame,
        options: LayerInitOptions,
        relaid: u64,
    ) -> Self {
        let desc = source.description();
        let preferred_spec = options
            .style
            .as_ref()
            .and_then(|s| s.resolve_stream_spec(desc))
            .unwrap_or_else(|| source.default_stream_spec());
        let meta = source
            .pyramid_meta(preferred_spec)
            .or_else(|_| source.pyramid_meta(veldmap_raster::DISPLAY_STREAM_SPEC))
            .unwrap_or_else(|_| PyramidMeta {
                source: source.source_id(),
                spec: veldmap_raster::DISPLAY_STREAM_SPEC,
                levels: desc.levels.clone(),
            });
        let style = LayerStyle::new(source.as_ref(), meta.spec, options.style.as_ref());
        let geo = GeoBinding::new(
            frame,
            desc.width,
            desc.height,
            meta.max_level(),
            options.crs_override,
            relaid,
        );
        let pin_root = matches!(origin, LayerOrigin::Local { .. });
        let fingerprint_hex = source.fingerprint().to_hex();

        Self {
            id,
            name,
            origin,
            source,
            streams: vec![LayerStream::new(meta)],
            style,
            geo,
            pin_root,
            fingerprint_hex,
            last_error: None,
            visible: options.visible,
            opacity: clamp_opacity(options.opacity),
            reload: ReloadState::default(),
        }
    }

    pub(super) fn from_source(
        id: LayerId,
        path: PathBuf,
        source: Arc<dyn RasterSource>,
        frame: Frame,
        options: LayerInitOptions,
        relaid: u64,
    ) -> Self {
        let name = display_name_from_path(&path);
        Self::from_parts(
            id,
            name,
            LayerOrigin::Local { path },
            source,
            frame,
            options,
            relaid,
        )
    }

    /// Construct a remote Cloud-Optimized GeoTIFF overlay layer. Remote layers
    /// do not pin root tiles in `PendingUploads` (`pin_root = false`) and carry
    /// their [`HttpRangeSource`] for asynchronous pinned prefetching.
    pub(super) fn from_remote_source(spec: RemoteLayerSpec) -> Self {
        let name = if spec.title.trim().is_empty() {
            spec.display_url.clone()
        } else {
            spec.title
        };
        Self::from_parts(
            spec.id,
            name,
            LayerOrigin::Remote {
                display_url: spec.display_url,
                range: spec.range_source,
            },
            spec.source,
            spec.frame,
            spec.options,
            spec.relaid,
        )
    }

    pub(super) fn id(&self) -> LayerId {
        self.id
    }

    pub(super) fn name(&self) -> &str {
        &self.name
    }

    pub(super) fn local_path(&self) -> Option<&Path> {
        match &self.origin {
            LayerOrigin::Local { path } => Some(path.as_path()),
            LayerOrigin::Remote { .. } => None,
        }
    }

    pub(super) fn is_remote(&self) -> bool {
        matches!(self.origin, LayerOrigin::Remote { .. })
    }

    pub(super) fn remote_url(&self) -> Option<&str> {
        match &self.origin {
            LayerOrigin::Remote { display_url, .. } => Some(display_url.as_str()),
            LayerOrigin::Local { .. } => None,
        }
    }

    pub(super) fn remote_range(&self) -> Option<&HttpRangeSource> {
        match &self.origin {
            LayerOrigin::Remote { range, .. } => Some(range),
            LayerOrigin::Local { .. } => None,
        }
    }

    pub(super) fn source(&self) -> &Arc<dyn RasterSource> {
        &self.source
    }

    /// Primary active stream (index 0).
    pub(super) fn primary_stream(&self) -> &LayerStream {
        &self.streams[0]
    }

    pub(super) fn primary_stream_mut(&mut self) -> &mut LayerStream {
        &mut self.streams[0]
    }

    pub(super) fn streams(&self) -> &[LayerStream] {
        &self.streams
    }

    pub(super) fn meta(&self) -> &PyramidMeta {
        self.primary_stream().meta()
    }

    pub(super) fn style(&self) -> &LayerStyle {
        &self.style
    }

    pub(super) fn style_mut(&mut self) -> &mut LayerStyle {
        &mut self.style
    }

    pub(super) fn geo(&self) -> &GeoBinding {
        &self.geo
    }

    pub(super) fn geo_mut(&mut self) -> &mut GeoBinding {
        &mut self.geo
    }

    pub(super) fn is_visible(&self) -> bool {
        self.visible
    }

    pub(super) fn opacity(&self) -> f32 {
        self.opacity
    }

    pub(super) fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub(super) fn fingerprint_hex(&self) -> &str {
        &self.fingerprint_hex
    }

    pub(super) fn pin_root(&self) -> bool {
        self.pin_root
    }

    #[cfg(test)]
    pub(super) fn set_pin_root(&mut self, pin_root: bool) {
        self.pin_root = pin_root;
    }

    #[cfg(test)]
    pub(super) fn push_aux_stream_for_test(&mut self, meta: PyramidMeta) {
        self.streams.push(LayerStream::new(meta));
    }

    pub(super) fn needs_band_stats(&self) -> Option<BandKey> {
        let spec = self.meta().spec;
        if self.is_active()
            && spec.format == PixelFormat::R32Float
            && self.style.scientific().needs_stats()
        {
            Some(spec.band)
        } else {
            None
        }
    }

    pub(super) fn begin_stats_job(&mut self, job_id: u64, cancel: CancelToken) {
        self.style.scientific_mut().begin_loading(job_id, cancel);
    }

    pub(super) fn cancel_stats_job(&mut self) {
        self.style.scientific_mut().cancel_loading();
    }

    pub(super) fn apply_band_stats_ready(&mut self, band: BandKey, stats: BandStats) -> bool {
        if self.meta().spec.band != band || self.meta().spec.format != PixelFormat::R32Float {
            return false;
        }
        self.style.scientific_mut().apply_ready(stats);
        true
    }

    pub(super) fn apply_band_stats_empty(&mut self, band: BandKey, job_id: u64) -> bool {
        if self.meta().spec.band != band || self.meta().spec.format != PixelFormat::R32Float {
            return false;
        }
        self.style.scientific_mut().apply_empty(job_id)
    }

    pub(super) fn apply_band_stats_cancelled(&mut self, band: BandKey, job_id: u64) -> bool {
        if self.meta().spec.band != band || self.meta().spec.format != PixelFormat::R32Float {
            return false;
        }
        self.style.scientific_mut().apply_cancelled(job_id)
    }

    /// Switch this layer's active stream (`Display` / `Scientific` / `Categorical`
    /// + band index). Returns `true` if the active stream changed.
    pub(super) fn set_active_stream(&mut self, spec: TileStreamSpec) -> Result<bool, String> {
        if self.meta().spec == spec {
            return Ok(false);
        }
        let meta = self.source.pyramid_meta(spec).map_err(|e| e.to_string())?;
        let desc = self.source.description();
        self.geo
            .rebuild_cull(desc.width, desc.height, meta.max_level());
        self.streams = vec![LayerStream::new(meta)];
        self.style.on_stream_switched(spec);
        Ok(true)
    }

    pub(super) fn draw_track(&self) -> DrawOverlayTrack {
        self.style
            .draw_track(self.source.as_ref(), self.meta().spec)
    }

    pub(super) fn saved_style(&self) -> Option<LayerStyleSpec> {
        self.style
            .saved_style(self.source.as_ref(), self.meta().spec)
    }

    pub(super) fn persist_entry(&self) -> Option<OverlayPersistEntry<'_>> {
        let path = self.local_path()?;
        Some(OverlayPersistEntry {
            path,
            visible: self.visible,
            opacity: self.opacity,
            crs_override: self.geo.crs_override(),
            style: self.saved_style(),
        })
    }

    pub(super) fn set_visible(&mut self, visible: bool) -> bool {
        if self.visible == visible {
            return false;
        }
        self.visible = visible;
        if !visible {
            self.clear_wanted();
        }
        true
    }

    pub(super) fn set_opacity(&mut self, opacity: f32) -> bool {
        let clamped = clamp_opacity(opacity);
        if (self.opacity - clamped).abs() <= f32::EPSILON {
            return false;
        }
        self.opacity = clamped;
        true
    }

    pub(super) fn observe_stream_tile_bytes(&mut self, stream: StreamKey, bytes: usize) {
        for s in &mut self.streams {
            if s.stream_key() == stream {
                s.observe_tile_bytes(bytes);
            }
        }
    }

    pub(super) fn apply_rebound_frame(
        &mut self,
        frame: Frame,
        crs_override: Option<CrsOverride>,
        relaid: u64,
    ) {
        let desc = self.source.description();
        let max_level = self.meta().max_level();
        self.geo.apply_rebound(
            frame,
            desc.width,
            desc.height,
            max_level,
            crs_override,
            relaid,
        );
        self.last_error = None;
    }

    pub(super) fn source_id(&self) -> SourceId {
        self.meta().source
    }

    pub(super) fn is_active(&self) -> bool {
        self.visible && !self.reload.in_progress
    }

    pub(super) fn is_reloading(&self) -> bool {
        self.reload.in_progress
    }

    pub(super) fn begin_reload(&mut self, cancel: CancelToken) -> ReloadPrep {
        let old_cancel = self.reload.cancel.replace(cancel);
        self.reload.seq = self.reload.seq.saturating_add(1);
        self.reload.in_progress = true;
        self.clear_wanted();
        ReloadPrep {
            reload_seq: self.reload.seq,
            old_source: self.source_id(),
            old_cancel,
        }
    }

    pub(super) fn abort_reload(&mut self) {
        self.reload.in_progress = false;
        self.reload.cancel = None;
    }

    pub(super) fn matches_reload_seq(&self, seq: u64) -> bool {
        self.reload.seq == seq
    }

    pub(super) fn fail_reload(&mut self, error: String) {
        self.reload.in_progress = false;
        self.reload.cancel = None;
        self.last_error = Some(error);
    }

    pub(super) fn replace_reloaded(
        &mut self,
        path: PathBuf,
        source: Arc<dyn RasterSource>,
        frame: Frame,
        relaid: u64,
    ) -> SourceId {
        let old_src = self.source_id();
        let saved_style = self.saved_style();
        let mut replacement = Self::from_source(
            self.id,
            path,
            source,
            frame,
            LayerInitOptions {
                crs_override: self.geo.crs_override().cloned(),
                visible: self.visible,
                opacity: self.opacity,
                style: saved_style,
            },
            relaid,
        );
        replacement.pin_root = self.pin_root;
        replacement.reload.seq = self.reload.seq;
        replacement.geo.crs_override_epoch = self.geo.crs_override_epoch;
        *self = replacement;
        old_src
    }

    pub(super) fn take_reload_cancel(&mut self) -> Option<CancelToken> {
        self.reload.cancel.take()
    }

    /// Total proportional VRAM appetite weight across all active streams of
    /// this layer (summing heterogeneous multi-band / multi-format streams).
    pub(super) fn appetite_weight_bytes(&self) -> u64 {
        if !self.is_active() {
            return 0;
        }
        self.streams.iter().fold(0u64, |acc, s| {
            acc.saturating_add((s.observed_tile_bytes() as u64).max(1024))
        })
    }

    /// Total observed bytes for one cell across all streams of this layer.
    pub(super) fn cell_observed_bytes(&self) -> usize {
        self.streams
            .iter()
            .fold(0usize, |acc, s| acc.saturating_add(s.observed_tile_bytes()))
            .max(1)
    }

    pub(super) fn wanted(&self) -> &HashSet<TileKey> {
        self.primary_stream().wanted()
    }

    pub(super) fn wanted_shared(&self) -> Arc<Mutex<HashSet<TileKey>>> {
        self.primary_stream().wanted_shared()
    }

    pub(super) fn replace_wanted(&mut self, tiles: impl IntoIterator<Item = TileKey>) {
        self.primary_stream_mut().replace_wanted(tiles);
    }

    pub(super) fn clear_wanted(&mut self) {
        for s in &mut self.streams {
            s.clear_wanted();
        }
    }
}
