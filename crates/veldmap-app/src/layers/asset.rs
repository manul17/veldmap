//! Cohesive single-asset raster binding (`RasterAsset`), georeferencing/culling
//! (`GeoBinding`), stream wanted-set tracking (`LayerStream`), and origin
//! descriptors shared by `LayerKind::SingleRaster` and `LayerKind::BandMath`
//! (and extensible to Slice 5B DEM mosaics / multi-temporal timelines).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use veldmap_core::geo::{CullFrustum, FootprintBounds, Frame};
use veldmap_core::sync::lock_unpoisoned;
use veldmap_core::tile::{
    OverlayCullTree, RASTER_TILE_SIZE, SourceId, StreamKey, TileKey, TileStreamSpec,
};
use veldmap_net::HttpRangeSource;
use veldmap_raster::{ExternalBandMeta, PyramidMeta, RasterSource};

use crate::gpu_tiles::tile_format_bytes;

use super::{BandAssetPersistEntry, CrsOverride};

/// Authentication policy for opening a remote Cloud-Optimized GeoTIFF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StacCogAuthMode {
    /// Zero-auth public scene (built-in sample scenes).
    None,
    /// Inherit auth mode & catalog origin from the STAC search that produced
    /// the displayed result list (decoupled from the live preset dropdown).
    FromSearchResult,
    /// Explicit user opt-in on a manually entered Direct COG URL.
    DirectCustom,
}

/// Cohesive georeferencing, CRS override, and frustum-culling state for one
/// [`RasterAsset`].
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
    pub(super) fn new(
        frame: Frame,
        width: u32,
        height: u32,
        max_level: u8,
        crs_override: Option<CrsOverride>,
        relaid: u64,
    ) -> Self {
        Self::from_arc(
            Arc::new(frame),
            width,
            height,
            max_level,
            crs_override,
            relaid,
        )
    }

    pub(super) fn from_arc(
        frame: Arc<Frame>,
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
            frame,
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

    pub(super) fn set_crs_override_epoch(&mut self, epoch: u64) {
        self.crs_override_epoch = epoch;
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

    pub(super) fn intersects_frustum(&mut self, frustum: &CullFrustum) -> bool {
        self.cull.intersects_frustum(&self.frame, frustum)
    }

    pub(super) fn visible_cells(&mut self, level: u8, frustum: &CullFrustum) -> Vec<(u32, u32)> {
        self.cull.visible_cells(&self.frame, level, frustum)
    }

    pub(super) fn rebuild_cull(&mut self, width: u32, height: u32, max_level: u8) {
        self.cull = OverlayCullTree::new(width, height, max_level);
    }

    pub(super) fn apply_rebound(
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

/// One active tile stream inside a [`RasterAsset`].
pub(super) struct LayerStream {
    meta: PyramidMeta,
    wanted_shared: Arc<Mutex<HashSet<TileKey>>>,
    wanted: HashSet<TileKey>,
    observed_tile_bytes: usize,
}

impl LayerStream {
    pub(super) fn new(meta: PyramidMeta) -> Self {
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

/// Sanitized metadata retained for a remote Cloud-Optimized GeoTIFF asset so
/// `session.json` and `.veldmap.json` can persist and reload remote layers and
/// STAC Band Math recipes without leaking query tokens or credentials.
#[derive(Clone, Debug)]
pub(super) struct RemoteAssetMeta {
    pub(super) canonical_url: String,
    pub(super) display_url: String,
    pub(super) auth_mode: StacCogAuthMode,
    pub(super) external_bands: Vec<ExternalBandMeta>,
    pub(super) is_scl: bool,
    pub(super) stac_item_href: Option<String>,
    pub(super) asset_key: Option<String>,
}

/// Origin of a [`RasterAsset`]: either a local file on disk or a remote
/// Cloud-Optimized GeoTIFF accessed via HTTP Range requests.
#[derive(Clone)]
pub(super) enum LayerOrigin {
    Local {
        path: PathBuf,
    },
    Remote {
        meta: RemoteAssetMeta,
        range: HttpRangeSource,
    },
}

/// Cohesive raster asset bundling its backing [`RasterSource`], optional
/// [`HttpRangeSource`], geodetic [`GeoBinding`], and active [`LayerStream`].
pub(super) struct RasterAsset {
    label: String,
    origin: LayerOrigin,
    source: Arc<dyn RasterSource>,
    fingerprint_hex: String,
    geo: GeoBinding,
    stream: LayerStream,
}

/// Return whether `(scale, offset)` declares a non-identity affine scaling
/// (`scale != 1` or `offset != 0`).
pub(super) fn is_non_identity_affine(scale: f64, offset: f64) -> bool {
    (scale - 1.0).abs() > 1e-12 || offset.abs() > 1e-12
}

/// Return whether `info` declares a non-identity affine scaling (`scale != 1` or `offset != 0`).
pub(super) fn has_affine_scaling(info: &veldmap_raster::BandInfo) -> bool {
    is_non_identity_affine(info.scale, info.offset)
}

impl RasterAsset {
    pub(super) fn new(
        label: String,
        origin: LayerOrigin,
        source: Arc<dyn RasterSource>,
        geo: GeoBinding,
        meta: PyramidMeta,
    ) -> Self {
        let fingerprint_hex = source.fingerprint().to_hex();
        Self {
            label,
            origin,
            source,
            fingerprint_hex,
            geo,
            stream: LayerStream::new(meta),
        }
    }

    /// Build a sibling [`RasterAsset`] reusing this asset's backing source,
    /// origin, and geodetic [`Frame`] for a different [`TileStreamSpec`] (e.g.,
    /// extracting Band 4 NIR alongside Band 1 Red from a single local 4-band
    /// GeoTIFF for band math).
    pub(super) fn clone_for_stream(
        &self,
        label: String,
        spec: TileStreamSpec,
    ) -> Result<Self, String> {
        let meta = self.source.pyramid_meta(spec).map_err(|e| e.to_string())?;
        let desc = self.source.description();
        let geo = GeoBinding::from_arc(
            Arc::clone(self.geo.frame()),
            desc.width,
            desc.height,
            meta.max_level(),
            self.geo.crs_override().cloned(),
            self.geo.relaid(),
        );
        Ok(Self {
            label,
            origin: self.origin.clone(),
            source: Arc::clone(&self.source),
            fingerprint_hex: self.fingerprint_hex.clone(),
            geo,
            stream: LayerStream::new(meta),
        })
    }

    pub(super) fn label(&self) -> &str {
        &self.label
    }

    pub(super) fn origin(&self) -> &LayerOrigin {
        &self.origin
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
            LayerOrigin::Remote { meta, .. } => Some(meta.display_url.as_str()),
            LayerOrigin::Local { .. } => None,
        }
    }

    pub(super) fn remote_range(&self) -> Option<&HttpRangeSource> {
        match &self.origin {
            LayerOrigin::Remote { range, .. } => Some(range),
            LayerOrigin::Local { .. } => None,
        }
    }

    /// Snapshot this constituent band asset for workspace persistence.
    pub(super) fn persist_band_asset(&self) -> BandAssetPersistEntry {
        let band_index = match self.stream.meta().spec.band {
            veldmap_core::tile::BandKey::Index(idx) => idx,
            _ => 0,
        };
        match &self.origin {
            LayerOrigin::Local { path } => BandAssetPersistEntry::Local {
                label: self.label.clone(),
                path: path.clone(),
                crs_override: self.geo.crs_override().cloned(),
                band_index,
            },
            LayerOrigin::Remote { meta, .. } => BandAssetPersistEntry::Remote {
                label: self.label.clone(),
                canonical_url: meta.canonical_url.clone(),
                auth_mode: meta.auth_mode,
                external_bands: meta.external_bands.clone(),
                stac_item_href: meta.stac_item_href.clone(),
                asset_key: meta.asset_key.clone(),
                band_index,
            },
        }
    }

    pub(super) fn source(&self) -> &Arc<dyn RasterSource> {
        &self.source
    }

    pub(super) fn source_id(&self) -> SourceId {
        self.stream.meta().source
    }

    pub(super) fn fingerprint_hex(&self) -> &str {
        &self.fingerprint_hex
    }

    pub(super) fn geo(&self) -> &GeoBinding {
        &self.geo
    }

    pub(super) fn geo_mut(&mut self) -> &mut GeoBinding {
        &mut self.geo
    }

    pub(super) fn stream(&self) -> &LayerStream {
        &self.stream
    }

    pub(super) fn stream_mut(&mut self) -> &mut LayerStream {
        &mut self.stream
    }

    pub(super) fn meta(&self) -> &PyramidMeta {
        self.stream.meta()
    }

    pub(super) fn band_scale_offset(&self) -> (f64, f64) {
        self.source
            .description()
            .band_info(self.meta().spec.band)
            .map_or((1.0, 0.0), |b| (b.scale, b.offset))
    }

    pub(super) fn band_units(&self) -> Option<String> {
        self.source
            .description()
            .band_info(self.meta().spec.band)
            .and_then(|b| b.units.clone())
    }

    /// Return the integer nodata value declared for this asset's active band,
    /// if finite and representable as `u32`.
    pub(super) fn categorical_nodata_value(&self) -> Option<u32> {
        self.source
            .description()
            .band_info(self.meta().spec.band)
            .and_then(|b| b.nodata)
            .filter(|v| v.is_finite() && *v >= 0.0 && *v <= f64::from(u32::MAX))
            .map(|v| v.round() as u32)
    }

    pub(super) fn set_active_stream(&mut self, spec: TileStreamSpec) -> Result<bool, String> {
        if self.meta().spec == spec {
            return Ok(false);
        }
        let meta = self.source.pyramid_meta(spec).map_err(|e| e.to_string())?;
        let desc = self.source.description();
        self.geo
            .rebuild_cull(desc.width, desc.height, meta.max_level());
        self.stream = LayerStream::new(meta);
        Ok(true)
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
    }

    pub(super) fn appetite_weight_bytes(&self) -> u64 {
        (self.stream.observed_tile_bytes() as u64).max(1024)
    }

    pub(super) fn observed_tile_bytes(&self) -> usize {
        self.stream.observed_tile_bytes().max(1)
    }

    pub(super) fn observe_stream_tile_bytes(&mut self, stream: StreamKey, bytes: usize) {
        if self.stream.stream_key() == stream {
            self.stream.observe_tile_bytes(bytes);
        }
    }

    pub(super) fn clear_wanted(&mut self) {
        self.stream.clear_wanted();
    }
}
