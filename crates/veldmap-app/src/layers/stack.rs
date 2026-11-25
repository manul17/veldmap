//! Ordered [`LayerStack`] and per-layer metadata/state (`OverlayLayer`).

use std::collections::HashSet;
#[cfg(test)]
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use veldmap_core::geo::{CullFrustum, FootprintBounds, Frame};
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{OverlayCullTree, SourceId, TileKey};
use veldmap_raster::{PyramidMeta, RasterSource};

use crate::{clamp_opacity, display_name_from_path};

use super::io_worker::ReloadRequest;
use super::{CrsOverride, LayerId, LayerSummary, OverlayPersistEntry, default_display_tile_bytes};

/// Initial display and georeference override options when inserting an opened
/// raster layer into [`LayerStack`].
#[derive(Clone, Debug)]
pub(super) struct LayerInitOptions {
    pub(super) visible: bool,
    pub(super) opacity: f32,
    pub(super) crs_override: Option<CrsOverride>,
}

/// Cohesive per-layer reload lifecycle state.
#[derive(Default)]
struct ReloadState {
    in_progress: bool,
    seq: u64,
    cancel: Option<CancelToken>,
}

/// Single georeferenced raster overlay inside [`LayerStack`].
///
/// Mutable fields stay private behind transition methods so visibility,
/// opacity, wanted-set synchronization, reload lifecycle, and CRS rebinding
/// invariants cannot be broken by callers.
pub(super) struct OverlayLayer {
    id: LayerId,
    name: String,
    path: PathBuf,
    source: Arc<dyn RasterSource>,
    meta: PyramidMeta,
    fingerprint_hex: String,
    frame: Arc<Frame>,
    crs_override: Option<CrsOverride>,
    crs_override_epoch: u64,
    last_error: Option<String>,
    relaid: u64,
    cull: OverlayCullTree,
    visible: bool,
    opacity: f32,
    source_pixel_size_m: Option<(f64, f64)>,
    footprint: Option<FootprintBounds>,
    /// Current wanted tile set (shared with active `ServeMode::Pass` jobs so
    /// mid-pass zooms deliver tiles matching the live viewport).
    wanted_shared: Arc<Mutex<HashSet<TileKey>>>,
    /// Snapshot of `wanted_shared` retained on the UI thread for
    /// `PendingUploads` retention without acquiring a lock inside upload drain.
    wanted: HashSet<TileKey>,
    observed_tile_bytes: usize,
    reload: ReloadState,
}

impl OverlayLayer {
    pub(super) fn from_source(
        id: LayerId,
        path: PathBuf,
        source: Arc<dyn RasterSource>,
        frame: Frame,
        options: LayerInitOptions,
        relaid: u64,
    ) -> Self {
        let name = display_name_from_path(&path);
        let meta = source.pyramid_meta();
        let fingerprint_hex = source.fingerprint().to_hex();
        let desc = source.description();
        let source_pixel_size_m = frame.pixel_size_m(desc.width, desc.height);
        let footprint = frame.footprint_bounds();
        let cull = OverlayCullTree::new(desc.width, desc.height, meta.max_level());
        Self {
            id,
            name,
            path,
            source,
            meta,
            fingerprint_hex,
            frame: Arc::new(frame),
            crs_override: options.crs_override,
            crs_override_epoch: 0,
            last_error: None,
            relaid,
            cull,
            visible: options.visible,
            opacity: clamp_opacity(options.opacity),
            source_pixel_size_m,
            footprint,
            wanted_shared: Arc::new(Mutex::new(HashSet::new())),
            wanted: HashSet::new(),
            observed_tile_bytes: default_display_tile_bytes(),
            reload: ReloadState::default(),
        }
    }

    pub(super) fn name(&self) -> &str {
        &self.name
    }

    pub(super) fn source(&self) -> &Arc<dyn RasterSource> {
        &self.source
    }

    pub(super) fn meta(&self) -> &PyramidMeta {
        &self.meta
    }

    pub(super) fn fingerprint_hex(&self) -> &str {
        &self.fingerprint_hex
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

    pub(super) fn opacity(&self) -> f32 {
        self.opacity
    }

    pub(super) fn source_pixel_size_m(&self) -> Option<(f64, f64)> {
        self.source_pixel_size_m
    }

    pub(super) fn footprint(&self) -> Option<FootprintBounds> {
        self.footprint
    }

    pub(super) fn observed_tile_bytes(&self) -> usize {
        self.observed_tile_bytes
    }

    pub(super) fn persist_entry(&self) -> OverlayPersistEntry<'_> {
        OverlayPersistEntry {
            path: &self.path,
            visible: self.visible,
            opacity: self.opacity,
            crs_override: self.crs_override.as_ref(),
        }
    }

    pub(super) fn visible_cells(&mut self, level: u8, frustum: &CullFrustum) -> Vec<(u32, u32)> {
        self.cull.visible_cells(&self.frame, level, frustum)
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

    pub(super) fn observe_tile_bytes(&mut self, bytes: usize) {
        self.observed_tile_bytes = self.observed_tile_bytes.max(bytes);
    }

    pub(super) fn to_summary(
        &self,
        in_flight: usize,
        pass_active: bool,
        can_move_up: bool,
        can_move_down: bool,
    ) -> LayerSummary {
        let desc = self.source.description();
        let sample_info = desc.bands.first().map_or_else(
            || "Display RGBA".to_owned(),
            |b| {
                let nodata_str = b
                    .nodata
                    .map(|v| format!(", nodata={v}"))
                    .unwrap_or_default();
                format!("{} ch, {:?}{nodata_str}", b.channels, b.sample_type)
            },
        );
        LayerSummary {
            id: self.id,
            name: self.name.clone(),
            path: self.path.clone(),
            visible: self.visible,
            opacity: self.opacity,
            width: desc.width,
            height: desc.height,
            source_pixel_size_m: self.source_pixel_size_m,
            footprint: self.footprint,
            in_flight,
            pass_active,
            reloading: self.is_reloading(),
            crs_label: self.frame.crs_label(),
            datum_warning: self.frame.datum_warning().map(|w| w.message()),
            sample_info,
            detail_limit_level: self
                .meta
                .finest_fitting_level()
                .unwrap_or(self.meta.max_level()),
            can_override_crs: desc.can_override_crs(),
            has_crs_override: self.crs_override.is_some(),
            last_error: self.last_error.clone(),
            can_move_up,
            can_move_down,
        }
    }

    pub(super) fn apply_rebound_frame(
        &mut self,
        frame: Frame,
        crs_override: Option<CrsOverride>,
        relaid: u64,
    ) {
        let desc = self.source.description();
        self.source_pixel_size_m = frame.pixel_size_m(desc.width, desc.height);
        self.footprint = frame.footprint_bounds();
        self.frame = Arc::new(frame);
        self.crs_override = crs_override;
        self.crs_override_epoch = self.crs_override_epoch.saturating_add(1);
        self.last_error = None;
        self.relaid = relaid;
        self.cull = OverlayCullTree::new(desc.width, desc.height, self.meta.max_level());
    }

    pub(super) fn source_id(&self) -> SourceId {
        self.meta.source
    }

    pub(super) fn is_active(&self) -> bool {
        self.visible && !self.reload.in_progress
    }

    pub(super) fn is_reloading(&self) -> bool {
        self.reload.in_progress
    }

    pub(super) fn begin_reload(
        &mut self,
        cancel: CancelToken,
    ) -> (ReloadRequest, SourceId, Option<CancelToken>) {
        let prev = self.reload.cancel.replace(cancel.clone());
        self.reload.seq = self.reload.seq.saturating_add(1);
        self.reload.in_progress = true;
        self.clear_wanted();
        let request = ReloadRequest {
            id: self.id,
            reload_seq: self.reload.seq,
            crs_override_epoch: self.crs_override_epoch,
            path: self.path.clone(),
            old_fingerprint_hex: self.fingerprint_hex.clone(),
            crs_override: self.crs_override.clone(),
            cancel,
        };
        (request, self.source_id(), prev)
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
        let mut replacement = Self::from_source(
            self.id,
            path,
            source,
            frame,
            LayerInitOptions {
                crs_override: self.crs_override.clone(),
                visible: self.visible,
                opacity: self.opacity,
            },
            relaid,
        );
        replacement.reload.seq = self.reload.seq;
        replacement.crs_override_epoch = self.crs_override_epoch;
        *self = replacement;
        old_src
    }

    pub(super) fn take_reload_cancel(&mut self) -> Option<CancelToken> {
        self.reload.cancel.take()
    }

    pub(super) fn appetite_weight_bytes(&self) -> u64 {
        if self.is_active() {
            (self.observed_tile_bytes as u64).max(1024)
        } else {
            0
        }
    }

    /// Borrow the UI-thread snapshot of this layer's currently wanted tiles.
    pub(super) fn wanted(&self) -> &HashSet<TileKey> {
        &self.wanted
    }

    /// Clone the shared wanted-tile handle passed to sequential `ServeMode::Pass` jobs.
    pub(super) fn wanted_shared(&self) -> Arc<Mutex<HashSet<TileKey>>> {
        Arc::clone(&self.wanted_shared)
    }

    /// Atomically replace this layer's wanted tile set and synchronize its
    /// worker-shared mirror so the two can never drift out of sync.
    pub(super) fn replace_wanted(&mut self, tiles: impl IntoIterator<Item = TileKey>) {
        self.wanted.clear();
        self.wanted.extend(tiles);
        lock_unpoisoned(&self.wanted_shared).clone_from(&self.wanted);
    }

    pub(super) fn clear_wanted(&mut self) {
        self.wanted.clear();
        lock_unpoisoned(&self.wanted_shared).clear();
    }

    #[cfg(test)]
    fn path(&self) -> &Path {
        &self.path
    }
}

/// Ordered collection of draped raster overlays in painter's order
/// (index `0` drawn right above the basemap, last index drawn on top beneath
/// the procedural graticule).
#[derive(Default)]
pub(super) struct LayerStack {
    layers: Vec<OverlayLayer>,
}

impl LayerStack {
    pub(super) fn push(&mut self, layer: OverlayLayer) {
        self.layers.push(layer);
    }

    pub(super) fn remove(&mut self, id: LayerId) -> Option<OverlayLayer> {
        let idx = self.layers.iter().position(|l| l.id == id)?;
        Some(self.layers.remove(idx))
    }

    /// Move layer toward the top of painter's order (higher index, drawn later).
    pub(super) fn move_up(&mut self, id: LayerId) -> bool {
        let Some(idx) = self.layers.iter().position(|l| l.id == id) else {
            return false;
        };
        if idx + 1 >= self.layers.len() {
            return false;
        }
        self.layers.swap(idx, idx + 1);
        true
    }

    /// Move layer toward the bottom of painter's order (lower index, drawn earlier).
    pub(super) fn move_down(&mut self, id: LayerId) -> bool {
        let Some(idx) = self.layers.iter().position(|l| l.id == id) else {
            return false;
        };
        if idx == 0 {
            return false;
        }
        self.layers.swap(idx - 1, idx);
        true
    }

    pub(super) fn get(&self, id: LayerId) -> Option<&OverlayLayer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub(super) fn get_mut(&mut self, id: LayerId) -> Option<&mut OverlayLayer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    pub(super) fn layers(&self) -> &[OverlayLayer] {
        &self.layers
    }

    pub(super) fn layers_mut(&mut self) -> &mut [OverlayLayer] {
        &mut self.layers
    }

    pub(super) fn count_sharing_source(&self, source_id: SourceId) -> usize {
        self.layers
            .iter()
            .filter(|l| l.source_id() == source_id)
            .count()
    }

    pub(super) fn has_visible_source(&self, source_id: SourceId) -> bool {
        self.layers
            .iter()
            .any(|l| l.is_active() && l.source_id() == source_id)
    }

    pub(super) fn total_appetite_weight_bytes(&self) -> u64 {
        self.layers
            .iter()
            .fold(0u64, |acc, l| acc.saturating_add(l.appetite_weight_bytes()))
    }

    pub(super) fn topmost_visible_source_pixel_size_m(&self) -> Option<(f64, f64)> {
        self.layers
            .iter()
            .rev()
            .find(|l| l.is_active())
            .and_then(|l| l.source_pixel_size_m)
    }

    pub(super) fn extend_pinned_root_sources(&self, out: &mut Vec<SourceId>) {
        for layer in &self.layers {
            if layer.is_active() {
                let src = layer.source_id();
                if !out.contains(&src) {
                    out.push(src);
                }
            }
        }
    }

    pub(super) fn extend_wanted_tile_sets<'a>(&'a self, out: &mut Vec<&'a HashSet<TileKey>>) {
        for layer in &self.layers {
            if layer.is_active() && !layer.wanted().is_empty() {
                out.push(layer.wanted());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use veldmap_core::geo::Projection;
    use veldmap_core::tile::SourceFingerprint;
    use veldmap_raster::{
        DecodeContext, LevelPlan, RasterDescription, RasterError, ServeMode, TileEmit,
    };

    struct StubSource {
        fp: SourceFingerprint,
        src: SourceId,
        desc: RasterDescription,
    }

    impl StubSource {
        fn arc(tag: u8, width: u32, height: u32) -> Arc<dyn RasterSource> {
            let fp = SourceFingerprint::from_digest([tag; 32]);
            let src = SourceId::from_stable_key(&format!("stub-{tag}"));
            let desc = RasterDescription {
                width,
                height,
                bands: Vec::new(),
                ties: Vec::new(),
                placement: None,
                geographic_crs: None,
                binding_trouble: None,
                levels: vec![LevelPlan {
                    serve: ServeMode::Pointwise,
                    estimated_peak_bytes: 1024,
                    fits: true,
                }],
            };
            Arc::new(Self { fp, src, desc })
        }
    }

    impl RasterSource for StubSource {
        fn fingerprint(&self) -> SourceFingerprint {
            self.fp
        }

        fn source_id(&self) -> SourceId {
            self.src
        }

        fn description(&self) -> &RasterDescription {
            &self.desc
        }

        fn pass_peak_bytes(&self) -> u64 {
            1024
        }

        fn produce_pass(
            &self,
            _ctx: &DecodeContext<'_>,
            _emit: TileEmit<'_>,
        ) -> Result<(), RasterError> {
            Ok(())
        }

        fn sample_display_rgba(
            &self,
            _px: u32,
            _py: u32,
            _ctx: &DecodeContext<'_>,
        ) -> Result<Option<[u8; 4]>, RasterError> {
            Ok(None)
        }
    }

    fn utm_frame(pixel_size_m: f64) -> Frame {
        let proj = Projection::from_epsg(32637).expect("UTM 37N");
        Frame::from_pixel_affine(
            proj,
            [
                pixel_size_m,
                0.0,
                500_000.0,
                0.0,
                -pixel_size_m,
                5_200_000.0,
            ],
            512,
            512,
        )
        .expect("valid UTM frame")
    }

    fn make_layer(id: u64, tag: u8, pixel_size_m: f64) -> OverlayLayer {
        OverlayLayer::from_source(
            LayerId(id),
            PathBuf::from(format!("/tmp/layer_{id}.tif")),
            StubSource::arc(tag, 512, 512),
            utm_frame(pixel_size_m),
            LayerInitOptions {
                visible: true,
                opacity: 1.0,
                crs_override: None,
            },
            id,
        )
    }

    #[test]
    fn layer_stack_reorder_visibility_opacity_and_reload_transitions() {
        let mut stack = LayerStack::default();
        stack.push(make_layer(1, 1, 10.0));
        stack.push(make_layer(2, 2, 2.0));
        stack.push(make_layer(3, 1, 5.0)); // shares source tag=1 with layer 1

        // Reorder respects painter's order bounds (0 = bottom, len-1 = top).
        assert!(!stack.move_down(LayerId(1)));
        assert!(!stack.move_up(LayerId(3)));
        assert!(stack.move_up(LayerId(1)));
        let ids: Vec<LayerId> = stack.layers().iter().map(|l| l.id).collect();
        assert_eq!(ids, vec![LayerId(2), LayerId(1), LayerId(3)]);
        assert!(stack.move_down(LayerId(1)));
        let ids: Vec<LayerId> = stack.layers().iter().map(|l| l.id).collect();
        assert_eq!(ids, vec![LayerId(1), LayerId(2), LayerId(3)]);

        // Shared-source queries and topmost active pixel size.
        let shared_src = stack.get(LayerId(1)).expect("layer 1").source_id();
        assert_eq!(stack.count_sharing_source(shared_src), 2);
        assert!(stack.has_visible_source(shared_src));
        let top_px = stack
            .topmost_visible_source_pixel_size_m()
            .expect("topmost pixel size");
        assert!((top_px.0 - 5.0).abs() < 0.1);

        // Visibility transition clears both `wanted` and `wanted_shared`.
        let l3 = stack.get_mut(LayerId(3)).expect("layer 3");
        let tk = l3.meta().tile_key(0, 0, 0);
        l3.replace_wanted([tk]);
        assert_eq!(l3.wanted().len(), 1);
        assert_eq!(lock_unpoisoned(&l3.wanted_shared()).len(), 1);
        assert!(!l3.set_visible(true));
        assert!(l3.set_visible(false));
        assert!(l3.wanted().is_empty());
        assert!(lock_unpoisoned(&l3.wanted_shared()).is_empty());

        // Opacity clamping & NaN guard.
        assert!(l3.set_opacity(-0.5));
        assert_eq!(l3.opacity(), 0.0);
        assert!(!l3.set_opacity(0.0));
        assert!(l3.set_opacity(f32::NAN));
        assert_eq!(l3.opacity(), 1.0);
        assert!(l3.set_visible(true));

        // Reload lifecycle: begin_reload -> fail_reload -> begin_reload -> replace_reloaded.
        let l1 = stack.get_mut(LayerId(1)).expect("layer 1");
        l1.replace_wanted([l1.meta().tile_key(0, 0, 0)]);
        let (req1, _, prev1) = l1.begin_reload(CancelToken::new());
        assert!(prev1.is_none());
        assert!(l1.is_reloading());
        assert!(!l1.is_active());
        assert!(l1.wanted().is_empty());
        assert!(l1.matches_reload_seq(req1.reload_seq));

        l1.fail_reload("disk read failed".to_owned());
        assert!(!l1.is_reloading());
        assert!(l1.is_active());
        assert_eq!(
            l1.to_summary(0, false, true, false).last_error.as_deref(),
            Some("disk read failed")
        );

        let (req2, _, _) = l1.begin_reload(CancelToken::new());
        assert!(req2.reload_seq > req1.reload_seq);
        assert!(!l1.matches_reload_seq(req1.reload_seq));
        assert!(l1.matches_reload_seq(req2.reload_seq));

        let new_src = StubSource::arc(9, 512, 512);
        let new_src_id = new_src.source_id();
        let old_src = l1.replace_reloaded(
            PathBuf::from("/tmp/reloaded.tif"),
            new_src,
            utm_frame(1.0),
            99,
        );
        assert_eq!(old_src, shared_src);
        assert_eq!(l1.source_id(), new_src_id);
        assert_eq!(l1.relaid(), 99);
        assert_eq!(l1.path(), Path::new("/tmp/reloaded.tif"));
        assert!(!l1.is_reloading());
        assert!(l1.matches_reload_seq(req2.reload_seq));
        assert!(l1.to_summary(0, false, true, false).last_error.is_none());

        // Rebinding CRS increments epoch and clears errors.
        let epoch_before = l1.crs_override_epoch();
        l1.apply_rebound_frame(utm_frame(20.0), Some(CrsOverride::Epsg(32637)), 100);
        assert_eq!(l1.crs_override_epoch(), epoch_before + 1);
        assert_eq!(l1.crs_override(), Some(&CrsOverride::Epsg(32637)));
        assert_eq!(l1.relaid(), 100);
    }
}
