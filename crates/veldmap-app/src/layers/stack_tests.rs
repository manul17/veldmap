use std::path::{Path, PathBuf};
use std::sync::Arc;

use veldmap_core::geo::{Frame, Projection};
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{SourceFingerprint, SourceId};
use veldmap_raster::{
    DecodeContext, LevelPlan, RasterDescription, RasterError, RasterSource, ServeMode, TileEmit,
};

use super::*;
use crate::layers::{CrsOverride, LayerInitOptions};

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

    fn pyramid_meta_for(
        &self,
        spec: veldmap_core::tile::TileStreamSpec,
    ) -> Result<veldmap_raster::PyramidMeta, RasterError> {
        Ok(veldmap_raster::PyramidMeta {
            source: self.src,
            spec,
            levels: self.desc.levels.clone(),
        })
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
            style: None,
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
    let ids: Vec<LayerId> = stack.layers().iter().map(OverlayLayer::id).collect();
    assert_eq!(ids, vec![LayerId(2), LayerId(1), LayerId(3)]);
    assert!(stack.move_down(LayerId(1)));
    let ids: Vec<LayerId> = stack.layers().iter().map(OverlayLayer::id).collect();
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
    let prep1 = l1.begin_reload(CancelToken::new());
    assert!(prep1.old_cancel.is_none());
    assert_eq!(prep1.old_source, shared_src);
    assert!(l1.is_reloading());
    assert!(!l1.is_active());
    assert!(l1.wanted().is_empty());
    assert!(l1.matches_reload_seq(prep1.reload_seq));

    l1.fail_reload("disk read failed".to_owned());
    assert!(!l1.is_reloading());
    assert!(l1.is_active());
    assert_eq!(
        l1.to_summary(0, false, true, false).last_error.as_deref(),
        Some("disk read failed")
    );

    let prep2 = l1.begin_reload(CancelToken::new());
    assert!(prep2.reload_seq > prep1.reload_seq);
    assert!(!l1.matches_reload_seq(prep1.reload_seq));
    assert!(l1.matches_reload_seq(prep2.reload_seq));

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
    assert!(l1.matches_reload_seq(prep2.reload_seq));
    assert!(l1.to_summary(0, false, true, false).last_error.is_none());

    // Rebinding CRS increments epoch and clears errors.
    let epoch_before = l1.crs_override_epoch();
    l1.apply_rebound_frame(utm_frame(20.0), Some(CrsOverride::Epsg(32637)), 100);
    assert_eq!(l1.crs_override_epoch(), epoch_before + 1);
    assert_eq!(l1.crs_override(), Some(&CrsOverride::Epsg(32637)));
    assert_eq!(l1.relaid(), 100);
}

#[test]
fn multi_stream_appetite_weight_and_opt_in_pin_root() {
    let mut stack = LayerStack::default();
    let mut layer = make_layer(10, 10, 10.0);
    assert!(
        layer.pin_root(),
        "local raster layers opt into root pinning by default"
    );
    layer.set_pin_root(false);
    assert!(!layer.pin_root());

    let primary_key = layer.meta().stream_key();
    let extra_spec = veldmap_core::tile::TileStreamSpec {
        band: veldmap_core::tile::BandKey::Index(1),
        format: veldmap_core::tile::PixelFormat::R8Uint,
        gutter: 0,
        decode_rev: 0,
    };
    let mut extra_meta = layer.meta().clone();
    extra_meta.spec = extra_spec;
    layer.push_aux_stream_for_test(extra_meta);
    let extra_key = layer.streams()[1].stream_key();

    // Distinct streams sharing one SourceId track independent tile-byte weights
    // and sum cleanly for multi-band VRAM appetite budgeting.
    layer.observe_stream_tile_bytes(primary_key, 2_000_000);
    layer.observe_stream_tile_bytes(extra_key, 262_144);
    assert_eq!(layer.streams()[0].observed_tile_bytes(), 2_000_000);
    assert_eq!(layer.streams()[1].observed_tile_bytes(), 262_144);
    assert_eq!(layer.appetite_weight_bytes(), 2_000_000 + 262_144);
    assert_eq!(layer.cell_observed_bytes(), 2_000_000 + 262_144);

    // Reloading preserves opt-in `pin_root` flag.
    let reloaded_src = StubSource::arc(11, 512, 512);
    layer.replace_reloaded(
        PathBuf::from("/tmp/reloaded.tif"),
        reloaded_src,
        utm_frame(10.0),
        2,
    );
    assert!(
        !layer.pin_root(),
        "replace_reloaded must preserve pin_root=false"
    );

    stack.push(layer);
    let mut pinned = Vec::new();
    stack.extend_pinned_root_streams(&mut pinned);
    assert!(
        pinned.is_empty(),
        "unpinned layer must not opt into pinned root streams"
    );
    stack
        .get_mut(LayerId(10))
        .expect("layer")
        .set_pin_root(true);
    stack.extend_pinned_root_streams(&mut pinned);
    assert_eq!(pinned.len(), 1);
    // Calling extend a second time deduplicates cleanly.
    stack.extend_pinned_root_streams(&mut pinned);
    assert_eq!(pinned.len(), 1);
}

#[test]
fn scientific_band_switch_stats_generation_and_saved_style_transitions() {
    use crate::layers::style::{SavedLayerStyle, StretchPreset};
    use crate::render::CategoricalPaletteId;
    use veldmap_core::tile::{BandKey, PixelFormat};
    use veldmap_raster::{BandStats, DISPLAY_STREAM_SPEC, stream_spec_for_band};

    let sample_stats = |lo: f64, hi: f64| BandStats {
        min: lo,
        max: hi,
        p2: lo + 1.0,
        p98: hi - 1.0,
        db_min: Some(-20.0),
        db_max: Some(0.0),
        db_p2: Some(-18.0),
        db_p98: Some(-2.0),
        sample_count: 100,
        histogram: [0; 64],
        db_histogram: [0; 64],
    };

    // Restoring `SourceColorMap` on a source without a TIFF ColorMap falls back cleanly.
    let src = StubSource::arc(21, 512, 512);
    let mut layer = OverlayLayer::from_source(
        LayerId(21),
        PathBuf::from("/tmp/sci.tif"),
        src,
        utm_frame(10.0),
        LayerInitOptions {
            visible: true,
            opacity: 1.0,
            crs_override: None,
            style: Some(SavedLayerStyle {
                categorical_palette: Some(CategoricalPaletteId::SourceColorMap),
                ..SavedLayerStyle::default()
            }),
        },
        1,
    );
    assert!(
        layer.saved_style().is_none(),
        "untouched default Display saves as None"
    );
    assert!(
        layer
            .to_summary(0, false, false, false)
            .categorical
            .is_none()
    );

    // Switch to Band 0 Scientific, load stats, apply custom range.
    let b0_spec = stream_spec_for_band(BandKey::Index(0), PixelFormat::R32Float);
    let b1_spec = stream_spec_for_band(BandKey::Index(1), PixelFormat::R32Float);
    assert!(layer.set_active_stream(b0_spec).expect("switch b0"));
    assert_eq!(layer.needs_band_stats(), Some(BandKey::Index(0)));

    let tok1 = CancelToken::new();
    layer.begin_stats_job(1, tok1.clone());
    assert!(layer.scientific_style().is_loading());
    // Stale job_id cannot clobber active job 1.
    assert!(!layer.apply_band_stats_empty(BandKey::Index(0), 99));
    assert!(layer.scientific_style().is_loading());
    // Cancelled job resets back to NotRequested (never poisons Empty).
    assert!(layer.apply_band_stats_cancelled(BandKey::Index(0), 1));
    assert_eq!(layer.needs_band_stats(), Some(BandKey::Index(0)));

    layer.begin_stats_job(2, CancelToken::new());
    assert!(layer.apply_band_stats_ready(BandKey::Index(0), sample_stats(0.0, 100.0)));
    assert_eq!(
        (layer.scientific_style().min, layer.scientific_style().max),
        (1.0, 99.0)
    );

    assert!(layer.set_stretch_range(10.0, 80.0));
    assert_eq!(layer.scientific_style().preset, StretchPreset::Custom);
    // No-op range and preset calls return false.
    assert!(!layer.set_stretch_range(10.0, 80.0));
    assert!(!layer.set_stretch_preset(StretchPreset::Custom));
    assert!(layer.saved_style().is_some());

    // Toggling Display -> back to SAME Band 0 preserves Ready stats and custom bounds.
    assert!(
        layer
            .set_active_stream(DISPLAY_STREAM_SPEC)
            .expect("display")
    );
    assert!(layer.set_active_stream(b0_spec).expect("back to b0"));
    assert_eq!(layer.needs_band_stats(), None);
    assert_eq!(
        (layer.scientific_style().min, layer.scientific_style().max),
        (10.0, 80.0)
    );
    assert_eq!(layer.scientific_style().preset, StretchPreset::Custom);

    // Switching to DIFFERENT Band 1 resets Custom -> Percentile2_98 so new stats apply cleanly.
    assert!(layer.set_active_stream(b1_spec).expect("switch b1"));
    assert_eq!(
        layer.scientific_style().preset,
        StretchPreset::Percentile2_98
    );
    assert_eq!(layer.needs_band_stats(), Some(BandKey::Index(1)));
    assert!(layer.apply_band_stats_ready(BandKey::Index(1), sample_stats(500.0, 1000.0)));
    assert_eq!(
        (layer.scientific_style().min, layer.scientific_style().max),
        (501.0, 999.0)
    );
}
