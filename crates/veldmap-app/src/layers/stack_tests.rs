use veldmap_core::geo::Projection;
use veldmap_core::tile::SourceFingerprint;
use veldmap_raster::{
    DecodeContext, LevelPlan, RasterDescription, RasterError, ServeMode, TileEmit,
};

use super::*;

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
