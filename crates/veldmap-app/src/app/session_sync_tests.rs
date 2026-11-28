use super::*;
use crate::colormap::ColormapId;
use crate::layers::{
    BandAssetPersistEntry, BandMathFormula, BandMathPreset, BandMathStyleSpec, CrsOverride,
    LayerStyleSpec, OverlayPersistEntry, StretchPreset,
};
use crate::stac::StacCogAuthMode;

#[test]
fn session_restore_tracker_preserves_local_remote_and_band_math_across_offline_failures() {
    let mut tracker = SessionRestoreTracker::default();
    let raster_a = PathBuf::from("/mnt/nas/a.tif");
    let vector_a = PathBuf::from("/mnt/nas/a.geojson");

    let custom_style = LayerStyleSpec {
        stretch_preset: Some(StretchPreset::Custom),
        stretch_min: Some(0.1),
        stretch_max: Some(0.9),
        custom_stretch_range: true,
        colormap: Some(ColormapId::Viridis),
        invert_colormap: true,
        ..LayerStyleSpec::default()
    };
    let saved_local = SavedLayerEntry::with_style(
        raster_a.clone(),
        false,
        0.5,
        Some(&CrsOverride::Epsg(32637)),
        Some(&custom_style),
    );
    let saved_remote = SavedLayerEntry::from_persist(OverlayPersistEntry::RemoteCog {
        canonical_url: "https://example.org/s2/B08.tif".to_owned(),
        title: "S2 NIR".to_owned(),
        auth_mode: StacCogAuthMode::DirectCustom,
        external_bands: Vec::new(),
        is_scl: false,
        stac_item_href: Some("https://stac.example.org/items/1".to_owned()),
        asset_key: Some("nir".to_owned()),
        visible: true,
        opacity: 0.9,
        style: Some(custom_style.clone()),
    });
    let saved_bm = SavedLayerEntry::from_persist(OverlayPersistEntry::BandMath {
        title: "NDVI + SCL".to_owned(),
        visible: true,
        opacity: 0.85,
        bands: vec![
            BandAssetPersistEntry::Remote {
                label: "NIR".to_owned(),
                canonical_url: "https://example.org/b08.tif".to_owned(),
                auth_mode: StacCogAuthMode::None,
                external_bands: Vec::new(),
                stac_item_href: None,
                asset_key: Some("B08".to_owned()),
                band_index: 0,
            },
            BandAssetPersistEntry::Remote {
                label: "Red".to_owned(),
                canonical_url: "https://example.org/b04.tif".to_owned(),
                auth_mode: StacCogAuthMode::None,
                external_bands: Vec::new(),
                stac_item_href: None,
                asset_key: Some("B04".to_owned()),
                band_index: 0,
            },
        ],
        scl: None,
        style: BandMathStyleSpec {
            preset: BandMathPreset::Ndvi,
            formula: BandMathFormula::NormalizedDiff,
            recipe_label: "NDVI".to_owned(),
            colormap: ColormapId::RdYlGn,
            invert_colormap: false,
            stretch_min: -0.2,
            stretch_max: 0.85,
            rgb_min: [0.0; 3],
            rgb_max: [0.35; 3],
            default_rgb_range: ([0.0; 3], [0.35; 3]),
            rgb_gamma: 1.0,
            scl_mask_enabled: false,
            scl_mask_bits: 0,
        },
    });

    let id_local = tracker.alloc_restore_id();
    let id_remote = tracker.alloc_restore_id();
    let id_bm = tracker.alloc_restore_id();
    let id_vec = tracker.alloc_restore_id();

    tracker.push_layer(id_local, Some(0), saved_local);
    tracker.push_layer(id_remote, Some(1), saved_remote.clone());
    tracker.push_layer(id_bm, Some(2), saved_bm.clone());
    tracker.push_vector(
        id_vec,
        SavedVectorEntry {
            path: vector_a.clone(),
            visible: true,
            opacity: 0.7,
            outline_only: true,
        },
    );

    // Non-restore failure (`None`) is ignored.
    assert!(!tracker.on_failed(None, "manual open error"));

    // Offline / 401 / 403 remote COG and BandMath restore failures preserve full entries!
    assert!(tracker.on_failed(Some(id_remote), "HTTP 401 Unauthorized"));
    assert!(tracker.on_failed(Some(id_bm), "network unreachable"));
    assert!(tracker.on_failed(Some(id_local), "drive offline"));
    assert_eq!(tracker.unavailable_count(), 3);
    assert_eq!(tracker.preserved_layers().count(), 3);
    assert!(tracker.preserved_layers().any(|e| e == &saved_remote));
    assert!(tracker.preserved_layers().any(|e| e == &saved_bm));

    // Manual reopen of `raster_a` deduplicates only the unavailable local entry.
    tracker.on_opened(None, None, RestoredEntryKind::LocalRaster, Some(&raster_a));
    assert_eq!(tracker.unavailable_count(), 2);
    assert_eq!(tracker.preserved_layers().count(), 2);

    // Hide / unhide / retry / forget preserve invariants.
    tracker.hide_notices();
    assert!(tracker.visible_notices().is_empty());
    tracker.unhide_notices();
    assert_eq!(tracker.visible_notices().len(), 2);

    let (retry_l, retry_v) = tracker.take_unavailable_for_retry();
    assert_eq!(retry_l.len(), 2);
    assert_eq!(retry_l[0].0, Some(1));
    assert_eq!(retry_l[1].0, Some(2));
    assert!(retry_v.is_empty());
    assert_eq!(tracker.unavailable_count(), 0);
    // Still-pending vector remains preserved!
    assert_eq!(tracker.preserved_vectors().count(), 1);
    assert!(tracker.on_failed(Some(id_vec), "missing"));
    assert!(tracker.forget_unavailable());
    assert_eq!(tracker.preserved_vectors().count(), 0);
}

#[test]
fn session_restore_tracker_preserves_order_and_swipe_indices_across_out_of_order_and_offline_restores()
 {
    let mut tracker = SessionRestoreTracker::default();
    let e0 = SavedLayerEntry::new(PathBuf::from("/data/l0.tif"), true, 1.0, None);
    let e1 = SavedLayerEntry::new(PathBuf::from("/data/l1.tif"), true, 1.0, None);
    let e2 = SavedLayerEntry::new(PathBuf::from("/data/l2.tif"), true, 1.0, None);

    tracker.set_pending_swipe(
        Some(SavedSwipeState {
            active: true,
            split_t: 0.4,
            left_layer_index: Some(0),
            right_layer_index: Some(2),
        }),
        AuthTrust::LocalSession,
    );
    let r0 = tracker.alloc_restore_id();
    let r1 = tracker.alloc_restore_id();
    let r2 = tracker.alloc_restore_id();
    tracker.push_layer(r0, Some(0), e0.clone());
    tracker.push_layer(r1, Some(1), e1.clone());
    tracker.push_layer(r2, Some(2), e2.clone());

    // Simulate out-of-order async open: slot 2 opens first, then slot 1.
    let lid2 = LayerId::from_raw_for_test(20);
    let lid1 = LayerId::from_raw_for_test(10);
    let mut live_ids = vec![lid2];
    let slot2 = tracker.on_opened(Some(r2), Some(lid2), RestoredEntryKind::LocalRaster, None);
    assert_eq!(slot2, Some(2));
    assert!(tracker.try_resolve_pending_swipe().is_none());

    live_ids.push(lid1);
    let slot1 = tracker.on_opened(Some(r1), Some(lid1), RestoredEntryKind::LocalRaster, None);
    assert_eq!(slot1, Some(1));
    let insert_idx = tracker.insertion_index_for_slot(&live_ids, lid1, 1);
    assert_eq!(insert_idx, 0, "slot 1 must be placed ahead of slot 2");
    live_ids.remove(1);
    live_ids.insert(insert_idx, lid1);
    assert_eq!(live_ids, vec![lid1, lid2]);

    // Slot 0 fails offline: pending_swipe must remain preserved (not wiped!).
    assert!(tracker.on_failed(Some(r0), "offline"));
    assert!(tracker.try_resolve_pending_swipe().is_none());

    // Mid-restore / offline snapshot assembly preserves [slot0, slot1, slot2]
    // order AND keeps swipe pointing at (0, 2).
    let (assembled, swipe) =
        tracker.assemble_layers_and_swipe(vec![(lid1, e1.clone()), (lid2, e2.clone())], None);
    assert_eq!(assembled, vec![e0.clone(), e1.clone(), e2.clone()]);
    assert_eq!(
        swipe,
        Some(SavedSwipeState {
            active: true,
            split_t: 0.4,
            left_layer_index: Some(0),
            right_layer_index: Some(2),
        })
    );

    // If the user actively swipes live layers (lid1 vs lid2) while slot 0 is
    // still unavailable, assembled indices must be (1, 2) rather than (0, 1).
    let (_, live_swipe) = tracker.assemble_layers_and_swipe(
        vec![(lid1, e1.clone()), (lid2, e2.clone())],
        Some((0.6, Some(lid1), Some(lid2))),
    );
    assert_eq!(
        live_swipe,
        Some(SavedSwipeState {
            active: true,
            split_t: 0.6,
            left_layer_index: Some(1),
            right_layer_index: Some(2),
        })
    );

    // Retrying slot 0 preserves its original slot index 0 and resolves pending_swipe.
    let (retry_l, _) = tracker.take_unavailable_for_retry();
    assert_eq!(retry_l, vec![(Some(0), e0.clone())]);
    let r0_retry = tracker.alloc_restore_id();
    tracker.push_layer(r0_retry, retry_l[0].0, retry_l[0].1.clone());
    let lid0 = LayerId::from_raw_for_test(30);
    let slot0 = tracker.on_opened(
        Some(r0_retry),
        Some(lid0),
        RestoredEntryKind::LocalRaster,
        None,
    );
    assert_eq!(slot0, Some(0));
    assert_eq!(
        tracker.try_resolve_pending_swipe(),
        Some((0.4, Some(lid0), Some(lid2)))
    );
}
