use super::*;
use crate::colormap::ColormapId;
use crate::layers::{CrsOverride, LayerStyleSpec, StretchPreset};

#[test]
fn session_restore_tracker_preserves_unavailable_layers_deduplicates_on_reopen_and_supports_forget()
{
    let mut tracker = SessionRestoreTracker::default();
    let raster_a = PathBuf::from("/mnt/nas/a.tif");
    let raster_b = PathBuf::from("/mnt/nas/b.tif");
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
    let saved_a = SavedLayerEntry::with_style(
        raster_a.clone(),
        false,
        0.5,
        Some(&CrsOverride::Epsg(32637)),
        Some(&custom_style),
    );

    tracker.push_layer(saved_a.clone());
    tracker.push_layer(SavedLayerEntry::new(raster_a.clone(), true, 1.0, None));
    tracker.push_layer(SavedLayerEntry::new(raster_b.clone(), true, 0.8, None));
    tracker.push_vector(SavedVectorEntry {
        path: vector_a.clone(),
        visible: true,
        opacity: 0.7,
    });

    // Non-existent path failure is ignored (not a restored entry).
    assert!(!tracker.on_failed(
        RestoredEntryKind::Raster,
        Path::new("/tmp/other.tif"),
        "not found"
    ));
    assert!(!tracker.on_failed(RestoredEntryKind::Vector, &raster_a, "wrong kind"));

    // Failing the first duplicate `raster_a` moves its full CRS/style entry to
    // `unavailable` while keeping it in `preserved_layers()`.
    assert!(tracker.on_failed(RestoredEntryKind::Raster, &raster_a, "drive offline"));
    assert_eq!(tracker.unavailable_count(), 1);
    assert_eq!(tracker.visible_notices().len(), 1);
    assert_eq!(tracker.visible_notices()[0].path, raster_a);
    assert_eq!(tracker.preserved_layers().count(), 3);
    let preserved_a = tracker
        .preserved_layers()
        .find(|e| e.path == raster_a && !e.visible)
        .expect("unavailable entry preserved");
    assert_eq!(preserved_a.crs_override(), Some(CrsOverride::Epsg(32637)));
    assert_eq!(preserved_a.style(), Some(custom_style));

    // Second duplicate `raster_a` opening consumes from `pending` first,
    // leaving the unavailable copy intact.
    tracker.on_opened(RestoredEntryKind::Raster, &raster_a);
    assert_eq!(tracker.unavailable_count(), 1);
    assert_eq!(tracker.preserved_layers().count(), 2);

    // A subsequent manual open of `raster_a` (no pending entry left) clears the
    // unavailable entry so `session.json` does not duplicate it.
    tracker.on_opened(RestoredEntryKind::Raster, &raster_a);
    assert_eq!(tracker.unavailable_count(), 0);
    assert_eq!(tracker.preserved_layers().count(), 1);
    assert_eq!(tracker.preserved_layers().next().expect("b").path, raster_b);

    // Failing vector moves it to unavailable; hiding silences UI notice without
    // losing the preserved entry or toolbar count.
    assert!(tracker.on_failed(RestoredEntryKind::Vector, &vector_a, "missing"));
    assert_eq!(tracker.unavailable_count(), 1);
    tracker.hide_notices();
    assert!(tracker.visible_notices().is_empty());
    assert_eq!(tracker.preserved_vectors().count(), 1);
    tracker.unhide_notices();
    assert_eq!(tracker.visible_notices().len(), 1);

    // Retry hands unavailable entries back cleanly, and Forget drops unavailable
    // entries while keeping still-pending entries untouched.
    let (retry_l, retry_v) = tracker.take_unavailable_for_retry();
    assert!(retry_l.is_empty());
    assert_eq!(retry_v.len(), 1);
    assert_eq!(tracker.unavailable_count(), 0);

    tracker.record_unavailable_vector(retry_v[0].clone(), "still offline".to_owned());
    assert!(tracker.forget_unavailable());
    assert!(!tracker.forget_unavailable());
    assert_eq!(tracker.unavailable_count(), 0);
    assert_eq!(tracker.preserved_vectors().count(), 0);
    assert_eq!(tracker.preserved_layers().count(), 1);
}
