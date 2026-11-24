use veldmap_core::tile::{BandKey, PixelFormat, PlanetaryScheme, QuadTileId, TileScheme};

use super::*;

fn wm_key(source: SourceId, level: u8, x: u32, y: u32) -> TileKey {
    TileKey::display_quad(
        source,
        QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level,
            x,
            y,
        },
    )
}

fn tile_1x1() -> DecodedTile {
    DecodedTile::try_new(1, 1, PixelFormat::Rgba8UnormSrgb, vec![1, 2, 3, 255]).expect("tile")
}

#[test]
fn pending_uploads_enforces_byte_cap_on_enqueue_and_preserves_roots() {
    let mut pending = PendingUploads::with_max_queued_bytes(10);
    let tx = pending.sender().expect("sender");
    let live_a = SourceId::from_stable_key("live-a");
    let k_live = wm_key(live_a, 1, 0, 0);
    let k_bundled = wm_key(SourceId::BUNDLED_BASEMAP, 0, 0, 0);
    let tile_4b = tile_1x1();
    tx.send((k_live, tile_4b.clone())).expect("send live");
    tx.send((k_bundled, tile_4b.clone())).expect("send bundled");

    pending.clear_source(live_a);
    assert!(!pending.contains(&k_live));
    assert!(pending.contains(&k_bundled));
    assert_eq!(pending.count_for_source(live_a), 0);
    assert_eq!(pending.count_for_source(SourceId::BUNDLED_BASEMAP), 1);
    assert_eq!(pending.queued_bytes(), 4);

    // Send finer z=3 first (8 B total <= 10 B), then coarser z=1 (would be 12 B > 10 B):
    // enqueue-time backpressure MUST immediately evict finest non-pinned z=3 without
    // requiring any UI drain, preserving pinned root z=0 and coarser z=1.
    let k_z3 = wm_key(live_a, 3, 0, 0);
    let k_z1 = wm_key(live_a, 1, 0, 0);
    tx.send((k_z3, tile_4b.clone())).expect("send z3");
    assert_eq!(pending.queued_bytes(), 8);
    assert!(pending.contains(&k_z3));

    tx.send((k_z1, tile_4b.clone())).expect("send z1");
    assert_eq!(
        pending.queued_bytes(),
        8,
        "enqueue must enforce max_queued_bytes without UI drain"
    );
    assert!(pending.contains(&k_bundled));
    assert!(pending.contains(&k_z1));
    assert!(!pending.contains(&k_z3));

    // Burst of 100 fine tiles with NO UI drain must stay strictly bounded <= 10 B.
    for x in 0..100u32 {
        let fine = wm_key(live_a, 8, x, 0);
        tx.send((fine, tile_4b.clone())).expect("send burst");
        assert!(pending.queued_bytes() <= 10);
    }
    assert!(pending.contains(&k_bundled));
    assert!(pending.contains(&k_z1));

    // Closing the queue rejects further sends and frees all queued bytes.
    pending.close();
    assert_eq!(pending.queued_bytes(), 0);
    assert!(pending.sender().is_none());
    assert_eq!(tx.send((k_z1, tile_4b)), Err(UploadChannelClosed));
}

#[test]
fn pinned_root_check_handles_planetary_and_raster_pyramid() {
    let live = SourceId::from_stable_key("eox");
    let wm_root = wm_key(live, 0, 0, 0);
    let wm_child = wm_key(live, 1, 0, 0);
    assert!(is_pinned_root(&wm_root, &[live]));
    assert!(!is_pinned_root(&wm_root, &[SourceId::BUNDLED_BASEMAP]));
    assert!(!is_pinned_root(&wm_child, &[live]));

    let pyr_root = TileKey {
        source: live,
        band: BandKey::Display,
        scheme: TileScheme::RasterPyramid { max_level: 4 },
        format: PixelFormat::Rgba8UnormSrgb,
        gutter: 0,
        decode_rev: 0,
        level: 4,
        x: 0,
        y: 0,
    };
    let pyr_finest = TileKey {
        level: 0,
        ..pyr_root
    };
    let pyr_oob = TileKey {
        level: 5,
        ..pyr_root
    };
    assert!(is_pinned_root(&pyr_root, &[live]));
    assert!(!is_pinned_root(&pyr_finest, &[live]));
    assert!(!is_pinned_root(&pyr_oob, &[live]));
}

#[test]
fn clear_where_and_proportional_budget_cover_pre_3a_hooks() {
    let mut pending = PendingUploads::new();
    let tx = pending.sender().expect("sender");
    let tile_4b = tile_1x1();
    let k0 = wm_key(SourceId::BUNDLED_BASEMAP, 0, 0, 0);
    let k2 = wm_key(SourceId::BUNDLED_BASEMAP, 2, 1, 1);
    tx.send((k0, tile_4b.clone())).expect("send k0");
    tx.send((k2, tile_4b)).expect("send k2");

    pending.clear_where(|k| k.level >= 2);
    assert!(pending.contains(&k0));
    assert!(!pending.contains(&k2));

    let cache = GpuTileCache::new(100 * 1024 * 1024);
    let weighted = cache.proportional_pyramid_byte_cap(4, 10);
    assert!(weighted > 0 && weighted * 2 < cache.max_bytes());
}

#[test]
fn dropping_pending_uploads_closes_channel_without_explicit_close() {
    let pending = PendingUploads::new();
    let tx = pending.sender().expect("sender");
    let tile = tile_1x1();
    let key = wm_key(SourceId::BUNDLED_BASEMAP, 0, 0, 0);
    tx.send((key, tile.clone())).expect("send while open");

    drop(pending);

    assert_eq!(tx.send((key, tile)), Err(UploadChannelClosed));
}
