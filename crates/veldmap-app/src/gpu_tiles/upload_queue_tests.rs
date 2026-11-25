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
    let live_display = StreamKey::new(live, BandKey::Display, PixelFormat::Rgba8UnormSrgb, 0);
    let bundled_display = StreamKey::new(
        SourceId::BUNDLED_BASEMAP,
        BandKey::Display,
        PixelFormat::Rgba8UnormSrgb,
        0,
    );
    let wm_root = wm_key(live, 0, 0, 0);
    let wm_child = wm_key(live, 1, 0, 0);
    assert!(is_pinned_root(&wm_root, &[live_display]));
    assert!(!is_pinned_root(&wm_root, &[bundled_display]));
    assert!(!is_pinned_root(&wm_child, &[live_display]));

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
    let pyr_band1_root = TileKey {
        band: BandKey::Index(1),
        format: PixelFormat::R32Float,
        ..pyr_root
    };
    assert!(is_pinned_root(&pyr_root, &[live_display]));
    assert!(!is_pinned_root(&pyr_finest, &[live_display]));
    assert!(!is_pinned_root(&pyr_oob, &[live_display]));
    // Pinning Display stream must NOT pin an obsolete Band 1 root on the same SourceId.
    assert!(!is_pinned_root(&pyr_band1_root, &[live_display]));
    assert!(is_pinned_root(
        &pyr_band1_root,
        &[pyr_band1_root.stream_key()]
    ));
}

#[test]
fn clear_where_and_proportional_budget_filter_and_cap_correctly() {
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

#[test]
fn upload_batch_report_isolates_stream_keys_and_tile_format_bytes() {
    use crate::gpu_tiles::tile_format_bytes;
    use veldmap_core::tile::StreamKey;

    let display_bytes = tile_format_bytes(512, PixelFormat::Rgba8UnormSrgb);
    let sci_bytes = tile_format_bytes(512, PixelFormat::R32Float);
    let cat8_bytes = tile_format_bytes(512, PixelFormat::R8Uint);
    let cat16_bytes = tile_format_bytes(512, PixelFormat::R16Uint);
    assert_eq!(display_bytes, sci_bytes, "both 4 B/px with full mip chain");
    assert!(sci_bytes > 512 * 512 * 4);
    assert_eq!(cat8_bytes, 512 * 512, "R8Uint is 1-mip 1 B/px");
    assert_eq!(cat16_bytes, 512 * 512 * 2, "R16Uint is 1-mip 2 B/px");

    let shared_src = SourceId::from_stable_key("multi-band-sentinel");
    let sci_stream = StreamKey::new(shared_src, BandKey::Index(0), PixelFormat::R32Float, 0);
    let cat_stream = StreamKey::new(shared_src, BandKey::Index(1), PixelFormat::R8Uint, 0);
    let mut report = UploadBatchReport::default();
    report.record_stream_tile_bytes(sci_stream, sci_bytes);
    report.record_stream_tile_bytes(cat_stream, cat8_bytes);
    assert_eq!(report.max_tile_bytes_for_stream(sci_stream), sci_bytes);
    assert_eq!(report.max_tile_bytes_for_stream(cat_stream), cat8_bytes);
    assert_ne!(sci_bytes, cat8_bytes);

    let mut cache = GpuTileCache::new(512 * 1024 * 1024);
    assert_eq!(cache.max_bytes(), 512 * 1024 * 1024);
    cache.set_max_bytes(256 * 1024 * 1024);
    assert_eq!(cache.max_bytes(), 256 * 1024 * 1024);
}
