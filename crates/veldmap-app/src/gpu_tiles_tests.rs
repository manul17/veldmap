use veldmap_core::tile::{BandKey, PlanetaryScheme, TileScheme};

use super::*;

#[test]
fn pending_uploads_clear_source_and_queued_byte_cap_preserve_roots() {
    let mut pending = PendingUploads::with_max_queued_bytes(10);
    let tx = pending.sender().expect("sender");
    let live_a = SourceId::from_stable_key("live-a");
    let k_live = source_tile_key(
        live_a,
        QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 1,
            x: 0,
            y: 0,
        },
    );
    let k_bundled = bundled_tile_key(QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 0,
        x: 0,
        y: 0,
    });
    let tile_4b =
        DecodedTile::try_new(1, 1, PixelFormat::Rgba8UnormSrgb, vec![1, 2, 3, 255]).expect("tile");
    tx.send((k_live, tile_4b.clone())).expect("send live");
    tx.send((k_bundled, tile_4b.clone())).expect("send bundled");

    pending.clear_source(live_a);
    assert!(!pending.contains(&k_live));
    assert!(pending.contains(&k_bundled));
    assert_eq!(pending.count_for_source(live_a), 0);
    assert_eq!(pending.count_for_source(SourceId::BUNDLED_BASEMAP), 1);
    assert_eq!(pending.queued_bytes(), 4);

    // Push two finer non-pinned tiles (4 B each -> 12 B > 10 B cap):
    // enforcing the cap must drop the finest non-pinned tile (z=3) while
    // keeping the pinned root (z=0) and coarser z=1 tile (8 B <= 10 B).
    let k_z1 = source_tile_key(
        live_a,
        QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 1,
            x: 0,
            y: 0,
        },
    );
    let k_z3 = source_tile_key(
        live_a,
        QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 3,
            x: 0,
            y: 0,
        },
    );
    pending.insert_queued(k_z1, tile_4b.clone());
    pending.insert_queued(k_z3, tile_4b);
    assert_eq!(pending.queued_bytes(), 12);

    pending.enforce_queued_byte_cap(&[SourceId::BUNDLED_BASEMAP]);
    assert_eq!(pending.queued_bytes(), 8);
    assert!(pending.contains(&k_bundled));
    assert!(pending.contains(&k_z1));
    assert!(!pending.contains(&k_z3));
}

#[test]
fn pinned_root_check_handles_planetary_and_raster_pyramid() {
    let live = SourceId::from_stable_key("eox");
    let wm_root = source_tile_key(
        live,
        QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 0,
            x: 0,
            y: 0,
        },
    );
    let wm_child = source_tile_key(
        live,
        QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 1,
            x: 0,
            y: 0,
        },
    );
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
