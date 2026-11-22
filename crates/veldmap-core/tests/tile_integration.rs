//! Public-API contract tests for `veldmap-core::tile` (`TileKey`, `TileScheme`,
//! `TileCache`, carrier lookup, active-frame protection, and proportional VRAM
//! appetite budgeting).

use veldmap_core::tile::{
    BandKey, PixelFormat, SourceId, TileCache, TileKey, TileScheme, compose_uv_rect,
};

fn make_key(scheme: TileScheme, level: u8, x: u32, y: u32) -> TileKey {
    TileKey {
        source: SourceId::BUNDLED_BASEMAP,
        band: BandKey::Display,
        scheme,
        format: PixelFormat::Rgba8UnormSrgb,
        gutter: 0,
        decode_rev: 0,
        level,
        x,
        y,
    }
}

#[test]
fn tile_scheme_parent_respects_level_direction_without_inverting() {
    for scheme in [
        TileScheme::WebMercator,
        TileScheme::Geodetic,
        TileScheme::PolarCap { north: true },
        TileScheme::PolarCap { north: false },
    ] {
        assert!(scheme.is_coarsest_zero());
        assert_eq!(scheme.parent(0, 0, 0), None);
        assert_eq!(scheme.parent(0, 1, 0), None);
        assert_eq!(scheme.parent(3, 5, 7), Some((2, 2, 3)));
        assert_eq!(scheme.parent(1, 3, 1), Some((0, 1, 0)));
        assert!(scheme.coarseness_rank(0) < scheme.coarseness_rank(3));
    }

    let pyr = TileScheme::RasterPyramid { max_level: 5 };
    assert!(!pyr.is_coarsest_zero());
    assert!(pyr.as_planetary().is_none());
    assert!(pyr.as_globe().is_none());
    assert_eq!(pyr.parent(5, 0, 0), None);
    assert_eq!(pyr.parent(6, 0, 0), None);
    assert_eq!(pyr.parent(0, 5, 7), Some((1, 2, 3)));
    assert_eq!(pyr.parent(4, 1, 1), Some((5, 0, 0)));
    assert!(pyr.coarseness_rank(5) < pyr.coarseness_rank(0));

    // RasterPyramid keys never convert to planetary QuadTileId and reject
    // out-of-range levels (`level > max_level`) as pinned roots.
    let pyr_root = make_key(pyr, 5, 0, 0);
    let pyr_oob = make_key(pyr, 6, 0, 0);
    assert!(pyr_root.quad_id().is_none());
    assert!(pyr_root.is_root_of(SourceId::BUNDLED_BASEMAP));
    assert!(!pyr_oob.is_root_of(SourceId::BUNDLED_BASEMAP));
}

#[test]
fn sub_rect_in_ancestor_maps_both_planetary_and_raster_pyramid_tiles() {
    let ancestor = make_key(TileScheme::Geodetic, 2, 3, 1);
    let child = TileKey {
        level: 4,
        x: (3 << 2) + 2,
        y: (1 << 2) + 3,
        ..ancestor
    };
    assert_eq!(
        child.sub_rect_in_ancestor(&ancestor),
        Some([0.5, 0.75, 0.25, 0.25])
    );
    assert_eq!(
        ancestor.sub_rect_in_ancestor(&ancestor),
        Some([0.0, 0.0, 1.0, 1.0])
    );
    assert_eq!(ancestor.sub_rect_in_ancestor(&child), None);

    let pyr_coarse = TileKey {
        scheme: TileScheme::RasterPyramid { max_level: 4 },
        level: 3,
        x: 1,
        y: 2,
        ..ancestor
    };
    let pyr_fine = TileKey {
        level: 1,
        x: (1 << 2) + 1,
        y: (2 << 2) + 3,
        ..pyr_coarse
    };
    assert_eq!(
        pyr_fine.sub_rect_in_ancestor(&pyr_coarse),
        Some([0.25, 0.75, 0.25, 0.25])
    );
    assert_eq!(pyr_coarse.sub_rect_in_ancestor(&pyr_fine), None);
}

#[test]
fn band_key_from_name_and_source_id_are_deterministic() {
    assert_eq!(BandKey::from_name("B08"), BandKey::from_name("B08"));
    assert_ne!(BandKey::from_name("B08"), BandKey::from_name("B04"));
    assert_ne!(BandKey::from_name("SCL"), BandKey::Display);

    let s1 = SourceId::from_stable_key("eox-s2cloudless-2023-3857");
    let s2 = SourceId::from_stable_key("nasa-gibs-bluemarble-l8");
    assert_eq!(s1, SourceId::from_stable_key("eox-s2cloudless-2023-3857"));
    assert_ne!(s1, s2);
    assert_ne!(s1, SourceId::BUNDLED_BASEMAP);
    assert_ne!(s1.0 & (1u64 << 63), 0, "derived SourceId sets high bit");
}

#[test]
fn grid_dims_sub_rect_and_coarsest_first_tie_breaking_are_deterministic() {
    assert_eq!(
        TileScheme::WebMercator.grid_dims(30),
        Some((1 << 30, 1 << 30))
    );
    assert_eq!(TileScheme::WebMercator.grid_dims(31), None);
    assert_eq!(TileScheme::Geodetic.grid_dims(29), Some((1 << 30, 1 << 29)));
    assert_eq!(TileScheme::Geodetic.grid_dims(30), None);
    assert_eq!(
        TileScheme::RasterPyramid { max_level: 4 }.grid_dims(0),
        None
    );

    assert_eq!(PixelFormat::Rgba8UnormSrgb.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::R32Float.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::R8Uint.bytes_per_pixel(), 1);
    assert_eq!(PixelFormat::R16Uint.bytes_per_pixel(), 2);

    let base = make_key(TileScheme::WebMercator, 0, 0, 0);
    let other_source = TileKey {
        source: SourceId::new(2),
        ..base
    };
    assert_eq!(base.sub_rect_in_ancestor(&other_source), None);

    let guttered = TileKey { gutter: 1, ..base };
    assert_eq!(guttered.sub_rect_in_ancestor(&base), None);
    assert_eq!(guttered.sub_rect_in_ancestor(&guttered), None);

    let too_deep = TileKey { level: 31, ..base };
    assert_eq!(too_deep.sub_rect_in_ancestor(&base), None);

    let wrong_branch = TileKey {
        level: 2,
        x: 2,
        y: 0,
        ..base
    };
    let other_parent = TileKey {
        level: 1,
        x: 0,
        y: 0,
        ..base
    };
    assert_eq!(wrong_branch.sub_rect_in_ancestor(&other_parent), None);

    let pyr = TileScheme::RasterPyramid { max_level: 3 };
    assert_eq!(pyr.level_delta(1, 4), None);
    assert_eq!(pyr.level_delta(4, 3), None);
    let pyr_invalid = TileKey {
        scheme: pyr,
        level: 4,
        x: 0,
        y: 0,
        ..base
    };
    let pyr_fine = TileKey {
        level: 1,
        ..pyr_invalid
    };
    assert_eq!(pyr_fine.sub_rect_in_ancestor(&pyr_invalid), None);

    // Coarsest-first key orders z=0 before z=2, max_level before 0, interleaves
    // bands at the same (y, x) cell, and breaks ties strictly across sources/bands.
    assert!(base.coarsest_first_key() < wrong_branch.coarsest_first_key());
    let pyr_root = TileKey {
        scheme: pyr,
        level: 3,
        ..base
    };
    assert!(pyr_root.coarsest_first_key() < pyr_fine.coarsest_first_key());

    let b04_cell0 = TileKey {
        band: BandKey::Index(4),
        format: PixelFormat::R32Float,
        x: 0,
        y: 0,
        ..pyr_fine
    };
    let b08_cell0 = TileKey {
        band: BandKey::Index(8),
        ..b04_cell0
    };
    let b04_cell1 = TileKey { x: 1, ..b04_cell0 };
    assert!(b04_cell0.coarsest_first_key() < b08_cell0.coarsest_first_key());
    assert!(b08_cell0.coarsest_first_key() < b04_cell1.coarsest_first_key());
    assert_ne!(base.coarsest_first_key(), other_source.coarsest_first_key());

    assert_eq!(
        compose_uv_rect([0.5, 0.25, 0.5, 0.5], [0.5, 0.5, 0.5, 0.25]),
        [0.75, 0.5, 0.25, 0.125]
    );
}

#[test]
fn byte_cap_evicts_lru_and_preserves_total_bytes_invariant() {
    let mut cache = TileCache::new(300);
    let k1 = make_key(TileScheme::Geodetic, 1, 0, 0);
    let k2 = make_key(TileScheme::Geodetic, 1, 1, 0);
    let k3 = make_key(TileScheme::Geodetic, 1, 2, 0);
    let k4 = make_key(TileScheme::Geodetic, 1, 3, 0);

    assert!(cache.insert(k1, 1u32, 100).expect("insert k1").is_empty());
    assert!(cache.insert(k2, 2u32, 100).expect("insert k2").is_empty());
    assert!(cache.insert(k3, 3u32, 100).expect("insert k3").is_empty());
    assert_eq!(cache.total_bytes(), 300);
    assert_eq!(cache.len(), 3);

    assert_eq!(cache.get(&k1), Some(&1));

    let evicted = cache.insert(k4, 4u32, 100).expect("insert k4");
    assert_eq!(evicted, vec![(k2, 2)]);
    assert_eq!(cache.total_bytes(), 300);
    assert!(!cache.contains(&k2));
    assert!(cache.contains(&k1));
    assert!(cache.contains(&k3));
    assert!(cache.contains(&k4));

    let replaced = cache.insert(k3, 30u32, 150).expect("replace k3");
    assert_eq!(replaced, vec![(k3, 3), (k1, 1)]);
    assert_eq!(cache.total_bytes(), 250);
    assert_eq!(cache.peek(&k3), Some(&30));

    let k_big = make_key(TileScheme::Geodetic, 2, 0, 0);
    let rejected = cache.insert(k_big, 99, 500);
    assert_eq!(rejected, Err((k_big, 99)));
    assert_eq!(cache.total_bytes(), 250);
    assert_eq!(cache.len(), 2);

    let mut zero_cache: TileCache<TileKey, u32> = TileCache::new(0);
    assert_eq!(zero_cache.insert(k1, 1, 0), Err((k1, 1)));

    let mut max_cache: TileCache<TileKey, u32> = TileCache::new(usize::MAX);
    let half_plus = (usize::MAX / 2) + 1;
    assert!(max_cache.insert(k1, 1, half_plus).expect("k1").is_empty());
    let ev = max_cache.insert(k2, 2, half_plus).expect("k2");
    assert_eq!(ev, vec![(k1, 1)]);
    assert_eq!(max_cache.total_bytes(), half_plus);
}

#[test]
fn remove_and_clear_maintain_intrusive_links_and_carrier_misses() {
    let mut cache = TileCache::new(400);
    let k1 = make_key(TileScheme::Geodetic, 1, 0, 0);
    let k2 = make_key(TileScheme::Geodetic, 1, 1, 0);
    let k3 = make_key(TileScheme::Geodetic, 1, 2, 0);
    assert!(cache.carrier(&k1).is_none());

    cache.insert(k1, "a", 100).expect("k1");
    cache.insert(k2, "b", 100).expect("k2");
    cache.insert(k3, "c", 100).expect("k3");

    assert_eq!(cache.remove(&k2), Some(("b", 100)));
    assert_eq!(cache.remove(&k3), Some(("c", 100)));
    assert_eq!(cache.remove(&k1), Some(("a", 100)));
    assert!(cache.is_empty());
    assert_eq!(cache.total_bytes(), 0);
    assert_eq!(cache.remove(&k1), None);

    cache.insert(k1, "x", 50).expect("re-insert k1");
    cache.insert(k2, "y", 50).expect("re-insert k2");
    let cleared = cache.clear();
    assert_eq!(cleared.len(), 2);
    assert!(cache.is_empty());
    assert_eq!(cache.total_bytes(), 0);
}

#[test]
fn ancestor_touch_protects_root_for_both_xyz_and_raster_pyramid() {
    let mut geo_cache = TileCache::new(300);
    let root = make_key(TileScheme::WebMercator, 0, 0, 0);
    let mid = make_key(TileScheme::WebMercator, 1, 1, 0);
    let decoy = make_key(TileScheme::WebMercator, 2, 0, 0);
    geo_cache.insert(root, "root", 100).expect("root");
    geo_cache.insert(mid, "mid", 100).expect("mid");
    geo_cache.insert(decoy, "decoy", 100).expect("decoy");

    let leaf = make_key(TileScheme::WebMercator, 3, 4, 0);
    let (carrier_key, uv, val) = geo_cache.carrier(&leaf).expect("carrier found");
    assert_eq!(carrier_key, mid);
    assert_eq!(*val, "mid");
    assert_eq!(uv, [0.0, 0.0, 0.25, 0.25]);

    let newcomer = make_key(TileScheme::WebMercator, 2, 3, 3);
    let evicted = geo_cache.insert(newcomer, "new", 100).expect("newcomer");
    assert_eq!(evicted, vec![(decoy, "decoy")]);
    assert!(geo_cache.contains(&root));
    assert!(geo_cache.contains(&mid));

    let pyr = TileScheme::RasterPyramid { max_level: 3 };
    let mut pyr_cache = TileCache::new(300);
    let p_root = make_key(pyr, 3, 0, 0);
    let p_mid = make_key(pyr, 2, 1, 1);
    let p_decoy = make_key(pyr, 1, 0, 0);
    pyr_cache.insert(p_root, "p_root", 100).expect("p_root");
    pyr_cache.insert(p_mid, "p_mid", 100).expect("p_mid");
    pyr_cache.insert(p_decoy, "p_decoy", 100).expect("p_decoy");

    let p_leaf = make_key(pyr, 0, 5, 6);
    let (p_carrier, p_uv, p_val) = pyr_cache.carrier(&p_leaf).expect("pyramid carrier");
    assert_eq!(p_carrier, p_mid);
    assert_eq!(*p_val, "p_mid");
    assert_eq!(p_uv, [0.25, 0.5, 0.25, 0.25]);

    let p_new = make_key(pyr, 1, 3, 3);
    let p_evicted = pyr_cache.insert(p_new, "p_new", 100).expect("p_new");
    assert_eq!(p_evicted, vec![(p_decoy, "p_decoy")]);
    assert!(pyr_cache.contains(&p_root));
    assert!(pyr_cache.contains(&p_mid));

    let g_root = TileKey {
        gutter: 1,
        ..p_root
    };
    let g_leaf = TileKey {
        gutter: 1,
        ..p_leaf
    };
    let mut g_cache = TileCache::new(200);
    g_cache.insert(g_root, "g_root", 100).expect("g_root");
    assert!(g_cache.carrier(&g_leaf).is_none());
    let (found_key, found_val) = g_cache.find_carrier(&g_leaf).expect("guttered carrier");
    assert_eq!(found_key, g_root);
    assert_eq!(*found_val, "g_root");
}

#[test]
fn appetite_cap_and_proportional_budget_stay_strictly_below_half() {
    let cache: TileCache<TileKey, u8> = TileCache::new(256 * 1024 * 1024);
    let cap1 = cache.pyramid_byte_cap(1);
    let cap2 = cache.pyramid_byte_cap(2);
    let cap4 = cache.pyramid_byte_cap(4);

    assert!(cap1 * 2 < cache.max_bytes());
    assert!(cap2 * 4 < cache.max_bytes());
    assert!(cap4 * 8 < cache.max_bytes());
    assert!(cache.max_tiles_for_pyramid(1, 256 * 256 * 4) < 512);

    // Heterogeneous multi-band weights: basemap (349 KiB) + 3x R32Float (1 MiB each) + SCL (256 KiB).
    let w_base = 350_000u64;
    let w_f32 = 1_048_576u64;
    let w_scl = 262_144u64;
    let total_w = w_base + 3 * w_f32 + w_scl;
    let c_base = cache.proportional_pyramid_byte_cap(w_base, total_w);
    let c_f32 = cache.proportional_pyramid_byte_cap(w_f32, total_w);
    let c_scl = cache.proportional_pyramid_byte_cap(w_scl, total_w);
    assert!(c_f32 > c_base && c_base > c_scl);
    assert!((c_base + 3 * c_f32 + c_scl) * 2 < cache.max_bytes());
    assert!(
        cache.max_tiles_for_weighted_pyramid(w_f32, total_w, w_f32 as usize) > 0,
        "R32Float band must receive proportional tile slots"
    );
    assert_eq!(cache.proportional_pyramid_byte_cap(0, total_w), 0);

    let small: TileCache<TileKey, u8> = TileCache::new(1000);
    let per_pyr = small.max_tiles_for_pyramid(4, 200);
    assert_eq!(per_pyr, 0);
    assert!(per_pyr * 4 * 200 * 2 < small.max_bytes());
}

#[test]
fn active_frame_protection_and_remove_where_preserve_invariants() {
    let mut cache = TileCache::new(300);
    let k1 = make_key(TileScheme::WebMercator, 1, 0, 0);
    let k2 = make_key(TileScheme::WebMercator, 1, 1, 0);
    let k3 = make_key(TileScheme::WebMercator, 1, 0, 1);
    let k4 = make_key(TileScheme::WebMercator, 1, 1, 1);
    let other = TileKey {
        source: SourceId::from_stable_key("other-source"),
        ..k1
    };

    cache.insert(k1, "k1", 100).expect("k1");
    cache.insert(k2, "k2", 100).expect("k2");
    cache.insert(k3, "k3", 100).expect("k3");

    let ev = cache
        .insert_protecting(k4, "k4", 100, |k| *k == k1)
        .expect("insert skipping protected");
    assert_eq!(ev, vec![(k2, "k2")]);
    assert!(cache.contains(&k1));
    assert!(!cache.contains(&k2));
    assert!(cache.contains(&k3));
    assert!(cache.contains(&k4));
    assert_eq!(cache.total_bytes(), 300);

    let rejected = cache.insert_protecting(other, "other", 100, |_| true);
    assert_eq!(rejected, Err((other, "other")));
    assert_eq!(cache.len(), 3);
    assert_eq!(cache.total_bytes(), 300);
    assert!(cache.contains(&k1));

    let rejected_replace = cache.insert_protecting(k3, "k3_big", 200, |k| *k != k3);
    assert_eq!(rejected_replace, Err((k3, "k3_big")));
    assert_eq!(cache.peek(&k3), Some(&"k3"));
    assert_eq!(cache.total_bytes(), 300);

    cache
        .insert_protecting(other, "other", 100, |k| *k == k1)
        .expect("insert other");
    let removed = cache.remove_where(|k| k.source == SourceId::BUNDLED_BASEMAP);
    assert_eq!(removed.len(), 2);
    assert_eq!(cache.len(), 1);
    assert!(cache.contains(&other));
    assert_eq!(cache.total_bytes(), 100);
}
