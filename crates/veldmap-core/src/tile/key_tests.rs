use super::*;
use crate::tile::{SourceFingerprint, make_key};

#[test]
fn tile_scheme_parent_respects_level_direction_without_inverting() {
    for scheme in [
        TileScheme::WebMercator,
        TileScheme::Geodetic,
        TileScheme::PolarCap { north: true },
        TileScheme::PolarCap { north: false },
    ] {
        assert!(scheme.as_planetary().is_some());
        assert_eq!(scheme.parent(0, 0, 0), None);
        assert_eq!(scheme.parent(0, 1, 0), None);
        assert_eq!(scheme.parent(3, 5, 7), Some((2, 2, 3)));
        assert_eq!(scheme.parent(1, 3, 1), Some((0, 1, 0)));
        assert!(scheme.coarseness_rank(0) < scheme.coarseness_rank(3));
    }

    let pyr = TileScheme::RasterPyramid { max_level: 5 };
    assert!(pyr.as_planetary().is_none());
    assert_eq!(pyr.parent(5, 0, 0), None);
    assert_eq!(pyr.parent(6, 0, 0), None);
    assert_eq!(pyr.parent(0, 5, 7), Some((1, 2, 3)));
    assert_eq!(pyr.parent(4, 1, 1), Some((5, 0, 0)));
    assert!(pyr.coarseness_rank(5) < pyr.coarseness_rank(0));

    // Fingerprint formatting & direct BLAKE3 -> SourceId derivation.
    let fp = SourceFingerprint::from_digest([0xAB; 32]);
    let hex = fp.to_hex();
    assert_eq!(hex.len(), 64);
    assert!(
        hex.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    );
    assert!(format!("{fp:?}").contains("abababababab.."));
    assert_eq!(
        SourceId::from_fingerprint(&fp).0,
        u64::from_le_bytes([0xAB; 8]) | (1u64 << 63)
    );

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
        PlanetaryScheme::WebMercator.grid_dims(30),
        Some((1 << 30, 1 << 30))
    );
    assert_eq!(PlanetaryScheme::WebMercator.grid_dims(31), None);
    assert_eq!(
        PlanetaryScheme::Geodetic.grid_dims(29),
        Some((1 << 30, 1 << 29))
    );
    assert_eq!(PlanetaryScheme::Geodetic.grid_dims(30), None);

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
    // Out-of-range RasterPyramid levels (`level > max_level`) both saturate
    // `coarseness_rank` to `0`, yet must remain strictly distinct under `Ord`
    // to preserve `Ord`/`Eq` consistency.
    assert_ne!(
        pyr_root.coarsest_first_key(),
        pyr_invalid.coarsest_first_key()
    );

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

    let fp = SourceFingerprint::from_digest([0xAB; 32]);
    let fp_id = SourceId::from_fingerprint(&fp);
    assert_ne!(fp_id, SourceId::BUNDLED_BASEMAP);
    assert_eq!(SourceId::from(&fp), fp_id);
}

#[test]
fn clamp_to_level_handles_large_level_delta_without_panicking() {
    let deep = QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 40,
        x: 0,
        y: 0,
    };
    let (clamped, uv) = deep.clamp_to_level(0);
    assert_eq!(clamped.level, 0);
    assert_eq!(uv, [0.0, 0.0, 0.0, 0.0]);
}
