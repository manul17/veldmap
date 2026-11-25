//! Integration tests for GeoTIFF user-defined GeoKeys (`3072 == 32767`,
//! `3076` feet, `3081` origin lat, `2062` TOWGS84, polar stereographic,
//! spherical ellipsoids, LCC 1SP/2SP), foreign-datum geographic GeoTIFF datum
//! shifts, and `.prj`-only sidecar binding.

use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::{
    CrsSpec, DatumWarning, KRASOVSKY, PULKOVO_1942, Projection, US_SURVEY_FOOT_M,
};
use veldmap_core::io::ByteSource;
use veldmap_core::io::testing::MemoryByteSource;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::SourceFingerprint;
use veldmap_raster::{
    PlacementCrs, RasterSource, SidecarGeoref, TiffSource, frame_from_description,
};

use crate::common::{build_rgb_4x4_tiff, tag_f64_vec, tag_u16_vec};

#[test]
fn geotiff_user_defined_32767_tm_feet_origin_lat_and_foreign_datum_shift() {
    // 1) User-defined projected CRS (3072 = 32767), CT_TransverseMercator (3075 = 1),
    //    US Survey Feet (3076 = 9003), origin_lat = 38.0 (3081), lon_0 = -75.0 (3080),
    //    false_easting = 656166.6667 ftUS (~200 km), scale = 0.9999 (3092).
    let fe_ft = 200_000.0 / US_SURVEY_FOOT_M;
    let doubles = [
        -75.0_f64,  // idx 0: lon_0
        38.0_f64,   // idx 1: lat_0
        fe_ft,      // idx 2: false easting (in ftUS!)
        0.0_f64,    // idx 3: false northing
        0.9999_f64, // idx 4: scale
    ];
    let geokeys: Vec<u16> = vec![
        1, 1, 0, 10, // header: 10 keys
        1024, 0, 1, 1, // ModelType = Projected
        1025, 0, 1, 1, // RasterType = PixelIsArea
        2048, 0, 1, 4326, // Geographic = WGS84
        3072, 0, 1, 32767, // Projected = User-Defined
        3075, 0, 1, 1, // CoordTrans = Transverse Mercator
        3076, 0, 1, 9003, // LinearUnits = US Survey Foot
        3080, 34736, 1, 0, // NatOriginLong
        3081, 34736, 1, 1, // NatOriginLat
        3082, 34736, 1, 2, // FalseEasting
        3092, 34736, 1, 4, // ScaleAtNatOrigin
    ];
    let tiff_bytes = build_rgb_4x4_tiff(
        &[
            tag_f64_vec(33550, &[10.0, 10.0, 0.0]), // 10 ftUS/px
            tag_f64_vec(33922, &[0.0, 0.0, 0.0, fe_ft, 20.0, 0.0]), // UL at (fe_ft, 20 ftUS)
            tag_u16_vec(34735, &geokeys),
            tag_f64_vec(34736, &doubles),
        ],
        200,
    );
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let src: Arc<dyn ByteSource> = Arc::new(MemoryByteSource::new(tiff_bytes.into()));
    let tiff = TiffSource::open(
        src,
        SourceFingerprint::from_digest([3; 32]),
        &budget,
        &cancel,
    )
    .expect("user-defined GeoTIFF opens");
    let frame = frame_from_description(tiff.description()).expect("user-defined frame binds");
    let (lat, lon) = frame.geodetic(0.0, 0.5);
    assert!((lat - 38.0).abs() < 1e-4, "expected lat ~38.0, got {lat}");
    assert!((lon + 75.0).abs() < 1e-4, "expected lon ~-75.0, got {lon}");
    // Pixel size must reflect US Survey Feet (~3.048 m/px, not 10 m/px!).
    let (px_x, px_y) = frame.pixel_size_m(4, 4).expect("pixel size");
    assert!((px_x - 10.0 * US_SURVEY_FOOT_M).abs() < 0.05, "got {px_x}");
    assert!((px_y - 10.0 * US_SURVEY_FOOT_M).abs() < 0.05, "got {px_y}");

    // 2) Foreign-datum geographic GeoTIFF (EPSG:4284 Pulkovo 1942) shifts to WGS84.
    let geokeys_p42: Vec<u16> = vec![
        1, 1, 0, 3, 1024, 0, 1, 2, // Geographic
        1025, 0, 1, 1, // PixelIsArea
        2048, 0, 1, 4284, // Pulkovo 1942
    ];
    let p42_bytes = build_rgb_4x4_tiff(
        &[
            tag_f64_vec(33550, &[0.01, 0.01, 0.0]),
            tag_f64_vec(33922, &[0.0, 0.0, 0.0, 39.70, 47.25, 0.0]),
            tag_u16_vec(34735, &geokeys_p42),
        ],
        200,
    );
    let p42_src: Arc<dyn ByteSource> = Arc::new(MemoryByteSource::new(p42_bytes.into()));
    let p42_tiff = TiffSource::open(
        p42_src,
        SourceFingerprint::from_digest([4; 32]),
        &budget,
        &cancel,
    )
    .expect("Pulkovo 42 geographic GeoTIFF opens");
    let p42_frame = frame_from_description(p42_tiff.description()).expect("P42 frame binds");
    let got = p42_frame.geodetic(0.5, 0.5);
    let expected = PULKOVO_1942.to_wgs84(47.23, 39.72);
    assert!((got.0 - expected.0).abs() < 1e-6);
    assert!((got.1 - expected.1).abs() < 1e-6);
}

#[test]
fn user_defined_geotiff_handles_datum_ellipsoid_polar_stere_sphere_and_warnings() {
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();

    // 1) GeoKey 2048=4284 (Pulkovo 1942) + GeoKey 2062 (TOWGS84) without 2056:
    //    must preserve Krasovsky ellipsoid (semi_major 6378245) rather than WGS84!
    let doubles_p42 = [
        39.0_f64,  // 0: lon_0
        0.0_f64,   // 1: lat_0
        500_000.0, // 2: false_easting
        1.0,       // 3: scale
        23.57, -140.95, -79.8, 0.0, -0.35, -0.79, -0.22, // 4..11: TOWGS84
    ];
    let geokeys_p42: Vec<u16> = vec![
        1, 1, 0, 10, 1024, 0, 1, 1, 1025, 0, 1, 1, 2048, 0, 1,
        4284, // Pulkovo 1942 (Krasovsky ellipsoid)
        2062, 34736, 7, 4, // GeogTOWGS84GeoKey
        3072, 0, 1, 32767, 3075, 0, 1, 1, // CT_TransverseMercator
        3080, 34736, 1, 0, 3081, 34736, 1, 1, 3082, 34736, 1, 2, 3092, 34736, 1, 3,
    ];
    let tiff_p42 = TiffSource::open(
        Arc::new(MemoryByteSource::new(
            build_rgb_4x4_tiff(
                &[
                    tag_f64_vec(33550, &[10.0, 10.0, 0.0]),
                    tag_f64_vec(33922, &[0.0, 0.0, 0.0, 500_000.0, 5_230_000.0, 0.0]),
                    tag_u16_vec(34735, &geokeys_p42),
                    tag_f64_vec(34736, &doubles_p42),
                ],
                180,
            )
            .into(),
        )),
        SourceFingerprint::from_digest([10; 32]),
        &budget,
        &cancel,
    )
    .expect("Pulkovo 42 + TOWGS84 user-defined TIFF opens");
    let placement_p42 = tiff_p42
        .description()
        .placement
        .as_ref()
        .expect("geo placement");
    match placement_p42.crs.as_ref() {
        Some(PlacementCrs::Resolved {
            crs: CrsSpec::Projected(Projection::Transverse { system: sys, .. }),
            ..
        }) => {
            assert!(
                (sys.datum.ellipsoid.semi_major_m - KRASOVSKY.semi_major_m).abs() < 1e-3,
                "expected Krasovsky ellipsoid, got {:?}",
                sys.datum.ellipsoid
            );
        }
        other => panic!("expected Transverse Mercator with Krasovsky ellipsoid, got {other:?}"),
    }

    // 2) CT_PolarStereographic (3075=15) using StraightVertPoleLong (3095) and
    //    southern latitude of true scale (-71.0): must project onto southern pole.
    let doubles_ps = [
        -71.0_f64, // 0: NatOriginLat (lat_ts = -71S)
        0.0_f64,   // 1: StraightVertPoleLong (lon_0 = 0)
    ];
    let geokeys_ps: Vec<u16> = vec![
        1, 1, 0, 7, 1024, 0, 1, 1, 1025, 0, 1, 1, 2048, 0, 1, 4326, 3072, 0, 1, 32767, 3075, 0, 1,
        15, // CT_PolarStereographic
        3081, 34736, 1, 0, // NatOriginLat (-71)
        3095, 34736, 1, 1, // StraightVertPoleLong (0)
    ];
    let tiff_ps = TiffSource::open(
        Arc::new(MemoryByteSource::new(
            build_rgb_4x4_tiff(
                &[
                    tag_f64_vec(33550, &[1000.0, 1000.0, 0.0]),
                    tag_f64_vec(33922, &[0.0, 0.0, 0.0, 0.0, 1_000_000.0, 0.0]),
                    tag_u16_vec(34735, &geokeys_ps),
                    tag_f64_vec(34736, &doubles_ps),
                ],
                180,
            )
            .into(),
        )),
        SourceFingerprint::from_digest([11; 32]),
        &budget,
        &cancel,
    )
    .expect("Polar Stereographic user-defined TIFF opens");
    let frame_ps =
        frame_from_description(tiff_ps.description()).expect("Polar Stereographic binds");
    let (lat_ps, _lon_ps) = frame_ps.geodetic(0.5, 0.5);
    assert!(
        lat_ps < -75.0 && lat_ps > -85.0,
        "Antarctic polar stereographic (x=0, y=1000 km) should lie around -81S, got {lat_ps}"
    );

    // 3) Spherical ellipsoid (a == b = 6378137, flattening == 0) in user-defined Mercator:
    //    must emit +b= instead of +rf=inf and bind cleanly!
    let doubles_sph = [6_378_137.0_f64, 6_378_137.0_f64];
    let geokeys_sph: Vec<u16> = vec![
        1, 1, 0, 7, 1024, 0, 1, 1, 1025, 0, 1, 1, 2057, 34736, 1, 0, // SemiMajorAxis
        2058, 34736, 1, 1, // SemiMinorAxis (sphere!)
        3072, 0, 1, 32767, 3075, 0, 1, 7, // CT_Mercator
        3076, 0, 1, 9001,
    ];
    let tiff_sph = TiffSource::open(
        Arc::new(MemoryByteSource::new(
            build_rgb_4x4_tiff(
                &[
                    tag_f64_vec(33550, &[100.0, 100.0, 0.0]),
                    tag_f64_vec(33922, &[0.0, 0.0, 0.0, 0.0, 1000.0, 0.0]),
                    tag_u16_vec(34735, &geokeys_sph),
                    tag_f64_vec(34736, &doubles_sph),
                ],
                180,
            )
            .into(),
        )),
        SourceFingerprint::from_digest([12; 32]),
        &budget,
        &cancel,
    )
    .expect("Spherical user-defined Mercator TIFF opens");
    let frame_sph =
        frame_from_description(tiff_sph.description()).expect("Spherical Mercator binds");
    let (lat_sph, lon_sph) = frame_sph.geodetic(0.5, 0.5);
    assert!(lat_sph.abs() < 0.1 && lon_sph.abs() < 0.1);

    // 4) User-defined TM on NAD83 (GeoKey 2048 = 4269) must attach DatumWarning::GridShiftRecommended.
    let doubles_nad = [-75.0_f64, 38.0_f64, 200_000.0_f64, 0.9999_f64];
    let geokeys_nad: Vec<u16> = vec![
        1, 1, 0, 9, 1024, 0, 1, 1, 1025, 0, 1, 1, 2048, 0, 1, 4269, // NAD83
        3072, 0, 1, 32767, 3075, 0, 1, 1, // CT_TransverseMercator
        3080, 34736, 1, 0, 3081, 34736, 1, 1, 3082, 34736, 1, 2, 3092, 34736, 1, 3,
    ];
    let tiff_nad = TiffSource::open(
        Arc::new(MemoryByteSource::new(
            build_rgb_4x4_tiff(
                &[
                    tag_f64_vec(33550, &[10.0, 10.0, 0.0]),
                    tag_f64_vec(33922, &[0.0, 0.0, 0.0, 200_000.0, 100.0, 0.0]),
                    tag_u16_vec(34735, &geokeys_nad),
                    tag_f64_vec(34736, &doubles_nad),
                ],
                180,
            )
            .into(),
        )),
        SourceFingerprint::from_digest([13; 32]),
        &budget,
        &cancel,
    )
    .expect("NAD83 user-defined TIFF opens");
    let frame_nad = frame_from_description(tiff_nad.description()).expect("NAD83 frame binds");
    assert!(
        matches!(
            frame_nad.datum_warning(),
            Some(DatumWarning::GridShiftRecommended { .. })
        ),
        "expected GridShiftRecommended warning for NAD83 user-defined CRS, got {:?}",
        frame_nad.datum_warning()
    );
}

#[test]
fn user_defined_geotiff_handles_false_origin_lcc_1sp_2sp_and_2048_32767_datum_fallback() {
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();

    // 1) GeoKey 2048=32767 (User-Defined) + 2050=6284 (Pulkovo 1942): must fall
    //    through to GeoKey 2050 and preserve Krasovsky ellipsoid instead of WGS84.
    let doubles_2050 = [39.0_f64, 0.0, 500_000.0, 1.0];
    let geokeys_2050: Vec<u16> = vec![
        1, 1, 0, 10, 1024, 0, 1, 1, 1025, 0, 1, 1, 2048, 0, 1, 32767, 2050, 0, 1, 6284, 3072, 0, 1,
        32767, 3075, 0, 1, 1, 3080, 34736, 1, 0, 3081, 34736, 1, 1, 3082, 34736, 1, 2, 3092, 34736,
        1, 3,
    ];
    let tiff_2050 = TiffSource::open(
        Arc::new(MemoryByteSource::new(
            build_rgb_4x4_tiff(
                &[
                    tag_f64_vec(33550, &[10.0, 10.0, 0.0]),
                    tag_f64_vec(33922, &[0.0, 0.0, 0.0, 500_000.0, 5_230_000.0, 0.0]),
                    tag_u16_vec(34735, &geokeys_2050),
                    tag_f64_vec(34736, &doubles_2050),
                ],
                128,
            )
            .into(),
        )),
        SourceFingerprint::from_digest([20; 32]),
        &budget,
        &cancel,
    )
    .expect("2048=32767 + 2050=6284 opens");
    match tiff_2050
        .description()
        .placement
        .as_ref()
        .and_then(|p| p.crs.as_ref())
    {
        Some(PlacementCrs::Resolved {
            crs: CrsSpec::Projected(Projection::Transverse { system, .. }),
            ..
        }) => {
            assert!((system.datum.ellipsoid.semi_major_m - KRASOVSKY.semi_major_m).abs() < 1e-3);
        }
        other => panic!("expected Krasovsky TM from GeoKey 2050, got {other:?}"),
    }

    // 2) CT_LambertConfConic_2SP (8) using ProjFalseOrigin* GeoKeys 3084-3087.
    let doubles_lcc2 = [
        -71.5_f64, 41.0, 200_000.0, 750_000.0, 42.683_333, 41.716_667,
    ];
    let geokeys_lcc2: Vec<u16> = vec![
        1, 1, 0, 11, 1024, 0, 1, 1, 1025, 0, 1, 1, 2048, 0, 1, 4326, 3072, 0, 1, 32767, 3075, 0, 1,
        8, 3084, 34736, 1, 0, 3085, 34736, 1, 1, 3086, 34736, 1, 2, 3087, 34736, 1, 3, 3078, 34736,
        1, 4, 3079, 34736, 1, 5,
    ];
    let tiff_lcc2 = TiffSource::open(
        Arc::new(MemoryByteSource::new(
            build_rgb_4x4_tiff(
                &[
                    tag_f64_vec(33550, &[10.0, 10.0, 0.0]),
                    tag_f64_vec(33922, &[0.0, 0.0, 0.0, 200_000.0, 750_000.0, 0.0]),
                    tag_u16_vec(34735, &geokeys_lcc2),
                    tag_f64_vec(34736, &doubles_lcc2),
                ],
                128,
            )
            .into(),
        )),
        SourceFingerprint::from_digest([21; 32]),
        &budget,
        &cancel,
    )
    .expect("LCC-2SP with 3084-3087 opens");
    let frame_lcc2 = frame_from_description(tiff_lcc2.description()).expect("LCC-2SP binds");
    let (lat_l2, lon_l2) = frame_lcc2.geodetic(0.0, 0.0);
    assert!((lat_l2 - 41.0).abs() < 1e-3 && (lon_l2 + 71.5).abs() < 1e-3);

    // 3) CT_LambertConfConic_1SP (9) using NatOriginLat (3081) -> emits +lat_1={lat_0}.
    let doubles_lcc1 = [2.337_229_f64, 46.8, 600_000.0, 2_200_000.0, 0.999_877];
    let geokeys_lcc1: Vec<u16> = vec![
        1, 1, 0, 10, 1024, 0, 1, 1, 1025, 0, 1, 1, 2048, 0, 1, 4326, 3072, 0, 1, 32767, 3075, 0, 1,
        9, 3080, 34736, 1, 0, 3081, 34736, 1, 1, 3082, 34736, 1, 2, 3083, 34736, 1, 3, 3092, 34736,
        1, 4,
    ];
    let tiff_lcc1 = TiffSource::open(
        Arc::new(MemoryByteSource::new(
            build_rgb_4x4_tiff(
                &[
                    tag_f64_vec(33550, &[10.0, 10.0, 0.0]),
                    tag_f64_vec(33922, &[0.0, 0.0, 0.0, 600_000.0, 2_200_000.0, 0.0]),
                    tag_u16_vec(34735, &geokeys_lcc1),
                    tag_f64_vec(34736, &doubles_lcc1),
                ],
                128,
            )
            .into(),
        )),
        SourceFingerprint::from_digest([22; 32]),
        &budget,
        &cancel,
    )
    .expect("LCC-1SP opens");
    let frame_lcc1 = frame_from_description(tiff_lcc1.description()).expect("LCC-1SP binds");
    let (lat_l1, lon_l1) = frame_lcc1.geodetic(0.0, 0.0);
    assert!((lat_l1 - 46.8).abs() < 1e-3 && (lon_l1 - 2.337_229).abs() < 1e-3);

    // 4) TIFF with embedded tiepoint+pixelscale (no GeoKeys) + `.prj`-only sidecar.
    let tiff_prj_only = TiffSource::open_with_sidecar(
        Arc::new(MemoryByteSource::new(
            build_rgb_4x4_tiff(
                &[
                    tag_f64_vec(33550, &[10.0, 10.0, 0.0]),
                    tag_f64_vec(33922, &[0.0, 0.0, 0.0, 500_000.0, 5_230_000.0, 0.0]),
                ],
                128,
            )
            .into(),
        )),
        SourceFingerprint::from_digest([23; 32]),
        Some(SidecarGeoref {
            affine: None,
            crs: Some(CrsSpec::from_epsg(32637).expect("UTM 37N")),
        }),
        &budget,
        &cancel,
    )
    .expect("TIFF with embedded affine + .prj-only sidecar opens");
    let frame_prj = frame_from_description(tiff_prj_only.description()).expect("binds via .prj");
    let (lat_p, lon_p) = frame_prj.geodetic(0.0, 0.0);
    assert!((lat_p - 47.22).abs() < 0.05 && (lon_p - 39.0).abs() < 0.05);
}
