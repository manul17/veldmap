//! Integration tests for GeoTIFF user-defined GeoKeys (`3072 == 32767`,
//! `3076` feet, `3081` origin lat, `2062` TOWGS84, polar stereographic,
//! spherical ellipsoids, LCC 1SP/2SP), BigTIFF `GeoDoubleParamsTag` /
//! `GeoAsciiParamsTag` / `ModelTransformationTag`, `PixelIsArea` vs.
//! `PixelIsPoint` warp-mesh placement, foreign-datum geographic GeoTIFF datum
//! shifts, and `.prj`-only sidecar binding.

use std::io::Cursor;
use std::sync::Arc;

use ::tiff::encoder::TiffEncoder;
use ::tiff::tags::Tag;
use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::geo::{
    CrsSpec, DatumWarning, Geodetic, KRASOVSKY, PULKOVO_1942, Projection, US_SURVEY_FOOT_M, WGS84,
};
use veldmap_core::io::ByteSource;
use veldmap_core::io::testing::MemoryByteSource;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{MIN_WARP_SEGMENTS, SourceFingerprint, build_warp_patch};
use veldmap_raster::{
    DISPLAY_STREAM_SPEC, DecodeContext, OpenRasterOptions, PlacementCrs, RasterError, RasterSource,
    SharedChunkCache, SidecarGeoref, TiffSource, frame_from_description, has_tiff_magic,
    open_raster_with_options,
};

use crate::common::{
    TiffIfdHeader, build_rgb_4x4_tiff, build_tiled_cog, fake_fingerprint, rgb_pattern,
    sample_display_rgba, tag_f64_vec, tag_u16_vec, write_ifd_header, write_single_strip_u32,
    write_single_strip_u64,
};

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
    let prj_sidecar = SidecarGeoref {
        affine: None,
        crs: Some(CrsSpec::from_epsg(32637).expect("UTM 37N")),
    };
    let tiff_prj_only = open_raster_with_options(
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
        OpenRasterOptions {
            sidecars: Some(&prj_sidecar),
            external_bands: &[],
        },
        &budget,
        &cancel,
    )
    .expect("TIFF with embedded affine + .prj-only sidecar opens");
    let frame_prj =
        frame_from_description(tiff_prj_only.source.description()).expect("binds via .prj");
    let (lat_p, lon_p) = frame_prj.geodetic(0.0, 0.0);
    assert!((lat_p - 47.22).abs() < 0.05 && (lon_p - 39.0).abs() < 0.05);
}

#[test]
fn decodes_bigtiff_with_geodouble_geoascii_pixel_is_point_and_model_transformation() {
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };

    // BigTIFF + user-defined TM CRS using GeoDoubleParamsTag + GeoAsciiParamsTag +
    // PixelIsPoint + ModelTransformationTag.
    let mut big_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new_big(&mut big_buf).expect("bigtiff encoder");
        let mut dir = enc.image_directory().expect("bigtiff dir");
        write_ifd_header(&mut dir, &TiffIfdHeader::rgb8(8, 8));
        let data = rgb_pattern(8, 8);
        write_single_strip_u64(&mut dir, 8, &data);

        let ascii = "VeldMap BigTIFF TM|";
        let geokeys: [u16; 44] = [
            1,
            1,
            0,
            10, // header: 10 keys
            1024,
            0,
            1,
            1, // GTModelTypeGeoKey = Projected
            1025,
            0,
            1,
            2, // GTRasterTypeGeoKey = PixelIsPoint
            2048,
            0,
            1,
            4326, // GeographicTypeGeoKey = WGS 84
            3072,
            0,
            1,
            32767, // ProjectedCSTypeGeoKey = User-Defined
            3073,
            34737,
            ascii.len() as u16,
            0, // PCSCitationGeoKey in GeoAsciiParamsTag
            3075,
            0,
            1,
            1, // ProjCoordTransGeoKey = Transverse Mercator
            3080,
            34736,
            1,
            0, // ProjNatOriginLongGeoKey = 39.0
            3081,
            34736,
            1,
            1, // ProjNatOriginLatGeoKey = 55.75
            3082,
            34736,
            1,
            2, // ProjFalseEastingGeoKey = 500_000.0
            3092,
            34736,
            1,
            3, // ProjScaleAtNatOriginGeoKey = 0.9996
        ];
        dir.write_tag(Tag::GeoKeyDirectoryTag, &geokeys[..])
            .expect("tag");
        dir.write_tag(
            Tag::GeoDoubleParamsTag,
            &[39.0_f64, 55.75, 500_000.0, 0.9996][..],
        )
        .expect("tag");
        dir.write_tag(Tag::GeoAsciiParamsTag, ascii).expect("tag");
        let matrix: [f64; 16] = [
            10.0, 0.0, 0.0, 500_000.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        dir.write_tag(Tag::ModelTransformationTag, &matrix[..])
            .expect("tag");
        dir.finish().expect("finish");
    }
    let big_bytes = big_buf.into_inner();
    assert!(has_tiff_magic(&big_bytes));
    let big_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(big_bytes.clone())),
        fake_fingerprint(&big_bytes),
        &budget,
        &cancel,
    )
    .expect("open BigTIFF");
    let placement = big_src
        .description()
        .placement
        .as_ref()
        .expect("projected placement");
    assert!((placement.affine[2] - 499_995.0).abs() < 1e-9);
    assert!((placement.affine[5] - 5.0).abs() < 1e-9);

    let frame = frame_from_description(big_src.description()).expect("BigTIFF user-defined frame");
    assert_eq!(frame.crs_label(), "VeldMap BigTIFF TM");
    // Pixel (0, 0) center sits at exact natural origin (500_000, 0) -> (55.75, 39.0).
    let (lat, lon) = frame.geodetic(0.5 / 8.0, 0.5 / 8.0);
    assert!((lat - 55.75).abs() < 1e-9, "got lat={lat}");
    assert!((lon - 39.0).abs() < 1e-9, "got lon={lon}");

    let tile = big_src
        .decode_pointwise_tile(DISPLAY_STREAM_SPEC, 0, 0, 0, &ctx)
        .expect("decode BigTIFF tile");
    assert_eq!((tile.width(), tile.height()), (8, 8));
}

#[test]
fn pixel_is_area_vs_pixel_is_point_shifts_warp_mesh_and_readout_by_half_pixel() {
    let build_geotiff = |raster_type_key: u16| -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut enc = TiffEncoder::new(&mut buf).expect("tiff encoder");
            let mut dir = enc.image_directory().expect("dir");
            let (w, h) = (64u32, 64u32);
            write_ifd_header(&mut dir, &TiffIfdHeader::rgb8(w, h));
            let data = rgb_pattern(w, h);
            write_single_strip_u32(&mut dir, h, &data);

            let geokeys: [u16; 16] = [
                1,
                1,
                0,
                3, // header: 3 keys
                1024,
                0,
                1,
                1, // GTModelTypeGeoKey = Projected
                1025,
                0,
                1,
                raster_type_key, // GTRasterTypeGeoKey
                3072,
                0,
                1,
                32637, // ProjectedCSTypeGeoKey = EPSG:32637
            ];
            dir.write_tag(Tag::GeoKeyDirectoryTag, &geokeys[..])
                .expect("tag");
            // Pixel scale = 10 m/px; tiepoint pixel (0, 0) at (550_000, 5_250_000).
            dir.write_tag(Tag::ModelPixelScaleTag, &[10.0_f64, 10.0, 0.0][..])
                .expect("tag");
            dir.write_tag(
                Tag::ModelTiepointTag,
                &[0.0_f64, 0.0, 0.0, 550_000.0, 5_250_000.0, 0.0][..],
            )
            .expect("tag");
            dir.finish().expect("finish");
        }
        buf.into_inner()
    };

    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };

    let bytes_area = build_geotiff(1);
    let bytes_point = build_geotiff(2);
    let src_area: Arc<dyn RasterSource> = Arc::new(
        TiffSource::open(
            Arc::new(MemoryByteSource::new(Arc::from(bytes_area.as_slice()))),
            fake_fingerprint(&bytes_area),
            &budget,
            &cancel,
        )
        .expect("open PixelIsArea"),
    );
    let src_point: Arc<dyn RasterSource> = Arc::new(
        TiffSource::open(
            Arc::new(MemoryByteSource::new(Arc::from(bytes_point.as_slice()))),
            fake_fingerprint(&bytes_point),
            &budget,
            &cancel,
        )
        .expect("open PixelIsPoint"),
    );

    let frame_area = frame_from_description(src_area.description()).expect("area frame");
    let frame_point = frame_from_description(src_point.description()).expect("point frame");

    let patch_area = build_warp_patch(
        &frame_area,
        64,
        64,
        [0.0, 0.0, 64.0, 64.0],
        MIN_WARP_SEGMENTS,
    );
    let patch_point = build_warp_patch(
        &frame_point,
        64,
        64,
        [0.0, 0.0, 64.0, 64.0],
        MIN_WARP_SEGMENTS,
    );

    let utm37 = Projection::from_epsg(32637).expect("UTM 37N");
    let geo_area = WGS84.from_ecef(patch_area.evaluate_ecef(0.0, 0.0).expect("area ul"));
    let geo_point = WGS84.from_ecef(patch_point.evaluate_ecef(0.0, 0.0).expect("point ul"));
    let (ea, na) = utm37.from_geodetic(geo_area.lat_deg, geo_area.lon_deg);
    let (ep, np) = utm37.from_geodetic(geo_point.lat_deg, geo_point.lon_deg);

    // PixelIsArea UL corner lands at (550_000, 5_250_000).
    assert!((ea - 550_000.0).abs() < 0.002, "area easting={ea}");
    assert!((na - 5_250_000.0).abs() < 0.002, "area northing={na}");
    // PixelIsPoint shifts UL corner by exact (-0.5 px, +0.5 px) = (-5.0 m, +5.0 m).
    assert!((ep - 549_995.0).abs() < 0.002, "point easting={ep}");
    assert!((np - 5_250_005.0).abs() < 0.002, "point northing={np}");
    let (pt_lat, pt_lon) = utm37.to_geodetic(549_995.0, 5_250_005.0);
    let exact_pt_ul = WGS84.to_ecef(Geodetic::surface(pt_lat, pt_lon));
    let drawn_pt_ul = patch_point.evaluate_ecef(0.0, 0.0).expect("point ul");
    assert!(exact_pt_ul.distance(drawn_pt_ul) < 0.002);

    // Zoom-invariant native base-IFD display readout matches synthetic RGB pattern.
    let (px, py) = (17u32, 23u32);
    let expected = [
        (px * 7 + py * 3) as u8,
        (px ^ py) as u8,
        ((px * py) / 97) as u8,
        255,
    ];
    assert_eq!(
        sample_display_rgba(src_area.as_ref(), px, py, &ctx).expect("sample area"),
        Some(expected)
    );
    assert_eq!(
        sample_display_rgba(src_point.as_ref(), px, py, &ctx).expect("sample point"),
        Some(expected)
    );
    assert_eq!(
        sample_display_rgba(src_area.as_ref(), 64, 0, &ctx).expect("out of bounds"),
        None
    );

    // Multi-chunk tiled COG readout: sample pixels in non-zero (cx, cy) tiles
    // and verify the large-chunk guard rejects oversized single-chunk readouts.
    let (cog_bytes, _, _) = build_tiled_cog(700, 600, 1);
    let cog_src = TiffSource::open(
        Arc::new(MemoryByteSource::new(Arc::from(cog_bytes.as_slice()))),
        fake_fingerprint(&cog_bytes),
        &budget,
        &cancel,
    )
    .expect("open multi-chunk cog");
    let (cpx, cpy) = (610u32, 540u32);
    let cog_expected = [
        (cpx * 7 + cpy * 3) as u8,
        (cpx ^ cpy) as u8,
        ((cpx * cpy) / 97) as u8,
        255,
    ];
    assert_eq!(
        sample_display_rgba(&cog_src, cpx, cpy, &ctx).expect("sample multi-chunk cog"),
        Some(cog_expected)
    );

    // Tight chunk cache budget triggers LimitExceeded before decoding on fast_pool.
    let tiny_budget = MemoryBudget {
        decoded_chunk_cache_bytes: 16 * 1024,
        ..budget
    };
    let tiny_ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &tiny_budget,
        cancel: &cancel,
    };
    assert!(matches!(
        sample_display_rgba(&cog_src, cpx, cpy, &tiny_ctx),
        Err(RasterError::LimitExceeded { .. })
    ));
}
