use veldmap_core::geo::{DatumWarning, PULKOVO_1942};

use super::geokeys::{GeoKeyValue, GeoRasterType};
use super::*;
use crate::pipeline::{RasterDescription, frame_from_description};

#[test]
fn geokey_parse_respects_count_and_nul_stripped_ascii_tail() {
    let raw_keys = [
        1, 1, 0, 3, // header
        1024, 0, 1, 1, // ModelType = Projected
        3073, 34737, 8, 4, // PCSCitation starting at 4, count 8 (stripped NUL)
        3080, 34736, 1, 0, // ProjFalseOriginLong
        0, 0, 0, 0, // trailing padding ignored
    ];
    let doubles = [39.0_f64, f64::NAN];
    let ascii = "Foo|Zone 37";
    let parsed = parse_geokeys(&raw_keys, &doubles, ascii);
    assert_eq!(parsed.model_type, Some(1));
    assert_eq!(parsed.get_ascii(3073), Some("Zone 37"));
    assert_eq!(parsed.get_f64(3080), Some(39.0));
    assert_eq!(parsed.keys.len(), 3);
}

#[test]
fn geographic_crs_resolution_placement_and_basic_geo_ties_work_consistently() {
    // Projected Pulkovo-42 GK zone does NOT populate geographic_crs, nor does default WGS84.
    let mut keys = GeoKeyParams {
        projected_crs: Some(28407),
        ..Default::default()
    };
    assert_eq!(resolve_geographic_crs(&keys), None);
    keys.geographic_crs = Some(4326);
    assert_eq!(resolve_geographic_crs(&keys), None);

    // Non-WGS84 geographic CRSs resolve to typed CrsSpec variants (or UnshiftedForeignDatum).
    keys.geographic_crs = Some(4284);
    assert!(matches!(
        resolve_geographic_crs(&keys),
        Some(CrsSpec::Geographic { datum, .. }) if datum == PULKOVO_1942
    ));
    keys.geographic_crs = Some(4267);
    assert!(matches!(
        resolve_geographic_crs(&keys),
        Some(CrsSpec::GeographicProj(_))
    ));
    keys.geographic_crs = Some(9999);
    assert!(matches!(
        resolve_geographic_crs(&keys),
        Some(CrsSpec::Geographic {
            warning: Some(DatumWarning::UnshiftedForeignDatum { .. }),
            ..
        })
    ));

    // User-defined projected CRS rejects non-positive scale or non-finite parameters before dispatch.
    let bad_scale = GeoKeyParams {
        model_type: Some(1),
        projected_crs: Some(32767),
        keys: vec![
            (3075, GeoKeyValue::Short(1)),
            (3092, GeoKeyValue::Double(vec![0.0])),
        ],
        ..Default::default()
    };
    assert!(build_user_defined_projected_crs(&bad_scale).is_none());

    let bad_merc_scale = GeoKeyParams {
        model_type: Some(1),
        projected_crs: Some(32767),
        keys: vec![
            (3075, GeoKeyValue::Short(7)),
            (3092, GeoKeyValue::Double(vec![-1.0])),
        ],
        ..Default::default()
    };
    assert!(build_user_defined_projected_crs(&bad_merc_scale).is_none());

    // PixelIsArea vs PixelIsPoint shifts affine origin by half a pixel.
    let area_keys = GeoKeyParams {
        model_type: Some(1),
        projected_crs: Some(32637),
        raster_type: GeoRasterType::PixelIsArea,
        ..Default::default()
    };
    let tie = [0.0, 0.0, 0.0, 500_000.0, 6_000_000.0, 0.0];
    let scale = [10.0, 10.0, 0.0];
    let p_area = geo_placement(&area_keys, &tie, &scale, &[], 100, 100).expect("area placement");
    assert_eq!(p_area.crs, Some(PlacementCrs::Epsg(32637)));
    assert_eq!(
        p_area.affine,
        [10.0, 0.0, 500_000.0, 0.0, -10.0, 6_000_000.0]
    );

    let mut point_keys = area_keys;
    point_keys.raster_type = GeoRasterType::PixelIsPoint;
    let p_point = geo_placement(&point_keys, &tie, &scale, &[], 100, 100).expect("point placement");
    assert_eq!(
        p_point.affine,
        [10.0, 0.0, 499_995.0, 0.0, -10.0, 6_000_005.0]
    );

    // Singular matrix is rejected cleanly.
    assert!(affine_from_matrix(&[0.0; 8], 0.0).is_none());

    // Geographic tie points produce 4 corners cleanly.
    let geo_keys = GeoKeyParams {
        model_type: Some(2),
        ..Default::default()
    };
    let ties = geo_ties(
        &geo_keys,
        &[0.0, 0.0, 0.0, 30.0, 60.0, 0.0],
        &[0.01, 0.01, 0.0],
        &[],
        10,
        10,
    );
    assert_eq!(ties.len(), 4);
    assert!((ties[3].lon - 30.1).abs() < 1e-12);
    assert!((ties[3].lat - 59.9).abs() < 1e-12);
}

#[test]
fn wide_and_global_pixel_is_point_geographic_geotiffs_bind_without_wrapping_or_rejection() {
    let make_desc = |width: u32, height: u32, ties: Vec<_>| RasterDescription {
        width,
        height,
        bands: Vec::new(),
        placement: None,
        ties,
        geographic_crs: None,
        binding_trouble: None,
        levels: Vec::new(),
    };

    // 1. Wide 240-deg affine geographic raster `[-120, +120]` must not wrap backwards.
    let area_geo = GeoKeyParams {
        model_type: Some(2),
        raster_type: GeoRasterType::PixelIsArea,
        ..Default::default()
    };
    let wide_ties = geo_ties(
        &area_geo,
        &[0.0, 0.0, 0.0, -120.0, 60.0, 0.0],
        &[1.0, 1.0, 0.0],
        &[],
        240,
        120,
    );
    assert!(wide_ties.len() > 4, "wide span should subdivide longitude");
    let wide_frame = frame_from_description(&make_desc(240, 120, wide_ties))
        .expect("wide geographic frame should bind");
    assert!(wide_frame.axial());
    let (lat_c, lon_c) = wide_frame.geodetic(0.5, 0.5);
    assert!(
        lat_c.abs() < 1e-9 && lon_c.abs() < 1e-9,
        "got ({lat_c}, {lon_c})"
    );
    let (fx, fy) = wide_frame.fraction_at(0.0, 0.0).expect("center hit");
    assert!((fx - 0.5).abs() < 1e-9 && (fy - 0.5).abs() < 1e-9);
    assert!(
        wide_frame.fraction_at(0.0, 170.0).is_none(),
        "Pacific gap outside [-120, +120] must not hit"
    );

    // 2. Rotated wide raster where longitude spans 240 deg along image Y.
    let rot_matrix = [
        0.0, 1.0, 0.0, -120.0, // lon = py - 120
        -1.0, 0.0, 0.0, 60.0, // lat = 60 - px
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let rot_ties = geo_ties(&area_geo, &[], &[], &rot_matrix, 120, 240);
    let rot_frame = frame_from_description(&make_desc(120, 240, rot_ties))
        .expect("rotated wide geographic frame should bind");
    let (rlat, rlon) = rot_frame.geodetic(0.5, 0.5);
    assert!(
        rlat.abs() < 1e-9 && rlon.abs() < 1e-9,
        "got ({rlat}, {rlon})"
    );

    // 3. Global 3601x1801 PixelIsPoint rasters in both [-180, 180] and [0, 360].
    let point_geo = GeoKeyParams {
        model_type: Some(2),
        raster_type: GeoRasterType::PixelIsPoint,
        ..Default::default()
    };
    for (lon0, expected_center_lon) in [(-180.0_f64, 0.0_f64), (0.0_f64, 180.0_f64)] {
        let ties = geo_ties(
            &point_geo,
            &[0.0, 0.0, 0.0, lon0, 90.0, 0.0],
            &[0.1, 0.1, 0.0],
            &[],
            3601,
            1801,
        );
        assert!(
            !ties.is_empty(),
            "global PixelIsPoint ties must be accepted"
        );
        let frame = frame_from_description(&make_desc(3601, 1801, ties))
            .expect("global PixelIsPoint frame should bind");
        assert!(frame.measurable());

        // Exact pixel centers: (0, 0) -> (90, lon0), (3600, 1800) -> (-90, lon0 + 360).
        let (lat0, lon_start) = frame.geodetic(0.5 / 3601.0, 0.5 / 1801.0);
        assert!((lat0 - 90.0).abs() < 1e-12);
        assert!((lon_start - lon0).abs() < 1e-12);
        let (lat1, lon_end) = frame.geodetic(3600.5 / 3601.0, 1800.5 / 1801.0);
        assert!((lat1 - -90.0).abs() < 1e-12);
        assert!((lon_end - (lon0 + 360.0)).abs() < 1e-12);
        let (mid_lat, mid_lon) = frame.geodetic(1800.5 / 3601.0, 900.5 / 1801.0);
        assert!(mid_lat.abs() < 1e-12);
        assert!((mid_lon - expected_center_lon).abs() < 1e-12);

        // Outer half-pixel edges clamp to ±90 deg.
        assert_eq!(frame.geodetic(0.0, 0.0).0, 90.0);
        assert_eq!(frame.geodetic(1.0, 1.0).0, -90.0);
    }
}
