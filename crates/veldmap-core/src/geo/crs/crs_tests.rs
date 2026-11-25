use super::*;
use crate::geo::ground_m;

#[test]
fn proj4_epsg_and_wkt_round_trip_and_warn_on_grid_datums() {
    // EPSG:3035 (ETRS89 / LAEA Europe)
    let laea = Projection::from_epsg(3035).expect("EPSG:3035");
    let (x, y) = laea.from_geodetic(52.52, 13.405);
    let back = laea.to_geodetic(x, y);
    assert!(ground_m(back, (52.52, 13.405)) < 0.05);

    // EPSG:3413 (WGS84 / Polar Stereographic North)
    let polar = Projection::from_epsg(3413).expect("EPSG:3413");
    let (px, py) = polar.from_geodetic(78.22, 15.63);
    let pback = polar.to_geodetic(px, py);
    assert!(ground_m(pback, (78.22, 15.63)) < 0.05);

    // WKT1 Lambert Conformal Conic (NAD83 Massachusetts)
    let wkt1 = concat!(
        r#"PROJCS["NAD83 / Massachusetts Mainland",GEOGCS["NAD83","#,
        r#"DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,"#,
        r#"AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,"#,
        r#"AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,"#,
        r#"AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],UNIT["metre",1,"#,
        r#"AUTHORITY["EPSG","9001"]],PROJECTION["Lambert_Conformal_Conic_2SP"],"#,
        r#"PARAMETER["standard_parallel_1",42.68333333333333],"#,
        r#"PARAMETER["standard_parallel_2",41.71666666666667],"#,
        r#"PARAMETER["latitude_of_origin",41],PARAMETER["central_meridian",-71.5],"#,
        r#"PARAMETER["false_easting",200000],PARAMETER["false_northing",750000],"#,
        r#"AUTHORITY["EPSG","26986"]]"#,
    );
    let spec = CrsSpec::from_wkt(wkt1).expect("WKT1 LCC");
    assert!(matches!(
        spec.datum_warning(),
        Some(DatumWarning::GridShiftRecommended { .. })
    ));
    let CrsSpec::Projected(lcc) = spec else {
        panic!("expected projected CRS");
    };
    let (lx, ly) = lcc.from_geodetic(42.36, -71.06);
    let lback = lcc.to_geodetic(lx, ly);
    assert!(ground_m(lback, (42.36, -71.06)) < 0.05);

    // Searchable presets find curated, regional MSK-61 code labels, and catalog EPSG codes.
    let hits = search_crs_presets("32637", 10);
    assert!(hits.iter().any(|p| p.epsg == 32637));
    let msk_hits = search_crs_presets("msk-61:z1", 10);
    assert!(msk_hits.iter().any(|p| p.epsg == 6_336_101));
    let catalog_hits = search_crs_presets("26911", 10);
    assert!(
        catalog_hits
            .iter()
            .any(|p| p.epsg == 26911 && p.category == "EPSG Catalog")
    );
    assert!(search_crs_presets("26911", 0).is_empty());
}

#[test]
fn crs_guards_geocentric_nad27_sphere_foreign_ellipsoids_and_named_custom_tm() {
    // 3D geocentric CRS (+proj=geocent) must be rejected as a 2D map CRS.
    assert!(CrsSpec::from_epsg(4978).is_err());
    assert!(Proj4Projection::from_proj_string("+proj=geocent +datum=WGS84", None).is_err());

    // NAD27 (missing NADCON grids) must construct via fallback Helmert + GridShiftRecommended.
    let nad27 = CrsSpec::from_epsg(26710).expect("NAD27 UTM 10N fallback");
    assert!(matches!(
        nad27.datum_warning(),
        Some(DatumWarning::GridShiftRecommended { .. })
    ));
    let CrsSpec::Projected(nad27_proj) = nad27 else {
        panic!("expected projected NAD27 CRS");
    };
    let (e, n) = nad27_proj.from_geodetic(37.7749, -122.4194);
    let back = nad27_proj.to_geodetic(e, n);
    assert!(ground_m(back, (37.7749, -122.4194)) < 1.0);

    // ETRS89 / RGF93 plate-fixed presets emit GridShiftRecommended consistently with EPSG:4258.
    for code in [3035, 25832, 2154] {
        let spec = CrsSpec::from_epsg(code).expect("ETRS89/RGF93 CRS");
        assert!(
            matches!(
                spec.datum_warning(),
                Some(DatumWarning::GridShiftRecommended { .. })
            ),
            "EPSG:{code} should emit GridShiftRecommended"
        );
    }

    // Foreign ellipsoid without +towgs84= or +datum= must warn (not just +ellps=krass).
    let bessel = Proj4Projection::from_proj_string("+proj=utm +zone=32 +ellps=bessel", None)
        .expect("bessel UTM");
    assert!(matches!(
        bessel.datum_warning(),
        Some(DatumWarning::UnshiftedForeignDatum { .. })
    ));

    // EPSG:4055 (Popular Visualisation Sphere) must NOT be mislabeled WGS 84.
    let sphere = CrsSpec::from_epsg(4055).expect("EPSG:4055");
    assert!(!sphere.label().contains("WGS 84"));

    // CustomTmParams::to_projection preserves user-supplied name in label().
    let params = CustomTmParams {
        name: "MSK-77 Test".to_string(),
        ..CustomTmParams::default()
    };
    let proj = params.to_projection().expect("valid custom TM");
    assert_eq!(proj.label(), "MSK-77 Test");
}

#[test]
fn parse_text_and_from_proj_string_labeled_classify_inputs_cleanly() {
    // Unified parse_text detects WKT vs PROJ/EPSG and preserves InvalidWkt on broken WKT.
    assert!(looks_like_wkt("\u{FEFF}  geogcs[\"WGS 84\"]"));
    assert!(!looks_like_wkt("+proj=utm +zone=37 +datum=WGS84"));
    let (crs, kind) = CrsSpec::parse_text("EPSG:32637").expect("EPSG shorthand");
    assert_eq!(kind, CrsTextKind::Epsg(32637));
    assert!(crs.label().contains("32637"));
    let (_, proj_kind) =
        CrsSpec::parse_text("+proj=utm +zone=37 +datum=WGS84 +units=m +no_defs").expect("PROJ");
    assert_eq!(proj_kind, CrsTextKind::Proj);
    let err = CrsSpec::parse_text("PROJCS[\"Broken\"").expect_err("malformed WKT");
    assert!(matches!(err, CrsError::InvalidWkt(_)));

    // Labeled PROJ.4 construction preserves explicit label for projected and geographic CRSs.
    let labeled = CrsSpec::from_proj_string_labeled(
        "+proj=merc +lon_0=0 +k_0=1 +x_0=0 +y_0=0 +datum=WGS84 +no_defs",
        Some("Custom Mercator".to_string()),
    )
    .expect("labeled PROJ");
    assert!(matches!(labeled, CrsSpec::Projected(_)));
    assert_eq!(labeled.label(), "Custom Mercator");
}
