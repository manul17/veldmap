use veldmap_core::geo::{CustomTmParams, DatumPreset, LinearUnit};
use veldmap_storage::test_support::TestTempDir;

use super::*;

#[test]
fn session_persists_all_crs_override_variants_flat_json_recent_files_and_bookmarks() {
    let tmp = TestTempDir::new("session-roundtrip");
    let paths = StoragePaths::with_root(tmp.path());

    let epsg_ov = CrsOverride::Epsg(32637);
    let wkt_ov = CrsOverride::Wkt(
        concat!(
            "GEOGCS[\"WGS 84\",",
            "DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],",
            "PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]"
        )
        .to_owned(),
    );
    let proj_ov = CrsOverride::Proj("+proj=utm +zone=37 +datum=WGS84 +units=m +no_defs".to_owned());
    let custom_tm_ov = CrsOverride::CustomTm(CustomTmParams {
        name: "MSK-61 Custom".to_string(),
        datum: DatumPreset::Pulkovo1942,
        central_meridian_deg: 37.983_333_333_33,
        origin_lat_deg: 0.0,
        scale: 1.0,
        false_easting_m: 1_300_000.0,
        false_northing_m: -4_811_057.628,
        unit: LinearUnit::Metre,
    });

    let mut session = AppSession::default();
    session.open_layers.push(SavedLayerEntry::new(
        PathBuf::from("/tmp/epsg.tif"),
        true,
        1.0,
        Some(&epsg_ov),
    ));
    session.open_layers.push(SavedLayerEntry::new(
        PathBuf::from("/tmp/wkt.tif"),
        false,
        0.5,
        Some(&wkt_ov),
    ));
    session.open_layers.push(SavedLayerEntry::new(
        PathBuf::from("/tmp/proj.tif"),
        true,
        0.8,
        Some(&proj_ov),
    ));
    session.open_layers.push(SavedLayerEntry::new(
        PathBuf::from("/tmp/ortho.tif"),
        true,
        0.75,
        Some(&custom_tm_ov),
    ));
    session.open_vectors.push(SavedVectorEntry {
        path: PathBuf::from("/tmp/parcels.geojson"),
        visible: true,
        opacity: 0.9,
        outline_only: true,
    });
    session.record_recent_file(Path::new("/tmp/a.tif"));
    session.record_recent_file(Path::new("/tmp/b.tif"));
    session.record_recent_file(Path::new("/tmp/a.tif"));
    let pose = SavedCameraPose {
        lat_deg: 47.23,
        lon_deg: 39.72,
        height_m: 15_000.0,
        heading_deg: 12.5,
    };
    session.bookmarks.push(CameraBookmark {
        name: "Rostov".to_string(),
        pose,
    });
    session.last_camera = Some(pose);

    session.save(&paths).expect("save session");
    // Verify on-disk JSON preserves the exact flat wire format for every
    // `CrsOverride` variant and `CameraBookmark`.
    let raw = std::fs::read_to_string(paths.session_path()).expect("read raw json");
    assert!(raw.contains(r#""kind": "epsg""#));
    assert!(raw.contains(r#""code": 32637"#));
    assert!(raw.contains(r#""kind": "wkt""#));
    assert!(raw.contains(r#""wkt": "GEOGCS"#));
    assert!(raw.contains(r#""kind": "proj""#));
    assert!(raw.contains(r#""proj": "+proj=utm"#));
    assert!(raw.contains(r#""kind": "custom_tm""#));
    assert!(raw.contains(r#""datum": "pulkovo1942""#));
    assert!(raw.contains(r#""central_meridian_deg""#));
    assert!(raw.contains(r#""unit": "metre""#));
    assert!(raw.contains(r#""name": "Rostov""#));
    assert!(!raw.contains(r#""pose""#));

    let loaded = AppSession::load_or_default(&paths);
    assert_eq!(loaded, session);
    assert_eq!(
        loaded.recent_files,
        vec![PathBuf::from("/tmp/a.tif"), PathBuf::from("/tmp/b.tif")]
    );

    let expected = [&epsg_ov, &wkt_ov, &proj_ov, &custom_tm_ov];
    for (entry, expected_ov) in loaded.open_layers.iter().zip(expected) {
        let roundtripped = entry.crs_override().expect("crs override");
        assert_eq!(&roundtripped, expected_ov);
        assert!(roundtripped.to_crs_spec().is_ok());
    }
    assert_eq!(
        loaded.open_layers[3]
            .crs_override()
            .expect("custom tm")
            .to_crs_spec()
            .expect("valid custom TM")
            .label(),
        "MSK-61 Custom"
    );
}

#[test]
fn session_custom_tm_omitted_datum_defaults_to_wgs84() {
    let tmp = TestTempDir::new("session-default-datum");
    let paths = StoragePaths::with_root(tmp.path());
    std::fs::create_dir_all(paths.session_path().parent().expect("parent")).expect("mkdir");
    std::fs::write(
        paths.session_path(),
        r#"{
          "version": 1,
          "open_layers": [{
            "path": "/tmp/legacy.tif",
            "visible": true,
            "opacity": 1.0,
            "crs_override": {
              "kind": "custom_tm",
              "name": "Legacy TM",
              "central_meridian_deg": 39.0,
              "origin_lat_deg": 0.0,
              "scale": 1.0,
              "false_easting_m": 500000.0,
              "false_northing_m": 0.0,
              "unit": "us_survey_foot"
            }
          }]
        }"#,
    )
    .expect("write legacy session");

    let loaded = AppSession::load_or_default(&paths);
    assert_eq!(loaded.open_layers.len(), 1);
    match loaded.open_layers[0].crs_override() {
        Some(CrsOverride::CustomTm(params)) => {
            assert_eq!(params.datum, DatumPreset::Wgs84);
            assert_eq!(params.unit, LinearUnit::UsSurveyFoot);
        }
        other => panic!("expected CustomTm, got {other:?}"),
    }
}

#[test]
fn session_normalize_rejects_out_of_range_poses_and_nan_opacity() {
    let mut session = AppSession::default();
    session
        .open_layers
        .push(SavedLayerEntry::LocalRaster(SavedLocalRasterEntry {
            path: PathBuf::from("/tmp/a.tif"),
            visible: true,
            opacity: f32::NAN,
            crs_override: None,
            style: None,
        }));
    session.bookmarks.push(CameraBookmark {
        name: "Invalid Lat".to_string(),
        pose: SavedCameraPose {
            lat_deg: 120.0,
            lon_deg: 0.0,
            height_m: 1000.0,
            heading_deg: 0.0,
        },
    });
    session.last_camera = Some(SavedCameraPose {
        lat_deg: -95.0,
        lon_deg: 0.0,
        height_m: 1000.0,
        heading_deg: 0.0,
    });
    session.normalize();
    match &session.open_layers[0] {
        SavedLayerEntry::LocalRaster(e) => assert_eq!(e.opacity, 1.0),
        other => panic!("expected LocalRaster, got {other:?}"),
    }
    assert!(session.bookmarks.is_empty());
    assert!(session.last_camera.is_none());
}

#[test]
fn session_layer_style_round_trips_and_tolerates_unknown_future_variants() {
    use crate::colormap::{CategoricalPaletteId, ColormapId};
    use crate::layers::{LayerStyleSpec, StretchPreset, TrackKind};

    let tmp = TestTempDir::new("veldmap-session-style");
    let paths = StoragePaths::with_root(tmp.path());

    let mut session = AppSession::default();
    let style = LayerStyleSpec {
        band_index: Some(2),
        track: Some(TrackKind::Scientific),
        stretch_preset: Some(StretchPreset::Log10Db),
        stretch_min: Some(-25.0),
        stretch_max: Some(-2.0),
        custom_stretch_range: true,
        colormap: Some(ColormapId::Inferno),
        invert_colormap: true,
        categorical_palette: Some(CategoricalPaletteId::Sentinel2Scl),
    };
    session.open_layers.push(SavedLayerEntry::with_style(
        PathBuf::from("/tmp/sar.tif"),
        true,
        0.85,
        None,
        Some(&style),
    ));
    session.save(&paths).expect("save session");
    let loaded = AppSession::load_or_default(&paths);
    assert_eq!(loaded.open_layers.len(), 1);
    assert_eq!(loaded.open_layers[0].style(), Some(style));

    // Unknown future enum variants must fall back to None instead of corrupting the whole session.
    std::fs::write(
        paths.session_path(),
        r#"{
          "version": 1,
          "open_layers": [{
            "path": "/tmp/future.tif",
            "visible": true,
            "opacity": 0.9,
            "style": {
              "band_index": 0,
              "track": "scientific",
              "stretch_preset": "future_histogram_eq",
              "colormap": "future_cividis",
              "categorical_palette": "future_modis_qa"
            }
          }]
        }"#,
    )
    .expect("write future session");
    let fwd = AppSession::load_or_default(&paths);
    assert_eq!(
        fwd.open_layers.len(),
        1,
        "session must NOT be wiped to .corrupt"
    );
    let st = fwd.open_layers[0].style().expect("style present");
    assert_eq!(st.track, Some(TrackKind::Scientific));
    assert_eq!(st.stretch_preset, None);
    assert_eq!(st.colormap, None);
    assert_eq!(st.categorical_palette, None);
}
