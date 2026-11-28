use veldmap_storage::test_support::TestTempDir;

use super::*;
use crate::layers::{
    BandAssetPersistEntry, BandMathStyleSpec, OverlayPersistEntry, StacCogAuthMode,
};

#[test]
fn project_path_detection_distinguishes_veldmap_project_from_generic_json() {
    assert!(is_project_path(Path::new("analysis.veldmap.json")));
    assert!(is_project_path(Path::new("/dir/MyProject.VELDMAP.JSON")));
    assert!(!is_project_path(Path::new("parcels.geojson")));
    assert!(!is_project_path(Path::new("parcels.json")));
}

#[test]
fn relative_path_round_trips_on_same_root_and_falls_back_across_roots() {
    let tmp = TestTempDir::new("project-rel");
    let base = tmp.path().join("projects").join("run1");
    let sibling = tmp.path().join("data").join("ortho.tif");
    let nested = base.join("bands").join("b04.tif");

    let rel_sib = make_path_relative_if_same_root(&base, &sibling);
    assert!(!rel_sib.is_absolute(), "expected relative, got {rel_sib:?}");
    assert_eq!(
        resolve_project_path(&base, &rel_sib),
        lexical_clean(&sibling)
    );

    let rel_nested = make_path_relative_if_same_root(&base, &nested);
    assert_eq!(rel_nested, PathBuf::from("bands").join("b04.tif"));
    assert_eq!(
        resolve_project_path(&base, &rel_nested),
        lexical_clean(&nested)
    );

    #[cfg(windows)]
    {
        let other_drive = Path::new(r"Z:\external\scene.tif");
        let c_base = Path::new(r"C:\work\proj");
        assert_eq!(
            make_path_relative_if_same_root(c_base, other_drive),
            PathBuf::from(r"Z:\external\scene.tif")
        );
    }
}

#[test]
fn project_file_strips_secrets_relativizes_paths_and_round_trips_swipe_and_band_math() {
    let tmp = TestTempDir::new("project-roundtrip");
    let proj_dir = tmp.path().join("workspace");
    let proj_path = proj_dir.join("study.veldmap.json");
    let local_tif = tmp.path().join("imagery").join("drone.tif");
    let local_vec = proj_dir.join("aoi.geojson");

    let mut proj = ProjectFile::default();
    proj.open_layers.push(SavedLayerEntry::with_style(
        local_tif.clone(),
        true,
        0.9,
        None,
        None,
    ));
    proj.open_layers
        .push(SavedLayerEntry::from_persist(OverlayPersistEntry::RemoteCog {
            canonical_url: "https://user:supersecret@sentinel-cogs.example.org/B04.tif?X-Amz-Signature=deadbeef&sig=secret#frag".to_owned(),
            title: "S2 Red".to_owned(),
            auth_mode: StacCogAuthMode::FromSearchResult,
            external_bands: vec![veldmap_raster::ExternalBandMeta {
                scale: Some(0.0001),
                offset: Some(-0.1),
                nodata: Some(0.0),
                unit: Some("reflectance".to_owned()),
            }],
            is_scl: false,
            stac_item_href: Some("https://user:pw@stac.example.org/items/s2?token=abc".to_owned()),
            asset_key: Some("red".to_owned()),
            visible: true,
            opacity: 0.85,
            style: None,
        }));
    proj.open_layers.push(SavedLayerEntry::from_persist(
        OverlayPersistEntry::BandMath {
            title: "Mixed NDVI".to_owned(),
            visible: true,
            opacity: 1.0,
            bands: vec![
                BandAssetPersistEntry::Local {
                    label: "NIR".to_owned(),
                    path: local_tif.clone(),
                    crs_override: None,
                    band_index: 3,
                },
                BandAssetPersistEntry::Remote {
                    label: "Red".to_owned(),
                    canonical_url: "https://api:key@example.org/b04.tif?sig=xyz".to_owned(),
                    auth_mode: StacCogAuthMode::DirectCustom,
                    external_bands: Vec::new(),
                    stac_item_href: None,
                    asset_key: Some("B04".to_owned()),
                    band_index: 0,
                },
            ],
            scl: None,
            style: BandMathStyleSpec {
                preset: crate::layers::BandMathPreset::Ndvi,
                formula: crate::layers::BandMathFormula::NormalizedDiff,
                recipe_label: "NDVI".to_owned(),
                colormap: crate::colormap::ColormapId::RdYlGn,
                invert_colormap: false,
                stretch_min: -0.2,
                stretch_max: 0.85,
                rgb_min: [0.0; 3],
                rgb_max: [0.35; 3],
                default_rgb_range: ([0.0; 3], [0.35; 3]),
                rgb_gamma: 1.0,
                scl_mask_enabled: false,
                scl_mask_bits: 0,
            },
        },
    ));
    proj.open_vectors.push(SavedVectorEntry {
        path: local_vec.clone(),
        visible: true,
        opacity: 0.7,
        outline_only: true,
    });
    proj.swipe = Some(SavedSwipeState {
        active: true,
        split_t: 0.35,
        left_layer_index: Some(0),
        right_layer_index: Some(2),
    });

    proj.save_to(&proj_path).expect("save project");
    let raw = std::fs::read_to_string(&proj_path).expect("read raw project");

    // Secret hygiene: no userinfo, query tokens, or signatures on disk.
    for forbidden in ["supersecret", "deadbeef", "sig=", "token=abc", "api:key"] {
        assert!(
            !raw.contains(forbidden),
            "leaked secret '{forbidden}' into project file: {raw}"
        );
    }
    assert!(raw.contains(r#""kind": "local_raster""#));
    assert!(raw.contains(r#""kind": "remote_cog""#));
    assert!(raw.contains(r#""kind": "band_math""#));
    assert!(raw.contains(r#""outline_only": true"#));
    assert!(raw.contains(r#""aoi.geojson""#));

    let loaded = ProjectFile::load_from(&proj_path).expect("load project");
    assert_eq!(loaded.open_layers.len(), 3);
    assert_eq!(loaded.open_vectors.len(), 1);
    assert!(loaded.open_vectors[0].outline_only);
    assert_eq!(loaded.open_vectors[0].path, lexical_clean(&local_vec));
    assert_eq!(loaded.swipe, proj.swipe);

    match &loaded.open_layers[0] {
        SavedLayerEntry::LocalRaster(e) => {
            assert_eq!(e.path, lexical_clean(&local_tif));
        }
        other => panic!("expected LocalRaster, got {other:?}"),
    }
    match &loaded.open_layers[1] {
        SavedLayerEntry::RemoteCog(e) => {
            assert_eq!(e.url, "https://sentinel-cogs.example.org/B04.tif");
            assert_eq!(
                e.stac_item_href.as_deref(),
                Some("https://stac.example.org/items/s2")
            );
            assert_eq!(e.asset_key.as_deref(), Some("red"));
            assert_eq!(e.auth_mode(), StacCogAuthMode::FromSearchResult);
        }
        other => panic!("expected RemoteCog, got {other:?}"),
    }
}
