use veldmap_net::{StacAsset, StacItem, StacRasterBandMeta};

use super::*;

fn cog_asset(key: &str, eo_bands: &[&str], roles: &[&str]) -> StacAsset {
    StacAsset {
        key: key.to_owned(),
        title: Some(key.to_owned()),
        href: format!("https://example.org/{key}.tif"),
        media_type: Some("image/tiff; application=geotiff; profile=cloud-optimized".to_owned()),
        roles: roles.iter().map(|s| (*s).to_owned()).collect(),
        eo_bands: eo_bands.iter().map(|s| (*s).to_owned()).collect(),
        gsd: Some(10.0),
        raster_bands: vec![StacRasterBandMeta {
            scale: Some(0.0001),
            offset: Some(-0.1),
            nodata: Some(0.0),
            unit: Some("reflectance".to_owned()),
        }],
    }
}

#[test]
fn build_item_view_generates_seven_stac_recipes_badges_and_scl_detection() {
    let item = StacItem {
        id: "S2B_32TMS_20240801_0_L2A".to_owned(),
        collection: Some("sentinel-2-l2a".to_owned()),
        self_href: Some("https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2B_32TMS_20240801_0_L2A".to_owned()),
        datetime: Some("2024-08-01T10:15:00Z".to_owned()),
        cloud_cover: Some(2.5),
        platform: Some("sentinel-2b".to_owned()),
        processing_level: Some("L2A".to_owned()),
        bbox: Some([7.5, 45.8, 9.2, 47.1]),
        geometry: None,
        mgrs_tile: Some("32TMS".to_owned()),
        assets: vec![
            cog_asset("blue", &["blue", "B02"], &["data"]),
            cog_asset("green", &["green", "B03"], &["data"]),
            cog_asset("red", &["red", "B04"], &["data"]),
            cog_asset("nir", &["nir", "B08"], &["data"]),
            cog_asset("swir16", &["swir16", "B11"], &["data"]),
            cog_asset("swir22", &["swir22", "B12"], &["data"]),
            cog_asset("scl", &[], &["data", "scl"]),
        ],
    };

    let view = build_item_view(&item);
    assert_eq!(view.processing_level.as_deref(), Some("L2A"));
    assert!(view.is_surface_reflectance);
    assert_eq!(view.platform.as_deref(), Some("sentinel-2b"));
    assert_eq!(
        view.meta_line.as_deref(),
        Some("2024-08-01 · 2.5% cloud · 32TMS")
    );
    let presets: Vec<BandMathPreset> = view.recipes.iter().map(|r| r.preset).collect();
    assert_eq!(
        presets,
        vec![
            BandMathPreset::TrueColor,
            BandMathPreset::Ndvi,
            BandMathPreset::Ndwi,
            BandMathPreset::Nbr,
            BandMathPreset::Ndsi,
            BandMathPreset::FalseColorNir,
            BandMathPreset::Swir,
        ]
    );
    let ndvi = view
        .recipes
        .iter()
        .find(|r| r.preset == BandMathPreset::Ndvi)
        .expect("ndvi recipe");
    assert!(ndvi.button_label.contains("SCL"));
    assert_eq!(ndvi.bands.len(), 2);
    assert!(ndvi.scl.is_some());
    assert!(
        view.cog_assets
            .iter()
            .find(|a| a.key == "scl")
            .is_some_and(|a| a.is_scl)
    );
}

#[test]
fn builtin_samples_and_recipes_use_https_cog_urls() {
    let samples = builtin_sample_scene_views();
    assert!(samples.len() >= 3);
    for s in samples {
        assert!(s.cog_url.starts_with("https://"), "bad URL: {}", s.cog_url);
        assert!(s.cog_url.ends_with(".tif"), "bad URL: {}", s.cog_url);
    }

    let recipes = builtin_sample_recipe_views();
    assert!(!recipes.is_empty());
    for r in recipes {
        assert!(!r.bands.is_empty());
        for b in &r.bands {
            assert!(b.href.as_str().starts_with("https://"));
            assert!(b.href.as_str().ends_with(".tif"));
        }
        if let Some(scl) = &r.scl {
            assert!(scl.href.as_str().starts_with("https://"));
            assert!(scl.href.as_str().ends_with(".tif"));
        }
    }
}
