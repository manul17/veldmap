//! Read-only STAC UI view projections (`build_item_view`, 1-click Band Math
//! scene recipes) and built-in zero-auth AWS Open Data onboarding sample scenes
//! and multi-band recipes.

use std::sync::OnceLock;

use veldmap_core::geo::GeoJsonGeometry;
use veldmap_net::{StacAsset, StacItem};
use veldmap_raster::ExternalBandMeta;

use crate::layers::BandMathPreset;

use super::aoi::{footprint_bounds_from_bbox, polygon_geometry_from_bbox};
use super::{
    RedactedUrl, SampleRecipeView, SampleSceneView, StacCogAssetView, StacItemView,
    StacRecipeAssetView, StacRecipeView,
};

const S2_ALPS_B08_URL: &str = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/T/MS/2023/8/S2A_32TMS_20230820_0_L2A/B08.tif";
const S2_ALPS_B04_URL: &str = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/T/MS/2023/8/S2A_32TMS_20230820_0_L2A/B04.tif";
const S2_ALPS_B03_URL: &str = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/T/MS/2023/8/S2A_32TMS_20230820_0_L2A/B03.tif";
const S2_ALPS_SCL_URL: &str = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/T/MS/2023/8/S2A_32TMS_20230820_0_L2A/SCL.tif";
const S2_ALPS_TCI_URL: &str = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/T/MS/2023/8/S2A_32TMS_20230820_0_L2A/TCI.tif";
const COP_DEM_MATTERHORN_URL: &str = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N46_00_E007_00_DEM/Copernicus_DSM_COG_10_N46_00_E007_00_DEM.tif";

fn external_bands_from_asset(asset: &StacAsset) -> Vec<ExternalBandMeta> {
    asset
        .raster_bands
        .iter()
        .map(|b| ExternalBandMeta {
            scale: b.scale,
            offset: b.offset,
            nodata: b.nodata,
            unit: b.unit.clone(),
        })
        .collect()
}

fn external_bands_from_sample(
    band_count: u8,
    scale: Option<f64>,
    offset: Option<f64>,
    nodata: Option<f64>,
    unit: Option<&str>,
) -> Vec<ExternalBandMeta> {
    let meta = ExternalBandMeta {
        scale,
        offset,
        nodata,
        unit: unit.map(str::to_owned),
    };
    if meta.has_any() {
        vec![meta; usize::from(band_count.max(1))]
    } else {
        Vec::new()
    }
}

fn s2_reflectance_band(label: &str, url: &str) -> StacRecipeAssetView {
    StacRecipeAssetView {
        label: label.to_owned(),
        href: RedactedUrl::new(url),
        external_bands: external_bands_from_sample(
            1,
            Some(0.0001),
            Some(-0.1),
            Some(0.0),
            Some("reflectance"),
        ),
        stac_item_href: None,
        asset_key: None,
    }
}

/// Cached UI-facing catalog of built-in zero-auth Sentinel-2 L2A & COP-DEM
/// GLO-30 COG sample scenes hosted on public AWS Open Data buckets.
pub(super) fn builtin_sample_scene_views() -> &'static [SampleSceneView] {
    static VIEWS: OnceLock<Vec<SampleSceneView>> = OnceLock::new();
    VIEWS.get_or_init(|| {
        vec![
            SampleSceneView {
                title: "Sentinel-2 L2A True Color (Swiss Alps)",
                description: "10 m Sentinel-2 L2A Cloud-Optimized GeoTIFF visual mosaic over the Swiss Alps (AWS Earth Search public bucket).",
                cog_url: S2_ALPS_TCI_URL,
                external_bands: external_bands_from_sample(3, None, None, Some(0.0), None),
            },
            SampleSceneView {
                title: "Sentinel-2 L2A Band 8 NIR (Swiss Alps)",
                description: "10 m scientific 16-bit NIR reflectance COG (scale=0.0001, offset=-0.1) over the Swiss Alps.",
                cog_url: S2_ALPS_B08_URL,
                external_bands: external_bands_from_sample(
                    1,
                    Some(0.0001),
                    Some(-0.1),
                    Some(0.0),
                    Some("reflectance"),
                ),
            },
            SampleSceneView {
                title: "COP-DEM GLO-30 Elevation (Matterhorn / Zermatt)",
                description: "30 m Float32 Copernicus DEM GLO-30 COG tile (N46 E007, metres EGM2008).",
                cog_url: COP_DEM_MATTERHORN_URL,
                external_bands: external_bands_from_sample(
                    1,
                    Some(1.0),
                    Some(0.0),
                    None,
                    Some("m"),
                ),
            },
        ]
    })
}

/// Cached UI-facing catalog of built-in zero-auth multi-band Sentinel-2 L2A
/// recipes (`NDVI + SCL`, `False Color NIR + SCL`) over the Swiss Alps.
pub(super) fn builtin_sample_recipe_views() -> &'static [SampleRecipeView] {
    static VIEWS: OnceLock<Vec<SampleRecipeView>> = OnceLock::new();
    VIEWS.get_or_init(|| {
        let b08 = s2_reflectance_band("B08 (NIR)", S2_ALPS_B08_URL);
        let b04 = s2_reflectance_band("B04 (Red)", S2_ALPS_B04_URL);
        let b03 = s2_reflectance_band("B03 (Green)", S2_ALPS_B03_URL);
        let scl = StacRecipeAssetView {
            label: "SCL".to_owned(),
            href: RedactedUrl::new(S2_ALPS_SCL_URL),
            external_bands: Vec::new(),
            stac_item_href: None,
            asset_key: None,
        };
        vec![
            SampleRecipeView {
                title: "Sentinel-2 L2A NDVI + SCL (Swiss Alps)",
                description: "10 m B08/B04 NDVI in physical BOA reflectance with 20 m ESA SCL cloud/shadow masking.",
                preset: BandMathPreset::Ndvi,
                bands: vec![b08.clone(), b04.clone()],
                scl: Some(scl.clone()),
            },
            SampleRecipeView {
                title: "Sentinel-2 L2A False Color NIR + SCL (Swiss Alps)",
                description: "10 m B08/B04/B03 NIR false-color composite with 20 m ESA SCL cloud/shadow masking.",
                preset: BandMathPreset::FalseColorNir,
                bands: vec![b08, b04, b03],
                scl: Some(scl),
            },
        ]
    })
}

fn format_recipe_band_label(key: &str, role: &str) -> String {
    if key.eq_ignore_ascii_case(role) {
        role.to_owned()
    } else {
        format!("{key} ({role})")
    }
}

fn build_asset_view(asset: &StacAsset) -> StacCogAssetView {
    StacCogAssetView {
        key: asset.key.clone(),
        title: asset.title.clone(),
        href: RedactedUrl::new(asset.href.clone()),
        gsd: asset.gsd,
        external_bands: external_bands_from_asset(asset),
        is_scl: asset.is_scl_asset(),
    }
}

fn recipe_asset_from_stac(
    label: &str,
    asset: &StacAsset,
    item_href: Option<&str>,
) -> StacRecipeAssetView {
    StacRecipeAssetView {
        label: label.to_owned(),
        href: RedactedUrl::new(asset.href.clone()),
        external_bands: external_bands_from_asset(asset),
        stac_item_href: item_href.map(str::to_owned),
        asset_key: Some(asset.key.clone()),
    }
}

fn push_recipe(
    out: &mut Vec<StacRecipeView>,
    preset: BandMathPreset,
    base_label: &str,
    tooltip: String,
    bands: &[(&str, &StacAsset)],
    scl: Option<StacRecipeAssetView>,
    item_href: Option<&str>,
) {
    let scl_tag = if scl.is_some() { " + SCL" } else { "" };
    out.push(StacRecipeView {
        preset,
        button_label: format!("{base_label}{scl_tag}"),
        tooltip: format!("{tooltip}{scl_tag}"),
        bands: bands
            .iter()
            .map(|(role, asset)| {
                recipe_asset_from_stac(
                    &format_recipe_band_label(&asset.key, role),
                    asset,
                    item_href,
                )
            })
            .collect(),
        scl,
    });
}

fn build_item_recipes(item: &StacItem) -> Vec<StacRecipeView> {
    let item_href = item.self_href.as_deref();
    let red = item.find_band_asset(&["red", "B04", "b04"]);
    let green = item.find_band_asset(&["green", "B03", "b03"]);
    let blue = item.find_band_asset(&["blue", "B02", "b02"]);
    let nir = item.find_band_asset(&["nir", "B08", "b08", "nir08", "B8A", "b8a"]);
    let swir = item.find_band_asset(&["swir22", "B12", "b12", "swir16", "B11", "b11"]);
    let scl = item
        .scl_asset()
        .map(|a| recipe_asset_from_stac("SCL", a, item_href));

    let mut recipes = Vec::new();
    if let (Some(r), Some(g), Some(b)) = (red, green, blue) {
        push_recipe(
            &mut recipes,
            BandMathPreset::TrueColor,
            "True Color",
            format!(
                "Physical reflectance True Color ({}, {}, {})",
                r.key, g.key, b.key
            ),
            &[("Red", r), ("Green", g), ("Blue", b)],
            scl.clone(),
            item_href,
        );
    }
    if let (Some(n), Some(r)) = (nir, red) {
        push_recipe(
            &mut recipes,
            BandMathPreset::Ndvi,
            "NDVI",
            format!(
                "Normalized Difference Vegetation Index ({} − {}) / ({} + {})",
                n.key, r.key, n.key, r.key
            ),
            &[("NIR", n), ("Red", r)],
            scl.clone(),
            item_href,
        );
    }
    if let (Some(g), Some(n)) = (green, nir) {
        push_recipe(
            &mut recipes,
            BandMathPreset::Ndwi,
            "NDWI",
            format!(
                "Normalized Difference Water Index ({} − {}) / ({} + {})",
                g.key, n.key, g.key, n.key
            ),
            &[("Green", g), ("NIR", n)],
            scl.clone(),
            item_href,
        );
    }
    if let (Some(n), Some(r), Some(g)) = (nir, red, green) {
        push_recipe(
            &mut recipes,
            BandMathPreset::FalseColorNir,
            "False Color NIR",
            format!(
                "False Color Infrared composite ({}, {}, {})",
                n.key, r.key, g.key
            ),
            &[("NIR", n), ("Red", r), ("Green", g)],
            scl.clone(),
            item_href,
        );
    }
    if let (Some(s), Some(n), Some(r)) = (swir, nir, red) {
        push_recipe(
            &mut recipes,
            BandMathPreset::Swir,
            "SWIR",
            format!(
                "Short-Wave Infrared composite ({}, {}, {})",
                s.key, n.key, r.key
            ),
            &[("SWIR", s), ("NIR", n), ("Red", r)],
            scl,
            item_href,
        );
    }
    recipes
}

fn is_surface_reflectance_level(level: &str) -> bool {
    let lower = level.to_ascii_lowercase();
    lower.contains("l2a") || lower.contains("boa")
}

pub(super) fn build_item_view(item: &StacItem) -> StacItemView {
    let short_date = item
        .datetime
        .as_deref()
        .map(|dt| dt.split('T').next().unwrap_or(dt));
    let mut meta_parts = Vec::with_capacity(3);
    if let Some(date) = short_date {
        meta_parts.push(date.to_owned());
    }
    if let Some(cc) = item.cloud_cover {
        meta_parts.push(format!("{cc:.1}% cloud"));
    }
    if let Some(tile) = &item.mgrs_tile {
        meta_parts.push(tile.clone());
    }
    let meta_line = (!meta_parts.is_empty()).then(|| meta_parts.join(" · "));

    let preferred_cog = item.preferred_cog_asset().map(build_asset_view);
    let recipes = build_item_recipes(item);
    let cog_assets = item
        .cog_assets()
        .into_iter()
        .map(build_asset_view)
        .collect();
    let zoom_bounds = item
        .bbox
        .or_else(|| item.geometry.as_ref().and_then(GeoJsonGeometry::bbox))
        .and_then(footprint_bounds_from_bbox);
    let is_surface_reflectance = item
        .processing_level
        .as_deref()
        .is_some_and(is_surface_reflectance_level);
    StacItemView {
        id: item.id.clone(),
        self_href: item.self_href.clone(),
        meta_line,
        platform: item.platform.clone(),
        processing_level: item.processing_level.clone(),
        is_surface_reflectance,
        preferred_cog,
        recipes,
        cog_assets,
        zoom_bounds,
    }
}

pub(super) fn item_geometry(item: &StacItem) -> Option<GeoJsonGeometry> {
    if let Some(geom) = &item.geometry {
        Some(geom.clone())
    } else {
        item.bbox.and_then(polygon_geometry_from_bbox)
    }
}

#[cfg(test)]
mod tests {
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
    fn build_item_view_generates_five_stac_recipes_badges_and_scl_detection() {
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
}
