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
    let swir16 = item.find_band_asset(&["swir16", "B11", "b11"]);
    let swir22 = item.find_band_asset(&["swir22", "B12", "b12"]);
    let swir = swir22.or(swir16);
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
    if let (Some(n), Some(s22)) = (nir, swir22) {
        push_recipe(
            &mut recipes,
            BandMathPreset::Nbr,
            "NBR",
            format!(
                "Normalized Burn Ratio ({} − {}) / ({} + {})",
                n.key, s22.key, n.key, s22.key
            ),
            &[("NIR", n), ("SWIR-2", s22)],
            scl.clone(),
            item_href,
        );
    }
    if let (Some(g), Some(s16)) = (green, swir16) {
        push_recipe(
            &mut recipes,
            BandMathPreset::Ndsi,
            "NDSI",
            format!(
                "Normalized Difference Snow Index ({} − {}) / ({} + {})",
                g.key, s16.key, g.key, s16.key
            ),
            &[("Green", g), ("SWIR-1", s16)],
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
#[path = "views_tests.rs"]
mod tests;
