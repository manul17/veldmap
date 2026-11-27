//! Built-in zero-auth public Cloud-Optimized GeoTIFF sample scenes for
//! 15-second onboarding without requiring credentials or custom endpoints.

use super::types::{COLLECTION_COP_DEM_GLO30, COLLECTION_SENTINEL2_L2A, combine_scale_offset};

/// Curated public zero-auth Cloud-Optimized GeoTIFF sample scene.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleScene {
    pub id: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub collection: &'static str,
    pub cog_url: &'static str,
    /// WGS84 framing bbox `[min_lon, min_lat, max_lon, max_lat]`.
    pub bbox: [f64; 4],
    pub scale: Option<f64>,
    pub offset: Option<f64>,
    pub nodata: Option<f64>,
    pub unit: Option<&'static str>,
}

impl SampleScene {
    /// Combined `(scale, offset)` metadata if either radiometric field is set.
    pub const fn scale_offset(&self) -> Option<(f64, f64)> {
        combine_scale_offset(self.scale, self.offset)
    }
}

/// One constituent COG asset inside a built-in multi-band sample recipe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleRecipeBand {
    pub label: &'static str,
    pub url: &'static str,
    pub scale: Option<f64>,
    pub offset: Option<f64>,
    pub nodata: Option<f64>,
    pub unit: Option<&'static str>,
}

/// Curated multi-band sample scene recipe (`NDVI + SCL`, `True Color + SCL`)
/// hosted on public AWS Open Data buckets for 1-click band-math onboarding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleRecipeScene {
    pub id: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    /// Recipe kind identifier (`"ndvi"`, `"true_color"`, `"false_color_nir"`).
    pub recipe_id: &'static str,
    pub bbox: [f64; 4],
    pub bands: &'static [SampleRecipeBand],
    pub scl: Option<SampleRecipeBand>,
}

/// Curated public zero-auth Sentinel-2 L2A & COP-DEM GLO-30 COG samples hosted
/// on public AWS Open Data buckets (HTTPS, zero credentials required).
pub fn builtin_sample_scenes() -> &'static [SampleScene] {
    static SAMPLES: &[SampleScene] = &[
        SampleScene {
            id: "s2-alps-tci",
            title: "Sentinel-2 L2A True Color (Swiss Alps)",
            description: "10 m Sentinel-2 L2A Cloud-Optimized GeoTIFF visual mosaic over the Swiss Alps (AWS Earth Search public bucket).",
            collection: COLLECTION_SENTINEL2_L2A,
            cog_url: "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/T/MS/2023/8/S2A_32TMS_20230820_0_L2A/TCI.tif",
            bbox: [7.6, 45.9, 9.1, 47.0],
            scale: None,
            offset: None,
            nodata: Some(0.0),
            unit: None,
        },
        SampleScene {
            id: "s2-alps-b08-nir",
            title: "Sentinel-2 L2A Band 8 NIR (Swiss Alps)",
            description: "10 m scientific 16-bit NIR reflectance COG (scale=0.0001, offset=-0.1) over the Swiss Alps.",
            collection: COLLECTION_SENTINEL2_L2A,
            cog_url: "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/T/MS/2023/8/S2A_32TMS_20230820_0_L2A/B08.tif",
            bbox: [7.6, 45.9, 9.1, 47.0],
            scale: Some(0.0001),
            offset: Some(-0.1),
            nodata: Some(0.0),
            unit: Some("reflectance"),
        },
        SampleScene {
            id: "cop-dem-matterhorn",
            title: "COP-DEM GLO-30 Elevation (Matterhorn / Zermatt)",
            description: "30 m Float32 Copernicus DEM GLO-30 COG tile (N46 E007, metres EGM2008).",
            collection: COLLECTION_COP_DEM_GLO30,
            cog_url: "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N46_00_E007_00_DEM/Copernicus_DSM_COG_10_N46_00_E007_00_DEM.tif",
            bbox: [7.0, 46.0, 8.0, 47.0],
            scale: Some(1.0),
            offset: Some(0.0),
            nodata: None,
            unit: Some("m"),
        },
    ];
    SAMPLES
}

/// Curated public zero-auth multi-band Sentinel-2 L2A recipes (`NDVI + SCL`,
/// `False Color NIR + SCL`) over the Swiss Alps.
pub fn builtin_sample_recipes() -> &'static [SampleRecipeScene] {
    const B08: SampleRecipeBand = SampleRecipeBand {
        label: "B08 (NIR)",
        url: "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/T/MS/2023/8/S2A_32TMS_20230820_0_L2A/B08.tif",
        scale: Some(0.0001),
        offset: Some(-0.1),
        nodata: Some(0.0),
        unit: Some("reflectance"),
    };
    const B04: SampleRecipeBand = SampleRecipeBand {
        label: "B04 (Red)",
        url: "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/T/MS/2023/8/S2A_32TMS_20230820_0_L2A/B04.tif",
        scale: Some(0.0001),
        offset: Some(-0.1),
        nodata: Some(0.0),
        unit: Some("reflectance"),
    };
    const B03: SampleRecipeBand = SampleRecipeBand {
        label: "B03 (Green)",
        url: "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/T/MS/2023/8/S2A_32TMS_20230820_0_L2A/B03.tif",
        scale: Some(0.0001),
        offset: Some(-0.1),
        nodata: Some(0.0),
        unit: Some("reflectance"),
    };
    const SCL: SampleRecipeBand = SampleRecipeBand {
        label: "SCL",
        url: "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/T/MS/2023/8/S2A_32TMS_20230820_0_L2A/SCL.tif",
        scale: None,
        offset: None,
        nodata: None,
        unit: None,
    };
    static NDVI_BANDS: &[SampleRecipeBand] = &[B08, B04];
    static FALSE_COLOR_BANDS: &[SampleRecipeBand] = &[B08, B04, B03];
    static RECIPES: &[SampleRecipeScene] = &[
        SampleRecipeScene {
            id: "s2-alps-ndvi-scl",
            title: "Sentinel-2 L2A NDVI + SCL (Swiss Alps)",
            description: "10 m B08/B04 NDVI in physical BOA reflectance with 20 m ESA SCL cloud/shadow masking.",
            recipe_id: "ndvi",
            bbox: [7.6, 45.9, 9.1, 47.0],
            bands: NDVI_BANDS,
            scl: Some(SCL),
        },
        SampleRecipeScene {
            id: "s2-alps-false-color-scl",
            title: "Sentinel-2 L2A False Color NIR + SCL (Swiss Alps)",
            description: "10 m B08/B04/B03 NIR false-color composite with 20 m ESA SCL cloud/shadow masking.",
            recipe_id: "false_color_nir",
            bbox: [7.6, 45.9, 9.1, 47.0],
            bands: FALSE_COLOR_BANDS,
            scl: Some(SCL),
        },
    ];
    RECIPES
}
