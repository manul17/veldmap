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
}

impl SampleScene {
    /// Combined `(scale, offset)` metadata if either radiometric field is set.
    pub const fn scale_offset(&self) -> Option<(f64, f64)> {
        combine_scale_offset(self.scale, self.offset)
    }
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
        },
    ];
    SAMPLES
}
