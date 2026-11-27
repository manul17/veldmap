//! Zero-auth and custom-header STAC API client (`StacClient`) and Sentinel-2
//! L2A & COP-DEM GLO-30 asset/band metadata parser (`raster:bands`
//! `scale`/`offset`).

mod client;
mod parse;
mod types;

pub use self::client::StacClient;
pub use self::types::{
    StacAsset, StacAuth, StacCatalogConfig, StacError, StacItem, StacRasterBandMeta,
    StacSearchQuery, StacSearchResult,
};
