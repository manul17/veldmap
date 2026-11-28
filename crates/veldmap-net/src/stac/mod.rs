//! Zero-auth and custom-header STAC API client (`StacClient`) and Sentinel-2
//! L2A & COP-DEM GLO-30 asset/band metadata parser (`raster:bands`
//! `scale`/`offset`).

mod client;
mod parse;
mod types;

pub use self::client::{StacClient, StacItemFlightCache};
pub use self::types::{
    MAX_STAC_PAGINATED_ITEMS, StacAsset, StacAuth, StacCatalogConfig, StacError, StacItem,
    StacNextLink, StacNextMethod, StacRasterBandMeta, StacSearchQuery, StacSearchResult,
    StacSortBy,
};
