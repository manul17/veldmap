//! Zero-auth and custom-header STAC API client (`StacClient`), Sentinel-2 L2A
//! & COP-DEM GLO-30 asset/band metadata parser (`raster:bands` `scale`/`offset`),
//! and built-in zero-auth onboarding sample scenes.

mod client;
mod parse;
mod samples;
mod types;

pub use self::client::StacClient;
pub use self::samples::{
    SampleRecipeBand, SampleRecipeKind, SampleRecipeScene, SampleScene, builtin_sample_recipes,
    builtin_sample_scenes,
};
pub use self::types::{
    StacAsset, StacAuth, StacCatalogConfig, StacError, StacItem, StacRasterBandMeta,
    StacSearchQuery, StacSearchResult,
};
