//! Library surface of `veldmap-app` exposing [`VeldmapApp`] and [`AppError`]
//! for the binary entry point plus the headless-testable rendering boundary
//! (`render`, `gpu_tiles`, `basemap_policy`, [`BUNDLED_BASEMAP_CONFIG`], and
//! [`BUNDLED_BASEMAP_JPEG`]) for GPU smoke tests and `naga` shader validation.

mod app;
mod basemap;
mod dialogs;
pub mod fonts;
pub mod gpu_tiles;
mod layers;
pub mod render;
mod scene;
mod session;
mod settings;
mod storage;
mod ui;
mod viewport_bridge;
mod viewport_input;

pub use app::{AppError, VeldmapApp};
pub use basemap::bundled::{BUNDLED_BASEMAP_CONFIG, BUNDLED_BASEMAP_JPEG, BUNDLED_MAX_LEVEL};
pub use basemap::policy as basemap_policy;
