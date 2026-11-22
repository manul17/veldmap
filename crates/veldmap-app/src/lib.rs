//! Library surface of `veldmap-app` exposing `VeldmapApp` for the binary entry
//! point plus the headless-testable rendering boundary (`render`, `gpu_tiles`,
//! and [`BUNDLED_BASEMAP_JPEG`]) for GPU smoke tests and `naga` shader
//! validation.

pub mod app;
pub(crate) mod basemap;
pub(crate) mod basemap_ui;
pub(crate) mod bundled_basemap;
pub mod gpu_tiles;
pub(crate) mod hud;
pub(crate) mod mesh;
pub mod render;
pub(crate) mod settings;
pub(crate) mod storage;

pub use bundled_basemap::BUNDLED_BASEMAP_JPEG;
