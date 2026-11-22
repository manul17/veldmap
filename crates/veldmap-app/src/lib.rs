//! Library surface of `veldmap-app` exposing the decoupled offscreen
//! `GlobeRenderer`, per-tile RTC quadtree mesh generator, `GpuTileCache`,
//! `PendingUploads`, and HUD formatting utilities for headless GPU smoke tests
//! and `naga` shader validation.

pub mod app;
pub mod basemap;
pub mod basemap_ui;
pub mod gpu_tiles;
pub mod hud;
pub mod mesh;
pub mod render;
