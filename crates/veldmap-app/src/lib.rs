//! Library surface of `veldmap-app` exposing the decoupled offscreen
//! `GlobeRenderer`, octant-RTC `GlobeMesh`, and HUD formatting utilities for
//! headless GPU smoke tests and `naga` shader validation.

pub mod app;
pub mod hud;
pub mod mesh;
pub mod render;
