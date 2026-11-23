//! Pure domain logic for VeldMap (zero `egui`, `eframe`, `wgpu`, `image`, or
//! network dependencies).

#![forbid(unsafe_code)]

pub const APP_NAME: &str = "VeldMap";

pub mod budget;
pub mod geo;
pub mod io;
pub mod sync;
pub mod tile;
