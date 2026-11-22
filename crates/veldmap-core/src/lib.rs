//! Pure domain logic for VeldMap (zero `egui`, `eframe`, `wgpu`, `image`, or
//! network dependencies).

pub const APP_NAME: &str = "VeldMap";

pub mod geo;
pub mod io;
pub mod tile;
