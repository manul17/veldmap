//! Shared scroll-wheel zoom step constants for both the 3D globe and 2D
//! raster inspector.

/// Scale factor applied per wheel click in the 2D inspector.
pub const ZOOM_PER_CLICK: f64 = 1.25;

/// Altitude multiplier applied per wheel click on the 3D globe (inverse of
/// [`ZOOM_PER_CLICK`]).
pub const ZOOM_PER_STEP: f64 = 1.0 / ZOOM_PER_CLICK;
