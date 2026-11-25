//! Pure georeference bindings mapping normalized raster fractions `(fx, fy)` in
//! `[0, 1]^2` onto the WGS84 ellipsoid (`Tie`, `TieGrid`, `Frame`).
//!
//! Kept strictly free of `crate::tile` dependencies so `veldmap-core::geo`
//! remains a pure geodesy/projection layer beneath `veldmap-core::tile`.

mod frame;
mod tie_grid;

pub use frame::Frame;
pub use tie_grid::{Tie, TieGrid};

#[cfg(test)]
#[path = "binding_tests.rs"]
mod tests;
