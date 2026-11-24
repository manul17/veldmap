//! WGS84 geodesy, map projections, MGRS indexing, nadir globe camera,
//! frustum culling, chord sagitta helpers, and footprint hit-testing.

pub mod camera;
pub mod coords;
pub mod cull;
pub mod footprint;
pub mod geodesy;
pub mod mgrs;
pub mod overlay;
pub mod projection;
pub mod sag;

pub use camera::Camera;
pub use coords::{LocationKind, ParseLocationError, ParsedLocation, parse_location};
pub use cull::{Ball, CullFrustum};
pub use footprint::{FootprintBounds, GeoPoint, Ring};
pub use geodesy::{Ellipsoid, Geodetic, WGS84};
pub use mgrs::{MgrsError, MgrsGranuleFrame};
pub use overlay::{
    Frame, MAX_WARP_SEGMENTS, MIN_WARP_SEGMENTS, OverlayCullTree, Tie, TieGrid, WarpPatch,
    build_warp_patch,
};
pub use projection::{Datum, Projection, System};
