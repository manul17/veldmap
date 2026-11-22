//! WGS84 geodesy, map projections, MGRS indexing, nadir globe camera,
//! frustum culling, chord sagitta helpers, and footprint hit-testing.

pub mod camera;
pub mod coords;
pub mod cull;
pub mod footprint;
pub mod geodesy;
pub mod mgrs;
pub mod projection;
pub mod sag;

pub use camera::Camera;
pub use coords::{LocationKind, ParseLocationError, ParsedLocation, parse_location};
pub use cull::{Ball, Frame as CullFrame};
pub use footprint::{Frame as FootprintFrame, GeoPoint, Ring};
pub use geodesy::{Ellipsoid, Geodetic, WGS84};
pub use mgrs::{Frame as MgrsFrame, MgrsError};
pub use projection::{Datum, Projection, System};
