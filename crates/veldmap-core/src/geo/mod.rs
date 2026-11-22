//! WGS84 geodesy, map projections, MGRS indexing, nadir globe camera,
//! frustum culling, footprint hit-testing, and surface outlines.

pub mod camera;
pub mod coords;
pub mod cull;
pub mod footprint;
pub mod geodesy;
pub mod mgrs;
pub mod outlines;
pub mod projection;
pub mod sag;

pub use camera::Camera;
pub use coords::{LocationKind, ParseLocationError, ParsedLocation, parse_location};
pub use cull::{Ball, Frame as CullFrame};
pub use footprint::{Frame as FootprintFrame, GeoPoint, Ring};
pub use geodesy::{Geodetic, Parallel, World};
pub use mgrs::{Frame as MgrsFrame, MgrsError};
pub use outlines::{Outline, OutlineStyle, OutlineVertex, Outlines, RibbonVertex};
pub use projection::{Datum, Ellipsoid, Projection, System};
