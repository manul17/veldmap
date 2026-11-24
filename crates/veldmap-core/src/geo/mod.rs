//! WGS84 geodesy, map projections, georeference bindings, MGRS indexing,
//! nadir globe camera, frustum culling, chord sagitta helpers, and footprint
//! hit-testing.

mod binding;
pub mod camera;
mod coords;
mod cull;
pub mod footprint;
pub mod geodesy;
pub mod mgrs;
pub mod projection;
pub(crate) mod sag;

pub use binding::{BindingError, Frame, Tie, TieGrid};
pub use camera::Camera;
pub use coords::{LocationKind, ParseLocationError, ParsedLocation, parse_location};
pub use cull::{Ball, CullFrustum};
pub use footprint::{FootprintBounds, GeoPoint, Ring};
pub use geodesy::{Ellipsoid, Geodetic, WGS84};
pub use mgrs::{MgrsError, MgrsGranuleFrame};
pub use projection::{Datum, Projection, System};
