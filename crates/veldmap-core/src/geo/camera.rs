//! Precision nadir WGS84 globe camera using `f64` `DMat4` infinite Reverse-Z
//! projection and orthonormal frame controls.
//!
//! All camera internals are private behind view-agnostic queries. The v1 globe
//! lies directly on the WGS84 ellipsoid (`height_m = 0.0`) with an altitude
//! floor of [`MIN_HEIGHT_M`] (2 m) for centimetre-scale orthophoto inspection.

use glam::{DMat3, DMat4, DQuat, DVec3, DVec4};

use super::cull;
use super::geodesy::{self, Geodetic};

/// Zoom scale factor per discrete step (shared between 3D globe pinch/wheel
/// scaling and future 2D inspection views).
pub const ZOOM_PER_CLICK: f64 = 1.25;

/// Altitude multiplier applied per wheel notch on the 3D globe (inverse of
/// [`ZOOM_PER_CLICK`]).
const ZOOM_PER_STEP: f64 = 1.0 / ZOOM_PER_CLICK;

/// Vertical field of view in degrees.
const FOV_Y_DEG: f64 = 50.0;

/// Minimum camera altitude above the WGS84 ellipsoid in metres.
///
/// At 2 m altitude with a 50-degree vertical FOV on a 1000-row viewport,
/// ground sample distance is ~1.87 mm/px — finer than 2 cm drone orthophotos.
pub const MIN_HEIGHT_M: f64 = 2.0;

/// Maximum camera altitude above the WGS84 ellipsoid in metres.
const MAX_HEIGHT_M: f64 = 80_000_000.0;

/// Allowed camera altitude bounds `(min_m, max_m)`.
const HEIGHT_RANGE_M: (f64, f64) = (MIN_HEIGHT_M, MAX_HEIGHT_M);

/// Grazing magnification factor at which cursor zoom hold begins softening
/// toward the center of the viewport.
const GRAZE_LIMIT: f64 = 2.0;

/// Framing margin applied around requested angular radii in [`Camera::focus`].
const FRAME_MARGIN: f64 = 1.3;

/// Duration of smooth quaternion flights in seconds.
const FLIGHT_S: f64 = 0.6;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Flight {
    from: DQuat,
    to: DQuat,
    from_height_m: f64,
    to_height_m: f64,
    passed_s: f64,
}

/// Nadir 3D WGS84 camera represented as an orthonormal frame `(at, up)` and
/// ellipsoidal altitude `height_m`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// Outward unit normal to the WGS84 ellipsoid directly under the camera.
    at: DVec3,
    /// Unit vector orthogonal to `at` indicating the top of the screen.
    up: DVec3,
    /// Ellipsoidal altitude above the WGS84 surface in metres.
    height_m: f64,
    /// Active smooth flight, if any.
    flight: Option<Flight>,
}

impl Default for Camera {
    fn default() -> Self {
        Self::at(45.0, 60.0, 14_000_000.0)
    }
}

const NORTH_ALIGNED_DOT: f64 = 1.0 - 1e-12;

fn half_fov_tan() -> f64 {
    (FOV_Y_DEG * 0.5).to_radians().tan()
}

fn sanitize_aspect(aspect: f64) -> f64 {
    if aspect.is_finite() && aspect > 0.0 {
        aspect
    } else {
        1.0
    }
}

impl Camera {
    /// Construct a North-up camera positioned over `(lat_deg, lon_deg)` at
    /// `height_m` metres above the WGS84 ellipsoid. Non-finite inputs fall back
    /// to `(0.0, 0.0)` / [`MIN_HEIGHT_M`].
    pub fn at(lat_deg: f64, lon_deg: f64, height_m: f64) -> Self {
        let lat_deg = if lat_deg.is_finite() {
            lat_deg.clamp(-90.0, 90.0)
        } else {
            0.0
        };
        let lon_deg = if lon_deg.is_finite() { lon_deg } else { 0.0 };
        let height_m = if height_m.is_finite() {
            height_m.clamp(HEIGHT_RANGE_M.0, HEIGHT_RANGE_M.1)
        } else {
            HEIGHT_RANGE_M.0
        };
        let at = geodesy::unit(lat_deg, lon_deg);
        Self {
            at,
            up: frame_up(at, DVec3::Z),
            height_m,
            flight: None,
        }
    }

    /// Geodetic `(lat_deg, lon_deg)` directly under the camera eye along the
    /// outward WGS84 ellipsoid normal (`at`), matching the center pixel of
    /// [`Camera::probe`].
    pub fn center(&self) -> (f64, f64) {
        geodesy::angles(self.at)
    }

    /// Current ellipsoidal altitude above the WGS84 surface in metres.
    pub fn height_m(&self) -> f64 {
        self.height_m
    }

    /// Vertical focal length in pixels for a viewport of `height_px` physical
    /// pixels, derived from the camera's active vertical field of view.
    pub(crate) fn focal_length_px(&self, height_px: f64) -> f64 {
        (height_px * 0.5) / half_fov_tan()
    }

    /// Clockwise heading of the top of the viewport from geographic North in
    /// degrees (`[-180, 180]`, `0` = North up). Returns `0.0` at the poles.
    pub fn heading_deg(&self) -> f64 {
        let Some(north) = north_up(self.at) else {
            return 0.0;
        };
        let east = north.cross(self.at);
        self.up.dot(east).atan2(self.up.dot(north)).to_degrees()
    }

    /// Return whether a smooth quaternion flight is currently active.
    pub fn is_flying(&self) -> bool {
        self.flight.is_some()
    }

    /// Exact `f64` ECEF position of the camera eye in **metres**.
    pub fn eye_ecef(&self) -> DVec3 {
        geodesy::world(self.geodetic())
    }

    /// Orthonormal camera basis `(right, up, forward)` in ECEF coordinates,
    /// with `forward = -self.at` aimed strictly along the geodetic nadir
    /// normal so the viewport center ray hits [`Camera::center`] with zero
    /// mid-latitude parallax or orbit-to-ground zoom drift.
    pub fn basis(&self) -> (DVec3, DVec3, DVec3) {
        let forward = -self.at;
        let side = forward.cross(self.up).normalize();
        (side, side.cross(forward), forward)
    }

    /// Angular radius in degrees visible from the center of the viewport to
    /// the vertical edge.
    fn visible_deg(&self) -> f64 {
        match self.arc_at((FOV_Y_DEG * 0.5).to_radians()) {
            Some(arc) => arc.to_degrees(),
            None => (1.0 / self.radius()).acos().to_degrees(),
        }
    }

    /// Surface metres per pixel along the vertical axis at viewport center
    /// (true nadir ground sample distance: `2 * height_m * tan(fov_y / 2) / height_px`).
    pub fn metres_per_pixel(&self, height_px: f64) -> f64 {
        (2.0 * self.height_m * half_fov_tan()) / height_px.max(1.0)
    }

    /// Rotation-only infinite Reverse-Z view-projection matrix in `f64`
    /// (operating on eye-relative ECEF coordinates in **metres**).
    ///
    /// Translation is intentionally omitted: vertices are shifted relative to
    /// the eye in `f64` via per-tile RTC offsets before multiplication.
    pub fn view_projection(&self, aspect: f64) -> DMat4 {
        let aspect = sanitize_aspect(aspect);
        let near = self.height_m * 0.5;
        perspective_infinite_reverse_rh(FOV_Y_DEG.to_radians(), aspect, near)
            * turn_to(self.basis())
    }

    /// Build a high-precision `f64` frustum + horizon culling frame.
    pub fn frustum(&self, aspect: f64) -> cull::CullFrustum {
        let vp = self.view_projection(aspect);
        cull::CullFrustum::new(&vp, self.eye_ecef())
    }

    /// Orbit the globe by drag deltas `(dx, dy)` expressed as fractions of the
    /// viewport height.
    pub fn orbit(&mut self, dx: f64, dy: f64) {
        if !(dx.is_finite() && dy.is_finite()) {
            return;
        }
        self.flight = None;

        let span = 2.0 * self.visible_deg();
        let turn = (self.up * -dx + self.basis().0 * -dy) * span;
        let angle_deg = turn.length();
        if angle_deg <= 0.0 {
            return;
        }

        let spin = DQuat::from_axis_angle(turn / angle_deg, angle_deg.to_radians());
        self.at = spin * self.at;
        self.up = spin * self.up;
        self.settle();
    }

    /// Rotate the camera heading clockwise around the nadir normal by
    /// `delta_rad` radians (positive values rotate the top of the viewport
    /// clockwise relative to North, increasing [`Self::heading_deg`]).
    pub fn rotate_heading(&mut self, delta_rad: f64) {
        if !delta_rad.is_finite() || delta_rad == 0.0 {
            return;
        }
        self.flight = None;
        let spin = DQuat::from_axis_angle(self.at, -delta_rad);
        self.up = spin * self.up;
        self.settle();
    }

    /// Zoom by `steps` scroll notches while holding the point under
    /// `(hold_x, hold_y)` (offset from viewport center in fractions of
    /// viewport height; positive `hold_y` is downward).
    pub fn zoom_at(&mut self, steps: f64, hold_x: f64, hold_y: f64) {
        if !(steps.is_finite() && hold_x.is_finite() && hold_y.is_finite()) {
            return;
        }
        self.flight = None;

        let (right, above) = frame_offset(hold_x, -hold_y);

        let height_m =
            (self.height_m * ZOOM_PER_STEP.powf(steps)).clamp(HEIGHT_RANGE_M.0, HEIGHT_RANGE_M.1);
        let alpha_seen = right.hypot(above).atan();
        let share = self
            .grip(alpha_seen)
            .min(Self { height_m, ..*self }.grip(alpha_seen));
        let (right, above) = (right * share, above * share);
        let alpha = right.hypot(above).atan();

        let held = geodesy::intersect_at(self.eye_ecef(), self.look(right, above), 0.0);
        let before = self.arc_at(alpha);

        let moved = (height_m - self.height_m).abs() > 0.0;
        self.height_m = height_m;

        let Some(held) = held.filter(|_| moved) else {
            return;
        };

        if let (Some(before), Some(after)) = (before, self.arc_at(alpha)) {
            let (side, up, _) = self.basis();
            self.swing(side * right + up * above, before - after);
        }

        let Some(slope) = self.slope_at(alpha) else {
            return;
        };
        self.correct_zoom_hold(held, right, above, alpha, slope);
    }

    fn correct_zoom_hold(&mut self, held: DVec3, right: f64, above: f64, alpha: f64, slope: f64) {
        let across_slope = match self.arc_at(alpha) {
            Some(arc) if alpha > 0.0 => arc.tan() / alpha,
            _ => slope,
        };
        let length = right.hypot(above);
        for _ in 0..2 {
            let Some((was_right, was_above)) = self.seen_at(held) else {
                return;
            };
            let now = was_right.hypot(was_above);
            let (along, across) = if length > 0.0 && now > 0.0 {
                (
                    now.atan() * (right * was_right + above * was_above) / (length * now) - alpha,
                    now.atan() * (right * was_above - above * was_right) / (length * now),
                )
            } else {
                (now.atan() - alpha, 0.0)
            };

            let (side, up, _) = self.basis();
            let (toward, sideways) = if length > 0.0 {
                (
                    (side * right + up * above) / length,
                    (up * right - side * above) / length,
                )
            } else if now > 0.0 {
                ((side * was_right + up * was_above) / now, up)
            } else {
                (side, up)
            };
            self.swing(toward, along * slope);
            self.swing(sideways, across * across_slope);
        }
    }

    /// Start a smooth North-up flight framing a circle of `radius_deg` around
    /// `(lat_deg, lon_deg)`.
    pub fn focus(&mut self, lat_deg: f64, lon_deg: f64, radius_deg: f64) {
        if !(lat_deg.is_finite() && lon_deg.is_finite() && radius_deg.is_finite()) {
            return;
        }
        let height_m = height_for(radius_deg * FRAME_MARGIN);
        self.fly_to(lat_deg, lon_deg, height_m);
    }

    fn make_flight(&self, to: DQuat, to_height_m: f64) -> Flight {
        Flight {
            from: orientation(self.at, self.up),
            to,
            from_height_m: self.height_m,
            to_height_m,
            passed_s: 0.0,
        }
    }

    /// Start a smooth North-up flight to `(lat_deg, lon_deg)` at `height_m`.
    pub fn fly_to(&mut self, lat_deg: f64, lon_deg: f64, height_m: f64) {
        self.fly_to_pose(lat_deg, lon_deg, height_m, 0.0);
    }

    /// Start a smooth flight to `(lat_deg, lon_deg)` at `height_m` with
    /// clockwise heading `heading_deg` degrees from North.
    pub fn fly_to_pose(&mut self, lat_deg: f64, lon_deg: f64, height_m: f64, heading_deg: f64) {
        if !(lat_deg.is_finite()
            && lon_deg.is_finite()
            && height_m.is_finite()
            && heading_deg.is_finite())
        {
            return;
        }
        let at = geodesy::unit(lat_deg.clamp(-90.0, 90.0), lon_deg);
        let mut up = frame_up(at, self.up);
        if heading_deg.abs() > 1e-12 {
            up = DQuat::from_axis_angle(at, -heading_deg.to_radians()) * up;
        }
        let height_m = height_m.clamp(HEIGHT_RANGE_M.0, HEIGHT_RANGE_M.1);
        let to = orientation(at, up);

        let already_there = self.at.dot(at) > NORTH_ALIGNED_DOT
            && self.up.dot(up) > NORTH_ALIGNED_DOT
            && (self.height_m - height_m).abs() <= height_m.max(1.0) * 1e-12;
        if already_there {
            self.flight = None;
            return;
        }

        if self
            .flight
            .is_some_and(|flight| flight.to == to && flight.to_height_m == height_m)
        {
            return;
        }

        self.flight = Some(self.make_flight(to, height_m));
    }

    /// Smoothly rotate the viewport back to North-up while keeping the current
    /// target position and altitude. Returns `true` if a rotation flight was
    /// started (`false` if already North-up).
    pub fn reset_north(&mut self) -> bool {
        let up = frame_up(self.at, self.up);
        if self.up.dot(up) > NORTH_ALIGNED_DOT {
            return false;
        }
        let to = orientation(self.at, up);
        self.flight = Some(self.make_flight(to, self.height_m));
        true
    }

    /// Advance any active flight by `dt_s` seconds. Returns `true` if the
    /// camera moved on this tick.
    pub fn advance(&mut self, dt_s: f64) -> bool {
        let Some(mut flight) = self.flight else {
            return false;
        };

        flight.passed_s += dt_s.max(0.0);
        let part = (flight.passed_s / FLIGHT_S).clamp(0.0, 1.0);
        let eased = part * part * (3.0 - 2.0 * part);

        let spin = flight.from.slerp(flight.to, eased);
        self.at = spin * DVec3::Z;
        self.up = spin * DVec3::Y;
        let times = flight.to_height_m / flight.from_height_m;
        self.height_m = flight.from_height_m * times.powf(eased);

        if part >= 1.0 {
            self.at = flight.to * DVec3::Z;
            self.up = flight.to * DVec3::Y;
            self.height_m = flight.to_height_m;
            self.flight = None;
        } else {
            self.flight = Some(flight);
        }
        true
    }

    /// World-space ray `(origin, direction)` through normalized viewport
    /// coordinates `(x, y)` where `(0, 0)` is top-left and `(1, 1)` is
    /// bottom-right. Non-positive or non-finite `aspect` falls back to `1.0`,
    /// matching [`Camera::view_projection`].
    fn ray(&self, x: f64, y: f64, aspect: f64) -> (DVec3, DVec3) {
        let aspect = sanitize_aspect(aspect);
        let (right, above) = frame_offset((x - 0.5) * aspect, 0.5 - y);
        (self.eye_ecef(), self.look(right, above))
    }

    /// Intersect the cursor ray at normalized `(x, y)` with the WGS84
    /// ellipsoid (`height_m = 0.0`), returning `(lat_deg, lon_deg)`.
    pub fn probe(&self, x: f64, y: f64, aspect: f64) -> Option<(f64, f64)> {
        let (eye, direction) = self.ray(x, y, aspect);
        geodesy::intersect_at(eye, direction, 0.0).map(geodesy::surface_at)
    }

    fn swing(&mut self, toward: DVec3, angle: f64) {
        let axis = self.at.cross(toward);
        let length = axis.length();
        if length <= 0.0 || angle == 0.0 || !angle.is_finite() {
            return;
        }
        let spin = DQuat::from_axis_angle(axis / length, angle);
        self.at = spin * self.at;
        self.up = spin * self.up;
        self.settle();
    }

    fn geodetic(&self) -> Geodetic {
        let (lat_deg, lon_deg) = geodesy::angles(self.at);
        Geodetic {
            lat_deg,
            lon_deg,
            height_m: self.height_m,
        }
    }

    fn settle(&mut self) {
        self.at = self.at.normalize();
        let up = self.up.reject_from_normalized(self.at);
        if up.length_squared() > 1e-12 {
            self.up = up.normalize();
        }
    }

    fn grazing(&self, alpha: f64) -> Option<f64> {
        let sine = self.radius() * alpha.sin();
        (sine < 1.0).then(|| (1.0 - sine * sine).sqrt())
    }

    fn grip(&self, alpha: f64) -> f64 {
        self.grazing(alpha)
            .map_or(0.0, |cosine| (cosine * GRAZE_LIMIT).min(1.0))
    }

    fn slope_at(&self, alpha: f64) -> Option<f64> {
        Some(self.radius() * alpha.cos() / self.grazing(alpha)? - 1.0)
    }

    fn arc_at(&self, alpha: f64) -> Option<f64> {
        let sine = self.radius() * alpha.sin();
        (sine < 1.0).then(|| sine.asin() - alpha)
    }

    fn radius(&self) -> f64 {
        1.0 + self.height_m / geodesy::SEMI_MAJOR_M
    }

    fn look(&self, right: f64, above: f64) -> DVec3 {
        let (side, up, forward) = self.basis();
        (forward + side * right + up * above).normalize()
    }

    fn seen_at(&self, point: DVec3) -> Option<(f64, f64)> {
        let (side, up, forward) = self.basis();
        let direction = point - self.eye_ecef();
        let ahead = direction.dot(forward);
        (ahead > 0.0).then(|| (direction.dot(side) / ahead, direction.dot(up) / ahead))
    }
}

fn frame_offset(right_share: f64, above_share: f64) -> (f64, f64) {
    let half = half_fov_tan();
    (2.0 * right_share * half, 2.0 * above_share * half)
}

fn north_up(at: DVec3) -> Option<DVec3> {
    let up = DVec3::Z.reject_from_normalized(at);
    (up.length_squared() > 1e-12).then(|| up.normalize())
}

fn frame_up(at: DVec3, previous: DVec3) -> DVec3 {
    north_up(at).unwrap_or_else(|| {
        let up = previous.reject_from_normalized(at);
        if up.length_squared() > 1e-12 {
            up.normalize()
        } else {
            at.any_orthonormal_vector()
        }
    })
}

fn orientation(at: DVec3, up: DVec3) -> DQuat {
    DQuat::from_mat3(&DMat3::from_cols(up.cross(at), up, at))
}

/// Ellipsoidal altitude in metres from which a circle of `radius_deg` fits
/// vertically inside the viewport.
fn height_for(radius_deg: f64) -> f64 {
    let half = (FOV_Y_DEG * 0.5).to_radians();
    let arc = radius_deg
        .clamp(0.0, (std::f64::consts::FRAC_PI_2 - half).to_degrees())
        .to_radians();
    let radius = (arc + half).sin() / half.sin();
    (radius - 1.0) * geodesy::SEMI_MAJOR_M
}

fn perspective_infinite_reverse_rh(fov_y_rad: f64, aspect_ratio: f64, z_near: f64) -> DMat4 {
    let f = 1.0 / (fov_y_rad * 0.5).tan();
    DMat4::from_cols(
        DVec4::new(f / aspect_ratio, 0.0, 0.0, 0.0),
        DVec4::new(0.0, f, 0.0, 0.0),
        DVec4::new(0.0, 0.0, 0.0, -1.0),
        DVec4::new(0.0, 0.0, z_near, 0.0),
    )
}

fn turn_to((side, up, forward): (DVec3, DVec3, DVec3)) -> DMat4 {
    let rotation = DMat3::from_cols(side, up, -forward).transpose();
    DMat4::from_cols(
        rotation.x_axis.extend(0.0),
        rotation.y_axis.extend(0.0),
        rotation.z_axis.extend(0.0),
        DVec3::ZERO.extend(1.0),
    )
}

#[cfg(test)]
#[path = "camera_tests.rs"]
mod tests;
