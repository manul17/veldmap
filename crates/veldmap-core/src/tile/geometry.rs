//! Planetary tile projection bounds and surface sampling (`Geodetic`,
//! `WebMercator`, and adaptive `PolarCap`), shared by quadtree traversal,
//! offline equirectangular tile resampling, and GPU mesh generation.

use crate::geo::projection::{WEB_MERCATOR_LIMIT_DEG, web_mercator_lat_deg};

use super::key::QuadTileId;
use super::scheme::PlanetaryScheme;

/// Stereographic disk radius corresponding to `|lat| = WEB_MERCATOR_LIMIT_DEG`
/// (`tan((90° - 85.05112878°) / 2)`).
pub const POLAR_R_CAP: f64 = 0.043_213_918_263_772_28;

/// Latitude/longitude bounds and raw texture range for a non-polar tile.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LatLonTileBounds {
    pub(crate) lon_west_deg: f64,
    pub(crate) lon_east_deg: f64,
    /// Clamped mesh north latitude (`<= WEB_MERCATOR_LIMIT_DEG`).
    pub(crate) lat_north_deg: f64,
    /// Clamped mesh south latitude (`>= -WEB_MERCATOR_LIMIT_DEG`).
    pub(crate) lat_south_deg: f64,
    /// Unclamped tile north latitude (for exact `Geodetic` texture UV mapping).
    pub(crate) raw_lat_north_deg: f64,
    /// Unclamped tile south latitude (for exact `Geodetic` texture UV mapping).
    pub(crate) raw_lat_south_deg: f64,
}

/// Surface sample evaluated at mesh-local fractions `(s, t)` on a planetary tile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileSurfaceSample {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub u_tex: f64,
    pub v_tex: f64,
    /// Stereographic polar disk `(dx, dy)` coordinates (`0.0` outside polar caps).
    pub polar_dx: f64,
    pub polar_dy: f64,
}

/// Compute geographic `(lon_west_deg, lon_east_deg)` bounds for column `x` of `cols`.
#[inline]
pub fn tile_lon_bounds(x: u32, cols: u32) -> (f64, f64) {
    let cols_f = f64::from(cols);
    let west = -180.0 + 360.0 * f64::from(x) / cols_f;
    let east = -180.0 + 360.0 * (f64::from(x) + 1.0) / cols_f;
    (west, east)
}

/// Compute unclamped geodetic `(lat_north_deg, lat_south_deg)` bounds in
/// `[-90, 90]` for tile row `y` out of `rows` rows.
#[inline]
pub fn geodetic_lat_bounds(y: u32, rows: u32) -> (f64, f64) {
    let rows_f = f64::from(rows);
    let north = 90.0 - 180.0 * f64::from(y) / rows_f;
    let south = 90.0 - 180.0 * (f64::from(y) + 1.0) / rows_f;
    (north, south)
}

/// Compute normalized cap square `(u_west, u_east, v_north, v_south)` bounds
/// in `[-1, 1]` for polar cap tile `(x, y)` inside an `n x n` grid.
#[inline]
pub fn polar_cap_uv_bounds(x: u32, y: u32, n: u32) -> (f64, f64, f64, f64) {
    let n_f = f64::from(n);
    let u0 = -1.0 + 2.0 * f64::from(x) / n_f;
    let u1 = -1.0 + 2.0 * (f64::from(x) + 1.0) / n_f;
    let v0 = -1.0 + 2.0 * f64::from(y) / n_f;
    let v1 = -1.0 + 2.0 * (f64::from(y) + 1.0) / n_f;
    (u0, u1, v0, v1)
}

/// Linearly interpolate along a 1D interval `[start, end]` at fraction `t`.
#[inline]
pub(crate) fn lerp_f64(start: f64, end: f64, t: f64) -> f64 {
    start + (end - start) * t
}

/// Compute Web Mercator north/south Y coordinates `(merc_n, merc_s)` in radians
/// for tile row `y` out of `rows` rows.
#[inline]
pub fn web_mercator_y_bounds(y: u32, rows: u32) -> (f64, f64) {
    let rows_f = f64::from(rows);
    let merc_n = std::f64::consts::PI * (1.0 - 2.0 * f64::from(y) / rows_f);
    let merc_s = std::f64::consts::PI * (1.0 - 2.0 * (f64::from(y) + 1.0) / rows_f);
    (merc_n, merc_s)
}

/// Compute geographic bounds for a `Geodetic` or `WebMercator` tile, returning
/// `None` if the tile is a `PolarCap`, out of bounds, or lies entirely inside
/// the `|lat| >= WEB_MERCATOR_LIMIT_DEG` polar caps.
pub(crate) fn lat_lon_bounds(id: QuadTileId) -> Option<LatLonTileBounds> {
    let (cols, rows) = id.checked_grid_dims()?;
    let (lon_west_deg, lon_east_deg) = tile_lon_bounds(id.x, cols);

    match id.scheme {
        PlanetaryScheme::Geodetic => {
            let (raw_lat_north_deg, raw_lat_south_deg) = geodetic_lat_bounds(id.y, rows);
            if raw_lat_south_deg >= WEB_MERCATOR_LIMIT_DEG
                || raw_lat_north_deg <= -WEB_MERCATOR_LIMIT_DEG
            {
                return None;
            }
            let lat_north_deg = raw_lat_north_deg.min(WEB_MERCATOR_LIMIT_DEG);
            let lat_south_deg = raw_lat_south_deg.max(-WEB_MERCATOR_LIMIT_DEG);
            Some(LatLonTileBounds {
                lon_west_deg,
                lon_east_deg,
                lat_north_deg,
                lat_south_deg,
                raw_lat_north_deg,
                raw_lat_south_deg,
            })
        }
        PlanetaryScheme::WebMercator => {
            let (merc_n, merc_s) = web_mercator_y_bounds(id.y, rows);
            let lat_north_deg = web_mercator_lat_deg(merc_n);
            let lat_south_deg = web_mercator_lat_deg(merc_s);
            Some(LatLonTileBounds {
                lon_west_deg,
                lon_east_deg,
                lat_north_deg,
                lat_south_deg,
                raw_lat_north_deg: lat_north_deg,
                raw_lat_south_deg: lat_south_deg,
            })
        }
        PlanetaryScheme::PolarCap { .. } => None,
    }
}

/// Evaluate a point on the adaptive Polar Stereographic cap at normalized cap
/// square coordinates `(u, v) in [-1, 1]^2` using the smooth elliptical
/// square-to-disk diffeomorphism (`dx = u * sqrt(1 - v^2 / 2)`,
/// `dy = v * sqrt(1 - u^2 / 2)`).
///
/// Returns `(lat_deg, lon_deg, dx, dy)` where `dx^2 + dy^2 == 1.0` maps
/// strictly onto `|lat_deg| == WEB_MERCATOR_LIMIT_DEG`, `(0, 0)` maps to the
/// exact pole, and standard `(row -> +v, col -> +u)` triangle winding faces
/// outward on both North and South caps.
pub fn polar_cap_point(north: bool, u: f64, v: f64) -> (f64, f64, f64, f64) {
    let dx = u * (1.0 - 0.5 * v * v).max(0.0).sqrt();
    let dy = v * (1.0 - 0.5 * u * u).max(0.0).sqrt();
    let rho = dx.hypot(dy).min(1.0);
    let colat_rad = 2.0 * (rho * POLAR_R_CAP).atan();
    let colat_deg = colat_rad.to_degrees();
    let lat_deg = if north {
        90.0 - colat_deg
    } else {
        -90.0 + colat_deg
    };
    let lon_deg = if rho == 0.0 {
        0.0
    } else if north {
        (-dy).atan2(dx).to_degrees()
    } else {
        dy.atan2(dx).to_degrees()
    };
    (lat_deg, lon_deg, dx, dy)
}

/// Evaluate a [`TileSurfaceSample`] on any planetary tile (`Geodetic`,
/// `WebMercator`, or `PolarCap`) at mesh-local fractions `(s, t) in [0, 1]^2`
/// (`s`: west->east / left->right, `t`: north->south / top->bottom).
pub fn sample_tile_surface(id: QuadTileId, s: f64, t: f64) -> Option<TileSurfaceSample> {
    match id.scheme {
        PlanetaryScheme::Geodetic => {
            let b = lat_lon_bounds(id)?;
            let lon_deg = lerp_f64(b.lon_west_deg, b.lon_east_deg, s);
            let lat_deg = lerp_f64(b.lat_north_deg, b.lat_south_deg, t);
            let raw_span = (b.raw_lat_north_deg - b.raw_lat_south_deg).max(1e-12);
            let v_tex = (b.raw_lat_north_deg - lat_deg) / raw_span;
            Some(TileSurfaceSample {
                lat_deg,
                lon_deg,
                u_tex: s,
                v_tex,
                polar_dx: 0.0,
                polar_dy: 0.0,
            })
        }
        PlanetaryScheme::WebMercator => {
            let (cols, rows) = id.checked_grid_dims()?;
            let (lon_west_deg, lon_east_deg) = tile_lon_bounds(id.x, cols);
            let lon_deg = lerp_f64(lon_west_deg, lon_east_deg, s);
            let (merc_n, merc_s) = web_mercator_y_bounds(id.y, rows);
            let merc = lerp_f64(merc_n, merc_s, t);
            let lat_deg = web_mercator_lat_deg(merc);
            Some(TileSurfaceSample {
                lat_deg,
                lon_deg,
                u_tex: s,
                v_tex: t,
                polar_dx: 0.0,
                polar_dy: 0.0,
            })
        }
        PlanetaryScheme::PolarCap { north } => {
            let (n, _) = id.checked_grid_dims()?;
            let (u0, u1, v0, v1) = polar_cap_uv_bounds(id.x, id.y, n);
            let u = lerp_f64(u0, u1, s);
            let v = lerp_f64(v0, v1, t);
            let (lat_deg, lon_deg, polar_dx, polar_dy) = polar_cap_point(north, u, v);
            Some(TileSurfaceSample {
                lat_deg,
                lon_deg,
                u_tex: s,
                v_tex: t,
                polar_dx,
                polar_dy,
            })
        }
    }
}

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod tests;
