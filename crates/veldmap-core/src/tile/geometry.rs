//! Planetary tile projection bounds and surface sampling (`Geodetic`,
//! `WebMercator`, and adaptive `PolarCap`), shared by quadtree traversal,
//! offline equirectangular tile resampling, and GPU mesh generation.

use crate::geo::projection::{WEB_MERCATOR_LIMIT_DEG, web_mercator_lat_deg};

use super::key::{PlanetaryScheme, QuadTileId};

/// Stereographic disk radius corresponding to `|lat| = WEB_MERCATOR_LIMIT_DEG`
/// (`tan((90° - 85.05112878°) / 2)`).
pub const POLAR_R_CAP: f64 = 0.043_213_918_263_772_28;

/// Latitude/longitude bounds and raw texture range for a non-polar tile.
#[derive(Clone, Copy, Debug)]
pub struct LatLonTileBounds {
    pub lon_west_deg: f64,
    pub lon_east_deg: f64,
    /// Clamped mesh north latitude (`<= WEB_MERCATOR_LIMIT_DEG`).
    pub lat_north_deg: f64,
    /// Clamped mesh south latitude (`>= -WEB_MERCATOR_LIMIT_DEG`).
    pub lat_south_deg: f64,
    /// Unclamped tile north latitude (for exact `Geodetic` texture UV mapping).
    pub raw_lat_north_deg: f64,
    /// Unclamped tile south latitude (for exact `Geodetic` texture UV mapping).
    pub raw_lat_south_deg: f64,
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
pub fn lat_lon_bounds(id: QuadTileId) -> Option<LatLonTileBounds> {
    let (cols, rows) = id.checked_grid_dims()?;
    let cols_f = f64::from(cols);
    let rows_f = f64::from(rows);
    let lon_west_deg = -180.0 + 360.0 * f64::from(id.x) / cols_f;
    let lon_east_deg = -180.0 + 360.0 * f64::from(id.x + 1) / cols_f;

    match id.scheme {
        PlanetaryScheme::Geodetic => {
            let raw_lat_north_deg = 90.0 - 180.0 * f64::from(id.y) / rows_f;
            let raw_lat_south_deg = 90.0 - 180.0 * f64::from(id.y + 1) / rows_f;
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

/// Evaluate a surface point `(lat_deg, lon_deg, u_tex, v_tex, dx, dy)` on any
/// planetary tile (`Geodetic`, `WebMercator`, or `PolarCap`) at mesh-local
/// fractions `(s, t) in [0, 1]^2` (`s`: west->east / left->right,
/// `t`: north->south / top->bottom).
pub fn sample_tile_surface(
    id: QuadTileId,
    s: f64,
    t: f64,
) -> Option<(f64, f64, f64, f64, f64, f64)> {
    match id.scheme {
        PlanetaryScheme::Geodetic => {
            let b = lat_lon_bounds(id)?;
            let lon = b.lon_west_deg + (b.lon_east_deg - b.lon_west_deg) * s;
            let lat = b.lat_north_deg + (b.lat_south_deg - b.lat_north_deg) * t;
            let raw_span = (b.raw_lat_north_deg - b.raw_lat_south_deg).max(1e-12);
            let v_tex = (b.raw_lat_north_deg - lat) / raw_span;
            Some((lat, lon, s, v_tex, 0.0, 0.0))
        }
        PlanetaryScheme::WebMercator => {
            let (cols, rows) = id.checked_grid_dims()?;
            let cols_f = f64::from(cols);
            let lon = -180.0 + 360.0 * (f64::from(id.x) + s) / cols_f;
            let (merc_n, merc_s) = web_mercator_y_bounds(id.y, rows);
            let merc = merc_n + (merc_s - merc_n) * t;
            let lat = web_mercator_lat_deg(merc);
            Some((lat, lon, s, t, 0.0, 0.0))
        }
        PlanetaryScheme::PolarCap { north } => {
            let (n, _) = id.checked_grid_dims()?;
            let n_f = f64::from(n);
            let u = -1.0 + 2.0 * (f64::from(id.x) + s) / n_f;
            let v = -1.0 + 2.0 * (f64::from(id.y) + t) / n_f;
            let (lat, lon, dx, dy) = polar_cap_point(north, u, v);
            Some((lat, lon, s, t, dx, dy))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mercator_y_bounds_and_lat_conversion_cover_web_mercator_extent() {
        let (n0, s0) = web_mercator_y_bounds(0, 1);
        assert!((n0 - std::f64::consts::PI).abs() < 1e-12);
        assert!((s0 + std::f64::consts::PI).abs() < 1e-12);
        assert!((web_mercator_lat_deg(n0) - WEB_MERCATOR_LIMIT_DEG).abs() < 1e-6);
        assert!((web_mercator_lat_deg(s0) + WEB_MERCATOR_LIMIT_DEG).abs() < 1e-6);
        assert!(web_mercator_lat_deg(0.0).abs() < 1e-12);

        let (n1, s1) = web_mercator_y_bounds(0, 2);
        assert!((n1 - std::f64::consts::PI).abs() < 1e-12);
        assert!(s1.abs() < 1e-12);
    }

    #[test]
    fn lat_lon_bounds_handles_polar_clamping_cap_exclusion_and_out_of_range() {
        // Level-0 Geodetic tiles clamp north/south to +/- WEB_MERCATOR_LIMIT_DEG
        // while keeping unclamped raw latitudes for texture UV mapping.
        let geo0 = QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level: 0,
            x: 0,
            y: 0,
        };
        let b = lat_lon_bounds(geo0).expect("geodetic z=0 valid");
        assert_eq!(b.lon_west_deg, -180.0);
        assert_eq!(b.lon_east_deg, 0.0);
        assert_eq!(b.raw_lat_north_deg, 90.0);
        assert_eq!(b.raw_lat_south_deg, -90.0);
        assert!((b.lat_north_deg - WEB_MERCATOR_LIMIT_DEG).abs() < 1e-12);
        assert!((b.lat_south_deg + WEB_MERCATOR_LIMIT_DEG).abs() < 1e-12);

        // At z=6 (rows=64, 2.8125 deg/row), row 0 covers [87.1875, 90] -> fully inside polar cap.
        let inside_cap = QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level: 6,
            x: 0,
            y: 0,
        };
        assert!(lat_lon_bounds(inside_cap).is_none());

        // PolarCap and out-of-range tiles return None.
        assert!(
            lat_lon_bounds(QuadTileId {
                scheme: PlanetaryScheme::PolarCap { north: true },
                level: 0,
                x: 0,
                y: 0,
            })
            .is_none()
        );
        assert!(
            lat_lon_bounds(QuadTileId {
                scheme: PlanetaryScheme::WebMercator,
                level: 0,
                x: 1,
                y: 0,
            })
            .is_none()
        );
    }

    #[test]
    fn sample_tile_surface_clamped_geodetic_v_tex_maps_unclamped_texture_range() {
        let geo0 = QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level: 0,
            x: 0,
            y: 0,
        };
        let (lat_n, lon_w, u0, v0, _, _) =
            sample_tile_surface(geo0, 0.0, 0.0).expect("north-west corner");
        let (lat_s, lon_e, u1, v1, _, _) =
            sample_tile_surface(geo0, 1.0, 1.0).expect("south-east corner");
        assert!((lat_n - WEB_MERCATOR_LIMIT_DEG).abs() < 1e-12);
        assert!((lat_s + WEB_MERCATOR_LIMIT_DEG).abs() < 1e-12);
        assert_eq!((lon_w, lon_e, u0, u1), (-180.0, 0.0, 0.0, 1.0));
        let expected_v0 = (90.0 - WEB_MERCATOR_LIMIT_DEG) / 180.0;
        assert!((v0 - expected_v0).abs() < 1e-12);
        assert!((v1 - (1.0 - expected_v0)).abs() < 1e-12);
    }
}
