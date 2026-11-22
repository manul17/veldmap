//! Scheme-matched WGS84 planetary quadtree and adaptive Polar Stereographic cap
//! with decoupled geometry vs. texture LOD, HiDPI-aware SSE subdivision, 2:1
//! neighbor LOD balancing, and per-cell sag-scaled skirts.

use std::collections::HashMap;

use glam::DVec3;

use crate::geo::camera::{Camera, FOV_Y_DEG};
use crate::geo::cull::{Ball, Frame as CullFrame};
use crate::geo::geodesy;
use crate::geo::projection::WEB_MERCATOR_LIMIT_DEG;
use crate::geo::sag;

use super::key::TileScheme;

/// Default mesh grid cells per quadtree tile side (`16 x 16` quads = 289
/// surface vertices + 68 skirt vertices).
pub const DEFAULT_GRID_CELLS: u32 = 16;

/// Stereographic disk radius corresponding to `|lat| = WEB_MERCATOR_LIMIT_DEG`
/// (`tan((90° - 85.05112878°) / 2)`).
pub const POLAR_R_CAP: f64 = 0.043_213_918_263_772_28;

/// Maximum WGS84 radius-of-curvature ratio relative to `SEMI_MAJOR_M`
/// (`a / (1 - f) / a`), used to keep spherical chord-sag bounds strictly
/// conservative at polar latitudes.
const POLAR_CURVATURE_SCALE: f64 = 1.0 / (1.0 - geodesy::FLATTENING);

/// Return [`POLAR_R_CAP`] without runtime trigonometry.
pub const fn polar_r_cap() -> f64 {
    POLAR_R_CAP
}

/// Geometry identifier for a single planetary quadtree or polar cap tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QuadTileId {
    pub scheme: TileScheme,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl QuadTileId {
    /// Return this tile's parent clamped to at most `max_level`, together with
    /// the exact UV sub-rectangle `[u_off, v_off, u_scale, v_scale]` mapping
    /// this tile into the clamped ancestor.
    pub fn clamp_to_level(self, max_level: u8) -> (Self, [f32; 4]) {
        if self.level <= max_level {
            return (self, [0.0, 0.0, 1.0, 1.0]);
        }
        let dz = self.level - max_level;
        if dz >= 31 {
            return (
                Self {
                    scheme: self.scheme,
                    level: max_level,
                    x: 0,
                    y: 0,
                },
                [0.0, 0.0, 0.0, 0.0],
            );
        }
        let mask = (1u32 << dz) - 1;
        let scale = 1.0_f64 / f64::from(1u32 << dz);
        let u0 = f64::from(self.x & mask) * scale;
        let v0 = f64::from(self.y & mask) * scale;
        (
            Self {
                scheme: self.scheme,
                level: max_level,
                x: self.x >> dz,
                y: self.y >> dz,
            },
            [u0 as f32, v0 as f32, scale as f32, scale as f32],
        )
    }

    /// Return the 4 child tiles of this planetary node.
    pub fn children(self) -> [Self; 4] {
        let next = self.level + 1;
        let cx = self.x << 1;
        let cy = self.y << 1;
        [
            Self {
                scheme: self.scheme,
                level: next,
                x: cx,
                y: cy,
            },
            Self {
                scheme: self.scheme,
                level: next,
                x: cx + 1,
                y: cy,
            },
            Self {
                scheme: self.scheme,
                level: next,
                x: cx,
                y: cy + 1,
            },
            Self {
                scheme: self.scheme,
                level: next,
                x: cx + 1,
                y: cy + 1,
            },
        ]
    }

    /// Return the coarser parent node, or `None` at `level == 0`.
    pub const fn parent(self) -> Option<Self> {
        let Some((level, x, y)) = self.scheme.parent(self.level, self.x, self.y) else {
            return None;
        };
        Some(Self {
            scheme: self.scheme,
            level,
            x,
            y,
        })
    }
}

/// Configuration for planetary quadtree traversal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadtreeConfig {
    /// Active non-polar planetary scheme (`TileScheme::Geodetic` or
    /// `TileScheme::WebMercator`).
    pub scheme: TileScheme,
    /// Maximum texture level offered by the active basemap (`0` requests are
    /// ever emitted above this level; geometry continues subdividing to
    /// `max_geom_level` and samples ancestor sub-rects).
    pub max_tex_level: u8,
    /// Maximum geometry subdivision level (`16` for `Geodetic`, `17` for
    /// `WebMercator` — avoids `f32` UV stepping while keeping sub-millimetre
    /// chord sag at `2 m` altitude).
    pub max_geom_level: u8,
    /// Maximum texture level offered for the adaptive Polar Stereographic caps.
    pub polar_max_tex_level: u8,
    /// Maximum geometry subdivision level for the adaptive Polar Stereographic
    /// caps (`14` yields `< 0.01 mm` chord sag at `2 m`).
    pub polar_max_geom_level: u8,
    /// Quads per tile side (default [`DEFAULT_GRID_CELLS`] = 16).
    pub grid_cells: u32,
    /// Texels per basemap tile side (typically 256).
    pub tile_px: u32,
    /// Maximum tolerated geometric chord-sag error in physical pixels.
    pub max_geom_sse_px: f64,
    /// Maximum tolerated basemap texel footprint in physical pixels (only
    /// evaluated while `level < max_tex_level`).
    pub max_tex_sse_px: f64,
}

impl Default for QuadtreeConfig {
    fn default() -> Self {
        Self::bundled_geodetic()
    }
}

impl QuadtreeConfig {
    /// Default configuration for the bundled offline `Geodetic` (`z = 0..3`)
    /// basemap with adaptive Polar Stereographic caps (`z = 0`).
    pub const fn bundled_geodetic() -> Self {
        Self {
            scheme: TileScheme::Geodetic,
            max_tex_level: 3,
            max_geom_level: 16,
            polar_max_tex_level: 0,
            polar_max_geom_level: 14,
            grid_cells: DEFAULT_GRID_CELLS,
            tile_px: 256,
            max_geom_sse_px: 0.5,
            max_tex_sse_px: 1.5,
        }
    }

    /// Configuration for a `WebMercator` basemap with given `max_tex_level`.
    pub const fn web_mercator(max_tex_level: u8) -> Self {
        Self {
            scheme: TileScheme::WebMercator,
            max_tex_level,
            max_geom_level: 17,
            polar_max_tex_level: 0,
            polar_max_geom_level: 14,
            grid_cells: DEFAULT_GRID_CELLS,
            tile_px: 256,
            max_geom_sse_px: 0.5,
            max_tex_sse_px: 1.5,
        }
    }

    const fn max_tex_for(&self, scheme: TileScheme) -> u8 {
        match scheme {
            TileScheme::PolarCap { .. } => self.polar_max_tex_level,
            _ => self.max_tex_level,
        }
    }

    const fn max_geom_for(&self, scheme: TileScheme) -> u8 {
        match scheme {
            TileScheme::PolarCap { .. } => self.polar_max_geom_level,
            _ => self.max_geom_level,
        }
    }
}

/// Geographic and mesh metadata for a single visible leaf tile produced by
/// [`traverse`].
#[derive(Clone, Copy, Debug)]
pub struct VisibleTile {
    /// Geometry quadtree tile ID (may be finer than `tex_id`).
    pub id: QuadTileId,
    /// Texture tile ID clamped to `<= max_tex_level` (`0` requests above
    /// `max_tex_level`).
    pub tex_id: QuadTileId,
    /// Exact UV sub-rectangle `[u_off, v_off, u_scale, v_scale]` mapping this
    /// geometry tile's `[0, 1]^2` UVs into `tex_id`.
    pub uv_rect: [f32; 4],
    /// Tile-local CPU `f64` RTC origin in ECEF metres.
    pub origin_ecef_m: DVec3,
    /// Conservative ECEF bounding sphere in metres.
    pub ball: Ball,
    /// Maximum diagonal chord sagitta of one `grid_cells x grid_cells` mesh
    /// cell in metres.
    pub sag_cell_diag_m: f64,
    /// Inward skirt depth along `-normal` in metres
    /// (`max(4.0 * sag_cell_diag_m, 0.001)`).
    pub skirt_depth_m: f64,
}

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

fn mercator_lat_deg(merc: f64) -> f64 {
    merc.sinh().atan().to_degrees()
}

/// Compute geographic bounds for a `Geodetic` or `WebMercator` tile, returning
/// `None` if the tile lies entirely inside the `|lat| >= WEB_MERCATOR_LIMIT_DEG`
/// polar caps (which are owned by `TileScheme::PolarCap`).
pub fn lat_lon_bounds(id: QuadTileId) -> Option<LatLonTileBounds> {
    let (cols, rows) = id.scheme.grid_dims(id.level)?;
    if id.x >= cols || id.y >= rows {
        return None;
    }
    let cols_f = f64::from(cols);
    let rows_f = f64::from(rows);
    let lon_west_deg = -180.0 + 360.0 * f64::from(id.x) / cols_f;
    let lon_east_deg = -180.0 + 360.0 * f64::from(id.x + 1) / cols_f;

    match id.scheme {
        TileScheme::Geodetic => {
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
        TileScheme::WebMercator => {
            let merc_n = std::f64::consts::PI * (1.0 - 2.0 * f64::from(id.y) / rows_f);
            let merc_s = std::f64::consts::PI * (1.0 - 2.0 * f64::from(id.y + 1) / rows_f);
            let lat_north_deg = mercator_lat_deg(merc_n);
            let lat_south_deg = mercator_lat_deg(merc_s);
            Some(LatLonTileBounds {
                lon_west_deg,
                lon_east_deg,
                lat_north_deg,
                lat_south_deg,
                raw_lat_north_deg: lat_north_deg,
                raw_lat_south_deg: lat_south_deg,
            })
        }
        TileScheme::PolarCap { .. } | TileScheme::RasterPyramid { .. } => None,
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
        TileScheme::Geodetic => {
            let b = lat_lon_bounds(id)?;
            let lon = b.lon_west_deg + (b.lon_east_deg - b.lon_west_deg) * s;
            let lat = b.lat_north_deg + (b.lat_south_deg - b.lat_north_deg) * t;
            let raw_span = (b.raw_lat_north_deg - b.raw_lat_south_deg).max(1e-12);
            let v_tex = (b.raw_lat_north_deg - lat) / raw_span;
            Some((lat, lon, s, v_tex, 0.0, 0.0))
        }
        TileScheme::WebMercator => {
            let (cols, rows) = id.scheme.grid_dims(id.level)?;
            if id.x >= cols || id.y >= rows {
                return None;
            }
            let cols_f = f64::from(cols);
            let rows_f = f64::from(rows);
            let lon = -180.0 + 360.0 * (f64::from(id.x) + s) / cols_f;
            let merc_n = std::f64::consts::PI * (1.0 - 2.0 * f64::from(id.y) / rows_f);
            let merc_s = std::f64::consts::PI * (1.0 - 2.0 * f64::from(id.y + 1) / rows_f);
            let merc = merc_n + (merc_s - merc_n) * t;
            let lat = mercator_lat_deg(merc);
            Some((lat, lon, s, t, 0.0, 0.0))
        }
        TileScheme::PolarCap { north } => {
            let (n, rows) = id.scheme.grid_dims(id.level)?;
            if id.x >= n || id.y >= rows {
                return None;
            }
            let n_f = f64::from(n);
            let u = -1.0 + 2.0 * (f64::from(id.x) + s) / n_f;
            let v = -1.0 + 2.0 * (f64::from(id.y) + t) / n_f;
            let (lat, lon, dx, dy) = polar_cap_point(north, u, v);
            Some((lat, lon, s, t, dx, dy))
        }
        TileScheme::RasterPyramid { .. } => None,
    }
}

/// Tile-local `f64` graticule anchor `(anchor_a, anchor_b)`:
/// - For `Geodetic` / `WebMercator`: `(lon_center_deg, lat_center_deg)`
/// - For `PolarCap`: `(dx_center, dy_center)` in unit-disk coordinates
pub fn tile_grid_anchor(id: QuadTileId) -> Option<(f64, f64)> {
    let (lat, lon, _, _, dx, dy) = sample_tile_surface(id, 0.5, 0.5)?;
    match id.scheme {
        TileScheme::PolarCap { .. } => Some((dx, dy)),
        _ => Some((lon, lat)),
    }
}

/// Compute tile metadata (`origin_ecef_m`, conservative `Ball`, cell-diagonal
/// chord sagitta, and skirt depth) for `id`, returning `None` if `id` is out of
/// domain (e.g. a `Geodetic` tile fully inside a polar cap).
pub fn describe_tile(id: QuadTileId, config: &QuadtreeConfig) -> Option<VisibleTile> {
    describe_tile_with_samples(id, config).map(|(tile, _, _, _)| tile)
}

fn describe_tile_with_samples(
    id: QuadTileId,
    config: &QuadtreeConfig,
) -> Option<(VisibleTile, [DVec3; 25], f64, f64)> {
    let grid_cells = f64::from(config.grid_cells.max(1));
    let mut samples = [DVec3::ZERO; 25];
    let mut sample_units = [DVec3::ZERO; 25];
    let e2 = geodesy::WGS84.eccentricity_sq();
    let one_minus_e2 = 1.0 - e2;

    let mut push_sample = |idx: usize, lat_deg: f64, lon_deg: f64| {
        let (sin_lat, cos_lat) = lat_deg.to_radians().sin_cos();
        let (sin_lon, cos_lon) = lon_deg.to_radians().sin_cos();
        let curvature = geodesy::SEMI_MAJOR_M / (1.0 - e2 * sin_lat * sin_lat).sqrt();
        sample_units[idx] = DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);
        samples[idx] = DVec3::new(
            curvature * cos_lat * cos_lon,
            curvature * cos_lat * sin_lon,
            curvature * one_minus_e2 * sin_lat,
        );
    };

    let mut unclamped_cell_diag_deg = 0.0_f64;
    let mut merc_equatorial_cell_diag_deg = 0.0_f64;

    match id.scheme {
        TileScheme::Geodetic => {
            let b = lat_lon_bounds(id)?;
            let mut idx = 0usize;
            for row in 0..5u32 {
                let t = f64::from(row) * 0.25;
                let lat = b.lat_north_deg + (b.lat_south_deg - b.lat_north_deg) * t;
                for col in 0..5u32 {
                    let s = f64::from(col) * 0.25;
                    let lon = b.lon_west_deg + (b.lon_east_deg - b.lon_west_deg) * s;
                    push_sample(idx, lat, lon);
                    idx += 1;
                }
            }
            if b.lat_north_deg < b.raw_lat_north_deg || b.lat_south_deg > b.raw_lat_south_deg {
                let raw_cell_lat = (b.raw_lat_north_deg - b.raw_lat_south_deg) / grid_cells;
                let min_abs_lat = b.lat_north_deg.abs().min(b.lat_south_deg.abs());
                let cell_lon = ((b.lon_east_deg - b.lon_west_deg) / grid_cells)
                    * min_abs_lat.to_radians().cos();
                unclamped_cell_diag_deg = raw_cell_lat.hypot(cell_lon);
            }
        }
        TileScheme::WebMercator => {
            let (cols, rows) = id.scheme.grid_dims(id.level)?;
            if id.x >= cols || id.y >= rows {
                return None;
            }
            let cols_f = f64::from(cols);
            let rows_f = f64::from(rows);
            let merc_n = std::f64::consts::PI * (1.0 - 2.0 * f64::from(id.y) / rows_f);
            let merc_s = std::f64::consts::PI * (1.0 - 2.0 * f64::from(id.y + 1) / rows_f);
            let mut idx = 0usize;
            for row in 0..5u32 {
                let t = f64::from(row) * 0.25;
                let lat = mercator_lat_deg(merc_n + (merc_s - merc_n) * t);
                for col in 0..5u32 {
                    let s = f64::from(col) * 0.25;
                    let lon = -180.0 + 360.0 * (f64::from(id.x) + s) / cols_f;
                    push_sample(idx, lat, lon);
                    idx += 1;
                }
            }
            // Evaluate the most equatorial mesh cell directly so non-linear
            // Mercator stretching at coarse z=0..1 does not underestimate sag.
            let d_merc = (merc_n - merc_s) / grid_cells;
            let merc_eq = if merc_s <= 0.0 && merc_n >= 0.0 {
                0.0
            } else if merc_n.abs() < merc_s.abs() {
                merc_n - 0.5 * d_merc
            } else {
                merc_s + 0.5 * d_merc
            };
            let cell_dlat = (mercator_lat_deg((merc_eq + 0.5 * d_merc).min(merc_n))
                - mercator_lat_deg((merc_eq - 0.5 * d_merc).max(merc_s)))
            .abs();
            let cell_dlon =
                (360.0 / (cols_f * grid_cells)) * mercator_lat_deg(merc_eq).to_radians().cos();
            merc_equatorial_cell_diag_deg = cell_dlat.hypot(cell_dlon);
        }
        TileScheme::PolarCap { north } => {
            let (n, rows) = id.scheme.grid_dims(id.level)?;
            if id.x >= n || id.y >= rows {
                return None;
            }
            let n_f = f64::from(n);
            let mut idx = 0usize;
            for row in 0..5u32 {
                let t = f64::from(row) * 0.25;
                let v = -1.0 + 2.0 * (f64::from(id.y) + t) / n_f;
                for col in 0..5u32 {
                    let s = f64::from(col) * 0.25;
                    let u = -1.0 + 2.0 * (f64::from(id.x) + s) / n_f;
                    let (lat, lon, _, _) = polar_cap_point(north, u, v);
                    push_sample(idx, lat, lon);
                    idx += 1;
                }
            }
        }
        TileScheme::RasterPyramid { .. } => return None,
    }

    // Center point `(s=0.5, t=0.5)` is index 12 (row 2, col 2).
    let origin_ecef_m = samples[12];

    // Estimate maximum 5x5 subcell diagonal arc from cached unit normals.
    let mut max_subcell_arc_deg = 0.0_f64;
    for row in 0..4usize {
        for col in 0..4usize {
            let a = sample_units[row * 5 + col];
            let b = sample_units[(row + 1) * 5 + (col + 1)];
            let c = sample_units[row * 5 + (col + 1)];
            let d = sample_units[(row + 1) * 5 + col];
            let arc_ab = a.cross(b).length().atan2(a.dot(b)).to_degrees();
            let arc_cd = c.cross(d).length().atan2(c.dot(d)).to_degrees();
            max_subcell_arc_deg = max_subcell_arc_deg.max(arc_ab).max(arc_cd);
        }
    }
    // Scale 5x5 subcell (`1/4` of tile) to `1/grid_cells`, taking the exact
    // equatorial WebMercator cell when larger.
    let cell_diag_deg =
        (max_subcell_arc_deg * (4.0 / grid_cells)).max(merc_equatorial_cell_diag_deg);
    let sag_cell_diag_m = sag::sag_m(cell_diag_deg) * POLAR_CURVATURE_SCALE;
    // For clamped Geodetic polar-adjacent tiles, derive skirt depth from the
    // unclamped latitude span so 2:1 coarser neighbors never out-sag a thin sliver.
    let skirt_sag_m =
        sag_cell_diag_m.max(sag::sag_m(unclamped_cell_diag_deg) * POLAR_CURVATURE_SCALE);
    let skirt_depth_m = (4.0 * skirt_sag_m).max(0.001);

    // Conservative subcell bulge margin (polar curvature + 15% trapezoid slack)
    // so the bounding sphere strictly encloses the curved ellipsoid surface.
    let sample_sag_m = sag::sag_m((max_subcell_arc_deg * 1.15).min(180.0)) * POLAR_CURVATURE_SCALE;
    let ball = Ball::over(&samples).widened(sample_sag_m + skirt_depth_m);

    let max_tex = config.max_tex_for(id.scheme);
    let (tex_id, uv_rect) = id.clamp_to_level(max_tex);

    let tile = VisibleTile {
        id,
        tex_id,
        uv_rect,
        origin_ecef_m,
        ball,
        sag_cell_diag_m,
        skirt_depth_m,
    };
    Some((tile, samples, sample_sag_m, max_subcell_arc_deg))
}

fn tile_visible(
    frustum: &CullFrame,
    tile: &VisibleTile,
    samples: &[DVec3; 25],
    sample_sag_m: f64,
    max_subcell_arc_deg: f64,
) -> bool {
    // Only apply the 5x5 sample horizon shortcut once subcells are small enough
    // that spherical-trapezoid circumradius stays within `1.15 * half_diag`;
    // coarse planetary roots (z = 0..1) rely solely on their enclosing `Ball`.
    if max_subcell_arc_deg <= 30.0 {
        let max_horizon = samples
            .iter()
            .map(|&p| frustum.horizon_margin(p))
            .fold(f64::NEG_INFINITY, f64::max);
        if max_horizon + sample_sag_m < 0.0 {
            return false;
        }
    }
    frustum.admits(&tile.ball)
}

/// Approximate ground width of one tile in metres (used for texture SSE while
/// `level < max_tex_level`).
fn approx_tile_width_m(tile: &VisibleTile) -> f64 {
    if tile.id.level <= 1
        && let Some((cols, _)) = tile.id.scheme.grid_dims(tile.id.level)
        && matches!(
            tile.id.scheme,
            TileScheme::Geodetic | TileScheme::WebMercator
        )
    {
        return (2.0 * std::f64::consts::PI * geodesy::SEMI_MAJOR_M) / f64::from(cols);
    }
    // Bounding-sphere radius is ~half the tile diagonal (`sqrt(2)/2 * width`).
    (tile.ball.radius * std::f64::consts::SQRT_2).max(1.0)
}

fn is_polar_perimeter(id: QuadTileId) -> bool {
    if !matches!(id.scheme, TileScheme::PolarCap { .. }) {
        return false;
    }
    let Some((n, _)) = id.scheme.grid_dims(id.level) else {
        return false;
    };
    id.x == 0 || id.y == 0 || id.x + 1 >= n || id.y + 1 >= n
}

fn should_subdivide(
    tile: &VisibleTile,
    eye: DVec3,
    camera_height_m: f64,
    focal_px: f64,
    config: &QuadtreeConfig,
) -> bool {
    let max_geom = config.max_geom_for(tile.id.scheme);
    if tile.id.level >= max_geom {
        return false;
    }

    let dist_m = (tile.ball.centre.distance(eye) - tile.ball.radius)
        .max(camera_height_m)
        .max(1.0);

    // 1. Geometric chord-sag criterion (HiDPI-aware SSE).
    let sse_geom_px = (tile.sag_cell_diag_m * focal_px) / dist_m;
    if sse_geom_px > config.max_geom_sse_px {
        return true;
    }

    // 2. Texture resolution criterion (clamped to `max_tex_level` so decoupled
    //    geometry LOD never emits texture requests above `max_tex_level`).
    //    For PolarCap perimeter tiles touching `±85.0511°`, couple geometry
    //    refinement to the adjacent main-scheme texture level (still clamping
    //    `tex_id` to `polar_max_tex_level`) so cross-scheme skirts stay matched.
    let mut tex_subdiv_cap = config.max_tex_for(tile.id.scheme);
    if is_polar_perimeter(tile.id) {
        let level_shift = match config.scheme {
            TileScheme::Geodetic => 1,
            _ => 2,
        };
        tex_subdiv_cap = tex_subdiv_cap.max(config.max_tex_level.saturating_sub(level_shift));
    }
    if tile.id.level < tex_subdiv_cap {
        let texel_m = approx_tile_width_m(tile) / f64::from(config.tile_px.max(1));
        let sse_tex_px = (texel_m * focal_px) / dist_m;
        if sse_tex_px > config.max_tex_sse_px {
            return true;
        }
    }

    false
}

/// Traverse the planetary quadtree (`config.scheme`) plus both adaptive Polar
/// Stereographic caps for `camera` on a physical `(width_px, height_px)`
/// viewport.
///
/// Guarantees:
/// - Every emitted tile's `tex_id.level <= max_tex_level` (`0` requests above
///   `max_tex_level`).
/// - Geometry subdivides up to `max_geom_level` for sub-millimetre sag at 2 m.
/// - Adjacent visible leaves in the same scheme differ by at most 1 LOD level
///   (2:1 balanced), matching the `4 * sag_cell_diag_m` skirt depth.
pub fn traverse(
    camera: &Camera,
    width_px: u32,
    height_px: u32,
    config: &QuadtreeConfig,
) -> Vec<VisibleTile> {
    if width_px == 0 || height_px == 0 {
        return Vec::new();
    }
    let width_f = f64::from(width_px);
    let height_f = f64::from(height_px);
    let aspect = width_f / height_f;
    let frustum = camera.frustum(aspect);
    let eye = camera.eye_ecef();
    let height_m = camera.height_m();
    let focal_px = (height_f * 0.5) / (FOV_Y_DEG * 0.5).to_radians().tan();

    let mut leaves: HashMap<QuadTileId, VisibleTile> = HashMap::new();

    // Seed roots: main planetary scheme + North & South Polar Stereographic caps.
    let mut stack: Vec<QuadTileId> = Vec::with_capacity(64);
    if let Some((cols, rows)) = config.scheme.grid_dims(0) {
        for y in 0..rows {
            for x in 0..cols {
                stack.push(QuadTileId {
                    scheme: config.scheme,
                    level: 0,
                    x,
                    y,
                });
            }
        }
    }
    for north in [true, false] {
        stack.push(QuadTileId {
            scheme: TileScheme::PolarCap { north },
            level: 0,
            x: 0,
            y: 0,
        });
    }

    while let Some(id) = stack.pop() {
        let Some((tile, samples, sample_sag_m, max_subcell_arc_deg)) =
            describe_tile_with_samples(id, config)
        else {
            continue;
        };
        if !tile_visible(&frustum, &tile, &samples, sample_sag_m, max_subcell_arc_deg) {
            continue;
        }
        if should_subdivide(&tile, eye, height_m, focal_px, config) {
            for child in id.children() {
                stack.push(child);
            }
        } else {
            leaves.insert(id, tile);
        }
    }

    // 2:1 neighbor LOD balancing within each scheme so adjacent visible tiles
    // differ by at most 1 LOD level (`4 * sag_cell_diag_m` skirt seals seams).
    balance_2_to_1(&mut leaves, &frustum, config);

    let mut out: Vec<VisibleTile> = leaves.into_values().collect();
    // Deterministic ordering for reproducible draws and stable snapshot equality.
    out.sort_by_key(|t| (t.id.scheme, t.id.level, t.id.y, t.id.x));
    out
}

fn neighbor_id(id: QuadTileId, dx: i32, dy: i32) -> Option<QuadTileId> {
    let (cols, rows) = id.scheme.grid_dims(id.level)?;
    let ny = i64::from(id.y) + i64::from(dy);
    if ny < 0 || ny >= i64::from(rows) {
        return None;
    }
    let nx = match id.scheme {
        TileScheme::Geodetic | TileScheme::WebMercator => {
            (i64::from(id.x) + i64::from(dx)).rem_euclid(i64::from(cols))
        }
        TileScheme::PolarCap { .. } => {
            let v = i64::from(id.x) + i64::from(dx);
            if v < 0 || v >= i64::from(cols) {
                return None;
            }
            v
        }
        TileScheme::RasterPyramid { .. } => return None,
    };
    Some(QuadTileId {
        scheme: id.scheme,
        level: id.level,
        x: nx as u32,
        y: ny as u32,
    })
}

fn balance_2_to_1(
    leaves: &mut HashMap<QuadTileId, VisibleTile>,
    frustum: &CullFrame,
    config: &QuadtreeConfig,
) {
    loop {
        let mut to_split: Vec<QuadTileId> = Vec::new();
        for &id in leaves.keys() {
            if id.level < 2 {
                continue;
            }
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let Some(nb) = neighbor_id(id, dx, dy) else {
                    continue;
                };
                // Walk up from `nb.parent()` -> `nb.parent().parent()` and check
                // if an ancestor at `<= id.level - 2` is currently a leaf.
                let mut anc = nb.parent().and_then(QuadTileId::parent);
                while let Some(a) = anc {
                    if leaves.contains_key(&a) {
                        if a.level < config.max_geom_for(a.scheme) {
                            to_split.push(a);
                        }
                        break;
                    }
                    anc = a.parent();
                }
            }
        }
        if to_split.is_empty() {
            break;
        }
        to_split.sort_unstable();
        to_split.dedup();
        for split_id in to_split {
            if leaves.remove(&split_id).is_none() {
                continue;
            }
            for child in split_id.children() {
                if let Some((tile, samples, sample_sag_m, max_subcell_arc_deg)) =
                    describe_tile_with_samples(child, config)
                    && tile_visible(frustum, &tile, &samples, sample_sag_m, max_subcell_arc_deg)
                {
                    leaves.insert(child, tile);
                }
            }
        }
    }
}

/// Adaptive graticule spacing in degrees (every candidate evenly divides `90°`
/// and `180°`), keeping lines spaced cleanly at ~85–425 logical points from
/// orbital zoom (`15°`) down to `2 m` close-ups (`0.00001° ~ 1.1 m`).
pub fn adaptive_grid_step_deg(metres_per_pt: f64) -> f64 {
    const STEPS_DEG: [f64; 13] = [
        0.000_01, 0.000_05, 0.000_1, 0.000_5, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 15.0,
    ];
    if !metres_per_pt.is_finite() || metres_per_pt <= 0.0 {
        return 15.0;
    }
    let target_deg = (metres_per_pt * 85.0) / 111_320.0;
    for &step in &STEPS_DEG {
        if step >= target_deg {
            return step;
        }
    }
    15.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::camera::MIN_HEIGHT_M;
    use crate::geo::geodesy::Geodetic;

    #[test]
    fn decoupled_lod_subdivides_past_max_tex_level_with_zero_requests_above_cap() {
        let cfg = QuadtreeConfig::bundled_geodetic();
        let cam = Camera::at(47.2357, 39.7015, MIN_HEIGHT_M);
        let tiles = traverse(&cam, 1280, 800, &cfg);

        assert!(!tiles.is_empty(), "must produce visible tiles at 2 m");
        let max_geom_seen = tiles.iter().map(|t| t.id.level).max().unwrap_or(0);
        assert!(
            max_geom_seen > cfg.max_tex_level,
            "geometry must subdivide past max_tex_level={} at 2 m (got {max_geom_seen})",
            cfg.max_tex_level
        );
        assert!(
            max_geom_seen <= cfg.max_geom_level,
            "geometry must respect max_geom_level={} (got {max_geom_seen})",
            cfg.max_geom_level
        );

        let nadir_ecef = geodesy::world(Geodetic::surface(47.2357, 39.7015));
        let focal_px = (800.0_f64 * 0.5) / (FOV_Y_DEG * 0.5).to_radians().tan();
        let eye = cam.eye_ecef();
        let mut nadir_tile_sag = f64::INFINITY;
        for tile in &tiles {
            let max_tex = cfg.max_tex_for(tile.id.scheme);
            assert!(
                tile.tex_id.level <= max_tex,
                "texture request {:?} exceeded max_tex_level={max_tex}",
                tile.tex_id
            );
            let dist_m = (tile.ball.centre.distance(eye) - tile.ball.radius)
                .max(cam.height_m())
                .max(1.0);
            let sse_px = (tile.sag_cell_diag_m * focal_px) / dist_m;
            assert!(
                sse_px <= cfg.max_geom_sse_px + 1e-9,
                "tile {:?} exceeded SSE cap: {sse_px} px",
                tile.id
            );
            if tile.ball.centre.distance(nadir_ecef) <= tile.ball.radius {
                nadir_tile_sag = nadir_tile_sag.min(tile.sag_cell_diag_m);
            }
            assert!(
                tile.skirt_depth_m >= (4.0 * tile.sag_cell_diag_m).max(0.001) - 1e-12,
                "skirt depth too shallow: {}",
                tile.skirt_depth_m
            );
        }
        assert!(
            nadir_tile_sag < 0.001,
            "nadir tile cell diagonal sag {nadir_tile_sag} m exceeded 1 mm at 2 m altitude"
        );
    }

    #[test]
    fn adaptive_polar_cap_covers_pole_with_sub_cm_sag_at_2m_and_exact_boundary() {
        let cfg = QuadtreeConfig::bundled_geodetic();
        for pole in [90.0, -90.0] {
            let cam = Camera::at(pole, 0.0, MIN_HEIGHT_M);
            let tiles = traverse(&cam, 1024, 768, &cfg);
            assert!(!tiles.is_empty());
            assert!(
                tiles
                    .iter()
                    .all(|t| matches!(t.id.scheme, TileScheme::PolarCap { north } if north == (pole > 0.0))),
                "at {pole} deg at 2 m, only matching PolarCap tiles should be visible"
            );
            for tile in &tiles {
                assert!(
                    tile.sag_cell_diag_m < 0.01,
                    "polar cap sag {} m exceeded 1 cm at 2 m altitude",
                    tile.sag_cell_diag_m
                );
                assert!(tile.tex_id.level <= cfg.polar_max_tex_level);
            }
        }

        // Verify exact boundary alignment along |u|=1 or |v|=1 -> |lat| == WEB_MERCATOR_LIMIT_DEG.
        for north in [true, false] {
            let sign = if north { 1.0 } else { -1.0 };
            for step in -10..=10 {
                let t = f64::from(step) / 10.0;
                for (u, v) in [(1.0, t), (-1.0, t), (t, 1.0), (t, -1.0)] {
                    let (lat, lon, dx, dy) = polar_cap_point(north, u, v);
                    assert!(
                        (lat - sign * WEB_MERCATOR_LIMIT_DEG).abs() < 1e-11,
                        "boundary ({u},{v}) landed at lat={lat}"
                    );
                    assert!(((dx * dx + dy * dy) - 1.0).abs() < 1e-12);
                    assert!(lon.is_finite());
                }
            }
            // Exact center maps to the exact pole.
            let (lat0, _, dx0, dy0) = polar_cap_point(north, 0.0, 0.0);
            assert!((lat0 - sign * 90.0).abs() < 1e-12);
            assert_eq!((dx0, dy0), (0.0, 0.0));
        }
    }

    #[test]
    fn quadtree_is_2_to_1_balanced_across_visible_neighbors_and_antimeridian() {
        let cfg = QuadtreeConfig::bundled_geodetic();
        for cam in [
            Camera::at(45.0, 39.0, 500_000.0),
            Camera::at(35.0, 180.0, 2_000_000.0),
        ] {
            let tiles = traverse(&cam, 1280, 800, &cfg);
            assert!(!tiles.is_empty());
            let map: HashMap<QuadTileId, VisibleTile> = tiles.iter().map(|&t| (t.id, t)).collect();

            for &id in map.keys() {
                if id.level < 2 {
                    continue;
                }
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let Some(nb) = neighbor_id(id, dx, dy) else {
                        continue;
                    };
                    let mut anc = nb.parent().and_then(QuadTileId::parent);
                    while let Some(a) = anc {
                        assert!(
                            !map.contains_key(&a),
                            "2:1 LOD violation: {:?} is adjacent to coarse leaf {:?}",
                            id,
                            a
                        );
                        anc = a.parent();
                    }
                }
            }
        }
    }

    #[test]
    fn hidpi_viewport_and_sse_config_control_subdivision_strictly() {
        // Use a high-altitude view where geometry SSE (not max_tex_level) governs refinement.
        let mut cfg = QuadtreeConfig::bundled_geodetic();
        cfg.max_tex_level = 0;
        let cam = Camera::at(35.0, 20.0, 1_500_000.0);
        let lod_1x = traverse(&cam, 600, 400, &cfg);
        let lod_2x = traverse(&cam, 2400, 1600, &cfg);
        let max_1x = lod_1x.iter().map(|t| t.id.level).max().unwrap_or(0);
        let max_2x = lod_2x.iter().map(|t| t.id.level).max().unwrap_or(0);
        assert!(
            max_2x > max_1x,
            "HiDPI 4x viewport must subdivide strictly finer than 1x: 1x={max_1x}, 4x={max_2x}"
        );

        // Relaxing max_geom_sse_px on a standard 800 px viewport must reduce subdivision.
        let mut loose = cfg;
        loose.max_geom_sse_px = 8.0;
        let tight_tiles = traverse(&cam, 1280, 800, &cfg);
        let loose_tiles = traverse(&cam, 1280, 800, &loose);
        let tight_max = tight_tiles.iter().map(|t| t.id.level).max().unwrap_or(0);
        let loose_max = loose_tiles.iter().map(|t| t.id.level).max().unwrap_or(0);
        assert!(
            loose_max < tight_max,
            "relaxing max_geom_sse_px must reduce LOD on 800 px viewport: tight={tight_max}, loose={loose_max}"
        );

        // Degenerate zero-area viewports return empty cleanly.
        assert!(traverse(&cam, 0, 800, &cfg).is_empty());
        assert!(traverse(&cam, 800, 0, &cfg).is_empty());
    }

    #[test]
    fn web_mercator_never_falsely_culls_across_globe_and_couples_polar_seam() {
        let cfg = QuadtreeConfig::web_mercator(12);
        // Sweep across mid-latitude quadrants including (25 N, -135 W) where
        // naive z=0 subcell horizon tests falsely cull the root tile.
        for lat in [-60.0, -25.0, 0.0, 25.0, 60.0] {
            for lon in [-135.0, -45.0, 0.0, 45.0, 135.0] {
                for height_m in [MIN_HEIGHT_M, 10_000.0, 400_000.0] {
                    let cam = Camera::at(lat, lon, height_m);
                    let tiles = traverse(&cam, 1280, 800, &cfg);
                    assert!(
                        tiles.iter().any(|t| t.id.scheme == TileScheme::WebMercator),
                        "WebMercator returned 0 tiles at ({lat}, {lon}, {height_m} m)"
                    );
                }
            }
        }

        // At 5 km over the 85.0511 deg polar seam with WebMercator(12), both
        // WebMercator and PolarCap must appear, PolarCap textures stay <= cap,
        // and PolarCap perimeter sag stays bounded against WebMercator skirts.
        let cam = Camera::at(WEB_MERCATOR_LIMIT_DEG, 15.0, 5_000.0);
        let tiles = traverse(&cam, 1280, 800, &cfg);
        let merc: Vec<&VisibleTile> = tiles
            .iter()
            .filter(|t| t.id.scheme == TileScheme::WebMercator)
            .collect();
        let polar: Vec<&VisibleTile> = tiles
            .iter()
            .filter(|t| t.id.scheme == TileScheme::PolarCap { north: true })
            .collect();
        assert!(!merc.is_empty() && !polar.is_empty());
        for t in &polar {
            assert!(t.tex_id.level <= cfg.polar_max_tex_level);
        }
        let max_merc_level = merc.iter().map(|t| t.id.level).max().unwrap_or(0);
        let max_polar_level = polar.iter().map(|t| t.id.level).max().unwrap_or(0);
        assert!(
            max_polar_level + 3 >= max_merc_level,
            "PolarCap perimeter lagged WebMercator at seam: merc={max_merc_level}, polar={max_polar_level}"
        );
    }

    #[test]
    fn clamped_geodetic_polar_adjacent_skirts_cover_coarser_neighbor_sag() {
        let cfg = QuadtreeConfig::bundled_geodetic();
        let coarse = describe_tile(
            QuadTileId {
                scheme: TileScheme::Geodetic,
                level: 4,
                x: 0,
                y: 0,
            },
            &cfg,
        )
        .expect("z=4 row 0");
        let fine = describe_tile(
            QuadTileId {
                scheme: TileScheme::Geodetic,
                level: 5,
                x: 2,
                y: 0,
            },
            &cfg,
        )
        .expect("z=5 row 0");
        assert!(
            fine.skirt_depth_m >= coarse.sag_cell_diag_m,
            "clamped Geodetic z=5 skirt ({} m) shallower than adjacent z=4 sag ({} m)",
            fine.skirt_depth_m,
            coarse.sag_cell_diag_m
        );
    }

    #[test]
    fn polar_r_cap_constant_and_bounds_guards_are_exact() {
        let expected = ((90.0 - WEB_MERCATOR_LIMIT_DEG) * 0.5).to_radians().tan();
        assert!((POLAR_R_CAP - expected).abs() < 1e-16);

        // Out-of-bounds WebMercator and PolarCap tile coordinates return None.
        assert!(
            sample_tile_surface(
                QuadTileId {
                    scheme: TileScheme::WebMercator,
                    level: 1,
                    x: 2,
                    y: 0,
                },
                0.5,
                0.5,
            )
            .is_none()
        );
        assert!(
            sample_tile_surface(
                QuadTileId {
                    scheme: TileScheme::PolarCap { north: true },
                    level: 0,
                    x: 1,
                    y: 0,
                },
                0.5,
                0.5,
            )
            .is_none()
        );
        // Huge dz in clamp_to_level must not panic.
        let deep = QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 40,
            x: 0,
            y: 0,
        };
        let (clamped, _) = deep.clamp_to_level(0);
        assert_eq!(clamped.level, 0);
    }

    #[test]
    fn adaptive_grid_step_divides_ninety_degrees_cleanly() {
        for m_per_pt in [1e-4, 0.002, 0.5, 10.0, 500.0, 16_000.0, 100_000.0] {
            let step = adaptive_grid_step_deg(m_per_pt);
            let ratio = 90.0 / step;
            assert!(
                (ratio - ratio.round()).abs() < 1e-6,
                "step {step} does not cleanly divide 90 deg (ratio={ratio})"
            );
        }
        assert_eq!(adaptive_grid_step_deg(16_320.0), 15.0);
        assert_eq!(adaptive_grid_step_deg(0.002), 0.000_01);
    }
}
