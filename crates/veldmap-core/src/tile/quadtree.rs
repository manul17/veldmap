//! Scheme-matched WGS84 planetary quadtree and adaptive Polar Stereographic cap
//! with decoupled geometry vs. texture LOD, HiDPI-aware SSE subdivision, 2:1
//! neighbor LOD balancing, and per-cell sag-scaled skirts.

use std::collections::HashMap;

use glam::DVec3;

use crate::geo::camera::Camera;
use crate::geo::cull::{Ball, Frame as CullFrame};
use crate::geo::geodesy;
use crate::geo::sag;

use super::geometry::{lat_lon_bounds, mercator_lat_deg, polar_cap_point, web_mercator_y_bounds};
use super::key::{GlobeScheme, PlanetaryScheme, QuadTileId};

/// Default mesh grid cells per quadtree tile side (`16 x 16` quads = 289
/// surface vertices + 68 skirt vertices).
pub const DEFAULT_GRID_CELLS: u32 = 16;

/// Neutral default planetary tile size in pixels for SSE estimation.
pub const DEFAULT_TILE_PX: u32 = 256;

/// Neutral default maximum `PolarCap` texture zoom level.
pub const DEFAULT_POLAR_MAX_TEX_LEVEL: u8 = 0;

/// Default `Geodetic` maximum texture level used by [`QuadtreeConfig::default`].
const DEFAULT_GEODETIC_MAX_TEX_LEVEL: u8 = 3;

/// Maximum WGS84 radius-of-curvature ratio relative to `SEMI_MAJOR_M`
/// (`a / (1 - f) / a`), used to keep spherical chord-sag bounds strictly
/// conservative at polar latitudes.
const POLAR_CURVATURE_SCALE: f64 = 1.0 / (1.0 - geodesy::FLATTENING);

/// Configuration for planetary quadtree traversal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadtreeConfig {
    /// Active non-polar planetary scheme (`GlobeScheme::Geodetic` or
    /// `GlobeScheme::WebMercator`).
    pub scheme: GlobeScheme,
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
        Self::geodetic(DEFAULT_GEODETIC_MAX_TEX_LEVEL)
    }
}

impl QuadtreeConfig {
    /// Configuration for a `Geodetic` basemap with given `max_tex_level` and
    /// adaptive Polar Stereographic caps (`z = 0` default texture cap).
    pub const fn geodetic(max_tex_level: u8) -> Self {
        Self {
            scheme: GlobeScheme::Geodetic,
            max_tex_level,
            max_geom_level: 16,
            polar_max_tex_level: DEFAULT_POLAR_MAX_TEX_LEVEL,
            polar_max_geom_level: 14,
            grid_cells: DEFAULT_GRID_CELLS,
            tile_px: DEFAULT_TILE_PX,
            max_geom_sse_px: 0.5,
            max_tex_sse_px: 1.5,
        }
    }

    /// Configuration for a `WebMercator` basemap with given `max_tex_level`.
    pub const fn web_mercator(max_tex_level: u8) -> Self {
        Self {
            scheme: GlobeScheme::WebMercator,
            max_tex_level,
            max_geom_level: 17,
            polar_max_tex_level: DEFAULT_POLAR_MAX_TEX_LEVEL,
            polar_max_geom_level: 14,
            grid_cells: DEFAULT_GRID_CELLS,
            tile_px: DEFAULT_TILE_PX,
            max_geom_sse_px: 0.5,
            max_tex_sse_px: 1.5,
        }
    }

    /// Configuration for a main-globe `scheme` (`Geodetic` or `WebMercator`)
    /// with given `max_tex_level`.
    pub const fn for_scheme(scheme: GlobeScheme, max_tex_level: u8) -> Self {
        match scheme {
            GlobeScheme::WebMercator => Self::web_mercator(max_tex_level),
            GlobeScheme::Geodetic => Self::geodetic(max_tex_level),
        }
    }

    const fn max_tex_for(&self, scheme: PlanetaryScheme) -> u8 {
        match scheme {
            PlanetaryScheme::PolarCap { .. } => self.polar_max_tex_level,
            PlanetaryScheme::Geodetic | PlanetaryScheme::WebMercator => self.max_tex_level,
        }
    }

    const fn max_geom_for(&self, scheme: PlanetaryScheme) -> u8 {
        match scheme {
            PlanetaryScheme::PolarCap { .. } => self.polar_max_geom_level,
            PlanetaryScheme::Geodetic | PlanetaryScheme::WebMercator => self.max_geom_level,
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
    /// (`max(4.0 * sag_cell_diag_m, 0.001)`, raised to the unclamped span sag
    /// on polar-clamped `Geodetic` tiles so coarser neighbors never out-sag a
    /// thin sliver).
    pub skirt_depth_m: f64,
}

struct DescribedTile {
    tile: VisibleTile,
    samples: [DVec3; 25],
    sample_sag_m: f64,
    max_subcell_arc_deg: f64,
}

/// Compute tile metadata (`origin_ecef_m`, conservative `Ball`, cell-diagonal
/// chord sagitta, and skirt depth) for `id`, returning `None` if `id` is out of
/// domain (e.g. a `Geodetic` tile fully inside a polar cap).
pub fn describe_tile(id: QuadTileId, config: &QuadtreeConfig) -> Option<VisibleTile> {
    describe_tile_with_samples(id, config).map(|described| described.tile)
}

/// Intentionally factors the 5x5 grid into 5 lat + 5 lon trig evaluations
/// (`10 sin_cos + 5 sqrt` instead of `50 sin_cos + 25 sqrt` per visited node).
fn fill_separable_grid(
    lats_deg: [f64; 5],
    lons_deg: [f64; 5],
    samples: &mut [DVec3; 25],
    sample_units: &mut [DVec3; 25],
) {
    let e2 = geodesy::WGS84.eccentricity_sq();
    let one_minus_e2 = 1.0 - e2;
    let mut lat_cache = [(0.0_f64, 0.0_f64, 0.0_f64); 5];
    for (slot, lat_deg) in lat_cache.iter_mut().zip(lats_deg) {
        let (sin_lat, cos_lat) = lat_deg.to_radians().sin_cos();
        let curvature = geodesy::SEMI_MAJOR_M / (1.0 - e2 * sin_lat * sin_lat).sqrt();
        *slot = (sin_lat, cos_lat, curvature);
    }
    let mut lon_cache = [(0.0_f64, 0.0_f64); 5];
    for (slot, lon_deg) in lon_cache.iter_mut().zip(lons_deg) {
        *slot = lon_deg.to_radians().sin_cos();
    }
    let mut idx = 0usize;
    for &(sin_lat, cos_lat, curvature) in &lat_cache {
        let eq = curvature * cos_lat;
        let z = curvature * one_minus_e2 * sin_lat;
        for &(sin_lon, cos_lon) in &lon_cache {
            sample_units[idx] = DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);
            samples[idx] = DVec3::new(eq * cos_lon, eq * sin_lon, z);
            idx += 1;
        }
    }
}

fn sample_geodetic_grid(
    id: QuadTileId,
    grid_cells: f64,
    samples: &mut [DVec3; 25],
    sample_units: &mut [DVec3; 25],
) -> Option<f64> {
    let b = lat_lon_bounds(id)?;
    let lats = [0u32, 1, 2, 3, 4]
        .map(|row| b.lat_north_deg + (b.lat_south_deg - b.lat_north_deg) * (f64::from(row) * 0.25));
    let lons = [0u32, 1, 2, 3, 4]
        .map(|col| b.lon_west_deg + (b.lon_east_deg - b.lon_west_deg) * (f64::from(col) * 0.25));
    fill_separable_grid(lats, lons, samples, sample_units);

    if b.lat_north_deg >= b.raw_lat_north_deg && b.lat_south_deg <= b.raw_lat_south_deg {
        return Some(0.0);
    }
    let raw_cell_lat = (b.raw_lat_north_deg - b.raw_lat_south_deg) / grid_cells;
    let min_abs_lat = if b.lat_south_deg <= 0.0 && b.lat_north_deg >= 0.0 {
        0.0
    } else {
        b.lat_north_deg.abs().min(b.lat_south_deg.abs())
    };
    let cell_lon =
        ((b.lon_east_deg - b.lon_west_deg) / grid_cells) * min_abs_lat.to_radians().cos();
    Some(raw_cell_lat.hypot(cell_lon))
}

fn sample_web_mercator_grid(
    id: QuadTileId,
    grid_cells: f64,
    samples: &mut [DVec3; 25],
    sample_units: &mut [DVec3; 25],
) -> Option<f64> {
    let (cols, rows) = id.checked_grid_dims()?;
    let cols_f = f64::from(cols);
    let (merc_n, merc_s) = web_mercator_y_bounds(id.y, rows);
    let lats = [0u32, 1, 2, 3, 4]
        .map(|row| mercator_lat_deg(merc_n + (merc_s - merc_n) * (f64::from(row) * 0.25)));
    let lons = [0u32, 1, 2, 3, 4]
        .map(|col| -180.0 + 360.0 * (f64::from(id.x) + f64::from(col) * 0.25) / cols_f);
    fill_separable_grid(lats, lons, samples, sample_units);

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
    let cell_dlon = (360.0 / (cols_f * grid_cells)) * mercator_lat_deg(merc_eq).to_radians().cos();
    Some(cell_dlat.hypot(cell_dlon))
}

fn sample_polar_cap_grid(
    id: QuadTileId,
    north: bool,
    samples: &mut [DVec3; 25],
    sample_units: &mut [DVec3; 25],
) -> Option<()> {
    let (n, _) = id.checked_grid_dims()?;
    let n_f = f64::from(n);
    let mut idx = 0usize;
    for row in 0..5u32 {
        let t = f64::from(row) * 0.25;
        let v = -1.0 + 2.0 * (f64::from(id.y) + t) / n_f;
        for col in 0..5u32 {
            let s = f64::from(col) * 0.25;
            let u = -1.0 + 2.0 * (f64::from(id.x) + s) / n_f;
            let (lat, lon, _, _) = polar_cap_point(north, u, v);
            let (ecef, unit) = geodesy::surface_ecef_and_normal(lat, lon);
            sample_units[idx] = unit;
            samples[idx] = ecef;
            idx += 1;
        }
    }
    Some(())
}

/// Estimate maximum 5x5 subcell diagonal arc from cached unit normals via
/// squared chord length (`|a - b|^2 = 4 sin^2(theta / 2)`, strictly monotonic on
/// `[0, pi]`), needing only one `asin` instead of 32 `atan2`s.
fn max_subcell_arc_deg(sample_units: &[DVec3; 25]) -> f64 {
    let mut max_chord_sq = 0.0_f64;
    for row in 0..4usize {
        for col in 0..4usize {
            let a = sample_units[row * 5 + col];
            let b = sample_units[(row + 1) * 5 + (col + 1)];
            let c = sample_units[row * 5 + (col + 1)];
            let d = sample_units[(row + 1) * 5 + col];
            max_chord_sq = max_chord_sq
                .max((a - b).length_squared())
                .max((c - d).length_squared());
        }
    }
    2.0 * (0.25 * max_chord_sq)
        .clamp(0.0, 1.0)
        .sqrt()
        .asin()
        .to_degrees()
}

fn describe_tile_with_samples(id: QuadTileId, config: &QuadtreeConfig) -> Option<DescribedTile> {
    let grid_cells = f64::from(config.grid_cells.max(1));
    let mut samples = [DVec3::ZERO; 25];
    let mut sample_units = [DVec3::ZERO; 25];

    let (unclamped_cell_diag_deg, merc_equatorial_cell_diag_deg) = match id.scheme {
        PlanetaryScheme::Geodetic => (
            sample_geodetic_grid(id, grid_cells, &mut samples, &mut sample_units)?,
            0.0,
        ),
        PlanetaryScheme::WebMercator => (
            0.0,
            sample_web_mercator_grid(id, grid_cells, &mut samples, &mut sample_units)?,
        ),
        PlanetaryScheme::PolarCap { north } => {
            sample_polar_cap_grid(id, north, &mut samples, &mut sample_units)?;
            (0.0, 0.0)
        }
    };

    // Center point `(s=0.5, t=0.5)` is index 12 (row 2, col 2).
    let origin_ecef_m = samples[12];
    let max_subcell_arc_deg = max_subcell_arc_deg(&sample_units);

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
    Some(DescribedTile {
        tile,
        samples,
        sample_sag_m,
        max_subcell_arc_deg,
    })
}

fn tile_visible(frustum: &CullFrame, described: &DescribedTile) -> bool {
    // Only apply the 5x5 sample horizon shortcut once subcells are small enough
    // that spherical-trapezoid circumradius stays within `1.15 * half_diag`;
    // coarse planetary roots (z = 0..1) rely solely on their enclosing `Ball`.
    if described.max_subcell_arc_deg <= 30.0 {
        let max_horizon = described
            .samples
            .iter()
            .map(|&p| frustum.horizon_margin(p))
            .fold(f64::NEG_INFINITY, f64::max);
        if max_horizon + described.sample_sag_m < 0.0 {
            return false;
        }
    }
    frustum.admits(&described.tile.ball)
}

/// Approximate ground width of one tile in metres (used for texture SSE while
/// `level < max_tex_level`).
fn approx_tile_width_m(tile: &VisibleTile) -> f64 {
    if tile.id.level <= 1
        && let Some((cols, _)) = tile.id.scheme.grid_dims(tile.id.level)
        && matches!(
            tile.id.scheme,
            PlanetaryScheme::Geodetic | PlanetaryScheme::WebMercator
        )
    {
        return (2.0 * std::f64::consts::PI * geodesy::SEMI_MAJOR_M) / f64::from(cols);
    }
    // Bounding-sphere radius is ~half the tile diagonal (`sqrt(2)/2 * width`).
    (tile.ball.radius * std::f64::consts::SQRT_2).max(1.0)
}

fn is_polar_perimeter(id: QuadTileId) -> bool {
    if !matches!(id.scheme, PlanetaryScheme::PolarCap { .. }) {
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
            GlobeScheme::Geodetic => 1,
            GlobeScheme::WebMercator => 2,
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
    let focal_px = camera.focal_length_px(height_f);

    let mut leaves: HashMap<QuadTileId, VisibleTile> = HashMap::new();

    // Seed roots: main planetary scheme (`Geodetic` or `WebMercator`) + North &
    // South Polar Stereographic caps.
    let mut stack: Vec<QuadTileId> = Vec::with_capacity(64);
    let main_scheme = config.scheme.into_planetary();
    if let Some((cols, rows)) = main_scheme.grid_dims(0) {
        for y in 0..rows {
            for x in 0..cols {
                stack.push(QuadTileId {
                    scheme: main_scheme,
                    level: 0,
                    x,
                    y,
                });
            }
        }
    }
    for north in [true, false] {
        stack.push(QuadTileId {
            scheme: PlanetaryScheme::PolarCap { north },
            level: 0,
            x: 0,
            y: 0,
        });
    }

    while let Some(id) = stack.pop() {
        let Some(described) = describe_tile_with_samples(id, config) else {
            continue;
        };
        if !tile_visible(&frustum, &described) {
            continue;
        }
        if should_subdivide(&described.tile, eye, height_m, focal_px, config) {
            for child in id.children() {
                stack.push(child);
            }
        } else {
            leaves.insert(id, described.tile);
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
        PlanetaryScheme::Geodetic | PlanetaryScheme::WebMercator => {
            (i64::from(id.x) + i64::from(dx)).rem_euclid(i64::from(cols))
        }
        PlanetaryScheme::PolarCap { .. } => {
            let v = i64::from(id.x) + i64::from(dx);
            if v < 0 || v >= i64::from(cols) {
                return None;
            }
            v
        }
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
                if let Some(described) = describe_tile_with_samples(child, config)
                    && tile_visible(frustum, &described)
                {
                    leaves.insert(child, described.tile);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "quadtree_tests.rs"]
mod tests;
