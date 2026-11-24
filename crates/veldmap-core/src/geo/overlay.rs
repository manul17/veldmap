//! Native-CRS raster overlay georeference bindings, top-down lazy `DVec3`
//! culling balls, and per-tile CPU `f64` RTC warp-mesh geometry.
//!
//! Raster tiles are never reprojected on CPU: each pyramid cell is draped over
//! the smooth WGS84 ellipsoid (`height_m = 0.0`) as a subdivided warp mesh
//! whose nodes are evaluated directly through [`Frame::geodetic`] in `f64` and
//! stored relative to a per-cell `f64` RTC origin (`origin_ecef_m`) in `f32`.

use std::collections::HashMap;

use glam::{DMat4, DVec3};

use crate::tile::{cell_image_rect, grid_dim, level_size};

use super::camera;
use super::cull::{Ball, CullFrustum};
use super::footprint::{self, FootprintBounds, GeoPoint, Ring};
use super::geodesy::{self, Geodetic, WGS84};
use super::projection::Projection;
use super::sag;

/// Minimum segments per warp-mesh cell side.
///
/// Using `16` segments per side keeps nonlinear transverse-Mercator curvature
/// and chord placement error strictly `< 2 cm` even on coarse pyramid levels
/// (`~110 km` cells) near UTM zone edges (`±3 deg` from the central meridian).
pub const MIN_WARP_SEGMENTS: u32 = 16;

/// Upper bound on segments per warp-mesh cell side (covers global cells up to
/// `360 deg` without exploding vertex counts).
pub const MAX_WARP_SEGMENTS: u32 = 64;

/// Target maximum surface arc per warp-mesh segment in degrees before clamping
/// to `[MIN_WARP_SEGMENTS, MAX_WARP_SEGMENTS]`.
const TARGET_WARP_SEGMENT_DEG: f64 = 0.5;

/// Tolerance for clustering tie-point raster fractions onto shared grid lines.
const SAME_LINE: f64 = 1e-6;

/// A node of a georeferenced tie-point lattice: normalized raster fractions in
/// `[0, 1]` paired with WGS84 geodetic coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tie {
    pub fx: f64,
    pub fy: f64,
    pub lat: f64,
    pub lon: f64,
}

/// Rectangular tie-point lattice mapping normalized raster fractions `(fx, fy)`
/// to WGS84 `(lat_deg, lon_deg)` with continuous neighbor-chain longitude
/// unwinding.
#[derive(Clone, Debug, PartialEq)]
pub struct TieGrid {
    xs: Vec<f64>,
    ys: Vec<f64>,
    nodes: Vec<(f64, f64)>,
    axial: bool,
    widest_row_m: f64,
    widest_col_m: f64,
}

fn line_index(axis: &[f64], at: f64) -> Option<usize> {
    let index = axis.partition_point(|line| *line < at - SAME_LINE);
    axis.get(index)
        .filter(|line| (**line - at).abs() <= SAME_LINE)
        .map(|_| index)
}

impl TieGrid {
    /// Assemble a rectangular tie-point grid from `ties`.
    ///
    /// Returns `None` if the points do not form a complete `>= 2 x 2` lattice,
    /// contain non-finite coordinates, or degenerate into a zero-width/height
    /// line on Earth.
    pub fn new(ties: &[Tie]) -> Option<Self> {
        if ties.len() < 4
            || ties.iter().any(|t| {
                !(t.fx.is_finite() && t.fy.is_finite() && t.lat.is_finite() && t.lon.is_finite())
            })
        {
            return None;
        }
        let build_axis = |mut values: Vec<f64>| {
            values.sort_by(f64::total_cmp);
            values.dedup_by(|here, before| (*here - *before).abs() <= SAME_LINE);
            values
        };
        let xs = build_axis(ties.iter().map(|t| t.fx).collect());
        let ys = build_axis(ties.iter().map(|t| t.fy).collect());
        if xs.len() < 2 || ys.len() < 2 || xs.len().checked_mul(ys.len()) != Some(ties.len()) {
            return None;
        }

        let mut nodes = vec![None; ties.len()];
        for tie in ties {
            let col = line_index(&xs, tie.fx)?;
            let row = line_index(&ys, tie.fy)?;
            nodes[row * xs.len() + col] = Some((tie.lat, tie.lon));
        }
        let nodes: Vec<(f64, f64)> = nodes.into_iter().collect::<Option<_>>()?;
        let mut grid = Self {
            xs,
            ys,
            nodes,
            axial: false,
            widest_row_m: 0.0,
            widest_col_m: 0.0,
        };
        grid.unwind();
        grid.axial = grid.compute_axial();
        grid.widest_row_m = grid.compute_widest_row();
        grid.widest_col_m = grid.compute_widest_column();
        (grid.widest_row_m.is_finite()
            && grid.widest_row_m > 0.0
            && grid.widest_col_m.is_finite()
            && grid.widest_col_m > 0.0)
            .then_some(grid)
    }

    /// Unwind longitudes along each row and align rows by majority full-turn
    /// vote so antimeridian and polar swaths remain continuous while global
    /// 2-column `-180 -> +180` grids keep their full 360-degree span.
    fn unwind(&mut self) {
        let across = self.xs.len();
        let whole_circle = across == 2;
        for row in 0..self.ys.len() {
            for col in 1..across {
                let previous = self.nodes[row * across + col - 1].1;
                let node = &mut self.nodes[row * across + col];
                if whole_circle && (node.1 - previous).abs() >= footprint::FULL_CIRCLE_DEG {
                    continue;
                }
                node.1 = geodesy::unwind(previous, node.1);
            }
            if row > 0 {
                let mut votes: Vec<i32> = (0..across)
                    .map(|col| {
                        let above = self.nodes[(row - 1) * across + col].1;
                        let here = self.nodes[row * across + col].1;
                        ((geodesy::unwind(above, here) - here) / geodesy::TURN_DEG).round() as i32
                    })
                    .collect();
                votes.sort_unstable();
                let mut chosen = votes[0];
                let mut best = 0usize;
                let mut at = 0usize;
                while at < votes.len() {
                    let same = votes[at..].partition_point(|vote| *vote == votes[at]);
                    if same > best || (same == best && votes[at].abs() < chosen.abs()) {
                        chosen = votes[at];
                        best = same;
                    }
                    at += same;
                }
                let shift = f64::from(chosen) * geodesy::TURN_DEG;
                if shift != 0.0 {
                    for col in 0..across {
                        self.nodes[row * across + col].1 += shift;
                    }
                }
            }
        }
    }

    fn compute_axial(&self) -> bool {
        let across = self.xs.len();
        let rows = self.nodes.chunks_exact(across);
        rows.clone()
            .all(|row| row.iter().all(|node| node.0 == row[0].0))
            && (0..across).all(|col| rows.clone().all(|row| row[col].1 == self.nodes[col].1))
    }

    fn compute_widest_row(&self) -> f64 {
        let across = self.xs.len();
        let width = (self.xs[across - 1] - self.xs[0]).max(f64::EPSILON);
        self.nodes
            .chunks_exact(across)
            .map(|row| {
                row.windows(2)
                    .map(|pair| ground_span_m(pair[0], pair[1]))
                    .sum::<f64>()
                    / width
            })
            .fold(0.0, f64::max)
    }

    fn compute_widest_column(&self) -> f64 {
        let across = self.xs.len();
        let height = (self.ys[self.ys.len() - 1] - self.ys[0]).max(f64::EPSILON);
        (0..across)
            .map(|col| {
                (1..self.ys.len())
                    .map(|row| {
                        ground_span_m(
                            self.nodes[(row - 1) * across + col],
                            self.nodes[row * across + col],
                        )
                    })
                    .sum::<f64>()
                    / height
            })
            .fold(0.0, f64::max)
    }

    /// Whether the grid carries nonlinear geometry beyond four corners.
    pub fn is_dense(&self) -> bool {
        self.xs.len() * self.ys.len() > 4
    }

    /// Whether every row shares a single latitude and every column shares a
    /// single longitude.
    pub fn is_axial(&self) -> bool {
        self.axial
    }

    /// Interpolate geodetic `(lat_deg, lon_deg)` for normalized raster fraction
    /// `(fx, fy)`.
    pub fn geodetic(&self, fx: f64, fy: f64) -> (f64, f64) {
        let (col, tx) = axis_cell(&self.xs, fx);
        let (row, ty) = axis_cell(&self.ys, fy);
        let across = self.xs.len();
        let node = |r: usize, c: usize| self.nodes[r * across + c];
        let top = lerp_pair(node(row, col), node(row, col + 1), tx);
        let bottom = lerp_pair(node(row + 1, col), node(row + 1, col + 1), tx);
        lerp_pair(top, bottom, ty)
    }

    /// Invert geodetic `(lat_deg, lon_deg)` back into normalized raster
    /// fractions `(fx, fy)` inside `[0, 1]^2`, returning `None` if outside the
    /// grid.
    pub fn fraction_at(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        if !(lat_deg.is_finite() && lon_deg.is_finite()) {
            return None;
        }
        let across = self.xs.len();
        if self.axial {
            let lats: Vec<f64> = (0..self.ys.len())
                .map(|row| self.nodes[row * across].0)
                .collect();
            let lons: Vec<f64> = (0..across).map(|col| self.nodes[col].1).collect();
            let fy = invert_1d_piecewise(&self.ys, &lats, lat_deg)?;
            let fx = invert_1d_lon_piecewise(&self.xs, &lons, lon_deg)?;
            return ((0.0..=1.0).contains(&fx) && (0.0..=1.0).contains(&fy)).then_some((fx, fy));
        }

        for row in 0..self.ys.len() - 1 {
            for col in 0..across - 1 {
                let n00 = self.nodes[row * across + col];
                let n10 = self.nodes[row * across + col + 1];
                let n01 = self.nodes[(row + 1) * across + col];
                let n11 = self.nodes[(row + 1) * across + col + 1];
                let lon_u = geodesy::unwind(n00.1, lon_deg);
                let min_lat = n00.0.min(n10.0).min(n01.0).min(n11.0);
                let max_lat = n00.0.max(n10.0).max(n01.0).max(n11.0);
                let min_lon = n00.1.min(n10.1).min(n01.1).min(n11.1);
                let max_lon = n00.1.max(n10.1).max(n01.1).max(n11.1);
                let eps = 1e-6;
                if lat_deg < min_lat - eps
                    || lat_deg > max_lat + eps
                    || lon_u < min_lon - eps
                    || lon_u > max_lon + eps
                {
                    continue;
                }
                if let Some((tx, ty)) = invert_bilinear_cell(n00, n10, n01, n11, (lat_deg, lon_u)) {
                    let fx = self.xs[col] + (self.xs[col + 1] - self.xs[col]) * tx;
                    let fy = self.ys[row] + (self.ys[row + 1] - self.ys[row]) * ty;
                    if (0.0..=1.0).contains(&fx) && (0.0..=1.0).contains(&fy) {
                        return Some((fx, fy));
                    }
                }
            }
        }
        None
    }
}

fn invert_1d_piecewise(axis: &[f64], values: &[f64], target: f64) -> Option<f64> {
    for i in 0..values.len().saturating_sub(1) {
        let (v0, v1) = (values[i], values[i + 1]);
        let lo = v0.min(v1);
        let hi = v0.max(v1);
        let span = v1 - v0;
        if span.abs() > 0.0 && target >= lo - 1e-9 && target <= hi + 1e-9 {
            let t = ((target - v0) / span).clamp(0.0, 1.0);
            return Some(axis[i] + (axis[i + 1] - axis[i]) * t);
        }
    }
    None
}

fn invert_1d_lon_piecewise(axis: &[f64], lons: &[f64], target_lon: f64) -> Option<f64> {
    let base = lons.first().copied().unwrap_or(0.0);
    for shift in [-geodesy::TURN_DEG, 0.0, geodesy::TURN_DEG] {
        let candidate = geodesy::unwind(base, target_lon) + shift;
        if let Some(fx) = invert_1d_piecewise(axis, lons, candidate) {
            return Some(fx);
        }
    }
    None
}

fn invert_bilinear_cell(
    n00: (f64, f64),
    n10: (f64, f64),
    n01: (f64, f64),
    n11: (f64, f64),
    target: (f64, f64),
) -> Option<(f64, f64)> {
    let mut tx = 0.5_f64;
    let mut ty = 0.5_f64;
    for _ in 0..12 {
        let top = lerp_pair(n00, n10, tx);
        let bot = lerp_pair(n01, n11, tx);
        let val = lerp_pair(top, bot, ty);
        let r0 = val.0 - target.0;
        let r1 = val.1 - target.1;
        if r0.hypot(r1) < 1e-11 {
            break;
        }
        let d_tx = (
            (1.0 - ty) * (n10.0 - n00.0) + ty * (n11.0 - n01.0),
            (1.0 - ty) * (n10.1 - n00.1) + ty * (n11.1 - n01.1),
        );
        let d_ty = (bot.0 - top.0, bot.1 - top.1);
        let det = d_tx.0 * d_ty.1 - d_tx.1 * d_ty.0;
        if !det.is_finite() || det.abs() < 1e-18 {
            return None;
        }
        let step_x = (r0 * d_ty.1 - r1 * d_ty.0) / det;
        let step_y = (d_tx.0 * r1 - d_tx.1 * r0) / det;
        tx -= step_x;
        ty -= step_y;
    }
    let eps = 1e-6;
    if tx >= -eps && tx <= 1.0 + eps && ty >= -eps && ty <= 1.0 + eps {
        let tx_c = tx.clamp(0.0, 1.0);
        let ty_c = ty.clamp(0.0, 1.0);
        let top = lerp_pair(n00, n10, tx_c);
        let bot = lerp_pair(n01, n11, tx_c);
        let val = lerp_pair(top, bot, ty_c);
        if (val.0 - target.0).hypot(val.1 - target.1) < 1e-5 {
            return Some((tx_c, ty_c));
        }
    }
    None
}

/// Georeference binding mapping normalized raster fractions `(fx, fy) in
/// [0, 1]^2` onto the WGS84 ellipsoid.
#[derive(Clone, Debug, PartialEq)]
pub enum Frame {
    /// Exact map-projection binding:
    /// `x = affine[0]*fx + affine[1]*fy + affine[2]`,
    /// `y = affine[3]*fx + affine[4]*fy + affine[5]`,
    /// evaluated at every warp-mesh node through `system.to_geodetic(x, y)`.
    Projected {
        system: Projection,
        affine: [f64; 6],
    },
    /// Rectangular tie-point lattice (for geographic or curvilinear swaths).
    Grid(TieGrid),
    /// Footprint quad in raster order `(UL, UR, LR, LL)` with unwound
    /// longitudes.
    Quad([(f64, f64); 4]),
}

impl Frame {
    /// Construct a north-up projected frame from zone bounds `[x0, x1] x [y0, y1]`.
    pub fn utm(system: impl Into<Projection>, x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        Self::Projected {
            system: system.into(),
            affine: [x1 - x0, 0.0, x0, 0.0, -(y1 - y0), y1],
        }
    }

    /// Construct a projected frame from a pixel-space affine transform
    /// `x = a[0]*i + a[1]*j + a[2]`, `y = a[3]*i + a[4]*j + a[5]` and raster
    /// dimensions `(width, height)`.
    ///
    /// Returns `None` if dimensions are zero, any coefficient is non-finite,
    /// or the resulting frame has zero ground extent.
    pub fn from_pixel_affine(
        system: Projection,
        pixel_affine: [f64; 6],
        width: u32,
        height: u32,
    ) -> Option<Self> {
        if width == 0 || height == 0 || !pixel_affine.iter().all(|v| v.is_finite()) {
            return None;
        }
        let across = f64::from(width);
        let down = f64::from(height);
        let frame = Self::Projected {
            system,
            affine: [
                pixel_affine[0] * across,
                pixel_affine[1] * down,
                pixel_affine[2],
                pixel_affine[3] * across,
                pixel_affine[4] * down,
                pixel_affine[5],
            ],
        };
        frame.measurable().then_some(frame)
    }

    /// Resolve `epsg` through [`Projection::from_epsg`] and build a projected
    /// frame from a pixel-space affine transform.
    pub fn from_epsg_pixel_affine(
        epsg: u32,
        pixel_affine: [f64; 6],
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let system = Projection::from_epsg(epsg)
            .ok_or_else(|| format!("unsupported projected CRS EPSG:{epsg}"))?;
        Self::from_pixel_affine(system, pixel_affine, width, height)
            .ok_or_else(|| format!("EPSG:{epsg} georeference transform has no valid extent"))
    }

    /// Construct a quad frame from `(UL, UR, LR, LL)` corners, unwinding
    /// longitudes continuously toward `UL`.
    pub fn quad(points: [(f64, f64); 4]) -> Self {
        let base = points[0].1;
        Self::Quad(points.map(|(lat, lon)| (lat, geodesy::unwind(base, lon))))
    }

    /// Evaluate WGS84 `(lat_deg, lon_deg)` at normalized raster fractions
    /// `(fx, fy)` (`0..1` left-to-right and top-to-bottom).
    pub fn geodetic(&self, fx: f64, fy: f64) -> (f64, f64) {
        match self {
            Self::Projected { system, affine } => {
                let x = affine[0] * fx + affine[1] * fy + affine[2];
                let y = affine[3] * fx + affine[4] * fy + affine[5];
                system.to_geodetic(x, y)
            }
            Self::Grid(grid) => grid.geodetic(fx, fy),
            Self::Quad([ul, ur, lr, ll]) => {
                let top = footprint::edge_point(*ul, *ur, fx);
                let bottom = footprint::edge_point(*ll, *lr, fx);
                footprint::edge_point(top, bottom, fy)
            }
        }
    }

    /// Invert WGS84 `(lat_deg, lon_deg)` into normalized raster fractions
    /// `(fx, fy) in [0, 1]^2` for zoom-invariant CPU cursor readouts.
    pub fn fraction_at(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        if !(lat_deg.is_finite() && lon_deg.is_finite()) {
            return None;
        }
        match self {
            Self::Projected { system, affine } => {
                let (x, y) = system.from_geodetic(lat_deg, lon_deg);
                if !(x.is_finite() && y.is_finite()) {
                    return None;
                }
                let det = affine[0] * affine[4] - affine[1] * affine[3];
                if !det.is_finite() || det.abs() < 1e-18 {
                    return None;
                }
                let dx = x - affine[2];
                let dy = y - affine[5];
                let fx = (dx * affine[4] - dy * affine[1]) / det;
                let fy = (affine[0] * dy - affine[3] * dx) / det;
                ((0.0..=1.0).contains(&fx) && (0.0..=1.0).contains(&fy)).then_some((fx, fy))
            }
            Self::Grid(grid) => grid.fraction_at(lat_deg, lon_deg),
            Self::Quad([ul, ur, lr, ll]) => {
                let lon_u = geodesy::unwind(ul.1, lon_deg);
                invert_bilinear_cell(*ul, *ur, *ll, *lr, (lat_deg, lon_u))
            }
        }
    }

    /// Ground metres per horizontal pixel for pyramid level selection.
    ///
    /// Probes the widest of top/middle/bottom rows for quads and grids so polar
    /// rasters never understate their ground resolution.
    pub fn ground_m_per_px(&self, width: u32) -> Option<f64> {
        let probed = || {
            [0.0, 0.5, 1.0]
                .iter()
                .map(|&fy| {
                    let at = |fx: f64| self.geodetic(fx, fy);
                    ground_span_m(at(0.0), at(0.5)) + ground_span_m(at(0.5), at(1.0))
                })
                .fold(0.0, f64::max)
        };
        let span = match self {
            Self::Projected {
                system: Projection::WebMercator,
                affine,
            } => {
                let (center_lat, _) = self.geodetic(0.5, 0.5);
                affine[0].hypot(affine[3]) * center_lat.to_radians().cos().abs().max(1e-6)
            }
            Self::Projected { affine, .. } => affine[0].hypot(affine[3]),
            Self::Quad(_) => probed(),
            Self::Grid(grid) => probed().max(grid.widest_row_m),
        };
        (span.is_finite() && span > 0.0).then(|| span / f64::from(width.max(1)))
    }

    /// Ground pixel size `(m_per_px_x, m_per_px_y)` at native resolution for
    /// HUD display (distinct from viewport GSD).
    pub fn pixel_size_m(&self, width: u32, height: u32) -> Option<(f64, f64)> {
        let mpp_x = self.ground_m_per_px(width)?;
        let span_y = match self {
            Self::Projected {
                system: Projection::WebMercator,
                affine,
            } => {
                let (center_lat, _) = self.geodetic(0.5, 0.5);
                affine[1].hypot(affine[4]) * center_lat.to_radians().cos().abs().max(1e-6)
            }
            Self::Projected { affine, .. } => affine[1].hypot(affine[4]),
            Self::Quad(_) => [0.0, 0.5, 1.0]
                .iter()
                .map(|&fx| {
                    let at = |fy: f64| self.geodetic(fx, fy);
                    ground_span_m(at(0.0), at(0.5)) + ground_span_m(at(0.5), at(1.0))
                })
                .fold(0.0, f64::max),
            Self::Grid(grid) => [0.0, 0.5, 1.0]
                .iter()
                .map(|&fx| {
                    let at = |fy: f64| self.geodetic(fx, fy);
                    ground_span_m(at(0.0), at(0.5)) + ground_span_m(at(0.5), at(1.0))
                })
                .fold(0.0, f64::max)
                .max(grid.widest_col_m),
        };
        let mpp_y = if span_y.is_finite() && span_y > 0.0 {
            span_y / f64::from(height.max(1))
        } else {
            mpp_x
        };
        Some((mpp_x, mpp_y))
    }

    /// Whether rows follow parallels and columns follow meridians (enabling
    /// `O(N)` separable trig evaluation during warp-mesh construction).
    pub fn axial(&self) -> bool {
        match self {
            Self::Projected {
                system: Projection::WebMercator,
                affine,
            } => affine[1] == 0.0 && affine[3] == 0.0,
            Self::Projected { .. } => false,
            Self::Grid(grid) => grid.axial,
            Self::Quad([ul, ur, lr, ll]) => {
                ul.0 == ur.0 && ll.0 == lr.0 && ul.1 == ll.1 && ur.1 == lr.1
            }
        }
    }

    /// Whether the binding has positive, finite ground extent.
    pub fn measurable(&self) -> bool {
        match self {
            Self::Projected { affine, .. } => {
                let det = affine[0] * affine[4] - affine[1] * affine[3];
                det.is_finite() && det.abs() > 0.0 && self.ground_m_per_px(1).is_some()
            }
            Self::Grid(_) | Self::Quad(_) => self.ground_m_per_px(1).is_some(),
        }
    }

    /// Select the sharpest raster pyramid level whose pixel size is no coarser
    /// than `viewport_mpp` metres per pixel, clamped to `0..=max_level`.
    pub fn sharpest_level(&self, width: u32, viewport_mpp: f64, max_level: u8) -> u8 {
        let Some(mpp_raster) = self.ground_m_per_px(width) else {
            return max_level;
        };
        if !viewport_mpp.is_finite() || viewport_mpp <= mpp_raster {
            return 0;
        }
        let raw = (viewport_mpp / mpp_raster).log2().floor();
        if !raw.is_finite() || raw <= 0.0 {
            0
        } else {
            (raw as u32).min(u32::from(max_level)) as u8
        }
    }

    /// Spherical bounding circle around the raster perimeter for "Zoom to
    /// extent" camera framing.
    pub fn footprint_bounds(&self) -> Option<FootprintBounds> {
        const EDGE_STEPS: usize = 4;
        let mut points = Vec::with_capacity(EDGE_STEPS * 4);
        let push = |out: &mut Vec<GeoPoint>, fx: f64, fy: f64| {
            let (lat, lon) = self.geodetic(fx, fy);
            if lat.is_finite() && lon.is_finite() {
                out.push(GeoPoint::new(lat, lon));
            }
        };
        for i in 0..EDGE_STEPS {
            let t = i as f64 / EDGE_STEPS as f64;
            push(&mut points, t, 0.0);
        }
        for i in 0..EDGE_STEPS {
            let t = i as f64 / EDGE_STEPS as f64;
            push(&mut points, 1.0, t);
        }
        for i in 0..EDGE_STEPS {
            let t = 1.0 - (i as f64 / EDGE_STEPS as f64);
            push(&mut points, t, 1.0);
        }
        for i in 0..EDGE_STEPS {
            let t = 1.0 - (i as f64 / EDGE_STEPS as f64);
            push(&mut points, 0.0, t);
        }
        if points.len() < 3 {
            return None;
        }
        footprint::frame(&[Ring { points }])
    }
}

/// Top-down lazy cache of `f64` ECEF [`Ball`] culling spheres per pyramid rung.
///
/// Computing balls lazily from `max_level` (coarsest) down to `target_level`
/// avoids eagerly allocating tens of thousands of level-0 spheres when opening
/// large orthophotos while preserving each rung's own 9-probe limb-culling
/// margin so coarse cells never vanish at the horizon.
#[derive(Clone, Debug)]
pub struct OverlayCullTree {
    width: u32,
    height: u32,
    max_level: u8,
    balls: Vec<HashMap<(u32, u32), Ball>>,
}

impl OverlayCullTree {
    /// Create an empty lazy culling tree for a raster of `(width, height)`
    /// pixels with coarsest level `max_level`.
    pub fn new(width: u32, height: u32, max_level: u8) -> Self {
        let levels = usize::from(max_level) + 1;
        Self {
            width: width.max(1),
            height: height.max(1),
            max_level,
            balls: vec![HashMap::new(); levels],
        }
    }

    /// Clear all cached balls (e.g. after re-laying the binding).
    pub fn clear(&mut self) {
        for map in &mut self.balls {
            map.clear();
        }
    }

    /// Total number of balls currently materialized in the lazy cache.
    pub fn cached_ball_count(&self) -> usize {
        self.balls.iter().map(HashMap::len).sum()
    }

    /// Return (computing and caching on demand) the conservative `f64` ECEF
    /// bounding [`Ball`] for cell `(level, x, y)`.
    pub fn ball(&mut self, frame: &Frame, level: u8, x: u32, y: u32) -> Ball {
        let clamped = level.min(self.max_level);
        let lvl_idx = usize::from(clamped);
        if let Some(&cached) = self.balls[lvl_idx].get(&(x, y)) {
            return cached;
        }
        let computed = compute_cell_ball(frame, self.width, self.height, u32::from(clamped), x, y);
        self.balls[lvl_idx].insert((x, y), computed);
        computed
    }

    /// Collect all visible `(x, y)` cells at `target_level` using top-down lazy
    /// ball pruning followed by exact 9-probe frustum + limb visibility checks.
    pub fn visible_cells(
        &mut self,
        frame: &Frame,
        target_level: u8,
        frustum: &CullFrustum,
        view_proj: &DMat4,
        eye: DVec3,
    ) -> Vec<(u32, u32)> {
        if !frame.measurable() {
            return Vec::new();
        }
        let target = target_level.min(self.max_level);
        let top = self.max_level;
        let top_w = grid_dim(level_size(self.width, u32::from(top)));
        let top_h = grid_dim(level_size(self.height, u32::from(top)));

        let mut stack: Vec<(u8, u32, u32)> = Vec::with_capacity((top_w * top_h) as usize);
        for y in (0..top_h).rev() {
            for x in (0..top_w).rev() {
                stack.push((top, x, y));
            }
        }

        let mut visible = Vec::new();
        while let Some((level, x, y)) = stack.pop() {
            let ball = self.ball(frame, level, x, y);
            if !frustum.admits(&ball) {
                continue;
            }
            if level == target {
                if cell_on_screen(
                    frame,
                    self.width,
                    self.height,
                    u32::from(level),
                    x,
                    y,
                    frustum,
                    view_proj,
                    eye,
                ) {
                    visible.push((x, y));
                }
                continue;
            }

            let child_level = level - 1;
            let child_w = grid_dim(level_size(self.width, u32::from(child_level)));
            let child_h = grid_dim(level_size(self.height, u32::from(child_level)));
            let base_x = x * 2;
            let base_y = y * 2;
            for dy in (0..2u32).rev() {
                let cy = base_y + dy;
                if cy >= child_h {
                    continue;
                }
                for dx in (0..2u32).rev() {
                    let cx = base_x + dx;
                    if cx >= child_w {
                        continue;
                    }
                    stack.push((child_level, cx, cy));
                }
            }
        }
        visible
    }
}

/// CPU `f64` RTC warp-mesh patch for a single raster pyramid cell.
#[derive(Clone, Debug)]
pub struct WarpPatch {
    /// Cell-center `f64` ECEF origin in metres.
    pub origin_ecef_m: DVec3,
    /// Segments per cell side (`[MIN_WARP_SEGMENTS, MAX_WARP_SEGMENTS]`).
    pub segments: u32,
    /// `true` when the binding has negative orientation (e.g. south-up or
    /// mirrored affine) so triangle indices must flip to keep CCW outward
    /// normals facing away from the WGS84 ellipsoid.
    pub flip_winding: bool,
    /// Row-major `(segments + 1)^2` surface positions relative to
    /// `origin_ecef_m` (`(P_ecef_m - origin_ecef_m) as f32`).
    pub rel_positions: Vec<[f32; 3]>,
}

/// Build the per-tile CPU `f64` RTC warp-mesh patch for a raster cell covering
/// pixel rectangle `cell_rect = [x0, y0, x1, y1]` inside a `(width, height)`
/// raster.
pub fn build_warp_patch(frame: &Frame, width: u32, height: u32, cell_rect: [f64; 4]) -> WarpPatch {
    let w = f64::from(width.max(1));
    let h = f64::from(height.max(1));
    let span_deg = cell_span_deg(frame, w, h, cell_rect);
    let segments = segments_for(span_deg);
    let side = (segments + 1) as usize;
    let total = side * side;
    let mut ecef_points = Vec::with_capacity(total);

    let at = |tx: f64, ty: f64| {
        let px = cell_rect[0] + (cell_rect[2] - cell_rect[0]) * tx;
        let py = cell_rect[1] + (cell_rect[3] - cell_rect[1]) * ty;
        frame.geodetic(px / w, py / h)
    };
    let share = |step: u32| f64::from(step) / f64::from(segments);

    if frame.axial() {
        let lats: Vec<f64> = (0..=segments).map(|row| at(0.0, share(row)).0).collect();
        let lons: Vec<f64> = (0..=segments).map(|col| at(share(col), 0.0).1).collect();
        WGS84.fill_separable_surface_points(&lats, &lons, &mut ecef_points);
    } else {
        for row in 0..=segments {
            let ty = share(row);
            for col in 0..=segments {
                let (lat, lon) = at(share(col), ty);
                ecef_points.push(WGS84.to_ecef(Geodetic::surface(lat, lon)));
            }
        }
    }

    let mid = (segments / 2) as usize;
    let origin_ecef_m = ecef_points[mid * side + mid];
    let flip_winding = needs_flipped_winding(&ecef_points, side);

    let rel_positions = ecef_points
        .iter()
        .map(|&p| (p - origin_ecef_m).as_vec3().to_array())
        .collect();

    WarpPatch {
        origin_ecef_m,
        segments,
        flip_winding,
        rel_positions,
    }
}

/// Evaluate the exact triangulated `f64` ECEF position produced by `patch`
/// (including `f32` RTC vertex quantization) at cell-local `(tx, ty) in
/// [0, 1]^2`, matching the GPU's quad triangulation `(top, bottom, top + 1)`
/// and `(top + 1, bottom, bottom + 1)`.
pub fn evaluate_warp_patch_ecef(patch: &WarpPatch, tx: f64, ty: f64) -> DVec3 {
    let segs = patch.segments.max(1);
    let side = (segs + 1) as usize;
    let sx = (tx.clamp(0.0, 1.0)) * f64::from(segs);
    let sy = (ty.clamp(0.0, 1.0)) * f64::from(segs);
    let col = (sx.floor() as usize).min((segs - 1) as usize);
    let row = (sy.floor() as usize).min((segs - 1) as usize);
    let u = sx - col as f64;
    let v = sy - row as f64;

    let v_at = |r: usize, c: usize| {
        let p = patch.rel_positions[r * side + c];
        DVec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]))
    };
    let p00 = v_at(row, col);
    let p10 = v_at(row, col + 1);
    let p01 = v_at(row + 1, col);
    let p11 = v_at(row + 1, col + 1);

    let rel = if u + v <= 1.0 {
        // Triangle (p00, p01, p10): weights (1 - u - v, v, u).
        p00 * (1.0 - u - v) + p01 * v + p10 * u
    } else {
        // Triangle (p10, p01, p11): weights (1 - v, 1 - u, u + v - 1).
        p10 * (1.0 - v) + p01 * (1.0 - u) + p11 * (u + v - 1.0)
    };
    patch.origin_ecef_m + rel
}

/// Check whether standard CCW quad winding `(top, bottom, top + 1)` points
/// inward relative to the outward WGS84 ellipsoid direction and therefore
/// requires flipped triangle winding.
fn needs_flipped_winding(ecef_points: &[DVec3], side: usize) -> bool {
    if side < 2 {
        return false;
    }
    let mut signed_outward = 0.0_f64;
    for row in 0..side - 1 {
        for col in 0..side - 1 {
            let top = ecef_points[row * side + col];
            let right = ecef_points[row * side + col + 1];
            let bottom = ecef_points[(row + 1) * side + col];
            let tri_normal = (bottom - top).cross(right - top);
            // WGS84 is strictly convex around the origin, so `top` always has
            // positive dot product with the true outward ellipsoidal normal.
            signed_outward += tri_normal.dot(top);
        }
    }
    signed_outward < 0.0
}

/// Number of segments per side for a cell spanning `span_deg` degrees.
pub fn segments_for(span_deg: f64) -> u32 {
    if !span_deg.is_finite() || span_deg <= 0.0 {
        return MIN_WARP_SEGMENTS;
    }
    let raw = (span_deg / TARGET_WARP_SEGMENT_DEG).ceil();
    if !raw.is_finite() {
        return MAX_WARP_SEGMENTS;
    }
    (raw as u32).clamp(MIN_WARP_SEGMENTS, MAX_WARP_SEGMENTS)
}

fn cell_probes(
    frame: &Frame,
    width: u32,
    height: u32,
    level: u32,
    x: u32,
    y: u32,
) -> ([DVec3; 9], [(f64, f64); 9]) {
    let cell = cell_image_rect(x, y, level, width, height);
    let w = f64::from(width.max(1));
    let h = f64::from(height.max(1));
    let along = |from: f64, to: f64, step: usize| from + (to - from) * (step as f64) * 0.5;
    let mut world = [DVec3::ZERO; 9];
    let mut seen = [(0.0_f64, 0.0_f64); 9];
    for at in 0..9 {
        let px = along(cell[0], cell[2], at % 3);
        let py = along(cell[1], cell[3], at / 3);
        let (lat, lon) = frame.geodetic(px / w, py / h);
        seen[at] = (lat, lon);
        world[at] = WGS84.to_ecef(Geodetic::surface(lat, lon));
    }
    (world, seen)
}

fn compute_cell_ball(frame: &Frame, width: u32, height: u32, level: u32, x: u32, y: u32) -> Ball {
    let (points, seen) = cell_probes(frame, width, height, level, x, y);
    let span = span_of_probes(&seen);
    let segs = f64::from(segments_for(span));
    let sag_margin_m = sag::sag_m((span / segs) * std::f64::consts::SQRT_2);
    // Preserve the rung's own 9-probe limb-culling margin (`covering_radius_deg`)
    // with a 5% conservative factor so parent balls strictly enclose children
    // during top-down lazy descent.
    let reach_m = covering_radius_deg(&seen).to_radians() * geodesy::SEMI_MAJOR_M * 1.05;
    Ball::over(&points).widened(reach_m + sag_margin_m)
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal cell visibility helper taking pre-computed camera frustum and matrices"
)]
fn cell_on_screen(
    frame: &Frame,
    width: u32,
    height: u32,
    level: u32,
    x: u32,
    y: u32,
    frustum: &CullFrustum,
    view_proj: &DMat4,
    eye: DVec3,
) -> bool {
    let (world, seen) = cell_probes(frame, width, height, level, x, y);
    let mut low = [f64::MAX; 2];
    let mut high = [f64::MIN; 2];
    let mut faces = false;

    for point in world {
        faces |= frustum.horizon_margin(point) >= 0.0;
        let clip = camera::project(view_proj, point, eye);
        if clip.w <= 0.0 {
            return true;
        }
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        low[0] = low[0].min(ndc_x);
        high[0] = high[0].max(ndc_x);
        low[1] = low[1].min(ndc_y);
        high[1] = high[1].max(ndc_y);
    }

    let faces = faces || faces_cap(&world, eye, covering_radius_deg(&seen));
    faces && low[0] <= 1.0 && high[0] >= -1.0 && low[1] <= 1.0 && high[1] >= -1.0
}

fn cell_span_deg(frame: &Frame, width: f64, height: f64, cell: [f64; 4]) -> f64 {
    let at = |x: f64, y: f64| frame.geodetic(x / width, y / height);
    let xs = [cell[0], (cell[0] + cell[2]) * 0.5, cell[2]];
    let ys = [cell[1], (cell[1] + cell[3]) * 0.5, cell[3]];
    let mut span = 0.0_f64;
    for &y in &ys {
        span = span.max(arc_deg(at(xs[0], y), at(xs[2], y)));
    }
    for &x in &xs {
        span = span.max(arc_deg(at(x, ys[0]), at(x, ys[2])));
    }
    span
}

fn span_of_probes(seen: &[(f64, f64); 9]) -> f64 {
    let mut span = 0.0_f64;
    for row in 0..3 {
        span = span.max(arc_deg(seen[row * 3], seen[row * 3 + 2]));
    }
    for col in 0..3 {
        span = span.max(arc_deg(seen[col], seen[6 + col]));
    }
    span
}

fn covering_radius_deg(seen: &[(f64, f64); 9]) -> f64 {
    let mut step_x = 0.0_f64;
    let mut step_y = 0.0_f64;
    for at in 0..3 {
        for step in 0..2 {
            let row = at * 3 + step;
            step_x = step_x.max(arc_deg(seen[row], seen[row + 1]));
            let col = step * 3 + at;
            step_y = step_y.max(arc_deg(seen[col], seen[col + 3]));
        }
    }
    0.5 * step_x.hypot(step_y)
}

fn faces_cap(points: &[DVec3; 9], eye: DVec3, spread_deg: f64) -> bool {
    let mut lengths_sq = [0.0_f64; 9];
    for (len_sq, point) in lengths_sq.iter_mut().zip(points) {
        *len_sq = point.length_squared();
    }
    let lowest = lengths_sq.iter().copied().fold(f64::MAX, f64::min);
    let radius = lowest.sqrt();
    let beyond = (eye.length_squared() - lowest).max(0.0).sqrt();

    let (sin_spread, cos_spread) = spread_deg.to_radians().sin_cos();
    if beyond * cos_spread + radius * sin_spread < 0.0 {
        return true;
    }
    let reach = radius * cos_spread - beyond * sin_spread;
    points
        .iter()
        .zip(&lengths_sq)
        .any(|(point, len_sq)| point.dot(eye) > len_sq.sqrt() * reach)
}

fn axis_cell(axis: &[f64], at: f64) -> (usize, f64) {
    let last = axis.len() - 2;
    let i = axis
        .partition_point(|edge| *edge <= at)
        .saturating_sub(1)
        .min(last);
    (i, (at - axis[i]) / (axis[i + 1] - axis[i]))
}

fn lerp_pair(a: (f64, f64), b: (f64, f64), t: f64) -> (f64, f64) {
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}

fn ground_span_m(from: (f64, f64), to: (f64, f64)) -> f64 {
    arc_deg(from, to).to_radians() * geodesy::SEMI_MAJOR_M
}

fn arc_deg(from: (f64, f64), to: (f64, f64)) -> f64 {
    let along = (to.1 - from.1) * ((from.0 + to.0) * 0.5).to_radians().cos();
    (to.0 - from.0).hypot(along)
}

#[cfg(test)]
#[path = "overlay_tests.rs"]
mod tests;
