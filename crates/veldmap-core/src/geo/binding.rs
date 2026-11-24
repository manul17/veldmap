//! Pure georeference bindings mapping normalized raster fractions `(fx, fy)` in
//! `[0, 1]^2` onto the WGS84 ellipsoid (`Tie`, `TieGrid`, `Frame`).
//!
//! Kept strictly free of `crate::tile` dependencies so `veldmap-core::geo`
//! remains a pure geodesy/projection layer beneath `veldmap-core::tile`.

use glam::DVec3;
use thiserror::Error;

use super::footprint::{self, FootprintBounds};
use super::geodesy;
use super::projection::{self, Projection};

/// Tolerance for clustering tie-point raster fractions onto shared grid lines.
const SAME_LINE: f64 = 1e-6;

/// Web Mercator equatorial circumference in metres (`2 * pi * a`).
const WEB_MERCATOR_WORLD_M: f64 = 2.0 * std::f64::consts::PI * geodesy::SEMI_MAJOR_M;

/// Errors returned when constructing a georeference [`Frame`].
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum BindingError {
    #[error("unsupported projected CRS EPSG:{0}")]
    UnsupportedEpsg(u32),
    #[error("EPSG:{0} georeference transform has no valid ground extent")]
    InvalidExtent(u32),
    #[error("raster carries no valid georeference binding")]
    InvalidBinding,
}

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
    axial_lats: Vec<f64>,
    axial_lons: Vec<f64>,
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
                !(t.fx.is_finite()
                    && t.fy.is_finite()
                    && t.lat.is_finite()
                    && t.lon.is_finite()
                    && (-90.0..=90.0).contains(&t.lat))
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
            axial_lats: Vec::new(),
            axial_lons: Vec::new(),
            widest_row_m: 0.0,
            widest_col_m: 0.0,
        };
        grid.unwind();
        grid.axial = grid.compute_axial();
        if grid.axial {
            let across = grid.xs.len();
            grid.axial_lats = (0..grid.ys.len())
                .map(|row| grid.nodes[row * across].0)
                .collect();
            grid.axial_lons = (0..across).map(|col| grid.nodes[col].1).collect();
        }
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
        let rows = self.ys.len();
        let width = (self.xs[across - 1] - self.xs[0]).max(f64::EPSILON);
        let mut widest = self
            .nodes
            .chunks_exact(across)
            .map(|row| {
                row.windows(2)
                    .map(|pair| unwound_span_m(pair[0], pair[1]))
                    .sum::<f64>()
                    / width
            })
            .fold(0.0, f64::max);

        // Also sample midpoints between consecutive tie rows so 2-row
        // pole-to-pole grids (`lat = ±90°`) measure their interior width
        // instead of degenerate `cos(±90°)` polar rows.
        for row in 1..rows {
            let prev = &self.nodes[(row - 1) * across..row * across];
            let curr = &self.nodes[row * across..(row + 1) * across];
            let mut mid_span = 0.0;
            for col in 0..across - 1 {
                let left = lerp_pair(prev[col], curr[col], 0.5);
                let right = lerp_pair(prev[col + 1], curr[col + 1], 0.5);
                mid_span += unwound_span_m(left, right);
            }
            widest = widest.max(mid_span / width);
        }

        widest
    }

    fn compute_widest_column(&self) -> f64 {
        let across = self.xs.len();
        let height = (self.ys[self.ys.len() - 1] - self.ys[0]).max(f64::EPSILON);
        (0..across)
            .map(|col| {
                (1..self.ys.len())
                    .map(|row| {
                        unwound_span_m(
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

    /// Widest normalized row and column ground spans in metres.
    pub(crate) fn widest_spans_m(&self) -> (f64, f64) {
        (self.widest_row_m, self.widest_col_m)
    }

    /// Interpolate geodetic `(lat_deg, lon_deg)` for normalized raster fraction
    /// `(fx, fy)`.
    ///
    /// Output latitude is clamped to `[-90°, +90°]` so outer half-pixel
    /// extrapolation on pole-touching `PixelIsPoint` lattices cannot fold over
    /// the poles while leaving every interior sample untouched.
    pub fn geodetic(&self, fx: f64, fy: f64) -> (f64, f64) {
        let (col, tx) = axis_cell(&self.xs, fx);
        let (row, ty) = axis_cell(&self.ys, fy);
        let across = self.xs.len();
        let node = |r: usize, c: usize| self.nodes[r * across + c];
        let top = lerp_pair(node(row, col), node(row, col + 1), tx);
        let bottom = lerp_pair(node(row + 1, col), node(row + 1, col + 1), tx);
        let (lat, lon) = lerp_pair(top, bottom, ty);
        (lat.clamp(-90.0, 90.0), lon)
    }

    /// Invert geodetic `(lat_deg, lon_deg)` back into normalized raster
    /// fractions `(fx, fy)` inside `[0, 1]^2`, returning `None` if outside the
    /// raster extent.
    ///
    /// Outer lattice cells allow linear extrapolation out to `[0, 1]^2` so
    /// inset tie lattices (such as `PixelIsPoint` half-pixel-inset bindings)
    /// match [`Self::geodetic`] cleanly along outer raster edges.
    pub fn fraction_at(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        if !(lat_deg.is_finite() && lon_deg.is_finite()) {
            return None;
        }
        if self.axial {
            let fy = invert_1d_piecewise(&self.ys, &self.axial_lats, lat_deg)?;
            let fx = invert_1d_lon_piecewise(&self.xs, &self.axial_lons, lon_deg)?;
            return ((0.0..=1.0).contains(&fx) && (0.0..=1.0).contains(&fy)).then_some((fx, fy));
        }

        let across = self.xs.len();
        let last_row = self.ys.len() - 2;
        let last_col = across - 2;

        for row in 0..=last_row {
            let (ty_min, ty_max) = axis_extrapolation_bounds(&self.ys, row);
            for col in 0..=last_col {
                let (tx_min, tx_max) = axis_extrapolation_bounds(&self.xs, col);
                let n00 = self.nodes[row * across + col];
                let n10 = self.nodes[row * across + col + 1];
                let n01 = self.nodes[(row + 1) * across + col];
                let n11 = self.nodes[(row + 1) * across + col + 1];

                let lon_mid = 0.25 * (n00.1 + n10.1 + n01.1 + n11.1);
                let base_lon = geodesy::unwind(lon_mid, lon_deg);
                for shift in [-geodesy::TURN_DEG, 0.0, geodesy::TURN_DEG] {
                    let lon_u = base_lon + shift;
                    if let Some((tx, ty)) = invert_bilinear_cell(
                        n00,
                        n10,
                        n01,
                        n11,
                        (lat_deg, lon_u),
                        (tx_min, tx_max),
                        (ty_min, ty_max),
                    ) {
                        let fx =
                            (self.xs[col] + (self.xs[col + 1] - self.xs[col]) * tx).clamp(0.0, 1.0);
                        let fy =
                            (self.ys[row] + (self.ys[row + 1] - self.ys[row]) * ty).clamp(0.0, 1.0);
                        return Some((fx, fy));
                    }
                }
            }
        }
        None
    }
}

fn axis_extrapolation_bounds(axis: &[f64], seg: usize) -> (f64, f64) {
    let a0 = axis[seg];
    let a1 = axis[seg + 1];
    let span = (a1 - a0).max(f64::EPSILON);
    let t_min = if seg == 0 && a0 > 0.0 {
        -a0 / span
    } else {
        0.0
    };
    let t_max = if seg + 2 == axis.len() && a1 < 1.0 {
        1.0 + (1.0 - a1) / span
    } else {
        1.0
    };
    (t_min, t_max)
}

fn invert_1d_piecewise(axis: &[f64], values: &[f64], target: f64) -> Option<f64> {
    let eps = 1e-9;
    for i in 0..values.len().saturating_sub(1) {
        let (v0, v1) = (values[i], values[i + 1]);
        let span = v1 - v0;
        if span.abs() <= 0.0 {
            continue;
        }
        let (t_min, t_max) = axis_extrapolation_bounds(axis, i);
        let t = (target - v0) / span;
        if t >= t_min - eps && t <= t_max + eps {
            let t_clamped = t.clamp(t_min, t_max);
            let frac = axis[i] + (axis[i + 1] - axis[i]) * t_clamped;
            if (-1e-7..=1.0 + 1e-7).contains(&frac) {
                return Some(frac.clamp(0.0, 1.0));
            }
        }
    }
    None
}

fn invert_1d_lon_piecewise(axis: &[f64], lons: &[f64], target_lon: f64) -> Option<f64> {
    let base = lons.first().copied().unwrap_or(0.0);
    let unwound = geodesy::unwind(base, target_lon);
    for shift in [-geodesy::TURN_DEG, 0.0, geodesy::TURN_DEG] {
        if let Some(fx) = invert_1d_piecewise(axis, lons, unwound + shift) {
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
    tx_bounds: (f64, f64),
    ty_bounds: (f64, f64),
) -> Option<(f64, f64)> {
    // Quick AABB rejection widened for any outer-cell extrapolation.
    let min_lat = n00.0.min(n10.0).min(n01.0).min(n11.0);
    let max_lat = n00.0.max(n10.0).max(n01.0).max(n11.0);
    let min_lon = n00.1.min(n10.1).min(n01.1).min(n11.1);
    let max_lon = n00.1.max(n10.1).max(n01.1).max(n11.1);
    let lat_pad = (max_lat - min_lat)
        * (tx_bounds.0.abs().max(tx_bounds.1 - 1.0)).max(ty_bounds.0.abs().max(ty_bounds.1 - 1.0))
        + 1e-5;
    let lon_pad = (max_lon - min_lon)
        * (tx_bounds.0.abs().max(tx_bounds.1 - 1.0)).max(ty_bounds.0.abs().max(ty_bounds.1 - 1.0))
        + 1e-5;
    if target.0 < min_lat - lat_pad
        || target.0 > max_lat + lat_pad
        || target.1 < min_lon - lon_pad
        || target.1 > max_lon + lon_pad
    {
        return None;
    }

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
    if tx >= tx_bounds.0 - eps
        && tx <= tx_bounds.1 + eps
        && ty >= ty_bounds.0 - eps
        && ty <= ty_bounds.1 + eps
    {
        let tx_c = tx.clamp(tx_bounds.0, tx_bounds.1);
        let ty_c = ty.clamp(ty_bounds.0, ty_bounds.1);
        let top = lerp_pair(n00, n10, tx_c);
        let bot = lerp_pair(n01, n11, tx_c);
        let val = lerp_pair(top, bot, ty_c);
        if (val.0 - target.0).hypot(val.1 - target.1) < 1e-5 {
            return Some((tx_c, ty_c));
        }
    }
    None
}

#[derive(Clone, Debug, PartialEq)]
enum FrameKind {
    /// Exact map-projection binding:
    /// `x = affine[0]*fx + affine[1]*fy + affine[2]`,
    /// `y = affine[3]*fx + affine[4]*fy + affine[5]`,
    /// evaluated through `system.to_geodetic(x, y)`.
    Projected {
        system: Projection,
        affine: [f64; 6],
    },
    /// Rectangular tie-point lattice (for geographic, quad, or curvilinear swaths).
    Grid(TieGrid),
}

/// Validated georeference binding mapping normalized raster fractions
/// `(fx, fy)` in `[0, 1]^2` onto the WGS84 ellipsoid.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    kind: FrameKind,
}

impl Frame {
    /// Construct a north-up projected frame from map-projection bounds
    /// `[x0, x1] x [y0, y1]`, returning `None` if non-finite or degenerate.
    pub fn projected_bounds(
        system: impl Into<Projection>,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
    ) -> Option<Self> {
        if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
            return None;
        }
        let frame = Self {
            kind: FrameKind::Projected {
                system: system.into(),
                affine: [x1 - x0, 0.0, x0, 0.0, -(y1 - y0), y1],
            },
        };
        frame.measurable().then_some(frame)
    }

    /// Construct a projected frame from a pixel-space affine transform
    /// `x = a[0]*i + a[1]*j + a[2]`, `y = a[3]*i + a[4]*j + a[5]` and raster
    /// dimensions `(width, height)`.
    ///
    /// Returns `None` if dimensions are zero, any coefficient is non-finite,
    /// or the resulting frame has no valid finite ground extent.
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
        let frame = Self {
            kind: FrameKind::Projected {
                system,
                affine: [
                    pixel_affine[0] * across,
                    pixel_affine[1] * down,
                    pixel_affine[2],
                    pixel_affine[3] * across,
                    pixel_affine[4] * down,
                    pixel_affine[5],
                ],
            },
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
    ) -> Result<Self, BindingError> {
        let system = Projection::from_epsg(epsg).ok_or(BindingError::UnsupportedEpsg(epsg))?;
        Self::from_pixel_affine(system, pixel_affine, width, height)
            .ok_or(BindingError::InvalidExtent(epsg))
    }

    /// Wrap a validated [`TieGrid`] as a georeference [`Frame`].
    pub fn from_grid(grid: TieGrid) -> Self {
        Self {
            kind: FrameKind::Grid(grid),
        }
    }

    /// Construct a 2x2 geographic tie-grid frame from `(UL, UR, LR, LL)`
    /// corners, preserving 360-degree global spans and matching forward/inverse
    /// bilinear interpolation.
    pub fn quad(points: [(f64, f64); 4]) -> Option<Self> {
        let [ul, ur, lr, ll] = points;
        let ties = [
            Tie {
                fx: 0.0,
                fy: 0.0,
                lat: ul.0,
                lon: ul.1,
            },
            Tie {
                fx: 1.0,
                fy: 0.0,
                lat: ur.0,
                lon: ur.1,
            },
            Tie {
                fx: 0.0,
                fy: 1.0,
                lat: ll.0,
                lon: ll.1,
            },
            Tie {
                fx: 1.0,
                fy: 1.0,
                lat: lr.0,
                lon: lr.1,
            },
        ];
        TieGrid::new(&ties).map(Self::from_grid)
    }

    /// Evaluate WGS84 `(lat_deg, lon_deg)` at normalized raster fractions
    /// `(fx, fy)` (`0..1` left-to-right and top-to-bottom).
    pub fn geodetic(&self, fx: f64, fy: f64) -> (f64, f64) {
        match &self.kind {
            FrameKind::Projected { system, affine } => {
                let x = affine[0] * fx + affine[1] * fy + affine[2];
                let y = affine[3] * fx + affine[4] * fy + affine[5];
                system.to_geodetic(x, y)
            }
            FrameKind::Grid(grid) => grid.geodetic(fx, fy),
        }
    }

    /// Invert WGS84 `(lat_deg, lon_deg)` into normalized raster fractions
    /// `(fx, fy) in [0, 1]^2` for zoom-invariant CPU cursor readouts.
    pub fn fraction_at(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        if !(lat_deg.is_finite() && lon_deg.is_finite()) {
            return None;
        }
        match &self.kind {
            FrameKind::Projected { system, affine } => {
                let (x_base, y) = match system {
                    Projection::WebMercator => {
                        projection::web_mercator_from_geodetic_strict(lat_deg, lon_deg)?
                    }
                    Projection::Transverse(tm) => tm.from_geodetic(lat_deg, lon_deg),
                };
                if !(x_base.is_finite() && y.is_finite()) {
                    return None;
                }
                let det = affine[0] * affine[4] - affine[1] * affine[3];
                if !det.is_finite() || det.abs() < 1e-18 {
                    return None;
                }
                let try_xy = |x: f64| {
                    let dx = x - affine[2];
                    let dy = y - affine[5];
                    let fx = (dx * affine[4] - dy * affine[1]) / det;
                    let fy = (affine[0] * dy - affine[3] * dx) / det;
                    ((0.0..=1.0).contains(&fx) && (0.0..=1.0).contains(&fy)).then_some((fx, fy))
                };
                if let Some(hit) = try_xy(x_base) {
                    return Some(hit);
                }
                if matches!(system, Projection::WebMercator) {
                    for shift in [-WEB_MERCATOR_WORLD_M, WEB_MERCATOR_WORLD_M] {
                        if let Some(hit) = try_xy(x_base + shift) {
                            return Some(hit);
                        }
                    }
                }
                None
            }
            FrameKind::Grid(grid) => grid.fraction_at(lat_deg, lon_deg),
        }
    }

    /// Ground metres per horizontal pixel at native resolution.
    pub fn ground_m_per_px(&self, width: u32) -> Option<f64> {
        let span = self.horizontal_span_m();
        (span.is_finite() && span > 0.0).then(|| span / f64::from(width.max(1)))
    }

    /// Ground pixel size `(m_per_px_x, m_per_px_y)` at native resolution for
    /// HUD display (distinct from viewport GSD) and pyramid level selection.
    pub fn pixel_size_m(&self, width: u32, height: u32) -> Option<(f64, f64)> {
        let mpp_x = self.ground_m_per_px(width)?;
        let span_y = self.vertical_span_m();
        let mpp_y = if span_y.is_finite() && span_y > 0.0 {
            span_y / f64::from(height.max(1))
        } else {
            return None;
        };
        Some((mpp_x, mpp_y))
    }

    /// Widest grid row/column ground spans in metres for [`TieGrid`] bindings.
    pub(crate) fn widest_grid_spans_m(&self) -> Option<(f64, f64)> {
        match &self.kind {
            FrameKind::Grid(grid) => Some(grid.widest_spans_m()),
            FrameKind::Projected { .. } => None,
        }
    }

    fn horizontal_span_m(&self) -> f64 {
        match &self.kind {
            FrameKind::Projected {
                system: Projection::WebMercator,
                affine,
            } => {
                let (center_lat, _) = self.geodetic(0.5, 0.5);
                affine[0].hypot(affine[3]) * center_lat.to_radians().cos().abs().max(1e-6)
            }
            FrameKind::Projected { affine, .. } => affine[0].hypot(affine[3]),
            FrameKind::Grid(grid) => {
                let probed = [0.0, 0.5, 1.0]
                    .iter()
                    .map(|&fy| {
                        let a = self.geodetic(0.0, fy);
                        let b = self.geodetic(0.5, fy);
                        let c = self.geodetic(1.0, fy);
                        geodesic_span_m(a, b) + geodesic_span_m(b, c)
                    })
                    .fold(0.0, f64::max);
                probed.max(grid.widest_row_m)
            }
        }
    }

    fn vertical_span_m(&self) -> f64 {
        match &self.kind {
            FrameKind::Projected {
                system: Projection::WebMercator,
                affine,
            } => {
                let (center_lat, _) = self.geodetic(0.5, 0.5);
                affine[1].hypot(affine[4]) * center_lat.to_radians().cos().abs().max(1e-6)
            }
            FrameKind::Projected { affine, .. } => affine[1].hypot(affine[4]),
            FrameKind::Grid(grid) => {
                let probed = [0.0, 0.5, 1.0]
                    .iter()
                    .map(|&fx| {
                        let a = self.geodetic(fx, 0.0);
                        let b = self.geodetic(fx, 0.5);
                        let c = self.geodetic(fx, 1.0);
                        geodesic_span_m(a, b) + geodesic_span_m(b, c)
                    })
                    .fold(0.0, f64::max);
                probed.max(grid.widest_col_m)
            }
        }
    }

    /// Whether rows follow parallels and columns follow meridians (enabling
    /// `O(N)` separable trig evaluation during culling and warp-mesh builds).
    pub fn axial(&self) -> bool {
        match &self.kind {
            FrameKind::Projected {
                system: Projection::WebMercator,
                affine,
            } => affine[1] == 0.0 && affine[3] == 0.0,
            FrameKind::Projected { .. } => false,
            FrameKind::Grid(grid) => grid.axial,
        }
    }

    /// Whether the binding has positive, finite ground extent on both axes and
    /// maps corners and center to finite geodetic coordinates.
    pub fn measurable(&self) -> bool {
        if let FrameKind::Projected { affine, .. } = &self.kind {
            let det = affine[0] * affine[4] - affine[1] * affine[3];
            if !det.is_finite() || det.abs() == 0.0 {
                return false;
            }
        }
        let samples_finite = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)]
            .into_iter()
            .all(|(fx, fy)| {
                let (lat, lon) = self.geodetic(fx, fy);
                lat.is_finite() && lon.is_finite() && (-90.0..=90.0).contains(&lat)
            });
        samples_finite && self.pixel_size_m(1, 1).is_some()
    }

    /// Select the sharpest raster pyramid level whose finer pixel axis is no
    /// coarser than `viewport_mpp` metres per pixel, clamped to `0..=max_level`.
    pub fn sharpest_level(&self, width: u32, height: u32, viewport_mpp: f64, max_level: u8) -> u8 {
        let Some((mpp_x, mpp_y)) = self.pixel_size_m(width, height) else {
            return max_level;
        };
        let mpp_raster = mpp_x.min(mpp_y);
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

    /// Spherical bounding circle around the raster for "Zoom to extent" camera
    /// framing.
    ///
    /// Computes the spherical centroid over a symmetric `4 x 4` interior
    /// midpoint grid (avoiding double-weighting `±180°` edges on global
    /// rasters) and falls back to `geodetic(0.5, 0.5)` when full-globe vectors
    /// cancel.
    pub fn footprint_bounds(&self) -> Option<FootprintBounds> {
        const INTERIOR_GRID: usize = 4;
        const EDGE_STEPS: usize = 4;

        let mut interior = Vec::with_capacity(INTERIOR_GRID * INTERIOR_GRID);
        let mut sum = DVec3::ZERO;
        for iy in 0..INTERIOR_GRID {
            let fy = (iy as f64 + 0.5) / INTERIOR_GRID as f64;
            for ix in 0..INTERIOR_GRID {
                let fx = (ix as f64 + 0.5) / INTERIOR_GRID as f64;
                let (lat, lon) = self.geodetic(fx, fy);
                if lat.is_finite() && lon.is_finite() {
                    interior.push((lat, lon));
                    sum += geodesy::unit(lat, lon);
                }
            }
        }
        if interior.is_empty() {
            return None;
        }

        let (center_lat, center_lon) = if sum.length() > 1e-9 {
            geodesy::angles(sum.normalize())
        } else {
            let (lat, lon) = self.geodetic(0.5, 0.5);
            if !(lat.is_finite() && lon.is_finite()) {
                interior[0]
            } else {
                (lat, lon)
            }
        };

        let mut radius_deg = interior
            .iter()
            .map(|&p| geodesy::separation((center_lat, center_lon), p))
            .fold(0.0_f64, f64::max);

        let mut check_edge = |fx: f64, fy: f64| {
            let (lat, lon) = self.geodetic(fx, fy);
            if lat.is_finite() && lon.is_finite() {
                radius_deg =
                    radius_deg.max(geodesy::separation((center_lat, center_lon), (lat, lon)));
            }
        };
        for i in 0..=EDGE_STEPS {
            let t = i as f64 / EDGE_STEPS as f64;
            check_edge(t, 0.0);
            check_edge(1.0, t);
            check_edge(t, 1.0);
            check_edge(0.0, t);
        }

        Some(FootprintBounds {
            lat: center_lat,
            lon: geodesy::unwind(0.0, center_lon),
            radius_deg,
        })
    }
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

fn unwound_span_m(from: (f64, f64), to: (f64, f64)) -> f64 {
    let along = (to.1 - from.1) * ((from.0 + to.0) * 0.5).to_radians().cos();
    (to.0 - from.0).hypot(along).to_radians() * geodesy::SEMI_MAJOR_M
}

fn geodesic_span_m(from: (f64, f64), to: (f64, f64)) -> f64 {
    geodesy::separation(from, to).to_radians() * geodesy::SEMI_MAJOR_M
}

#[cfg(test)]
#[path = "binding_tests.rs"]
mod tests;
