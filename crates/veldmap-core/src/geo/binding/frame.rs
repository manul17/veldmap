//! Validated georeference frame (`Frame`) mapping normalized raster fractions
//! `(fx, fy)` in `[0, 1]^2` onto the WGS84 ellipsoid via either an exact map
//! projection (`Projection` + affine) or a rectangular tie-point lattice
//! (`TieGrid` + optional horizontal datum shift).

use std::sync::Arc;

use super::super::crs::{CrsSpec, Datum, DatumWarning, Proj4Projection, Projection, WGS84_DATUM};
use super::super::footprint::FootprintBounds;
use super::super::geodesy;
#[cfg(any(test, feature = "test-support"))]
use super::tie_grid::Tie;
use super::tie_grid::TieGrid;

/// Web Mercator equatorial circumference in metres (`2 * pi * a`).
const WEB_MERCATOR_WORLD_M: f64 = geodesy::EQUATORIAL_CIRCUMFERENCE_M;

#[derive(Clone, Debug, PartialEq)]
enum GeographicShift {
    /// Static 7-parameter Helmert datum (identity when [`WGS84_DATUM`]).
    Datum(Datum),
    /// General `proj4rs` geographic CRS (e.g. from `.prj` WKT).
    Proj4(Arc<Proj4Projection>),
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
    /// Rectangular tie-point lattice (for geographic, quad, or curvilinear
    /// swaths) with optional horizontal datum conversion to WGS84.
    Grid {
        grid: TieGrid,
        shift: GeographicShift,
        label: String,
    },
}

/// Validated georeference binding mapping normalized raster fractions
/// `(fx, fy)` in `[0, 1]^2` onto the WGS84 ellipsoid.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    kind: FrameKind,
    datum_warning: Option<DatumWarning>,
    /// Cached positive finite `(span_x_m, span_y_m)` computed once at
    /// construction so per-frame culling and LOD queries stay `O(1)`.
    cached_ground_spans_m: Option<(f64, f64)>,
    /// Cached [`Self::measurable`] result computed once at construction.
    cached_measurable: bool,
}

impl Frame {
    fn from_parts(kind: FrameKind, datum_warning: Option<DatumWarning>) -> Self {
        let mut frame = Self {
            kind,
            datum_warning,
            cached_ground_spans_m: None,
            cached_measurable: false,
        };
        let (span_x, span_y) = frame.compute_ground_spans_m();
        if span_x.is_finite() && span_x > 0.0 && span_y.is_finite() && span_y > 0.0 {
            frame.cached_ground_spans_m = Some((span_x, span_y));
        }
        frame.cached_measurable = frame.compute_measurable();
        frame
    }

    /// Construct a north-up projected frame from map-projection bounds
    /// `[x0, x1] x [y0, y1]`, returning `None` if non-finite or degenerate.
    #[cfg(test)]
    pub(crate) fn projected_bounds(
        system: impl Into<Projection>,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
    ) -> Option<Self> {
        if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
            return None;
        }
        let system = system.into();
        let datum_warning = system.datum_warning().cloned();
        let frame = Self::from_parts(
            FrameKind::Projected {
                system,
                affine: [x1 - x0, 0.0, x0, 0.0, -(y1 - y0), y1],
            },
            datum_warning,
        );
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
        let datum_warning = system.datum_warning().cloned();
        let frame = Self::from_parts(
            FrameKind::Projected {
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
            datum_warning,
        );
        frame.measurable().then_some(frame)
    }

    /// Construct a frame from any [`CrsSpec`] (projected or geographic) and a
    /// pixel-space affine transform `x = a[0]*i + a[1]*j + a[2]`,
    /// `y = a[3]*i + a[4]*j + a[5]`.
    pub fn from_crs_pixel_affine(
        crs: &CrsSpec,
        pixel_affine: [f64; 6],
        width: u32,
        height: u32,
    ) -> Option<Self> {
        if width == 0 || height == 0 || !pixel_affine.iter().all(|v| v.is_finite()) {
            return None;
        }
        match crs {
            CrsSpec::Projected(proj) => {
                Self::from_pixel_affine(proj.clone(), pixel_affine, width, height)
            }
            CrsSpec::Geographic { .. } | CrsSpec::GeographicProj(_) => {
                let grid = TieGrid::from_geographic_affine(pixel_affine, width, height)?;
                let frame = Self::from_grid_with_crs(grid, Some(crs));
                frame.measurable().then_some(frame)
            }
        }
    }

    /// Wrap a validated [`TieGrid`] on WGS84 as a georeference [`Frame`].
    pub fn from_grid(grid: TieGrid) -> Self {
        Self::from_parts(
            FrameKind::Grid {
                grid,
                shift: GeographicShift::Datum(WGS84_DATUM),
                label: "EPSG:4326 (WGS 84)".to_string(),
            },
            None,
        )
    }

    /// Wrap a validated [`TieGrid`] using the horizontal datum or geographic
    /// CRS described by `crs` (defaulting to WGS84 when `crs` is `None` or
    /// projected).
    pub fn from_grid_with_crs(grid: TieGrid, crs: Option<&CrsSpec>) -> Self {
        match crs {
            Some(CrsSpec::Geographic {
                datum,
                label,
                warning,
            }) => Self::from_grid_with_datum(grid, *datum, label.clone())
                .with_datum_warning(warning.clone()),
            Some(CrsSpec::GeographicProj(proj)) => {
                Self::from_grid_with_proj(grid, Arc::clone(proj))
            }
            None | Some(CrsSpec::Projected(_)) => Self::from_grid(grid),
        }
    }

    /// Wrap a validated [`TieGrid`] on a specific horizontal [`Datum`],
    /// shifting coordinates to/from WGS84 via [`Datum::to_wgs84`] /
    /// [`Datum::from_wgs84`] (discarding phantom Helmert `Delta h`).
    fn from_grid_with_datum(grid: TieGrid, datum: Datum, label: impl Into<String>) -> Self {
        Self::from_parts(
            FrameKind::Grid {
                grid,
                shift: GeographicShift::Datum(datum),
                label: label.into(),
            },
            None,
        )
    }

    /// Wrap a validated [`TieGrid`] on an arbitrary `proj4rs` geographic CRS.
    fn from_grid_with_proj(grid: TieGrid, proj: Arc<Proj4Projection>) -> Self {
        let datum_warning = proj.datum_warning().cloned();
        let label = proj.label().to_string();
        Self::from_parts(
            FrameKind::Grid {
                grid,
                shift: GeographicShift::Proj4(proj),
                label,
            },
            datum_warning,
        )
    }

    /// Attach a diagnostic [`DatumWarning`] on this frame (leaving any existing
    /// warning intact when `warning` is `None`).
    pub fn with_datum_warning(mut self, warning: Option<DatumWarning>) -> Self {
        if warning.is_some() {
            self.datum_warning = warning;
        }
        self
    }

    /// Optional datum diagnostic warning for UI banners.
    pub fn datum_warning(&self) -> Option<&DatumWarning> {
        self.datum_warning.as_ref()
    }

    /// Concise human-readable CRS label for UI metadata inspectors.
    pub fn crs_label(&self) -> String {
        match &self.kind {
            FrameKind::Projected { system, .. } => system.label(),
            FrameKind::Grid { label, .. } => label.clone(),
        }
    }

    /// Construct a 2x2 geographic tie-grid frame from `(UL, UR, LR, LL)`
    /// corners, preserving 360-degree global spans and matching forward/inverse
    /// bilinear interpolation.
    #[cfg(any(test, feature = "test-support"))]
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
            FrameKind::Grid { grid, shift, .. } => {
                let (lat, lon) = grid.geodetic(fx, fy);
                match shift {
                    GeographicShift::Datum(datum) => datum.to_wgs84(lat, lon),
                    GeographicShift::Proj4(proj) => proj.to_wgs84_geodetic(lon, lat),
                }
            }
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
                let (x_base, y) = system.project_geodetic_strict(lat_deg, lon_deg)?;
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
            FrameKind::Grid { grid, shift, .. } => {
                let (src_lat, src_lon) = match shift {
                    GeographicShift::Datum(datum) => datum.from_wgs84(lat_deg, lon_deg),
                    GeographicShift::Proj4(proj) => {
                        let (lon, lat) = proj.transform_from_wgs84_geodetic(lat_deg, lon_deg);
                        (lat, lon)
                    }
                };
                if !(src_lat.is_finite() && src_lon.is_finite()) {
                    return None;
                }
                grid.fraction_at(src_lat, src_lon)
            }
        }
    }

    /// Ground pixel size `(m_per_px_x, m_per_px_y)` at native resolution for
    /// HUD display (distinct from viewport GSD) and pyramid level selection.
    pub fn pixel_size_m(&self, width: u32, height: u32) -> Option<(f64, f64)> {
        self.cached_ground_spans_m.map(|(span_x, span_y)| {
            (
                span_x / f64::from(width.max(1)),
                span_y / f64::from(height.max(1)),
            )
        })
    }

    /// Widest grid row/column ground spans in metres for [`TieGrid`] bindings.
    pub(crate) fn widest_grid_spans_m(&self) -> Option<(f64, f64)> {
        match &self.kind {
            FrameKind::Grid { grid, .. } => Some(grid.widest_spans_m()),
            FrameKind::Projected { .. } => None,
        }
    }

    /// Probe a `3 x 3` grid over normalized rectangle `[fx0, fx1] x [fy0, fy1]`
    /// in a single 9-point pass, returning the widest horizontal and vertical
    /// great-circle arc spans in degrees.
    pub(crate) fn probe_rect_spans_deg(
        &self,
        fx0: f64,
        fx1: f64,
        fy0: f64,
        fy1: f64,
    ) -> (f64, f64) {
        let xs = [fx0, (fx0 + fx1) * 0.5, fx1];
        let ys = [fy0, (fy0 + fy1) * 0.5, fy1];
        let mut probes = [(0.0_f64, 0.0_f64); 9];
        for row in 0..3 {
            for col in 0..3 {
                probes[row * 3 + col] = self.geodetic(xs[col], ys[row]);
            }
        }
        let mut span_x = 0.0_f64;
        for row in 0..3 {
            let a = probes[row * 3];
            let b = probes[row * 3 + 1];
            let c = probes[row * 3 + 2];
            span_x = span_x.max(geodesy::separation(a, b) + geodesy::separation(b, c));
        }
        let mut span_y = 0.0_f64;
        for col in 0..3 {
            let a = probes[col];
            let b = probes[3 + col];
            let c = probes[6 + col];
            span_y = span_y.max(geodesy::separation(a, b) + geodesy::separation(b, c));
        }
        (span_x, span_y)
    }

    fn compute_ground_spans_m(&self) -> (f64, f64) {
        match &self.kind {
            FrameKind::Projected {
                system: Projection::WebMercator,
                affine,
            } => {
                let (center_lat, _) = self.geodetic(0.5, 0.5);
                let scale = center_lat.to_radians().cos().abs().max(1e-6);
                (
                    affine[0].hypot(affine[3]) * scale,
                    affine[1].hypot(affine[4]) * scale,
                )
            }
            FrameKind::Projected {
                system: sys @ Projection::Transverse { .. },
                affine,
            } => {
                let unit = sys.unit_to_metres();
                (
                    affine[0].hypot(affine[3]) * unit,
                    affine[1].hypot(affine[4]) * unit,
                )
            }
            FrameKind::Projected {
                system: Projection::Proj4(proj),
                affine,
            } => {
                let unit = proj.unit_to_metres();
                let nominal_x = affine[0].hypot(affine[3]) * unit;
                let nominal_y = affine[1].hypot(affine[4]) * unit;
                let (deg_x, deg_y) = self.probe_rect_spans_deg(0.0, 1.0, 0.0, 1.0);
                let probed_x = deg_x.to_radians() * geodesy::SEMI_MAJOR_M;
                let probed_y = deg_y.to_radians() * geodesy::SEMI_MAJOR_M;
                (
                    if probed_x > 0.0 { probed_x } else { nominal_x },
                    if probed_y > 0.0 { probed_y } else { nominal_y },
                )
            }
            FrameKind::Grid { grid, .. } => {
                let (widest_x, widest_y) = grid.widest_spans_m();
                let (deg_x, deg_y) = self.probe_rect_spans_deg(0.0, 1.0, 0.0, 1.0);
                let probed_x = deg_x.to_radians() * geodesy::SEMI_MAJOR_M;
                let probed_y = deg_y.to_radians() * geodesy::SEMI_MAJOR_M;
                (probed_x.max(widest_x), probed_y.max(widest_y))
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
            FrameKind::Grid { grid, shift, .. } => {
                matches!(shift, GeographicShift::Datum(d) if d.is_wgs84()) && grid.is_axial()
            }
        }
    }

    /// Whether the binding has positive, finite ground extent on both axes and
    /// maps corners and center to finite geodetic coordinates.
    pub fn measurable(&self) -> bool {
        self.cached_measurable
    }

    fn compute_measurable(&self) -> bool {
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
        samples_finite && self.cached_ground_spans_m.is_some()
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
        for iy in 0..INTERIOR_GRID {
            let fy = (iy as f64 + 0.5) / INTERIOR_GRID as f64;
            for ix in 0..INTERIOR_GRID {
                let fx = (ix as f64 + 0.5) / INTERIOR_GRID as f64;
                let (lat, lon) = self.geodetic(fx, fy);
                if lat.is_finite() && lon.is_finite() {
                    interior.push((lat, lon));
                }
            }
        }
        if interior.is_empty() {
            return None;
        }

        let (center_lat, center_lon) = geodesy::spherical_centroid(interior.iter().copied())
            .unwrap_or_else(|| {
                let (lat, lon) = self.geodetic(0.5, 0.5);
                if lat.is_finite() && lon.is_finite() {
                    (lat, lon)
                } else {
                    interior[0]
                }
            });

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
