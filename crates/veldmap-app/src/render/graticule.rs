//! Adaptive procedural graticule step selection and per-tile `f64` graticule
//! anchor uniform packing shared by [`super::snapshot`], [`super::mesh`], and
//! [`super::globe`].

use veldmap_core::tile::{PlanetaryScheme, QuadTileId, TileSurfaceSample, sample_tile_surface};

/// `TileUniform::tile_kind` tag for `Geodetic` / `WebMercator` tiles.
pub(super) const TILE_KIND_GEO: f32 = 0.0;
/// `TileUniform::tile_kind` tag for North PolarCap tiles near the pole.
const TILE_KIND_POLAR_NORTH: f32 = 1.0;
/// `TileUniform::tile_kind` tag for South PolarCap tiles near the pole.
const TILE_KIND_POLAR_SOUTH: f32 = -1.0;
/// Scale multiplier applied to `TILE_KIND_POLAR_*` (`±2.0`) when a PolarCap
/// tile is strictly clear of the pole and uses cancellation-free relative
/// longitude deltas in `shaders/globe.wgsl` (`abs(tile.tile_kind) > 1.5`).
const TILE_KIND_POLAR_CLEAR_SCALE: f32 = 2.0;

/// Choose a clean graticule step in degrees (dividing `90°` evenly) such that
/// lines stay roughly `~85 pt` apart on screen across orbital-to-2 m zoom.
pub(super) fn adaptive_grid_step_deg(metres_per_pt: f64) -> f64 {
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

/// Cached tile-center `f64` graticule anchor sampled once during mesh build.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TileGridAnchor {
    pub(super) lat0: f64,
    pub(super) lon0: f64,
    pub(super) dx0: f64,
    pub(super) dy0: f64,
}

impl TileGridAnchor {
    pub(super) fn for_tile(id: QuadTileId) -> Self {
        match sample_tile_surface(id, 0.5, 0.5) {
            Some(TileSurfaceSample {
                lat_deg,
                lon_deg,
                polar_dx,
                polar_dy,
                ..
            }) => Self {
                lat0: lat_deg,
                lon0: lon_deg,
                dx0: polar_dx,
                dy0: polar_dy,
            },
            None => Self {
                lat0: match id.scheme {
                    PlanetaryScheme::PolarCap { north } => {
                        if north {
                            90.0
                        } else {
                            -90.0
                        }
                    }
                    PlanetaryScheme::Geodetic | PlanetaryScheme::WebMercator => 0.0,
                },
                lon0: 0.0,
                dx0: 0.0,
                dy0: 0.0,
            },
        }
    }
}

/// Compute per-tile `(tile_kind, grid_anchor)` uniform values from a cached
/// [`TileGridAnchor`] so `GlobeRenderer` never re-samples `sample_tile_surface`
/// on every dirty frame.
pub(super) fn tile_graticule_uniform(
    id: QuadTileId,
    cached_anchor: Option<TileGridAnchor>,
    step_deg: f64,
) -> (f32, [f32; 4]) {
    let anchor = cached_anchor.unwrap_or_else(|| TileGridAnchor::for_tile(id));
    match id.scheme {
        PlanetaryScheme::PolarCap { north } => {
            let colat_phase = (90.0 - anchor.lat0.abs()).rem_euclid(step_deg);
            let lon_phase = anchor.lon0.rem_euclid(step_deg);

            // Use cancellation-free relative longitude deltas when the tile's
            // disk center is strictly clear of the pole (`rho0 > 4 * tile_half_diag`).
            let rho0 = anchor.dx0.hypot(anchor.dy0);
            let n = f64::from(1u32 << id.level.min(30));
            let tile_half_diag = std::f64::consts::SQRT_2 / n;
            let clear_of_pole = id.level >= 2 && rho0 > 4.0 * tile_half_diag;
            let sign = if north {
                TILE_KIND_POLAR_NORTH
            } else {
                TILE_KIND_POLAR_SOUTH
            };
            let kind = if clear_of_pole {
                sign * TILE_KIND_POLAR_CLEAR_SCALE
            } else {
                sign
            };
            (
                kind,
                [
                    anchor.dx0 as f32,
                    anchor.dy0 as f32,
                    colat_phase as f32,
                    lon_phase as f32,
                ],
            )
        }
        PlanetaryScheme::Geodetic | PlanetaryScheme::WebMercator => {
            // Wrap tile anchor phase modulo `grid_step_deg` in f64 on CPU so
            // `f32` fragment arithmetic preserves sub-millimetre accuracy even
            // at `2 m` altitude.
            let lon_phase = anchor.lon0.rem_euclid(step_deg);
            let lat_phase = anchor.lat0.rem_euclid(step_deg);
            (
                TILE_KIND_GEO,
                [lon_phase as f32, lat_phase as f32, 0.0, 0.0],
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::adaptive_grid_step_deg;

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
