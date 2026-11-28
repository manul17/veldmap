//! Seamless polar-cap harmonic synthesis from live `WebMercator` polar-rim
//! texels (`±85.05112878°`).
//!
//! Standard Web Mercator basemap servers (`EPSG:3857`) terminate strictly at
//! `±85.05112878°`. Building a 1D wrap-around linear-light mip pyramid around
//! that boundary ring and evaluating an isotropic Poisson-width lookup inward
//! from `rho = 1.0` (exact live rim) to `rho = 0.0` (global rim mean at the
//! pole) eliminates the hard circular cutout artifact without issuing any live
//! HTTP requests for `PolarCap` tiles.

use std::f64::consts::PI;

use veldmap_core::tile::{
    DecodedTile, PixelFormat, PlanetaryScheme, QuadTileId, polar_cap_point, polar_cap_uv_bounds,
};

use crate::RasterError;
use crate::color::{linear_to_srgb_lut, linear_to_srgb_u8, srgb_to_linear_lut};
use crate::mipmap::generate_tile_mipmaps;

/// Longitudinal resolution of the `±85.05112878°` polar rim ring (`2048` bins
/// ≈ `0.176°` / `~1.6 km` along the polar cap perimeter).
pub const POLAR_RIM_BINS: usize = 2048;

/// Maximum single polar cap tile dimension accepted by
/// [`render_harmonic_polar_cap_with_mipmaps`].
const MAX_POLAR_TILE_DIM: u32 = 2048;

/// Precomputed 1D wrap-around linear-light mip pyramid around the `±85.05112878°`
/// polar cap boundary ring.
#[derive(Clone, Debug)]
pub struct PolarRimPyramid {
    /// Mip levels from finest (`levels[0].len() == POLAR_RIM_BINS`) down to the
    /// 1-bin global mean (`levels.last().len() == 1`).
    levels: Vec<Vec<[f32; 3]>>,
}

impl PolarRimPyramid {
    /// Build a 1D wrap-around linear-light mip pyramid from `POLAR_RIM_BINS`
    /// `(linear_rgb, populated)` slots, interpolating cleanly across any
    /// unpopulated angular gaps. Returns `None` if no bin is populated or the
    /// slice length does not match [`POLAR_RIM_BINS`].
    pub fn from_bins(bins: &[([f32; 3], bool)]) -> Option<Self> {
        if bins.len() != POLAR_RIM_BINS {
            return None;
        }
        let populated_indices: Vec<usize> = bins
            .iter()
            .enumerate()
            .filter_map(|(i, (_, ok))| ok.then_some(i))
            .collect();
        if populated_indices.is_empty() {
            return None;
        }

        let mut base = vec![[0.0_f32; 3]; POLAR_RIM_BINS];
        if populated_indices.len() == 1 {
            base.fill(bins[populated_indices[0]].0);
        } else {
            // Fill populated bins directly and linearly interpolate across
            // circular gaps between adjacent populated bins.
            let count = populated_indices.len();
            for p in 0..count {
                let start = populated_indices[p];
                let end = populated_indices[(p + 1) % count];
                let c0 = bins[start].0;
                let c1 = bins[end].0;
                base[start] = c0;

                let span = if end > start {
                    end - start
                } else {
                    (POLAR_RIM_BINS - start) + end
                };
                for step in 1..span {
                    let t = (step as f32) / (span as f32);
                    let inv = 1.0 - t;
                    let idx = (start + step) % POLAR_RIM_BINS;
                    base[idx] = [
                        c0[0] * inv + c1[0] * t,
                        c0[1] * inv + c1[1] * t,
                        c0[2] * inv + c1[2] * t,
                    ];
                }
            }
        }

        let mut levels = Vec::with_capacity(12);
        levels.push(base);
        while let Some(prev) = levels.last() {
            if prev.len() <= 1 {
                break;
            }
            let next_len = prev.len() / 2;
            let mut next = Vec::with_capacity(next_len);
            for i in 0..next_len {
                let a = prev[i * 2];
                let b = prev[i * 2 + 1];
                next.push([
                    0.5 * (a[0] + b[0]),
                    0.5 * (a[1] + b[1]),
                    0.5 * (a[2] + b[2]),
                ]);
            }
            levels.push(next);
        }

        Some(Self { levels })
    }

    #[inline]
    fn sample_level(&self, level_idx: usize, u_wrap: f64) -> [f32; 3] {
        let row = &self.levels[level_idx];
        let n = row.len();
        if n == 1 {
            return row[0];
        }
        let fx = u_wrap * (n as f64) - 0.5;
        let x0_i = fx.floor() as isize;
        let t = (fx - (x0_i as f64)) as f32;
        let inv = 1.0 - t;
        let x0 = x0_i.rem_euclid(n as isize) as usize;
        let x1 = (x0 + 1) % n;
        let c0 = row[x0];
        let c1 = row[x1];
        [
            c0[0] * inv + c1[0] * t,
            c0[1] * inv + c1[1] * t,
            c0[2] * inv + c1[2] * t,
        ]
    }

    /// Evaluate the smooth harmonic extension of the rim ring at `(lon_deg, rho)`
    /// where `rho = 1.0` lies on `±85.05112878°` (exact live rim color) and
    /// `rho = 0.0` lies at the exact pole (global mean rim color).
    #[inline]
    pub fn sample_harmonic(&self, lon_deg: f64, rho: f64) -> [f32; 3] {
        let u = ((lon_deg + 180.0) / 360.0).rem_euclid(1.0);
        let r = rho.clamp(0.0, 1.0);
        // Approximate isotropic Poisson filter width near `rho = 1`
        // (`~(1-rho)/(2*pi)` of the circle) smoothly expanding to `1.0`
        // (global mean) at `rho = 0` via the dyadic mipmap pyramid.
        let frac = (1.0 - r) / (1.0 + r * (2.0 * PI - 1.0));
        let max_lod = (self.levels.len().saturating_sub(1)) as f64;
        let lod = (1.0 + frac * (POLAR_RIM_BINS as f64))
            .log2()
            .clamp(0.0, max_lod);
        let l0 = (lod.floor() as usize).min(self.levels.len() - 1);
        let l1 = (l0 + 1).min(self.levels.len() - 1);
        let frac_lod = (lod - (l0 as f64)) as f32;
        let c0 = self.sample_level(l0, u);
        if l0 == l1 || frac_lod <= 1e-6 {
            return c0;
        }
        let c1 = self.sample_level(l1, u);
        let inv = 1.0 - frac_lod;
        [
            c0[0] * inv + c1[0] * frac_lod,
            c0[1] * inv + c1[1] * frac_lod,
            c0[2] * inv + c1[2] * frac_lod,
        ]
    }
}

/// Extract linear-light RGB texels along the polar-facing edge of an sRGB
/// `Rgba8UnormSrgb` `WebMercator` tile (`north = true` -> top edge `y = 0`,
/// `north = false` -> bottom edge `y = height - 1`).
///
/// Averages up to 2 opaque edge rows in linear light to suppress 1-pixel JPEG
/// block ringing along the seam; transparent nodata pixels return `None`.
pub fn extract_srgb_rim_row(tile: &DecodedTile, north: bool) -> Vec<Option<[f32; 3]>> {
    if tile.format() != PixelFormat::Rgba8UnormSrgb || tile.width() == 0 || tile.height() == 0 {
        return Vec::new();
    }
    let w = tile.width() as usize;
    let h = tile.height() as usize;
    let data = tile.data();
    let needed = w.saturating_mul(h).saturating_mul(4);
    if data.len() < needed {
        return Vec::new();
    }
    let to_lin = srgb_to_linear_lut();
    let (r0, r1) = if north {
        (0usize, 1usize.min(h - 1))
    } else {
        (h - 1, h.saturating_sub(2))
    };
    let mut out = Vec::with_capacity(w);
    for x in 0..w {
        let o0 = (r0 * w + x) * 4;
        let o1 = (r1 * w + x) * 4;
        let a0 = f32::from(data[o0 + 3]) / 255.0;
        let a1 = if r0 == r1 {
            0.0
        } else {
            f32::from(data[o1 + 3]) / 255.0
        };
        let w0 = 0.7 * a0;
        let w1 = 0.3 * a1;
        let w_sum = w0 + w1;
        if w_sum <= 1e-4 {
            out.push(None);
        } else {
            let inv = 1.0 / w_sum;
            out.push(Some([
                (to_lin[usize::from(data[o0])] * w0 + to_lin[usize::from(data[o1])] * w1) * inv,
                (to_lin[usize::from(data[o0 + 1])] * w0 + to_lin[usize::from(data[o1 + 1])] * w1)
                    * inv,
                (to_lin[usize::from(data[o0 + 2])] * w0 + to_lin[usize::from(data[o1 + 2])] * w1)
                    * inv,
            ]));
        }
    }
    out
}

/// Render one `tile_px x tile_px` `PolarCap` tile with CPU linear-light mipmaps
/// by evaluating [`PolarRimPyramid::sample_harmonic`] across the elliptical
/// square-to-disk polar stereographic cap.
pub fn render_harmonic_polar_cap_with_mipmaps(
    id: QuadTileId,
    tile_px: u32,
    max_polar_level: u8,
    rim: &PolarRimPyramid,
) -> Result<DecodedTile, RasterError> {
    if tile_px == 0 || tile_px > MAX_POLAR_TILE_DIM {
        return Err(RasterError::InvalidDimensions {
            width: tile_px,
            height: tile_px,
        });
    }
    let PlanetaryScheme::PolarCap { north } = id.scheme else {
        return Err(RasterError::TileOutOfRange {
            scheme: id.scheme.into_tile_scheme(),
            level: id.level,
            x: id.x,
            y: id.y,
        });
    };
    if id.level > max_polar_level {
        return Err(RasterError::TileOutOfRange {
            scheme: id.scheme.into_tile_scheme(),
            level: id.level,
            x: id.x,
            y: id.y,
        });
    }
    let (cols, _) = id.checked_grid_dims().ok_or(RasterError::TileOutOfRange {
        scheme: id.scheme.into_tile_scheme(),
        level: id.level,
        x: id.x,
        y: id.y,
    })?;

    let tile_usize = tile_px as usize;
    let total_bytes = tile_usize
        .checked_mul(tile_usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or(RasterError::InvalidDimensions {
            width: tile_px,
            height: tile_px,
        })?;
    let mut data = vec![0u8; total_bytes];

    let (u0, u1, v0, v1) = polar_cap_uv_bounds(id.x, id.y, cols);
    let tile_f = f64::from(tile_px);
    let to_srgb = linear_to_srgb_lut();

    for py in 0..tile_usize {
        let t = (py as f64 + 0.5) / tile_f;
        let v = v0 + (v1 - v0) * t;
        let row_off = py * tile_usize * 4;
        for px in 0..tile_usize {
            let s = (px as f64 + 0.5) / tile_f;
            let u = u0 + (u1 - u0) * s;
            let (_, lon_deg, dx, dy) = polar_cap_point(north, u, v);
            let rho = dx.hypot(dy).min(1.0);
            let lin = rim.sample_harmonic(lon_deg, rho);
            let off = row_off + px * 4;
            data[off] = linear_to_srgb_u8(lin[0], to_srgb);
            data[off + 1] = linear_to_srgb_u8(lin[1], to_srgb);
            data[off + 2] = linear_to_srgb_u8(lin[2], to_srgb);
            data[off + 3] = 255;
        }
    }

    let mut tile = DecodedTile::try_new(tile_px, tile_px, PixelFormat::Rgba8UnormSrgb, data)
        .map_err(|_| RasterError::InvalidDimensions {
            width: tile_px,
            height: tile_px,
        })?;
    generate_tile_mipmaps(&mut tile);
    Ok(tile)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harmonic_rim_matches_boundary_and_converges_to_mean_at_pole() {
        let mut bins = vec![([0.0_f32; 3], false); POLAR_RIM_BINS];
        // West half = pure red (1,0,0), East half = pure blue (0,0,1).
        for (i, slot) in bins.iter_mut().enumerate() {
            if i < POLAR_RIM_BINS / 2 {
                *slot = ([1.0, 0.0, 0.0], true);
            } else {
                *slot = ([0.0, 0.0, 1.0], true);
            }
        }
        let rim = PolarRimPyramid::from_bins(&bins).expect("valid rim");

        // At rho = 1.0: exact boundary match on both sides.
        let west_edge = rim.sample_harmonic(-90.0, 1.0);
        assert!(
            west_edge[0] > 0.99 && west_edge[2] < 0.01,
            "west={west_edge:?}"
        );
        let east_edge = rim.sample_harmonic(90.0, 1.0);
        assert!(
            east_edge[2] > 0.99 && east_edge[0] < 0.01,
            "east={east_edge:?}"
        );

        // At rho = 0.0 (exact pole): exact global mean (0.5, 0, 0.5) regardless of lon.
        let pole_a = rim.sample_harmonic(-90.0, 0.0);
        let pole_b = rim.sample_harmonic(90.0, 0.0);
        assert!((pole_a[0] - 0.5).abs() < 1e-4 && (pole_a[2] - 0.5).abs() < 1e-4);
        assert!((pole_b[0] - 0.5).abs() < 1e-4 && (pole_b[2] - 0.5).abs() < 1e-4);

        let tile = render_harmonic_polar_cap_with_mipmaps(
            QuadTileId {
                scheme: PlanetaryScheme::PolarCap { north: false },
                level: 1,
                x: 0,
                y: 0,
            },
            32,
            2,
            &rim,
        )
        .expect("render polar cap");
        assert_eq!(tile.width(), 32);
        assert!(tile.mip_level_count() > 1);
    }
}
