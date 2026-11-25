//! Virtual offline global basemap provider backed by a bundled public-domain
//! equirectangular image (e.g. NASA Blue Marble Next Generation).
//!
//! Stores the opaque source image as compact `Rgb8` with a linear-light mip
//! pyramid and serves `256 x 256` `Rgba8UnormSrgb` tiles for:
//! - `PlanetaryScheme::Geodetic` (`z = 0..=3`, `2 x 1` root up to `16 x 8` tiles)
//! - `PlanetaryScheme::PolarCap { north }` (`z = 0`, reprojected via the
//!   elliptical square-to-disk Polar Stereographic mapping matching the
//!   `±85.05112878 deg` cap geometry)
//! - `PlanetaryScheme::WebMercator` (`z = 0..=3`, mip-filtered per row to avoid
//!   coarse-level aliasing)

use std::fmt;

use veldmap_core::geo::web_mercator_lat_deg;
use veldmap_core::tile::{
    DecodedTile, PixelFormat, PlanetaryScheme, QuadTileId, geodetic_lat_bounds, polar_cap_point,
    polar_cap_uv_bounds, tile_lon_bounds, web_mercator_y_bounds,
};

use crate::RasterError;
use crate::color::{
    LINEAR_TO_SRGB_LUT_LEN, linear_to_srgb_lut, mix_2x2_srgb_rgb, srgb_to_linear_lut,
};

use crate::mipmap::generate_linear_mipmaps;

use super::map_tile::decode_rgb8_limited;

/// Downsample an opaque packed `Rgb8` sRGB image 2x in linear light, wrapping
/// horizontally for equirectangular continuity.
fn downsample_rgb8_srgb_wrap_x_2x(src: &[u8], src_w: u32, src_h: u32) -> (u32, u32, Vec<u8>) {
    let to_lin = srgb_to_linear_lut();
    let to_srgb = linear_to_srgb_lut();
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = (sw / 2).max(1);
    let dh = (sh / 2).max(1);
    let mut dst = vec![0u8; dw * dh * 3];

    for y in 0..dh {
        let sy0 = (y * 2).min(sh - 1);
        let sy1 = (sy0 + 1).min(sh - 1);
        for x in 0..dw {
            let sx0 = (x * 2).min(sw - 1);
            let sx1 = (sx0 + 1) % sw;
            let offsets = [
                (sy0 * sw + sx0) * 3,
                (sy0 * sw + sx1) * 3,
                (sy1 * sw + sx0) * 3,
                (sy1 * sw + sx1) * 3,
            ];
            let out = (y * dw + x) * 3;
            mix_2x2_srgb_rgb(
                src,
                offsets,
                [0.25; 4],
                to_lin,
                to_srgb,
                &mut dst[out..out + 3],
            );
        }
    }

    (dw as u32, dh as u32, dst)
}

/// Maximum single virtual tile dimension accepted by [`BundledBasemap::render_tile_sized`].
const MAX_BASEMAP_TILE_DIM: u32 = 2048;

/// Maximum dimension accepted for a bundled global equirectangular basemap.
const MAX_EQUIR_DIM: u32 = 8192;

/// Maximum peak allocation budget for decoding a bundled global equirectangular
/// basemap image (`128 MiB`).
const MAX_EQUIR_ALLOC_BYTES: u64 = 128 * 1024 * 1024;

/// Maximum number of prefiltered half-resolution mip levels built above base.
const MAX_MIP_STEPS: usize = 3;

/// Configuration for [`BundledBasemap`] tiling limits and default tile size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BundledBasemapConfig {
    pub tile_px: u32,
    pub max_level: u8,
    pub polar_max_level: u8,
}

impl BundledBasemapConfig {
    /// Default bundled basemap tiling configuration (`256 px`, `max_level = 3`,
    /// `polar_max_level = 0`).
    pub const DEFAULT: Self = Self {
        tile_px: 256,
        max_level: 3,
        polar_max_level: 0,
    };
}

impl Default for BundledBasemapConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

struct RgbLevel {
    width: u32,
    height: u32,
    rgb: Box<[u8]>,
}

impl fmt::Debug for RgbLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RgbLevel")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("byte_size", &self.rgb.len())
            .finish()
    }
}

impl RgbLevel {
    fn downsample_2x(&self) -> Self {
        let (width, height, rgb) =
            downsample_rgb8_srgb_wrap_x_2x(&self.rgb, self.width, self.height);
        Self {
            width,
            height,
            rgb: rgb.into_boxed_slice(),
        }
    }

    #[inline]
    fn sample_bilinear_srgb_with_luts(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        to_lin: &[f32; 256],
        to_srgb: &[u8; LINEAR_TO_SRGB_LUT_LEN],
    ) -> [u8; 4] {
        let w = self.width as usize;
        let h = self.height as usize;

        let u = ((lon_deg + 180.0) / 360.0).rem_euclid(1.0);
        let v = ((90.0 - lat_deg) / 180.0).clamp(0.0, 1.0);

        let fx = u * (w as f64) - 0.5;
        let fy = v * (h as f64) - 0.5;

        let x0_i = fx.floor() as isize;
        let y0_i = fy.floor() as isize;
        let tx = (fx - (x0_i as f64)) as f32;
        let ty = (fy - (y0_i as f64)) as f32;

        let x0 = x0_i.rem_euclid(w as isize) as usize;
        let x1 = (x0 + 1) % w;
        let y0 = y0_i.clamp(0, (h - 1) as isize) as usize;
        let y1 = (y0_i + 1).clamp(0, (h - 1) as isize) as usize;

        let o00 = (y0 * w + x0) * 3;
        let o10 = (y0 * w + x1) * 3;
        let o01 = (y1 * w + x0) * 3;
        let o11 = (y1 * w + x1) * 3;

        let w00 = (1.0 - tx) * (1.0 - ty);
        let w10 = tx * (1.0 - ty);
        let w01 = (1.0 - tx) * ty;
        let w11 = tx * ty;

        let mut out = [255u8; 4];
        mix_2x2_srgb_rgb(
            &self.rgb,
            [o00, o10, o01, o11],
            [w00, w10, w01, w11],
            to_lin,
            to_srgb,
            &mut out,
        );
        out
    }
}

/// Decoded equirectangular source image with a compact `Rgb8` linear-light mip
/// pyramid backing virtual offline basemap tiles.
pub struct BundledBasemap {
    levels: Vec<RgbLevel>,
    config: BundledBasemapConfig,
}

impl fmt::Debug for BundledBasemap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BundledBasemap")
            .field("dimensions", &self.dimensions())
            .field("mip_levels", &self.levels.len())
            .field("config", &self.config)
            .finish()
    }
}

impl BundledBasemap {
    /// Decode a bundled global equirectangular PNG or JPEG (`-180..+180 deg`
    /// west-to-east, `+90..-90 deg` north-to-south) with explicit tiling
    /// limits (`tile_px`, `max_level`, `polar_max_level`).
    pub fn from_encoded_bytes_with_config(
        bytes: &[u8],
        config: BundledBasemapConfig,
    ) -> Result<Self, RasterError> {
        let (width, height, rgb) =
            decode_rgb8_limited(bytes, MAX_EQUIR_DIM, MAX_EQUIR_ALLOC_BYTES)?;
        Ok(Self::from_rgb_box(width, height, rgb, config))
    }

    fn from_rgb_box(width: u32, height: u32, rgb: Box<[u8]>, config: BundledBasemapConfig) -> Self {
        let mut levels = Vec::with_capacity(MAX_MIP_STEPS + 1);
        levels.push(RgbLevel { width, height, rgb });
        for _ in 0..MAX_MIP_STEPS {
            let Some(prev) = levels.last() else {
                break;
            };
            // Stop before halving odd dimensions so a mip level never silently
            // truncates the last column/row and shifts geographic alignment.
            if prev.width <= 64
                || prev.height <= 32
                || !prev.width.is_multiple_of(2)
                || !prev.height.is_multiple_of(2)
            {
                break;
            }
            levels.push(prev.downsample_2x());
        }
        Self { levels, config }
    }

    /// Base equirectangular source dimensions `(width, height)`.
    pub fn dimensions(&self) -> (u32, u32) {
        let base = &self.levels[0];
        (base.width, base.height)
    }

    fn select_mip(&self, src_px_per_dst_px: f64) -> &RgbLevel {
        let max_idx = self.levels.len().saturating_sub(1);
        let idx = if src_px_per_dst_px <= 1.0 {
            0
        } else {
            (src_px_per_dst_px.log2().floor() as usize).min(max_idx)
        };
        &self.levels[idx]
    }

    /// Render one virtual tile at `self.config.tile_px`, rejecting any
    /// `level > max_level` or out-of-bounds `(x, y)` cleanly.
    pub fn render_tile(&self, id: QuadTileId) -> Result<DecodedTile, RasterError> {
        self.render_tile_sized(id, self.config.tile_px)
    }

    /// Render one virtual tile at `self.config.tile_px` and populate its CPU
    /// linear-light mipmap chain.
    pub fn render_tile_with_mipmaps(&self, id: QuadTileId) -> Result<DecodedTile, RasterError> {
        let mut tile = self.render_tile(id)?;
        generate_linear_mipmaps(&mut tile);
        Ok(tile)
    }

    /// Render one `tile_px x tile_px` virtual tile for `id`.
    pub fn render_tile_sized(
        &self,
        id: QuadTileId,
        tile_px: u32,
    ) -> Result<DecodedTile, RasterError> {
        if tile_px == 0 || tile_px > MAX_BASEMAP_TILE_DIM {
            return Err(RasterError::InvalidDimensions {
                width: tile_px,
                height: tile_px,
            });
        }

        let out_of_range = || RasterError::TileOutOfRange {
            scheme: id.scheme.into_tile_scheme(),
            level: id.level,
            x: id.x,
            y: id.y,
        };

        let max_level = match id.scheme {
            PlanetaryScheme::PolarCap { .. } => self.config.polar_max_level,
            PlanetaryScheme::Geodetic | PlanetaryScheme::WebMercator => self.config.max_level,
        };
        if id.level > max_level {
            return Err(out_of_range());
        }
        let (cols, rows) = id.checked_grid_dims().ok_or_else(out_of_range)?;

        let tile_usize = tile_px as usize;
        let total_bytes = tile_usize
            .checked_mul(tile_usize)
            .and_then(|px| px.checked_mul(4))
            .ok_or(RasterError::InvalidDimensions {
                width: tile_px,
                height: tile_px,
            })?;
        let mut data = vec![0u8; total_bytes];

        let to_lin = srgb_to_linear_lut();
        let to_srgb = linear_to_srgb_lut();
        let ctx = TileRenderContext {
            id,
            tile_px,
            cols,
            rows,
            to_lin,
            to_srgb,
        };

        match id.scheme {
            PlanetaryScheme::Geodetic => self.render_geodetic_tile(&ctx, &mut data),
            PlanetaryScheme::WebMercator => self.render_web_mercator_tile(&ctx, &mut data),
            PlanetaryScheme::PolarCap { north } => {
                self.render_polar_cap_tile(&ctx, north, &mut data);
            }
        }

        DecodedTile::try_new(tile_px, tile_px, PixelFormat::Rgba8UnormSrgb, data).map_err(|_| {
            RasterError::InvalidDimensions {
                width: tile_px,
                height: tile_px,
            }
        })
    }

    fn render_pixel_grid<'a, RowState>(
        ctx: &TileRenderContext<'_>,
        data: &mut [u8],
        mut row_setup: impl FnMut(f64) -> (&'a RgbLevel, RowState),
        mut sample_lat_lon: impl FnMut(&RowState, f64) -> (f64, f64),
    ) {
        let tile_usize = ctx.tile_px as usize;
        let tile_f = f64::from(ctx.tile_px);

        for py in 0..tile_usize {
            let t = (py as f64 + 0.5) / tile_f;
            let (mip, row_state) = row_setup(t);
            let row_off = py * tile_usize * 4;
            for px in 0..tile_usize {
                let s = (px as f64 + 0.5) / tile_f;
                let (lat, lon) = sample_lat_lon(&row_state, s);
                let off = row_off + px * 4;
                data[off..off + 4].copy_from_slice(&mip.sample_bilinear_srgb_with_luts(
                    lat,
                    lon,
                    ctx.to_lin,
                    ctx.to_srgb,
                ));
            }
        }
    }

    fn render_geodetic_tile(&self, ctx: &TileRenderContext<'_>, data: &mut [u8]) {
        let (base_w, base_h) = self.dimensions();
        let tile_f = f64::from(ctx.tile_px);
        let cols_f = f64::from(ctx.cols);
        let rows_f = f64::from(ctx.rows);
        let scale_x = f64::from(base_w) / (cols_f * tile_f);
        let scale_y = f64::from(base_h) / (rows_f * tile_f);
        let mip = self.select_mip(scale_x.max(scale_y));
        let (lon_west, lon_east) = tile_lon_bounds(ctx.id.x, ctx.cols);
        let (lat_north, lat_south) = geodetic_lat_bounds(ctx.id.y, ctx.rows);

        Self::render_pixel_grid(
            ctx,
            data,
            |t| (mip, lat_north + (lat_south - lat_north) * t),
            |&lat, s| (lat, lon_west + (lon_east - lon_west) * s),
        );
    }

    fn render_web_mercator_tile(&self, ctx: &TileRenderContext<'_>, data: &mut [u8]) {
        let (base_w, base_h) = self.dimensions();
        let tile_f = f64::from(ctx.tile_px);
        let cols_f = f64::from(ctx.cols);
        let (lon_west, lon_east) = tile_lon_bounds(ctx.id.x, ctx.cols);
        let (merc_n, merc_s) = web_mercator_y_bounds(ctx.id.y, ctx.rows);
        let d_merc_per_px = (merc_n - merc_s) / tile_f;
        let scale_x = f64::from(base_w) / (cols_f * tile_f);

        Self::render_pixel_grid(
            ctx,
            data,
            |t| {
                let merc = merc_n + (merc_s - merc_n) * t;
                let lat = web_mercator_lat_deg(merc);
                // Local d(lat_deg)/d(merc) = (180 / pi) * cos(lat).
                let dlat_per_px =
                    (180.0 / std::f64::consts::PI) * lat.to_radians().cos() * d_merc_per_px;
                let scale_y = (f64::from(base_h) / 180.0) * dlat_per_px;
                (self.select_mip(scale_x.max(scale_y)), lat)
            },
            |&lat, s| (lat, lon_west + (lon_east - lon_west) * s),
        );
    }

    fn render_polar_cap_tile(&self, ctx: &TileRenderContext<'_>, north: bool, data: &mut [u8]) {
        let (u0, u1, v0, v1) = polar_cap_uv_bounds(ctx.id.x, ctx.id.y, ctx.cols);
        let mip = &self.levels[0];

        Self::render_pixel_grid(
            ctx,
            data,
            |t| (mip, v0 + (v1 - v0) * t),
            |&v, s| {
                let u = u0 + (u1 - u0) * s;
                let (lat, lon, _, _) = polar_cap_point(north, u, v);
                (lat, lon)
            },
        );
    }
}

struct TileRenderContext<'a> {
    id: QuadTileId,
    tile_px: u32,
    cols: u32,
    rows: u32,
    to_lin: &'a [f32; 256],
    to_srgb: &'a [u8; LINEAR_TO_SRGB_LUT_LEN],
}

#[cfg(test)]
#[path = "bundled_tests.rs"]
mod tests;
