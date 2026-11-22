//! Virtual offline global basemap provider backed by a bundled public-domain
//! equirectangular image (e.g. NASA Blue Marble Next Generation).
//!
//! Stores the opaque source image as compact `Rgb8` with a linear-light mip
//! pyramid and serves `256 x 256` `Rgba8UnormSrgb` tiles for:
//! - `TileScheme::Geodetic` (`z = 0..=3`, `2 x 1` root up to `16 x 8` tiles)
//! - `TileScheme::PolarCap { north }` (`z = 0`, reprojected via the elliptical
//!   square-to-disk Polar Stereographic mapping matching the `±85.05112878 deg`
//!   cap geometry)
//! - `TileScheme::WebMercator` (`z = 0..=3`, mip-filtered per row to avoid
//!   coarse-level aliasing)

use std::fmt;

use veldmap_core::tile::{DecodedTile, PixelFormat, QuadTileId, TileScheme, polar_cap_point};

use crate::mipmap::{
    LINEAR_TO_SRGB_LUT_LEN, generate_linear_mipmaps, linear_to_srgb_lut, linear_to_srgb_u8,
    srgb_to_linear_lut,
};
use crate::{
    MAX_EQUIR_ALLOC_BYTES, MAX_EQUIR_DIM, MAX_MAP_TILE_DIM, RasterError, decode_rgb8_limited,
};

/// Default tile size produced by [`BundledBasemap`].
pub const BUNDLED_TILE_PX: u32 = 256;

/// Maximum `Geodetic` / `WebMercator` zoom level served by [`BundledBasemap`].
pub const BUNDLED_MAX_LEVEL: u8 = 3;

/// Maximum `PolarCap` zoom level served by [`BundledBasemap`].
///
/// At `3600 x 1800` (`10 px/deg`), the `±85.0511°..±90°` cap spans only ~99
/// source pixels in diameter, so `z = 0` (`256 x 256`) already exceeds native
/// source resolution.
pub const BUNDLED_POLAR_MAX_LEVEL: u8 = 0;

/// Maximum number of prefiltered half-resolution mip levels built above base.
const MAX_MIP_STEPS: usize = 3;

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
        let to_lin = srgb_to_linear_lut();
        let to_srgb = linear_to_srgb_lut();
        let src_w = self.width as usize;
        let src_h = self.height as usize;
        let dst_w = (src_w / 2).max(1);
        let dst_h = (src_h / 2).max(1);
        let mut out = vec![0u8; dst_w * dst_h * 3];

        for y in 0..dst_h {
            let sy0 = (y * 2).min(src_h - 1);
            let sy1 = (sy0 + 1).min(src_h - 1);
            for x in 0..dst_w {
                let sx0 = (x * 2).min(src_w - 1);
                let sx1 = (sx0 + 1) % src_w;
                let o00 = (sy0 * src_w + sx0) * 3;
                let o10 = (sy0 * src_w + sx1) * 3;
                let o01 = (sy1 * src_w + sx0) * 3;
                let o11 = (sy1 * src_w + sx1) * 3;
                let dst = (y * dst_w + x) * 3;
                for ch in 0..3 {
                    let avg = 0.25
                        * (to_lin[usize::from(self.rgb[o00 + ch])]
                            + to_lin[usize::from(self.rgb[o10 + ch])]
                            + to_lin[usize::from(self.rgb[o01 + ch])]
                            + to_lin[usize::from(self.rgb[o11 + ch])]);
                    out[dst + ch] = linear_to_srgb_u8(avg, to_srgb);
                }
            }
        }

        Self {
            width: dst_w as u32,
            height: dst_h as u32,
            rgb: out.into_boxed_slice(),
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
        for ch in 0..3 {
            let lin = to_lin[usize::from(self.rgb[o00 + ch])] * w00
                + to_lin[usize::from(self.rgb[o10 + ch])] * w10
                + to_lin[usize::from(self.rgb[o01 + ch])] * w01
                + to_lin[usize::from(self.rgb[o11 + ch])] * w11;
            out[ch] = linear_to_srgb_u8(lin, to_srgb);
        }
        out
    }
}

/// Decoded equirectangular source image with a compact `Rgb8` linear-light mip
/// pyramid backing virtual offline basemap tiles.
pub struct BundledBasemap {
    levels: Vec<RgbLevel>,
}

impl fmt::Debug for BundledBasemap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BundledBasemap")
            .field("dimensions", &self.dimensions())
            .field("mip_levels", &self.levels.len())
            .finish()
    }
}

impl BundledBasemap {
    /// Decode a bundled global equirectangular PNG or JPEG (`-180..+180 deg`
    /// west-to-east, `+90..-90 deg` north-to-south) directly into compact
    /// `Rgb8` and build its linear-light mip pyramid.
    pub fn from_encoded_bytes(bytes: &[u8]) -> Result<Self, RasterError> {
        let (width, height, rgb) =
            decode_rgb8_limited(bytes, MAX_EQUIR_DIM, MAX_EQUIR_ALLOC_BYTES)?;
        Ok(Self::from_rgb_box(width, height, rgb))
    }

    /// Construct directly from raw `Rgba8UnormSrgb` pixels (extracting RGB;
    /// used in synthetic unit tests).
    #[cfg(test)]
    fn from_rgba(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self, RasterError> {
        let px_count = (width as usize)
            .checked_mul(height as usize)
            .ok_or(RasterError::InvalidDimensions { width, height })?;
        let expected = px_count
            .checked_mul(4)
            .ok_or(RasterError::InvalidDimensions { width, height })?;
        if width == 0 || height == 0 || rgba.len() != expected {
            return Err(RasterError::InvalidDimensions { width, height });
        }
        let mut rgb = Vec::with_capacity(px_count * 3);
        for px in rgba.chunks_exact(4) {
            rgb.extend_from_slice(&px[0..3]);
        }
        Ok(Self::from_rgb_box(width, height, rgb.into_boxed_slice()))
    }

    fn from_rgb_box(width: u32, height: u32, rgb: Box<[u8]>) -> Self {
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
        Self { levels }
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

    /// Render one `BUNDLED_TILE_PX x BUNDLED_TILE_PX` virtual tile for `id`,
    /// rejecting any `level > max_level` or out-of-bounds `(x, y)` cleanly.
    pub fn render_tile(&self, id: QuadTileId) -> Result<DecodedTile, RasterError> {
        self.render_tile_sized(id, BUNDLED_TILE_PX)
    }

    /// Render one `BUNDLED_TILE_PX x BUNDLED_TILE_PX` virtual tile for `id`
    /// and populate its CPU linear-light mipmap chain.
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
        if tile_px == 0 || tile_px > MAX_MAP_TILE_DIM {
            return Err(RasterError::InvalidDimensions {
                width: tile_px,
                height: tile_px,
            });
        }

        let out_of_range = || RasterError::TileOutOfRange {
            scheme: id.scheme,
            level: id.level,
            x: id.x,
            y: id.y,
        };

        let max_level = match id.scheme {
            TileScheme::PolarCap { .. } => BUNDLED_POLAR_MAX_LEVEL,
            TileScheme::Geodetic | TileScheme::WebMercator => BUNDLED_MAX_LEVEL,
            TileScheme::RasterPyramid { .. } => return Err(out_of_range()),
        };
        if id.level > max_level {
            return Err(out_of_range());
        }
        let (cols, rows) = id.scheme.grid_dims(id.level).ok_or_else(out_of_range)?;
        if id.x >= cols || id.y >= rows {
            return Err(out_of_range());
        }

        let tile_usize = tile_px as usize;
        let total_bytes = tile_usize
            .checked_mul(tile_usize)
            .and_then(|px| px.checked_mul(4))
            .ok_or(RasterError::InvalidDimensions {
                width: tile_px,
                height: tile_px,
            })?;
        let mut data = vec![0u8; total_bytes];

        let (base_w, base_h) = self.dimensions();
        let base_w_f = f64::from(base_w);
        let base_h_f = f64::from(base_h);
        let tile_f = f64::from(tile_px);
        let cols_f = f64::from(cols);
        let rows_f = f64::from(rows);

        let to_lin = srgb_to_linear_lut();
        let to_srgb = linear_to_srgb_lut();

        match id.scheme {
            TileScheme::Geodetic => {
                let scale_x = base_w_f / (cols_f * tile_f);
                let scale_y = base_h_f / (rows_f * tile_f);
                let mip = self.select_mip(scale_x.max(scale_y));

                for py in 0..tile_usize {
                    let t = (py as f64 + 0.5) / tile_f;
                    let lat = 90.0 - 180.0 * (f64::from(id.y) + t) / rows_f;
                    let row_off = py * tile_usize * 4;
                    for px in 0..tile_usize {
                        let s = (px as f64 + 0.5) / tile_f;
                        let lon = -180.0 + 360.0 * (f64::from(id.x) + s) / cols_f;
                        let off = row_off + px * 4;
                        data[off..off + 4].copy_from_slice(
                            &mip.sample_bilinear_srgb_with_luts(lat, lon, to_lin, to_srgb),
                        );
                    }
                }
            }
            TileScheme::WebMercator => {
                let merc_n = std::f64::consts::PI * (1.0 - 2.0 * f64::from(id.y) / rows_f);
                let merc_s = std::f64::consts::PI * (1.0 - 2.0 * f64::from(id.y + 1) / rows_f);
                let d_merc_per_px = (merc_n - merc_s) / tile_f;
                let scale_x = base_w_f / (cols_f * tile_f);

                for py in 0..tile_usize {
                    let t = (py as f64 + 0.5) / tile_f;
                    let merc = merc_n + (merc_s - merc_n) * t;
                    let lat = merc.sinh().atan().to_degrees();
                    // Local d(lat_deg)/d(merc) = (180 / pi) * cos(lat).
                    let dlat_per_px =
                        (180.0 / std::f64::consts::PI) * lat.to_radians().cos() * d_merc_per_px;
                    let scale_y = (base_h_f / 180.0) * dlat_per_px;
                    let mip = self.select_mip(scale_x.max(scale_y));
                    let row_off = py * tile_usize * 4;

                    for px in 0..tile_usize {
                        let s = (px as f64 + 0.5) / tile_f;
                        let lon = -180.0 + 360.0 * (f64::from(id.x) + s) / cols_f;
                        let off = row_off + px * 4;
                        data[off..off + 4].copy_from_slice(
                            &mip.sample_bilinear_srgb_with_luts(lat, lon, to_lin, to_srgb),
                        );
                    }
                }
            }
            TileScheme::PolarCap { north } => {
                let mip = &self.levels[0];
                for py in 0..tile_usize {
                    let t = (py as f64 + 0.5) / tile_f;
                    let v = -1.0 + 2.0 * (f64::from(id.y) + t) / rows_f;
                    let row_off = py * tile_usize * 4;
                    for px in 0..tile_usize {
                        let s = (px as f64 + 0.5) / tile_f;
                        let u = -1.0 + 2.0 * (f64::from(id.x) + s) / cols_f;
                        let (lat, lon, _, _) = polar_cap_point(north, u, v);
                        let off = row_off + px * 4;
                        data[off..off + 4].copy_from_slice(
                            &mip.sample_bilinear_srgb_with_luts(lat, lon, to_lin, to_srgb),
                        );
                    }
                }
            }
            TileScheme::RasterPyramid { .. } => return Err(out_of_range()),
        }

        Ok(DecodedTile::new(
            tile_px,
            tile_px,
            PixelFormat::Rgba8UnormSrgb,
            data,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn synthetic_quadrant_basemap() -> BundledBasemap {
        // Synthetic 128x64 equirectangular image with distinct N/S & E/W quadrants
        // plus white South Pole and cyan North Pole so flips/rotations fail loudly:
        //   lat > +80: cyan   (40, 220, 230)
        //   lat < -80: white  (240, 240, 240)
        //   NW (lat >= 0, lon < 0):  red    (210, 50, 40)
        //   NE (lat >= 0, lon >= 0): green  (40, 190, 60)
        //   SW (lat < 0, lon < 0):   blue   (40, 70, 210)
        //   SE (lat < 0, lon >= 0):  yellow (220, 200, 40)
        let (w, h) = (128u32, 64u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            let lat = 90.0 - 180.0 * (f64::from(y) + 0.5) / f64::from(h);
            for x in 0..w {
                let lon = -180.0 + 360.0 * (f64::from(x) + 0.5) / f64::from(w);
                let color = if lat > 80.0 {
                    [40, 220, 230, 255]
                } else if lat < -80.0 {
                    [240, 240, 240, 255]
                } else {
                    match (lat >= 0.0, lon >= 0.0) {
                        (true, false) => [210, 50, 40, 255],
                        (true, true) => [40, 190, 60, 255],
                        (false, false) => [40, 70, 210, 255],
                        (false, true) => [220, 200, 40, 255],
                    }
                };
                let off = ((y * w + x) * 4) as usize;
                rgba[off..off + 4].copy_from_slice(&color);
            }
        }
        BundledBasemap::from_rgba(w, h, rgba).expect("build synthetic basemap")
    }

    #[test]
    fn renders_geodetic_web_mercator_and_polar_tiles_with_orientation_and_bounds_checks() {
        let basemap = synthetic_quadrant_basemap();

        // 1. Geodetic z = 1: NW (x=0,y=0) = red, NE (x=2,y=0) = green, SE (x=2,y=1) = yellow.
        let nw = basemap
            .render_tile_sized(
                QuadTileId {
                    scheme: TileScheme::Geodetic,
                    level: 1,
                    x: 0,
                    y: 0,
                },
                16,
            )
            .expect("geodetic NW");
        let mid = (8 * 16 + 8) * 4;
        assert!(
            nw.data[mid] > 180 && nw.data[mid + 1] < 80,
            "NW={:?}",
            &nw.data[mid..mid + 4]
        );

        let ne = basemap
            .render_tile_sized(
                QuadTileId {
                    scheme: TileScheme::Geodetic,
                    level: 1,
                    x: 2,
                    y: 0,
                },
                16,
            )
            .expect("geodetic NE");
        assert!(
            ne.data[mid + 1] > 160 && ne.data[mid] < 80,
            "NE={:?}",
            &ne.data[mid..mid + 4]
        );

        // 2. WebMercator z = 1: NW (0,0) = red, NE (1,0) = green, SW (0,1) = blue, SE (1,1) = yellow.
        let wm_nw = basemap
            .render_tile_sized(
                QuadTileId {
                    scheme: TileScheme::WebMercator,
                    level: 1,
                    x: 0,
                    y: 0,
                },
                16,
            )
            .expect("web mercator NW");
        assert!(wm_nw.data[mid] > 180 && wm_nw.data[mid + 1] < 80);

        let wm_se = basemap
            .render_tile_sized(
                QuadTileId {
                    scheme: TileScheme::WebMercator,
                    level: 1,
                    x: 1,
                    y: 1,
                },
                16,
            )
            .expect("web mercator SE");
        assert!(wm_se.data[mid] > 180 && wm_se.data[mid + 1] > 160 && wm_se.data[mid + 2] < 80);

        // 3. Both North (cyan) and South (white) PolarCap tiles render cleanly.
        let north_cap = basemap
            .render_tile_sized(
                QuadTileId {
                    scheme: TileScheme::PolarCap { north: true },
                    level: 0,
                    x: 0,
                    y: 0,
                },
                16,
            )
            .expect("north polar cap");
        assert!(
            north_cap.data[mid] < 80
                && north_cap.data[mid + 1] > 190
                && north_cap.data[mid + 2] > 200
        );

        let south_cap = basemap
            .render_tile_sized(
                QuadTileId {
                    scheme: TileScheme::PolarCap { north: false },
                    level: 0,
                    x: 0,
                    y: 0,
                },
                16,
            )
            .expect("south polar cap");
        assert!(
            south_cap.data[mid] > 220
                && south_cap.data[mid + 1] > 220
                && south_cap.data[mid + 2] > 220
        );

        // 4. Out-of-range levels, OOB x/y, RasterPyramid, and invalid tile_px fail cleanly.
        for bad in [
            QuadTileId {
                scheme: TileScheme::Geodetic,
                level: BUNDLED_MAX_LEVEL + 1,
                x: 0,
                y: 0,
            },
            QuadTileId {
                scheme: TileScheme::PolarCap { north: true },
                level: BUNDLED_POLAR_MAX_LEVEL + 1,
                x: 0,
                y: 0,
            },
            QuadTileId {
                scheme: TileScheme::WebMercator,
                level: 1,
                x: 2,
                y: 0,
            },
            QuadTileId {
                scheme: TileScheme::RasterPyramid { max_level: 2 },
                level: 0,
                x: 0,
                y: 0,
            },
        ] {
            assert!(matches!(
                basemap.render_tile(bad),
                Err(RasterError::TileOutOfRange { .. })
            ));
        }
        assert!(matches!(
            basemap.render_tile_sized(
                QuadTileId {
                    scheme: TileScheme::Geodetic,
                    level: 0,
                    x: 0,
                    y: 0,
                },
                0,
            ),
            Err(RasterError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            basemap.render_tile_sized(
                QuadTileId {
                    scheme: TileScheme::Geodetic,
                    level: 0,
                    x: 0,
                    y: 0,
                },
                MAX_MAP_TILE_DIM + 1,
            ),
            Err(RasterError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn bundled_blue_marble_asset_decodes_and_preserves_geographic_landmarks() {
        let asset_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/basemap/blue_marble_ng.jpg");
        let bytes = std::fs::read(&asset_path).expect("read bundled blue marble asset");
        let basemap = BundledBasemap::from_encoded_bytes(&bytes).expect("decode blue marble");
        assert_eq!(basemap.dimensions(), (3600, 1800));
        assert!(basemap.levels.len() >= 4, "expected base + 3 mip levels");

        let to_lin = srgb_to_linear_lut();
        let to_srgb = linear_to_srgb_lut();
        // Verify exact sRGB -> linear -> sRGB LUT round-trip across all 256 values.
        for v in 0..=255u8 {
            assert_eq!(linear_to_srgb_u8(to_lin[usize::from(v)], to_srgb), v);
        }

        // South pole (Antarctica) must be bright white/ice.
        let sp = basemap.levels[0].sample_bilinear_srgb_with_luts(-89.0, 0.0, to_lin, to_srgb);
        assert!(
            sp[0] > 200 && sp[1] > 200 && sp[2] > 200,
            "south pole={sp:?}"
        );

        // Sahara desert (23 N, 13 E) must be bright tan (R > G > B).
        let sahara = basemap.levels[0].sample_bilinear_srgb_with_luts(23.0, 13.0, to_lin, to_srgb);
        assert!(
            sahara[0] > 150 && sahara[0] > sahara[2].saturating_add(40),
            "sahara={sahara:?}"
        );

        // Central Pacific (0 N, -150 E) must be deep ocean blue (B > R).
        let pacific =
            basemap.levels[0].sample_bilinear_srgb_with_luts(0.0, -150.0, to_lin, to_srgb);
        assert!(
            pacific[2] > pacific[0].saturating_add(20) && pacific[0] < 40,
            "pacific={pacific:?}"
        );
    }
}
