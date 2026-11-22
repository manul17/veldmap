//! Virtual offline global basemap provider backed by a bundled public-domain
//! equirectangular image (e.g. NASA Blue Marble Next Generation).
//!
//! Serves `256 x 256` `Rgba8UnormSrgb` tiles for:
//! - `TileScheme::Geodetic` (`z = 0..=3`, `2 x 1` root up to `16 x 8` tiles)
//! - `TileScheme::PolarCap { north }` (`z = 0..=1`, reprojected smoothly via
//!   the exact elliptical square-to-disk Polar Stereographic mapping so the
//!   `±85.05112878 deg` boundary matches the main globe pixel-for-pixel)
//! - `TileScheme::WebMercator` (`z = 0..=3`, reprojected cleanly if requested)

use std::sync::Arc;

use veldmap_core::tile::{PixelFormat, QuadTileId, TileScheme, polar_cap_point};

use crate::{DecodedTile, MAX_EQUIR_ALLOC_BYTES, RasterError, decode_rgba8_limited};

/// Default tile size produced by [`BundledBasemap`].
pub const BUNDLED_TILE_PX: u32 = 256;

/// Maximum `Geodetic` / `WebMercator` zoom level served by [`BundledBasemap`].
pub const BUNDLED_MAX_LEVEL: u8 = 3;

/// Maximum `PolarCap` zoom level served by [`BundledBasemap`].
pub const BUNDLED_POLAR_MAX_LEVEL: u8 = 1;

/// Decoded equirectangular source image backing virtual offline basemap tiles.
#[derive(Clone, Debug)]
pub struct BundledBasemap {
    width: u32,
    height: u32,
    rgba: Arc<[u8]>,
}

impl BundledBasemap {
    /// Decode a bundled global equirectangular PNG or JPEG (`-180..+180 deg`
    /// west-to-east, `+90..-90 deg` north-to-south).
    pub fn from_encoded_bytes(bytes: &[u8]) -> Result<Self, RasterError> {
        let decoded = decode_rgba8_limited(bytes, 8192, MAX_EQUIR_ALLOC_BYTES)?;
        Ok(Self {
            width: decoded.width,
            height: decoded.height,
            rgba: decoded.data.into(),
        })
    }

    /// Construct directly from raw `Rgba8UnormSrgb` pixels (useful in synthetic
    /// tests).
    pub fn from_rgba(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self, RasterError> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|px| px.checked_mul(4))
            .ok_or(RasterError::InvalidDimensions { width, height })?;
        if width == 0 || height == 0 || rgba.len() != expected {
            return Err(RasterError::InvalidDimensions { width, height });
        }
        Ok(Self {
            width,
            height,
            rgba: rgba.into(),
        })
    }

    /// Source equirectangular image dimensions `(width, height)`.
    pub const fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Render one `BUNDLED_TILE_PX x BUNDLED_TILE_PX` virtual tile for `id`,
    /// rejecting any `level > max_level` or out-of-bounds `(x, y)` cleanly.
    pub fn render_tile(&self, id: QuadTileId) -> Result<DecodedTile, RasterError> {
        self.render_tile_sized(id, BUNDLED_TILE_PX)
    }

    /// Render one `tile_px x tile_px` virtual tile for `id`.
    pub fn render_tile_sized(
        &self,
        id: QuadTileId,
        tile_px: u32,
    ) -> Result<DecodedTile, RasterError> {
        if tile_px == 0 {
            return Err(RasterError::InvalidDimensions {
                width: tile_px,
                height: tile_px,
            });
        }
        let max_level = match id.scheme {
            TileScheme::PolarCap { .. } => BUNDLED_POLAR_MAX_LEVEL,
            TileScheme::Geodetic | TileScheme::WebMercator => BUNDLED_MAX_LEVEL,
            TileScheme::RasterPyramid { .. } => {
                return Err(RasterError::TileOutOfRange {
                    level: id.level,
                    x: id.x,
                    y: id.y,
                });
            }
        };
        if id.level > max_level {
            return Err(RasterError::TileOutOfRange {
                level: id.level,
                x: id.x,
                y: id.y,
            });
        }
        let Some((cols, rows)) = id.scheme.grid_dims(id.level) else {
            return Err(RasterError::TileOutOfRange {
                level: id.level,
                x: id.x,
                y: id.y,
            });
        };
        if id.x >= cols || id.y >= rows {
            return Err(RasterError::TileOutOfRange {
                level: id.level,
                x: id.x,
                y: id.y,
            });
        }

        let mut data = vec![0u8; (tile_px as usize) * (tile_px as usize) * 4];
        let tile_f = f64::from(tile_px);
        let cols_f = f64::from(cols);
        let rows_f = f64::from(rows);

        match id.scheme {
            TileScheme::Geodetic => {
                // Use 2x2 supersampling on coarse levels (`z <= 2`) so downsampling
                // from 3600x1800 stays crisp without aliasing.
                let ss = if id.level <= 1 {
                    4u32
                } else if id.level == 2 {
                    2
                } else {
                    1
                };
                let ss_f = f64::from(ss);
                let inv_ss2 = 1.0 / (ss_f * ss_f);

                for py in 0..tile_px {
                    for px in 0..tile_px {
                        let mut acc = [0.0_f64; 4];
                        for sy in 0..ss {
                            let t = (f64::from(py) + (f64::from(sy) + 0.5) / ss_f) / tile_f;
                            let lat = 90.0 - 180.0 * (f64::from(id.y) + t) / rows_f;
                            for sx in 0..ss {
                                let s = (f64::from(px) + (f64::from(sx) + 0.5) / ss_f) / tile_f;
                                let lon = -180.0 + 360.0 * (f64::from(id.x) + s) / cols_f;
                                let c = self.sample_bilinear(lat, lon);
                                for ch in 0..4 {
                                    acc[ch] += c[ch];
                                }
                            }
                        }
                        let off = ((py * tile_px + px) * 4) as usize;
                        for ch in 0..4 {
                            data[off + ch] = (acc[ch] * inv_ss2).round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
            TileScheme::WebMercator => {
                let merc_n = std::f64::consts::PI * (1.0 - 2.0 * f64::from(id.y) / rows_f);
                let merc_s = std::f64::consts::PI * (1.0 - 2.0 * f64::from(id.y + 1) / rows_f);
                for py in 0..tile_px {
                    let t = (f64::from(py) + 0.5) / tile_f;
                    let merc = merc_n + (merc_s - merc_n) * t;
                    let lat = merc.sinh().atan().to_degrees();
                    for px in 0..tile_px {
                        let s = (f64::from(px) + 0.5) / tile_f;
                        let lon = -180.0 + 360.0 * (f64::from(id.x) + s) / cols_f;
                        let c = self.sample_bilinear(lat, lon);
                        let off = ((py * tile_px + px) * 4) as usize;
                        for ch in 0..4 {
                            data[off + ch] = c[ch].round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
            TileScheme::PolarCap { north } => {
                for py in 0..tile_px {
                    let t = (f64::from(py) + 0.5) / tile_f;
                    let v = -1.0 + 2.0 * (f64::from(id.y) + t) / rows_f;
                    for px in 0..tile_px {
                        let s = (f64::from(px) + 0.5) / tile_f;
                        let u = -1.0 + 2.0 * (f64::from(id.x) + s) / cols_f;
                        let (lat, lon, _, _) = polar_cap_point(north, u, v);
                        let c = self.sample_bilinear(lat, lon);
                        let off = ((py * tile_px + px) * 4) as usize;
                        for ch in 0..4 {
                            data[off + ch] = c[ch].round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
            TileScheme::RasterPyramid { .. } => unreachable!(),
        }

        Ok(DecodedTile {
            width: tile_px,
            height: tile_px,
            format: PixelFormat::Rgba8UnormSrgb,
            data,
        })
    }

    /// Bilinear sample at geodetic `(lat_deg, lon_deg)` with horizontal
    /// longitude wraparound across `±180 deg` and polar latitude clamping.
    fn sample_bilinear(&self, lat_deg: f64, lon_deg: f64) -> [f64; 4] {
        let w = self.width as usize;
        let h = self.height as usize;

        let u = ((lon_deg + 180.0) / 360.0).rem_euclid(1.0);
        let v = ((90.0 - lat_deg) / 180.0).clamp(0.0, 1.0);

        let fx = u * (w as f64) - 0.5;
        let fy = v * (h as f64) - 0.5;

        let x0_i = fx.floor() as isize;
        let y0_i = fy.floor() as isize;
        let tx = fx - (x0_i as f64);
        let ty = fy - (y0_i as f64);

        let x0 = x0_i.rem_euclid(w as isize) as usize;
        let x1 = (x0 + 1) % w;
        let y0 = y0_i.clamp(0, (h - 1) as isize) as usize;
        let y1 = (y0_i + 1).clamp(0, (h - 1) as isize) as usize;

        let p00 = self.pixel(x0, y0);
        let p10 = self.pixel(x1, y0);
        let p01 = self.pixel(x0, y1);
        let p11 = self.pixel(x1, y1);

        let w00 = (1.0 - tx) * (1.0 - ty);
        let w10 = tx * (1.0 - ty);
        let w01 = (1.0 - tx) * ty;
        let w11 = tx * ty;

        let mut out = [0.0_f64; 4];
        for ch in 0..4 {
            out[ch] = f64::from(p00[ch]) * w00
                + f64::from(p10[ch]) * w10
                + f64::from(p01[ch]) * w01
                + f64::from(p11[ch]) * w11;
        }
        out
    }

    fn pixel(&self, x: usize, y: usize) -> [u8; 4] {
        let off = (y * (self.width as usize) + x) * 4;
        [
            self.rgba[off],
            self.rgba[off + 1],
            self.rgba[off + 2],
            self.rgba[off + 3],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_geodetic_polar_and_web_mercator_tiles_and_rejects_above_max_level() {
        // Synthetic 64x32 equirectangular image: north hemisphere = green, south = blue,
        // south pole cap = white.
        let mut rgba = vec![0u8; 64 * 32 * 4];
        for y in 0..32u32 {
            let lat = 90.0 - 180.0 * (f64::from(y) + 0.5) / 32.0;
            let color = if lat < -80.0 {
                [240, 240, 240, 255]
            } else if lat >= 0.0 {
                [40, 180, 60, 255]
            } else {
                [20, 60, 200, 255]
            };
            for x in 0..64u32 {
                let off = ((y * 64 + x) * 4) as usize;
                rgba[off..off + 4].copy_from_slice(&color);
            }
        }
        let basemap = BundledBasemap::from_rgba(64, 32, rgba).expect("build synthetic basemap");

        // Geodetic z = 1 north-east tile should be green.
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
            .expect("render geodetic tile");
        assert_eq!(ne.byte_size(), 16 * 16 * 4);
        assert!(ne.data[1] > 150 && ne.data[2] < 80);

        // South polar cap tile should be bright white.
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
            .expect("render south polar cap");
        let mid = (8 * 16 + 8) * 4;
        assert!(
            south_cap.data[mid] > 220
                && south_cap.data[mid + 1] > 220
                && south_cap.data[mid + 2] > 220
        );

        // Requests above max_level must fail cleanly.
        assert!(
            basemap
                .render_tile(QuadTileId {
                    scheme: TileScheme::Geodetic,
                    level: BUNDLED_MAX_LEVEL + 1,
                    x: 0,
                    y: 0,
                })
                .is_err()
        );
        assert!(
            basemap
                .render_tile(QuadTileId {
                    scheme: TileScheme::PolarCap { north: true },
                    level: BUNDLED_POLAR_MAX_LEVEL + 1,
                    x: 0,
                    y: 0,
                })
                .is_err()
        );
    }

    #[test]
    fn bundled_blue_marble_asset_decodes_and_preserves_geographic_landmarks() {
        let bytes = include_bytes!("../../../assets/basemap/blue_marble_ng.jpg");
        let basemap = BundledBasemap::from_encoded_bytes(bytes).expect("decode blue marble");
        assert_eq!(basemap.dimensions(), (3600, 1800));

        // South pole (Antarctica) must be bright white/ice.
        let sp = basemap.sample_bilinear(-89.0, 0.0);
        assert!(
            sp[0] > 200.0 && sp[1] > 200.0 && sp[2] > 200.0,
            "south pole={sp:?}"
        );

        // Sahara desert (23 N, 13 E) must be bright tan (R > G > B).
        let sahara = basemap.sample_bilinear(23.0, 13.0);
        assert!(
            sahara[0] > 150.0 && sahara[0] > sahara[2] + 40.0,
            "sahara={sahara:?}"
        );

        // Central Pacific (0 N, -150 E) must be deep ocean blue (B > R).
        let pacific = basemap.sample_bilinear(0.0, -150.0);
        assert!(
            pacific[2] > pacific[0] + 20.0 && pacific[0] < 40.0,
            "pacific={pacific:?}"
        );
    }
}
