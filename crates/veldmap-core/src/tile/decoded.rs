//! CPU-decoded tile pixel storage (`DecodedTile`) with construction-time
//! invariant enforcement and optional precomputed mipmap levels shared across
//! `veldmap-raster`, `veldmap-net`, and `veldmap-app`.

use std::fmt;

use thiserror::Error;

use super::key::PixelFormat;

/// Maximum number of mip levels (including base level 0) for a 2D texture of
/// dimensions `(width, height)`.
pub const fn max_mip_levels(width: u32, height: u32) -> u32 {
    if width == 0 || height == 0 {
        return 0;
    }
    let max_dim = if width > height { width } else { height };
    32 - max_dim.leading_zeros()
}

/// Dimensions `(mip_width, mip_height)` at `mip_level` (`0` = base image).
/// Returns `(0, 0)` when either base dimension is zero.
pub const fn mip_dimensions(width: u32, height: u32, mip_level: u32) -> (u32, u32) {
    if width == 0 || height == 0 {
        return (0, 0);
    }
    (
        mip_axis_dim(width, mip_level),
        mip_axis_dim(height, mip_level),
    )
}

const fn mip_axis_dim(dim: u32, mip_level: u32) -> u32 {
    if mip_level >= 32 {
        return 1;
    }
    let shifted = dim >> mip_level;
    if shifted == 0 { 1 } else { shifted }
}

/// Exact unpacked byte size of a 2D tile texture with `mip_level_count` mip
/// levels (clamped to [`max_mip_levels`]).
pub fn mip_chain_byte_size(
    width: u32,
    height: u32,
    format: PixelFormat,
    mip_level_count: u32,
) -> usize {
    let bpp = format.bytes_per_pixel();
    let max_levels = max_mip_levels(width, height);
    let levels = mip_level_count.min(max_levels);
    let mut total = 0usize;
    for level in 0..levels {
        let (mw, mh) = mip_dimensions(width, height, level);
        let level_bytes = (mw as usize)
            .saturating_mul(mh as usize)
            .saturating_mul(bpp);
        total = total.saturating_add(level_bytes);
    }
    total
}

/// Validation errors returned when constructing or attaching mip levels to a
/// [`DecodedTile`].
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum DecodedTileError {
    #[error(
        "tile dimensions and valid interior must be non-zero ({width}x{height}, valid {valid_w}x{valid_h})"
    )]
    ZeroDimension {
        width: u32,
        height: u32,
        valid_w: u32,
        valid_h: u32,
    },
    #[error(
        "valid interior {valid_w}x{valid_h} with gutter {gutter} exceeds tile {width}x{height}"
    )]
    InvalidValidRegion {
        width: u32,
        height: u32,
        valid_w: u32,
        valid_h: u32,
        gutter: u8,
    },
    #[error(
        "guttered or padded tiles ({width}x{height}, valid {valid_w}x{valid_h}, gutter {gutter}) cannot carry mip levels"
    )]
    MipsUnsupportedForPaddedOrGuttered {
        width: u32,
        height: u32,
        valid_w: u32,
        valid_h: u32,
        gutter: u8,
    },
    #[error("mip level count {count} exceeds maximum {max} for {width}x{height}")]
    TooManyMipLevels {
        width: u32,
        height: u32,
        count: u32,
        max: u32,
    },
    #[error("mip level {level} byte length mismatch: expected {expected} B, got {actual} B")]
    MipByteLengthMismatch {
        level: u32,
        expected: usize,
        actual: usize,
    },
    #[error("byte size overflow for mip level {level} ({width}x{height}, {format:?})")]
    ByteSizeOverflow {
        level: u32,
        width: u32,
        height: u32,
        format: PixelFormat,
    },
}

/// CPU-decoded tile pixels ready for GPU texture upload, with invariants
/// enforced at construction and optional CPU-generated mipmap levels (`1..N`).
#[derive(Clone)]
pub struct DecodedTile {
    width: u32,
    height: u32,
    /// Valid interior pixel width excluding border gutters
    /// (`valid_w + 2 * gutter <= width`).
    valid_w: u32,
    /// Valid interior pixel height excluding border gutters
    /// (`valid_h + 2 * gutter <= height`).
    valid_h: u32,
    /// Border gutter width in texels (`0` for standard tiles, `1` for DEM
    /// derivative tiles).
    gutter: u8,
    format: PixelFormat,
    /// Tightly packed mip level 0 pixels (`width * height * bytes_per_pixel`).
    data: Vec<u8>,
    /// Optional precomputed mip levels `1..N`, where `mips[i]` holds mip level
    /// `i + 1` with dimensions `mip_dimensions(width, height, i + 1)`.
    mips: Vec<Vec<u8>>,
}

impl fmt::Debug for DecodedTile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecodedTile")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("valid_w", &self.valid_w)
            .field("valid_h", &self.valid_h)
            .field("gutter", &self.gutter)
            .field("format", &self.format)
            .field("mip_levels", &self.mip_level_count())
            .field("byte_size", &self.byte_size())
            .finish()
    }
}

impl DecodedTile {
    fn validate_with_mips(&self, mips: &[Vec<u8>]) -> Result<(), DecodedTileError> {
        if self.width == 0 || self.height == 0 || self.valid_w == 0 || self.valid_h == 0 {
            return Err(DecodedTileError::ZeroDimension {
                width: self.width,
                height: self.height,
                valid_w: self.valid_w,
                valid_h: self.valid_h,
            });
        }
        let two_gutter = u32::from(self.gutter) * 2;
        if self
            .valid_w
            .checked_add(two_gutter)
            .is_none_or(|w| w > self.width)
            || self
                .valid_h
                .checked_add(two_gutter)
                .is_none_or(|h| h > self.height)
        {
            return Err(DecodedTileError::InvalidValidRegion {
                width: self.width,
                height: self.height,
                valid_w: self.valid_w,
                valid_h: self.valid_h,
                gutter: self.gutter,
            });
        }
        if (self.gutter != 0 || self.valid_w != self.width || self.valid_h != self.height)
            && !mips.is_empty()
        {
            return Err(DecodedTileError::MipsUnsupportedForPaddedOrGuttered {
                width: self.width,
                height: self.height,
                valid_w: self.valid_w,
                valid_h: self.valid_h,
                gutter: self.gutter,
            });
        }
        let max_mips = max_mip_levels(self.width, self.height);
        let extra = u32::try_from(mips.len()).unwrap_or(u32::MAX);
        let count = 1u32.saturating_add(extra);
        if count > max_mips {
            return Err(DecodedTileError::TooManyMipLevels {
                width: self.width,
                height: self.height,
                count,
                max: max_mips,
            });
        }
        let bpp = self.format.bytes_per_pixel();
        for level in 0..count {
            let (mw, mh) = mip_dimensions(self.width, self.height, level);
            let expected = (mw as usize)
                .checked_mul(mh as usize)
                .and_then(|px| px.checked_mul(bpp))
                .ok_or(DecodedTileError::ByteSizeOverflow {
                    level,
                    width: mw,
                    height: mh,
                    format: self.format,
                })?;
            let buf = if level == 0 {
                self.data.as_slice()
            } else {
                mips[(level - 1) as usize].as_slice()
            };
            if buf.len() != expected {
                return Err(DecodedTileError::MipByteLengthMismatch {
                    level,
                    expected,
                    actual: buf.len(),
                });
            }
        }
        Ok(())
    }

    /// Construct a validated single-level (`mip_level_count == 1`, `gutter = 0`,
    /// full valid interior) decoded tile.
    pub fn try_new(
        width: u32,
        height: u32,
        format: PixelFormat,
        data: Vec<u8>,
    ) -> Result<Self, DecodedTileError> {
        Self::try_with_valid_region(width, height, width, height, 0, format, data)
    }

    /// Construct a validated single-level decoded tile with explicit valid
    /// interior dimensions `(valid_w, valid_h)` and border `gutter` (e.g.
    /// clipped `RasterPyramid` edge tiles or `514x514` `gutter = 1` DEM tiles).
    pub fn try_with_valid_region(
        width: u32,
        height: u32,
        valid_w: u32,
        valid_h: u32,
        gutter: u8,
        format: PixelFormat,
        data: Vec<u8>,
    ) -> Result<Self, DecodedTileError> {
        let tile = Self {
            width,
            height,
            valid_w,
            valid_h,
            gutter,
            format,
            data,
            mips: Vec::new(),
        };
        tile.validate_with_mips(&tile.mips)?;
        Ok(tile)
    }

    /// Construct a validated decoded tile carrying precomputed mip levels
    /// `1..N` (`gutter = 0`, full valid interior).
    pub fn try_with_mips(
        width: u32,
        height: u32,
        format: PixelFormat,
        data: Vec<u8>,
        mips: Vec<Vec<u8>>,
    ) -> Result<Self, DecodedTileError> {
        let tile = Self {
            width,
            height,
            valid_w: width,
            valid_h: height,
            gutter: 0,
            format,
            data,
            mips,
        };
        tile.validate_with_mips(&tile.mips)?;
        Ok(tile)
    }

    /// Attach a validated mipmap chain (`1..N`) to this tile, replacing any
    /// existing mip levels.
    pub fn try_set_mips(&mut self, mips: Vec<Vec<u8>>) -> Result<(), DecodedTileError> {
        self.validate_with_mips(&mips)?;
        self.mips = mips;
        Ok(())
    }

    /// Clear any precomputed mip levels (`1..N`), keeping base level 0 intact.
    pub fn clear_mips(&mut self) {
        self.mips.clear();
    }

    /// Allocated tile width in pixels.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Allocated tile height in pixels.
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Valid interior width in pixels excluding border gutters.
    pub const fn valid_w(&self) -> u32 {
        self.valid_w
    }

    /// Valid interior height in pixels excluding border gutters.
    pub const fn valid_h(&self) -> u32 {
        self.valid_h
    }

    /// Border gutter width in texels (`0` for standard tiles, `1` for DEMs).
    pub const fn gutter(&self) -> u8 {
        self.gutter
    }

    /// Unpacked pixel storage format.
    pub const fn format(&self) -> PixelFormat {
        self.format
    }

    /// Tightly packed base level 0 pixels.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Precomputed mipmap buffers (`1..N`).
    pub fn mips(&self) -> &[Vec<u8>] {
        &self.mips
    }

    /// Total number of mip levels present (`1` for base only, up to
    /// [`max_mip_levels`]).
    pub fn mip_level_count(&self) -> u32 {
        let extra = u32::try_from(self.mips.len()).unwrap_or(u32::MAX);
        1u32.saturating_add(extra)
    }

    /// Dimensions `(w, h)` of mip level `level` (`0` = base).
    pub const fn mip_dims(&self, level: u32) -> (u32, u32) {
        mip_dimensions(self.width, self.height, level)
    }

    /// Raw pixel slice for mip level `level` (`0` = base), or `None` if absent.
    pub fn mip_data(&self, level: u32) -> Option<&[u8]> {
        if level == 0 {
            Some(&self.data)
        } else {
            self.mips.get((level - 1) as usize).map(Vec::as_slice)
        }
    }

    /// Exact unpacked byte size across base level `0` and all present mip
    /// levels.
    pub fn byte_size(&self) -> usize {
        self.mips
            .iter()
            .fold(self.data.len(), |acc, mip| acc.saturating_add(mip.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_dimensions_and_validation_handle_pow2_and_non_pow2() {
        assert_eq!(max_mip_levels(0, 0), 0);
        assert_eq!(max_mip_levels(0, 10), 0);
        assert_eq!(max_mip_levels(10, 0), 0);
        assert_eq!(mip_dimensions(0, 0, 0), (0, 0));
        assert_eq!(mip_dimensions(0, 10, 0), (0, 0));
        assert_eq!(mip_dimensions(10, 0, 2), (0, 0));
        assert_eq!(max_mip_levels(1, 1), 1);
        assert_eq!(max_mip_levels(256, 256), 9);
        assert_eq!(max_mip_levels(5, 3), 3);

        assert!(matches!(
            DecodedTile::try_new(0, 4, PixelFormat::Rgba8UnormSrgb, Vec::new()),
            Err(DecodedTileError::ZeroDimension { .. })
        ));
        assert!(matches!(
            DecodedTile::try_new(4, 0, PixelFormat::Rgba8UnormSrgb, Vec::new()),
            Err(DecodedTileError::ZeroDimension { .. })
        ));

        assert_eq!(mip_dimensions(5, 3, 0), (5, 3));
        assert_eq!(mip_dimensions(5, 3, 1), (2, 1));
        assert_eq!(mip_dimensions(5, 3, 2), (1, 1));
        assert_eq!(
            mip_chain_byte_size(4, 4, PixelFormat::Rgba8UnormSrgb, 3),
            64 + 16 + 4
        );

        let mut valid = DecodedTile::try_with_mips(
            4,
            4,
            PixelFormat::Rgba8UnormSrgb,
            vec![0u8; 4 * 4 * 4],
            vec![vec![0u8; 2 * 2 * 4], vec![0u8; 4]],
        )
        .expect("valid mipmapped tile");
        assert_eq!(valid.mip_level_count(), 3);
        assert_eq!(valid.byte_size(), 64 + 16 + 4);
        assert!(format!("{valid:?}").contains("mip_levels: 3"));

        // Too many mip levels rejected without mutating existing mips.
        assert!(matches!(
            valid.try_set_mips(vec![vec![0u8; 16], vec![0u8; 4], vec![0u8; 4]]),
            Err(DecodedTileError::TooManyMipLevels { .. })
        ));
        assert_eq!(valid.mip_level_count(), 3);

        // Mismatched mip byte length rejected.
        assert!(matches!(
            valid.try_set_mips(vec![vec![0u8; 15], vec![0u8; 4]]),
            Err(DecodedTileError::MipByteLengthMismatch { level: 1, .. })
        ));

        // Guttered DEM tile (6x6 storing 4x4 interior with gutter = 1) constructs cleanly.
        let mut dem = DecodedTile::try_with_valid_region(
            6,
            6,
            4,
            4,
            1,
            PixelFormat::R32Float,
            vec![0u8; 6 * 6 * 4],
        )
        .expect("valid guttered DEM");

        // Guttered tile rejects extra mip levels.
        assert!(matches!(
            dem.try_set_mips(vec![vec![0u8; 3 * 3 * 4]]),
            Err(DecodedTileError::MipsUnsupportedForPaddedOrGuttered { .. })
        ));

        // Overflowing valid_w + 2*gutter > width is rejected.
        assert!(matches!(
            DecodedTile::try_with_valid_region(
                6,
                6,
                5,
                4,
                1,
                PixelFormat::R32Float,
                vec![0u8; 6 * 6 * 4],
            ),
            Err(DecodedTileError::InvalidValidRegion { .. })
        ));
    }
}
