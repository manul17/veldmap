//! CPU-decoded tile pixel storage (`DecodedTile`) with optional precomputed
//! mipmap levels shared across `veldmap-raster`, `veldmap-net`, and
//! `veldmap-app`.

use std::fmt;

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
    let mw = if mip_level >= 32 {
        1
    } else {
        let shifted = width >> mip_level;
        if shifted == 0 { 1 } else { shifted }
    };
    let mh = if mip_level >= 32 {
        1
    } else {
        let shifted = height >> mip_level;
        if shifted == 0 { 1 } else { shifted }
    };
    (mw, mh)
}

/// CPU-decoded tile pixels ready for GPU texture upload, with optional
/// CPU-generated mipmap levels (`1..N`) and explicit valid interior / gutter
/// dimensions for clipped edge tiles and `gutter = 1` DEM tiles.
#[derive(Clone)]
pub struct DecodedTile {
    pub width: u32,
    pub height: u32,
    /// Valid interior pixel width excluding border gutters
    /// (`valid_w + 2 * gutter <= width`).
    pub valid_w: u32,
    /// Valid interior pixel height excluding border gutters
    /// (`valid_h + 2 * gutter <= height`).
    pub valid_h: u32,
    /// Border gutter width in texels (`0` for standard tiles, `1` for DEM
    /// derivative tiles).
    pub gutter: u8,
    pub format: PixelFormat,
    /// Tightly packed mip level 0 pixels (`width * height * bytes_per_pixel`).
    pub data: Vec<u8>,
    /// Optional precomputed mip levels `1..N`, where `mips[i]` holds mip level
    /// `i + 1` with dimensions `mip_dimensions(width, height, i + 1)`.
    pub mips: Vec<Vec<u8>>,
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
    /// Create a single-level (`mip_level_count == 1`, `gutter = 0`, full valid
    /// interior) decoded tile.
    pub fn new(width: u32, height: u32, format: PixelFormat, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            valid_w: width,
            valid_h: height,
            gutter: 0,
            format,
            data,
            mips: Vec::new(),
        }
    }

    /// Create a single-level decoded tile with explicit valid interior
    /// dimensions `(valid_w, valid_h)` and border `gutter` (e.g. clipped
    /// `RasterPyramid` edge tiles or `514x514` `gutter = 1` DEM tiles).
    pub fn with_valid_region(
        width: u32,
        height: u32,
        valid_w: u32,
        valid_h: u32,
        gutter: u8,
        format: PixelFormat,
        data: Vec<u8>,
    ) -> Self {
        Self {
            width,
            height,
            valid_w,
            valid_h,
            gutter,
            format,
            data,
            mips: Vec::new(),
        }
    }

    /// Create a decoded tile carrying precomputed mip levels `1..N` (`gutter = 0`,
    /// full valid interior).
    pub fn with_mips(
        width: u32,
        height: u32,
        format: PixelFormat,
        data: Vec<u8>,
        mips: Vec<Vec<u8>>,
    ) -> Self {
        Self {
            width,
            height,
            valid_w: width,
            valid_h: height,
            gutter: 0,
            format,
            data,
            mips,
        }
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

    /// Validate that dimensions and valid interior are non-zero, gutters fit
    /// cleanly (`valid_w + 2*gutter <= width`, `valid_h + 2*gutter <= height`),
    /// guttered/padded tiles carry no mip chain, mip count does not exceed
    /// [`max_mip_levels`], and every mip buffer has the exact expected byte
    /// length for its `(w, h, format)`.
    pub fn validate(&self) -> bool {
        if self.width == 0 || self.height == 0 || self.valid_w == 0 || self.valid_h == 0 {
            return false;
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
            return false;
        }
        if (self.gutter != 0 || self.valid_w != self.width || self.valid_h != self.height)
            && !self.mips.is_empty()
        {
            return false;
        }
        let max_mips = max_mip_levels(self.width, self.height);
        let count = self.mip_level_count();
        if count > max_mips {
            return false;
        }
        let bpp = self.format.bytes_per_pixel();
        for level in 0..count {
            let (mw, mh) = self.mip_dims(level);
            let Some(expected) = (mw as usize)
                .checked_mul(mh as usize)
                .and_then(|px| px.checked_mul(bpp))
            else {
                return false;
            };
            let Some(buf) = self.mip_data(level) else {
                return false;
            };
            if buf.len() != expected {
                return false;
            }
        }
        true
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

        assert!(
            !DecodedTile::new(0, 4, PixelFormat::Rgba8UnormSrgb, Vec::new()).validate(),
            "0xN tile must fail validation"
        );
        assert!(
            !DecodedTile::new(4, 0, PixelFormat::Rgba8UnormSrgb, Vec::new()).validate(),
            "Nx0 tile must fail validation"
        );

        assert_eq!(mip_dimensions(5, 3, 0), (5, 3));
        assert_eq!(mip_dimensions(5, 3, 1), (2, 1));
        assert_eq!(mip_dimensions(5, 3, 2), (1, 1));

        let valid = DecodedTile::with_mips(
            4,
            4,
            PixelFormat::Rgba8UnormSrgb,
            vec![0u8; 4 * 4 * 4],
            vec![vec![0u8; 2 * 2 * 4], vec![0u8; 4]],
        );
        assert!(valid.validate());
        assert_eq!(valid.mip_level_count(), 3);
        assert_eq!(valid.byte_size(), 64 + 16 + 4);
        assert!(format!("{valid:?}").contains("mip_levels: 3"));

        // Too many mip levels rejected.
        let mut too_many = valid.clone();
        too_many.mips.push(vec![0u8; 4]);
        assert!(!too_many.validate());

        // Mismatched mip byte length rejected.
        let mut bad_mip = valid;
        bad_mip.mips[0].pop();
        assert!(!bad_mip.validate());

        // Guttered DEM tile (6x6 storing 4x4 interior with gutter = 1) validates cleanly.
        let dem = DecodedTile::with_valid_region(
            6,
            6,
            4,
            4,
            1,
            PixelFormat::R32Float,
            vec![0u8; 6 * 6 * 4],
        );
        assert!(dem.validate());

        // Guttered tile with extra mip levels is rejected.
        let mut dem_with_mip = dem.clone();
        dem_with_mip.mips.push(vec![0u8; 3 * 3 * 4]);
        assert!(!dem_with_mip.validate());

        // Overflowing valid_w + 2*gutter > width is rejected.
        let bad_gutter = DecodedTile::with_valid_region(
            6,
            6,
            5,
            4,
            1,
            PixelFormat::R32Float,
            vec![0u8; 6 * 6 * 4],
        );
        assert!(!bad_gutter.validate());
    }
}
