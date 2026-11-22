//! Unified `Copy` tile addressing (`TileKey`, `TileScheme`, `BandKey`,
//! `PixelFormat`, `SourceId`) shared by planetary basemaps, polar caps, and
//! local/cloud raster pyramids.

/// Compact identifier for a tile data source (bundled basemap, live XYZ/TMS
/// layer, local GeoTIFF, or cloud COG).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(pub u64);

impl SourceId {
    /// Reserved source ID for the bundled offline global basemap.
    pub const BUNDLED_BASEMAP: Self = Self(1);

    pub const fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Deterministic, `Copy` band selector avoiding heap allocations in `TileKey`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BandKey {
    /// Display-ready color composite (e.g. sRGB basemap or RGB orthophoto).
    Display,
    /// Zero-based numeric band index in a multi-band raster.
    Index(u16),
    /// Deterministic 64-bit FNV-1a hash of a named band/variable (e.g. `"B08"`,
    /// `"SCL"`, or NetCDF variable name).
    NamedHash(u64),
}

impl BandKey {
    /// Compute a deterministic 64-bit FNV-1a hash for a band or variable name.
    pub const fn from_name(name: &str) -> Self {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let bytes = name.as_bytes();
        let mut hash = FNV_OFFSET;
        let mut i = 0;
        while i < bytes.len() {
            hash ^= bytes[i] as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
            i += 1;
        }
        Self::NamedHash(hash)
    }
}

/// Pixel storage format for GPU and CPU tile caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 8-bit sRGB RGBA display imagery (4 bytes/pixel).
    Rgba8UnormSrgb,
    /// 32-bit IEEE float scientific samples (4 bytes/pixel, NaN = nodata).
    R32Float,
    /// 8-bit unsigned integer categorical/QA masks (1 byte/pixel).
    R8Uint,
    /// 16-bit unsigned integer categorical/QA masks (2 bytes/pixel).
    R16Uint,
}

impl PixelFormat {
    /// Exact unpacked byte size of a single pixel in this format.
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgba8UnormSrgb | Self::R32Float => 4,
            Self::R8Uint => 1,
            Self::R16Uint => 2,
        }
    }
}

/// Hierarchical tiling scheme governing level direction and parent traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TileScheme {
    /// Web Mercator XYZ (`1 x 1` root at `z = 0` coarsest, `2^z x 2^z` tiles,
    /// `|lat| <= 85.05112878 deg`).
    WebMercator,
    /// Equirectangular / EPSG:4326 (`2 x 1` root at `z = 0` coarsest,
    /// `(2 * 2^z) x 2^z` tiles).
    Geodetic,
    /// Adaptive Polar Stereographic square cap (`1 x 1` root at `z = 0`
    /// coarsest, `2^z x 2^z` tiles covering `|lat| >= 85.05112878 deg`).
    PolarCap { north: bool },
    /// Local or cloud raster pyramid (`level = 0` finest native resolution,
    /// `level = max_level` coarsest overview).
    RasterPyramid { max_level: u8 },
}

impl TileScheme {
    /// Return whether `level = 0` is the coarsest level (`true` for planetary
    /// XYZ/Geodetic/PolarCap schemes, `false` for `RasterPyramid`).
    pub const fn is_coarsest_zero(self) -> bool {
        !matches!(self, Self::RasterPyramid { .. })
    }

    /// Return the grid dimensions `(cols, rows)` at `level` for planetary
    /// schemes, or `None` for raster pyramids (whose dimensions depend on the
    /// underlying raster) or out-of-range shift levels.
    pub const fn grid_dims(self, level: u8) -> Option<(u32, u32)> {
        match self {
            Self::WebMercator | Self::PolarCap { .. } => {
                if level <= 30 {
                    let n = 1u32 << level;
                    Some((n, n))
                } else {
                    None
                }
            }
            Self::Geodetic => {
                if level <= 29 {
                    let rows = 1u32 << level;
                    Some((rows << 1, rows))
                } else {
                    None
                }
            }
            Self::RasterPyramid { .. } => None,
        }
    }

    /// Return the coarser parent tile `(parent_level, parent_x, parent_y)`, or
    /// `None` if `(level, x, y)` is already at the coarsest root of the scheme.
    ///
    /// Crucially, `WebMercator`/`Geodetic`/`PolarCap` walk `level -> level - 1`
    /// toward `z = 0`, whereas `RasterPyramid { max_level }` walks
    /// `level -> level + 1` toward `max_level`, so ancestor lookups and LRU
    /// ancestor touches never invert.
    pub const fn parent(self, level: u8, x: u32, y: u32) -> Option<(u8, u32, u32)> {
        match self {
            Self::WebMercator | Self::Geodetic | Self::PolarCap { .. } => {
                if level == 0 {
                    None
                } else {
                    Some((level - 1, x >> 1, y >> 1))
                }
            }
            Self::RasterPyramid { max_level } => {
                if level >= max_level {
                    None
                } else {
                    Some((level + 1, x >> 1, y >> 1))
                }
            }
        }
    }

    /// Return the coarseness rank of `level` where smaller values are coarser
    /// (for coarsest-first upload/scheduling priority queues).
    pub const fn coarseness_rank(self, level: u8) -> u8 {
        match self {
            Self::WebMercator | Self::Geodetic | Self::PolarCap { .. } => level,
            Self::RasterPyramid { max_level } => max_level.saturating_sub(level),
        }
    }

    /// Return how many pyramid steps `finer_level` is below `coarser_level`,
    /// or `None` if `coarser_level` is strictly finer than `finer_level` or
    /// either level exceeds `RasterPyramid::max_level`.
    pub const fn level_delta(self, finer_level: u8, coarser_level: u8) -> Option<u8> {
        match self {
            Self::WebMercator | Self::Geodetic | Self::PolarCap { .. } => {
                finer_level.checked_sub(coarser_level)
            }
            Self::RasterPyramid { max_level } => {
                if finer_level > max_level || coarser_level > max_level {
                    None
                } else {
                    coarser_level.checked_sub(finer_level)
                }
            }
        }
    }
}

/// Complete, `Copy + Eq + Hash` cache key for any GPU or CPU tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileKey {
    pub source: SourceId,
    pub band: BandKey,
    pub scheme: TileScheme,
    pub format: PixelFormat,
    /// Border gutter width in texels (`0` for standard tiles, `1` for DEM
    /// relief tiles) so guttered and unguttered tiles never alias in caches.
    pub gutter: u8,
    pub decode_rev: u32,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl TileKey {
    /// Deterministic coarsest-first ordering key for upload and request queues.
    pub const fn coarsest_first_key(&self) -> (u8, TileScheme, u32, u32) {
        (
            self.scheme.coarseness_rank(self.level),
            self.scheme,
            self.y,
            self.x,
        )
    }

    /// Return this tile's immediate coarser parent key, or `None` at the root.
    pub const fn parent(&self) -> Option<Self> {
        let Some((level, x, y)) = self.scheme.parent(self.level, self.x, self.y) else {
            return None;
        };
        Some(Self {
            level,
            x,
            y,
            ..*self
        })
    }

    /// Compute the power-of-two UV sub-rectangle
    /// `[u_offset, v_offset, u_scale, v_scale]` mapping `self`'s local
    /// `[0, 1]^2` UV space into `ancestor`'s texture space.
    ///
    /// Intended for unguttered (`gutter == 0`) power-of-two quadtrees. Returns
    /// `None` if `ancestor` is not a valid ancestor of `self`, differs in
    /// source/band/scheme/format/gutter/decode_rev, or if `gutter != 0`
    /// (guttered tiles and clipped `RasterPyramid` edge tiles should resolve
    /// carriers via `TileCache::find_carrier` and compute UVs from the
    /// carrier's `(width, height, valid_w, valid_h, gutter)`).
    pub fn sub_rect_in_ancestor(&self, ancestor: &Self) -> Option<[f32; 4]> {
        if self.source != ancestor.source
            || self.band != ancestor.band
            || self.scheme != ancestor.scheme
            || self.format != ancestor.format
            || self.gutter != ancestor.gutter
            || self.gutter != 0
            || self.decode_rev != ancestor.decode_rev
        {
            return None;
        }
        let dz = self.scheme.level_delta(self.level, ancestor.level)?;
        if dz == 0 {
            return (self.x == ancestor.x && self.y == ancestor.y).then_some([0.0, 0.0, 1.0, 1.0]);
        }
        if dz >= 31 {
            return None;
        }
        if (self.x >> dz) != ancestor.x || (self.y >> dz) != ancestor.y {
            return None;
        }
        let mask = (1u32 << dz) - 1;
        let scale = 1.0_f64 / f64::from(1u32 << dz);
        let u0 = f64::from(self.x & mask) * scale;
        let v0 = f64::from(self.y & mask) * scale;
        Some([u0 as f32, v0 as f32, scale as f32, scale as f32])
    }
}

/// Compose two UV sub-rectangles `[u0, v0, us, vs]`: apply `inner` inside
/// `outer`, returning the combined sub-rectangle in `outer`'s parent space.
pub fn compose_uv_rect(outer: [f32; 4], inner: [f32; 4]) -> [f32; 4] {
    [
        outer[0] + inner[0] * outer[2],
        outer[1] + inner[1] * outer[3],
        outer[2] * inner[2],
        outer[3] * inner[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_scheme_parent_respects_level_direction_without_inverting() {
        // Planetary XYZ / Geodetic / PolarCap: z = 0 is coarsest.
        for scheme in [
            TileScheme::WebMercator,
            TileScheme::Geodetic,
            TileScheme::PolarCap { north: true },
            TileScheme::PolarCap { north: false },
        ] {
            assert!(scheme.is_coarsest_zero());
            assert_eq!(scheme.parent(0, 0, 0), None);
            assert_eq!(scheme.parent(0, 1, 0), None);
            assert_eq!(scheme.parent(3, 5, 7), Some((2, 2, 3)));
            assert_eq!(scheme.parent(1, 3, 1), Some((0, 1, 0)));
            assert!(scheme.coarseness_rank(0) < scheme.coarseness_rank(3));
        }

        // RasterPyramid: level = 0 is finest, level = max_level is coarsest.
        let pyr = TileScheme::RasterPyramid { max_level: 5 };
        assert!(!pyr.is_coarsest_zero());
        assert_eq!(pyr.parent(5, 0, 0), None);
        assert_eq!(pyr.parent(6, 0, 0), None);
        assert_eq!(pyr.parent(0, 5, 7), Some((1, 2, 3)));
        assert_eq!(pyr.parent(4, 1, 1), Some((5, 0, 0)));
        assert!(pyr.coarseness_rank(5) < pyr.coarseness_rank(0));
    }

    #[test]
    fn sub_rect_in_ancestor_maps_both_planetary_and_raster_pyramid_tiles() {
        let ancestor = TileKey {
            source: SourceId::BUNDLED_BASEMAP,
            band: BandKey::Display,
            scheme: TileScheme::Geodetic,
            format: PixelFormat::Rgba8UnormSrgb,
            gutter: 0,
            decode_rev: 0,
            level: 2,
            x: 3,
            y: 1,
        };
        let child = TileKey {
            level: 4,
            x: (3 << 2) + 2,
            y: (1 << 2) + 3,
            ..ancestor
        };
        assert_eq!(
            child.sub_rect_in_ancestor(&ancestor),
            Some([0.5, 0.75, 0.25, 0.25])
        );
        assert_eq!(
            ancestor.sub_rect_in_ancestor(&ancestor),
            Some([0.0, 0.0, 1.0, 1.0])
        );
        // Reversed direction must return None.
        assert_eq!(ancestor.sub_rect_in_ancestor(&child), None);

        // RasterPyramid ancestor has LARGER level index.
        let pyr_coarse = TileKey {
            scheme: TileScheme::RasterPyramid { max_level: 4 },
            level: 3,
            x: 1,
            y: 2,
            ..ancestor
        };
        let pyr_fine = TileKey {
            level: 1,
            x: (1 << 2) + 1,
            y: (2 << 2) + 3,
            ..pyr_coarse
        };
        assert_eq!(
            pyr_fine.sub_rect_in_ancestor(&pyr_coarse),
            Some([0.25, 0.75, 0.25, 0.25])
        );
        assert_eq!(pyr_coarse.sub_rect_in_ancestor(&pyr_fine), None);
    }

    #[test]
    fn band_key_from_name_is_deterministic() {
        assert_eq!(BandKey::from_name("B08"), BandKey::from_name("B08"));
        assert_ne!(BandKey::from_name("B08"), BandKey::from_name("B04"));
        assert_ne!(BandKey::from_name("SCL"), BandKey::Display);
    }

    #[test]
    fn grid_dims_and_sub_rect_edge_cases_are_guarded() {
        assert_eq!(
            TileScheme::WebMercator.grid_dims(30),
            Some((1 << 30, 1 << 30))
        );
        assert_eq!(TileScheme::WebMercator.grid_dims(31), None);
        assert_eq!(TileScheme::Geodetic.grid_dims(29), Some((1 << 30, 1 << 29)));
        assert_eq!(TileScheme::Geodetic.grid_dims(30), None);
        assert_eq!(
            TileScheme::RasterPyramid { max_level: 4 }.grid_dims(0),
            None
        );

        assert_eq!(PixelFormat::Rgba8UnormSrgb.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::R32Float.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::R8Uint.bytes_per_pixel(), 1);
        assert_eq!(PixelFormat::R16Uint.bytes_per_pixel(), 2);

        let base = TileKey {
            source: SourceId::BUNDLED_BASEMAP,
            band: BandKey::Display,
            scheme: TileScheme::WebMercator,
            format: PixelFormat::Rgba8UnormSrgb,
            gutter: 0,
            decode_rev: 0,
            level: 0,
            x: 0,
            y: 0,
        };
        let other_source = TileKey {
            source: SourceId::new(2),
            ..base
        };
        assert_eq!(base.sub_rect_in_ancestor(&other_source), None);

        let guttered = TileKey { gutter: 1, ..base };
        assert_eq!(guttered.sub_rect_in_ancestor(&base), None);
        assert_eq!(guttered.sub_rect_in_ancestor(&guttered), None);

        let too_deep = TileKey { level: 31, ..base };
        assert_eq!(too_deep.sub_rect_in_ancestor(&base), None);

        let wrong_branch = TileKey {
            level: 2,
            x: 2,
            y: 0,
            ..base
        };
        let other_parent = TileKey {
            level: 1,
            x: 0,
            y: 0,
            ..base
        };
        assert_eq!(wrong_branch.sub_rect_in_ancestor(&other_parent), None);

        // RasterPyramid rejects levels strictly above max_level in both level_delta and sub_rect.
        let pyr = TileScheme::RasterPyramid { max_level: 3 };
        assert_eq!(pyr.level_delta(1, 4), None);
        assert_eq!(pyr.level_delta(4, 3), None);
        let pyr_invalid = TileKey {
            scheme: pyr,
            level: 4,
            x: 0,
            y: 0,
            ..base
        };
        let pyr_fine = TileKey {
            level: 1,
            ..pyr_invalid
        };
        assert_eq!(pyr_fine.sub_rect_in_ancestor(&pyr_invalid), None);

        // Coarsest-first key orders z=0 before z=2 for planetary, and max_level before 0 for RasterPyramid.
        assert!(base.coarsest_first_key() < wrong_branch.coarsest_first_key());
        let pyr_root = TileKey {
            scheme: pyr,
            level: 3,
            ..base
        };
        assert!(pyr_root.coarsest_first_key() < pyr_fine.coarsest_first_key());

        assert_eq!(
            compose_uv_rect([0.5, 0.25, 0.5, 0.5], [0.5, 0.5, 0.5, 0.25]),
            [0.75, 0.5, 0.25, 0.125]
        );
    }
}
