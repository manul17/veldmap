//! Unified `Copy` tile addressing (`TileKey`, `QuadTileId`, `TileScheme`,
//! `BandKey`, `PixelFormat`, `SourceId`) shared by planetary basemaps, polar
//! caps, and local/cloud raster pyramids.

/// Deterministic 64-bit FNV-1a hash used for stable `Copy` source/band keys.
pub const fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    hash
}

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

    /// Derive a deterministic `SourceId` from a stable ASCII/UTF-8 source key
    /// (e.g. a live basemap layer ID or normalized URL template).
    ///
    /// Sets the high bit so derived IDs never collide with small reserved
    /// constants such as [`Self::BUNDLED_BASEMAP`].
    pub const fn from_stable_key(key: &str) -> Self {
        Self(fnv1a_64(key.as_bytes()) | (1u64 << 63))
    }

    /// Derive a deterministic [`SourceId`] from a content [`crate::io::SourceFingerprint`].
    pub const fn from_fingerprint(fingerprint: &crate::io::SourceFingerprint) -> Self {
        Self(fnv1a_64(fingerprint.as_bytes()) | (1u64 << 63))
    }
}

impl From<&crate::io::SourceFingerprint> for SourceId {
    fn from(fingerprint: &crate::io::SourceFingerprint) -> Self {
        Self::from_fingerprint(fingerprint)
    }
}

/// Deterministic, `Copy` band selector avoiding heap allocations in `TileKey`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
        Self::NamedHash(fnv1a_64(name.as_bytes()))
    }
}

/// Pixel storage format for GPU and CPU tile caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

/// Main-globe (non-polar) coarsest-zero planetary tiling scheme for quadtree
/// traversal (`Geodetic` or `WebMercator`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GlobeScheme {
    /// Web Mercator XYZ (`1 x 1` root at `z = 0` coarsest, `2^z x 2^z` tiles,
    /// `|lat| <= 85.05112878 deg`).
    WebMercator,
    /// Equirectangular / EPSG:4326 (`2 x 1` root at `z = 0` coarsest,
    /// `(2 * 2^z) x 2^z` tiles).
    Geodetic,
}

impl GlobeScheme {
    /// Convert to [`PlanetaryScheme`].
    pub const fn into_planetary(self) -> PlanetaryScheme {
        match self {
            Self::WebMercator => PlanetaryScheme::WebMercator,
            Self::Geodetic => PlanetaryScheme::Geodetic,
        }
    }

    /// Convert to [`TileScheme`].
    pub const fn into_tile_scheme(self) -> TileScheme {
        self.into_planetary().into_tile_scheme()
    }

    /// Return the grid dimensions `(cols, rows)` at `level`, or `None` for
    /// out-of-range shift levels.
    pub const fn grid_dims(self, level: u8) -> Option<(u32, u32)> {
        self.into_planetary().grid_dims(level)
    }
}

impl From<GlobeScheme> for PlanetaryScheme {
    fn from(scheme: GlobeScheme) -> Self {
        scheme.into_planetary()
    }
}

impl From<GlobeScheme> for TileScheme {
    fn from(scheme: GlobeScheme) -> Self {
        scheme.into_tile_scheme()
    }
}

/// Coarsest-zero planetary tiling scheme (excludes `TileScheme::RasterPyramid`
/// so planetary quadtree arithmetic can never invert finest-zero raster levels).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PlanetaryScheme {
    /// Web Mercator XYZ (`1 x 1` root at `z = 0` coarsest, `2^z x 2^z` tiles,
    /// `|lat| <= 85.05112878 deg`).
    WebMercator,
    /// Equirectangular / EPSG:4326 (`2 x 1` root at `z = 0` coarsest,
    /// `(2 * 2^z) x 2^z` tiles).
    Geodetic,
    /// Adaptive Polar Stereographic square cap (`1 x 1` root at `z = 0`
    /// coarsest, `2^z x 2^z` tiles covering `|lat| >= 85.05112878 deg`).
    PolarCap { north: bool },
}

impl PlanetaryScheme {
    /// Convert to [`TileScheme`].
    pub const fn into_tile_scheme(self) -> TileScheme {
        match self {
            Self::WebMercator => TileScheme::WebMercator,
            Self::Geodetic => TileScheme::Geodetic,
            Self::PolarCap { north } => TileScheme::PolarCap { north },
        }
    }

    /// Return the grid dimensions `(cols, rows)` at `level`, or `None` for
    /// out-of-range shift levels.
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
        }
    }

    /// Return the coarser parent `(level - 1, x >> 1, y >> 1)`, or `None` at
    /// `level == 0`.
    pub const fn parent(self, level: u8, x: u32, y: u32) -> Option<(u8, u32, u32)> {
        if level == 0 {
            None
        } else {
            Some((level - 1, x >> 1, y >> 1))
        }
    }
}

impl From<PlanetaryScheme> for TileScheme {
    fn from(scheme: PlanetaryScheme) -> Self {
        scheme.into_tile_scheme()
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
    /// Narrow to [`PlanetaryScheme`], returning `None` for `RasterPyramid`.
    pub const fn as_planetary(self) -> Option<PlanetaryScheme> {
        match self {
            Self::WebMercator => Some(PlanetaryScheme::WebMercator),
            Self::Geodetic => Some(PlanetaryScheme::Geodetic),
            Self::PolarCap { north } => Some(PlanetaryScheme::PolarCap { north }),
            Self::RasterPyramid { .. } => None,
        }
    }

    /// Narrow to main-globe [`GlobeScheme`], returning `None` for `PolarCap` or
    /// `RasterPyramid`.
    pub const fn as_globe(self) -> Option<GlobeScheme> {
        match self {
            Self::WebMercator => Some(GlobeScheme::WebMercator),
            Self::Geodetic => Some(GlobeScheme::Geodetic),
            Self::PolarCap { .. } | Self::RasterPyramid { .. } => None,
        }
    }

    /// Return the grid dimensions `(cols, rows)` at `level` for planetary
    /// schemes, or `None` for raster pyramids (whose dimensions depend on the
    /// underlying raster) or out-of-range shift levels.
    pub const fn grid_dims(self, level: u8) -> Option<(u32, u32)> {
        match self.as_planetary() {
            Some(p) => p.grid_dims(level),
            None => None,
        }
    }

    /// Return the coarser parent tile `(parent_level, parent_x, parent_y)`, or
    /// `None` if `(level, x, y)` is already at the coarsest root of the scheme
    /// (or above `max_level` for `RasterPyramid`).
    ///
    /// Crucially, `WebMercator`/`Geodetic`/`PolarCap` walk `level -> level - 1`
    /// toward `z = 0`, whereas `RasterPyramid { max_level }` walks
    /// `level -> level + 1` toward `max_level`, so ancestor lookups and LRU
    /// ancestor touches never invert.
    pub const fn parent(self, level: u8, x: u32, y: u32) -> Option<(u8, u32, u32)> {
        match self {
            Self::RasterPyramid { max_level } => {
                if level >= max_level {
                    None
                } else {
                    Some((level + 1, x >> 1, y >> 1))
                }
            }
            _ => match self.as_planetary() {
                Some(planetary) => planetary.parent(level, x, y),
                None => None,
            },
        }
    }

    /// Return the coarseness rank of `level` where smaller values are coarser
    /// (for coarsest-first upload/scheduling priority queues).
    pub const fn coarseness_rank(self, level: u8) -> u8 {
        match self {
            Self::RasterPyramid { max_level } => max_level.saturating_sub(level),
            _ => level,
        }
    }

    /// Return how many pyramid steps `finer_level` is below `coarser_level`,
    /// or `None` if `coarser_level` is strictly finer than `finer_level` or
    /// either level exceeds `RasterPyramid::max_level`.
    pub const fn level_delta(self, finer_level: u8, coarser_level: u8) -> Option<u8> {
        match self {
            Self::RasterPyramid { max_level } => {
                if finer_level > max_level || coarser_level > max_level {
                    None
                } else {
                    coarser_level.checked_sub(finer_level)
                }
            }
            _ => finer_level.checked_sub(coarser_level),
        }
    }
}

/// Geometry identifier for a single planetary quadtree or polar cap tile
/// (strictly coarsest-zero [`PlanetaryScheme`], statically excluding
/// finest-zero `TileScheme::RasterPyramid`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QuadTileId {
    pub scheme: PlanetaryScheme,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

/// Power-of-two UV sub-rectangle `[u0, v0, scale, scale]` for a tile `(x, y)`
/// nested `dz` levels (`1 <= dz < 31`) below an ancestor.
#[inline]
fn sub_rect_for_shift(x: u32, y: u32, dz: u8) -> [f32; 4] {
    let mask = (1u32 << dz) - 1;
    let scale = 1.0_f64 / f64::from(1u32 << dz);
    let u0 = f64::from(x & mask) * scale;
    let v0 = f64::from(y & mask) * scale;
    [u0 as f32, v0 as f32, scale as f32, scale as f32]
}

impl QuadTileId {
    /// Validate `self.level`, `self.x`, and `self.y` against the planetary
    /// grid, returning `Some((cols, rows))` iff the tile lies strictly inside
    /// the grid bounds at `self.level`.
    pub const fn checked_grid_dims(self) -> Option<(u32, u32)> {
        let Some((cols, rows)) = self.scheme.grid_dims(self.level) else {
            return None;
        };
        if self.x < cols && self.y < rows {
            Some((cols, rows))
        } else {
            None
        }
    }

    /// Return this tile's parent clamped to at most `max_level`, together with
    /// the exact UV sub-rectangle `[u_off, v_off, u_scale, v_scale]` mapping
    /// this tile into the clamped ancestor.
    pub fn clamp_to_level(self, max_level: u8) -> (Self, [f32; 4]) {
        if self.level <= max_level {
            return (self, [0.0, 0.0, 1.0, 1.0]);
        }
        let dz = self.level - max_level;
        if dz >= 31 {
            return (
                Self {
                    scheme: self.scheme,
                    level: max_level,
                    x: 0,
                    y: 0,
                },
                [0.0, 0.0, 0.0, 0.0],
            );
        }
        (
            Self {
                scheme: self.scheme,
                level: max_level,
                x: self.x >> dz,
                y: self.y >> dz,
            },
            sub_rect_for_shift(self.x, self.y, dz),
        )
    }

    /// Return the 4 child tiles of this planetary node.
    pub fn children(self) -> [Self; 4] {
        debug_assert!(self.level < u8::MAX, "QuadTileId level overflow");
        let next = self.level.saturating_add(1);
        let cx = self.x << 1;
        let cy = self.y << 1;
        [
            Self {
                scheme: self.scheme,
                level: next,
                x: cx,
                y: cy,
            },
            Self {
                scheme: self.scheme,
                level: next,
                x: cx + 1,
                y: cy,
            },
            Self {
                scheme: self.scheme,
                level: next,
                x: cx,
                y: cy + 1,
            },
            Self {
                scheme: self.scheme,
                level: next,
                x: cx + 1,
                y: cy + 1,
            },
        ]
    }

    /// Return the coarser parent node, or `None` at `level == 0`.
    pub const fn parent(self) -> Option<Self> {
        let Some((level, x, y)) = self.scheme.parent(self.level, self.x, self.y) else {
            return None;
        };
        Some(Self {
            scheme: self.scheme,
            level,
            x,
            y,
        })
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
    /// Build the canonical display `TileKey` (`BandKey::Display`,
    /// `PixelFormat::Rgba8UnormSrgb`, `gutter = 0`, `decode_rev = 0`) for
    /// `(source, id)`.
    pub const fn display_quad(source: SourceId, id: QuadTileId) -> Self {
        Self {
            source,
            band: BandKey::Display,
            scheme: id.scheme.into_tile_scheme(),
            format: PixelFormat::Rgba8UnormSrgb,
            gutter: 0,
            decode_rev: 0,
            level: id.level,
            x: id.x,
            y: id.y,
        }
    }

    /// Extract the planetary geometric `(scheme, level, x, y)` identifier,
    /// returning `None` for non-planetary `TileScheme::RasterPyramid` keys.
    pub const fn quad_id(&self) -> Option<QuadTileId> {
        let Some(scheme) = self.scheme.as_planetary() else {
            return None;
        };
        Some(QuadTileId {
            scheme,
            level: self.level,
            x: self.x,
            y: self.y,
        })
    }

    /// Return whether `self` is a valid coarsest root tile of `source`
    /// (`level == 0` for planetary schemes, or `level == max_level` for
    /// `RasterPyramid`; out-of-range `level > max_level` returns `false`).
    pub const fn is_root_of(&self, source: SourceId) -> bool {
        if self.source.0 != source.0 {
            return false;
        }
        match self.scheme {
            TileScheme::RasterPyramid { max_level } => self.level == max_level,
            _ => self.level == 0,
        }
    }

    /// Deterministic coarsest-first ordering key for upload and request queues.
    ///
    /// Orders by `(coarseness_rank, scheme, y, x)` first so multi-band tiles
    /// (`B04`, `B08`, `SCL`) interleave at each spatial cell, then breaks ties
    /// deterministically by `(source, band, format, gutter, decode_rev)`.
    pub const fn coarsest_first_key(
        &self,
    ) -> (
        u8,
        TileScheme,
        u32,
        u32,
        SourceId,
        BandKey,
        PixelFormat,
        u8,
        u32,
    ) {
        (
            self.scheme.coarseness_rank(self.level),
            self.scheme,
            self.y,
            self.x,
            self.source,
            self.band,
            self.format,
            self.gutter,
            self.decode_rev,
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
        Some(sub_rect_for_shift(self.x, self.y, dz))
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
#[path = "key_tests.rs"]
mod tests;
