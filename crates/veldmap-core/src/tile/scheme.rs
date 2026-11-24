//! Hierarchical tiling schemes (`GlobeScheme`, `PlanetaryScheme`, `TileScheme`)
//! governing grid dimensions, coarseness ordering, and parent traversal.

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
            Self::WebMercator | Self::Geodetic | Self::PolarCap { .. } => {
                if level == 0 {
                    None
                } else {
                    Some((level - 1, x >> 1, y >> 1))
                }
            }
        }
    }

    /// Return the coarseness rank of `level` where smaller values are coarser
    /// (for coarsest-first upload/scheduling priority queues).
    pub const fn coarseness_rank(self, level: u8) -> u8 {
        match self {
            Self::RasterPyramid { max_level } => max_level.saturating_sub(level),
            Self::WebMercator | Self::Geodetic | Self::PolarCap { .. } => level,
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
            Self::WebMercator | Self::Geodetic | Self::PolarCap { .. } => {
                finer_level.checked_sub(coarser_level)
            }
        }
    }
}
