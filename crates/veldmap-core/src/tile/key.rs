//! Spatial tile coordinates (`QuadTileId`), unified cache keys (`TileKey`),
//! and power-of-two ancestor UV sub-rectangle composition (`compose_uv_rect`).

use super::identity::{BandKey, PixelFormat, SourceId, StreamKey, TileStreamSpec};
use super::scheme::{PlanetaryScheme, TileScheme};

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
    pub(super) fn children(self) -> [Self; 4] {
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
        Self::from_stream(
            source,
            TileStreamSpec::DISPLAY,
            id.scheme.into_tile_scheme(),
            id.level,
            id.x,
            id.y,
        )
    }

    /// Build a [`TileKey`] from `(source, spec, scheme, level, x, y)`.
    pub const fn from_stream(
        source: SourceId,
        spec: TileStreamSpec,
        scheme: TileScheme,
        level: u8,
        x: u32,
        y: u32,
    ) -> Self {
        Self {
            source,
            band: spec.band,
            scheme,
            format: spec.format,
            gutter: spec.gutter,
            decode_rev: spec.decode_rev,
            level,
            x,
            y,
        }
    }

    /// Extract the source-independent [`TileStreamSpec`] partitioning fields.
    pub const fn stream_spec(&self) -> TileStreamSpec {
        TileStreamSpec {
            band: self.band,
            format: self.format,
            gutter: self.gutter,
            decode_rev: self.decode_rev,
        }
    }

    /// Extract the [`StreamKey`] identifying this tile's source/band/format/gutter
    /// stream.
    pub const fn stream_key(&self) -> StreamKey {
        StreamKey {
            source: self.source,
            band: self.band,
            format: self.format,
            gutter: self.gutter,
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
            TileScheme::WebMercator | TileScheme::Geodetic | TileScheme::PolarCap { .. } => {
                self.level == 0
            }
        }
    }

    /// Deterministic coarsest-first ordering key for upload and request queues.
    ///
    /// Orders by `(coarseness_rank, scheme, y, x)` first so multi-band tiles
    /// (`B04`, `B08`, `SCL`) interleave at each spatial cell, then breaks ties
    /// deterministically by `(source, stream_spec, level)`. Including `level`
    /// ensures strict injectivity with [`Eq`] even for out-of-range
    /// `RasterPyramid` levels (`level > max_level`) whose `coarseness_rank`
    /// saturates to `0`.
    pub const fn coarsest_first_key(
        &self,
    ) -> (u8, TileScheme, u32, u32, SourceId, TileStreamSpec, u8) {
        (
            self.scheme.coarseness_rank(self.level),
            self.scheme,
            self.y,
            self.x,
            self.source,
            self.stream_spec(),
            self.level,
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
    /// source/scheme/stream_spec, or if `gutter != 0` (guttered tiles and
    /// clipped `RasterPyramid` edge tiles should resolve carriers via
    /// `TileCache::find_carrier` and compute UVs via [`super::cell_uv_rect`]).
    pub(super) fn sub_rect_in_ancestor(&self, ancestor: &Self) -> Option<[f32; 4]> {
        if self.source != ancestor.source
            || self.scheme != ancestor.scheme
            || self.stream_spec() != ancestor.stream_spec()
            || self.gutter != 0
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
