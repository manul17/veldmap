//! Shared planetary basemap carrier-routing policy and canonical display
//! [`TileKey`] constructors (`PolarCap` -> bundled offline cap; live ->
//! bundled offline fallback).

use std::sync::Arc;

use veldmap_core::tile::{PlanetaryScheme, QuadTileId, SourceId, TileKey, compose_uv_rect};

use crate::gpu_tiles::{GpuTile, GpuTileCache};
use crate::render::TileCarrierPolicy;

/// Pure planetary basemap carrier-routing policy shared by `DrawSnapshot::build`
/// and `BasemapManager::schedule_for_prioritized_tiles`.
///
/// Enforces the explicit `PolarCap` + live-basemap contract:
/// - `PolarCap` tiles always resolve against `SourceId::BUNDLED_BASEMAP` (never
///   a live HTTP source).
/// - Non-polar live tiles fall back to cached `SourceId::BUNDLED_BASEMAP`
///   ancestors while streaming or offline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BasemapCarrierPolicy {
    active_source: SourceId,
    /// Active harmonic rim revision per pole (`[north, south]`, `0` = offline
    /// bundled Blue Marble).
    polar_revs: [u32; 2],
    /// Fallback harmonic rim revision per pole (`[north, south]`) kept warm
    /// while a newly bumped `polar_revs` tile renders, avoiding any 1-frame
    /// visual flash.
    polar_fallback_revs: [u32; 2],
}

impl BasemapCarrierPolicy {
    pub const fn new(active_source: SourceId) -> Self {
        Self {
            active_source,
            polar_revs: [0, 0],
            polar_fallback_revs: [0, 0],
        }
    }

    pub(crate) const fn with_polar_revs(
        active_source: SourceId,
        polar_revs: [u32; 2],
        polar_fallback_revs: [u32; 2],
    ) -> Self {
        Self {
            active_source,
            polar_revs,
            polar_fallback_revs,
        }
    }

    /// Primary [`TileKey`] serving `tex_id` under this basemap policy.
    pub(crate) const fn primary_key(self, tex_id: QuadTileId) -> TileKey {
        match tex_id.scheme {
            PlanetaryScheme::PolarCap { north } => {
                let idx = if north { 0 } else { 1 };
                bundled_polar_tile_key(tex_id, self.polar_revs[idx])
            }
            PlanetaryScheme::Geodetic | PlanetaryScheme::WebMercator => {
                source_tile_key(self.active_source, tex_id)
            }
        }
    }

    /// Optional offline bundled fallback [`TileKey`] when `primary_key(tex_id)`
    /// has no cached carrier in VRAM yet.
    pub(crate) const fn fallback_key(self, tex_id: QuadTileId) -> Option<TileKey> {
        match tex_id.scheme {
            PlanetaryScheme::PolarCap { north } => {
                let idx = if north { 0 } else { 1 };
                let primary_rev = self.polar_revs[idx];
                if primary_rev == 0 {
                    None
                } else {
                    let fb_rev = self.polar_fallback_revs[idx];
                    Some(bundled_polar_tile_key(
                        tex_id,
                        if fb_rev == primary_rev { 0 } else { fb_rev },
                    ))
                }
            }
            PlanetaryScheme::Geodetic | PlanetaryScheme::WebMercator => {
                if self.active_source.is_bundled() {
                    None
                } else {
                    Some(bundled_tile_key(tex_id))
                }
            }
        }
    }
}

impl TileCarrierPolicy for BasemapCarrierPolicy {
    fn resolve_carrier(
        &self,
        tex_id: QuadTileId,
        uv_rect: [f32; 4],
        gpu_cache: &mut GpuTileCache,
    ) -> Option<(TileKey, [f32; 4], Arc<GpuTile>)> {
        let primary = self.primary_key(tex_id);
        if let Some((ck, c_uv, tile)) = gpu_cache.carrier(&primary) {
            return Some((ck, compose_uv_rect(c_uv, uv_rect), tile));
        }
        let fallback = self.fallback_key(tex_id)?;
        if let Some((ck, c_uv, tile)) = gpu_cache.carrier(&fallback) {
            return Some((ck, compose_uv_rect(c_uv, uv_rect), tile));
        }
        if matches!(tex_id.scheme, PlanetaryScheme::PolarCap { .. }) && fallback.decode_rev > 0 {
            let root_fb = bundled_tile_key(tex_id);
            return gpu_cache
                .carrier(&root_fb)
                .map(|(ck, c_uv, tile)| (ck, compose_uv_rect(c_uv, uv_rect), tile));
        }
        None
    }
}

/// Build the canonical display `TileKey` for a tile belonging to `source`.
pub const fn source_tile_key(source: SourceId, id: QuadTileId) -> TileKey {
    TileKey::display_quad(source, id)
}

/// Build the canonical display `TileKey` for a bundled offline basemap tile.
pub const fn bundled_tile_key(id: QuadTileId) -> TileKey {
    TileKey::display_quad(SourceId::BUNDLED_BASEMAP, id)
}

/// Build the display `TileKey` for a bundled/harmonic `PolarCap` tile at
/// `decode_rev`.
pub const fn bundled_polar_tile_key(id: QuadTileId, decode_rev: u32) -> TileKey {
    let mut key = TileKey::display_quad(SourceId::BUNDLED_BASEMAP, id);
    key.decode_rev = decode_rev;
    key
}
