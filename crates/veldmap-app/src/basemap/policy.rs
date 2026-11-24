//! Shared planetary basemap carrier-routing policy and canonical display
//! [`TileKey`] constructors (`PolarCap` -> bundled offline cap; live ->
//! bundled offline fallback).

use std::sync::Arc;

use veldmap_core::tile::{PlanetaryScheme, QuadTileId, SourceId, TileKey, compose_uv_rect};

use crate::gpu_tiles::{GpuTile, GpuTileCache};
use crate::render::TileCarrierPolicy;

/// Pure planetary basemap carrier-routing policy shared by `DrawSnapshot::build`
/// and `BasemapManager::schedule_for_tiles`.
///
/// Enforces the explicit `PolarCap` + live-basemap contract:
/// - `PolarCap` tiles always resolve against `SourceId::BUNDLED_BASEMAP` (never
///   a live HTTP source).
/// - Non-polar live tiles fall back to cached `SourceId::BUNDLED_BASEMAP`
///   ancestors while streaming or offline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BasemapCarrierPolicy {
    active_source: SourceId,
}

impl BasemapCarrierPolicy {
    pub const fn new(active_source: SourceId) -> Self {
        Self { active_source }
    }

    /// Primary [`TileKey`] serving `tex_id` under this basemap policy.
    pub(crate) const fn primary_key(self, tex_id: QuadTileId) -> TileKey {
        if matches!(tex_id.scheme, PlanetaryScheme::PolarCap { .. }) {
            bundled_tile_key(tex_id)
        } else {
            source_tile_key(self.active_source, tex_id)
        }
    }

    /// Optional offline bundled fallback [`TileKey`] when `primary_key(tex_id)`
    /// has no cached carrier in VRAM yet.
    pub(crate) const fn fallback_key(self, tex_id: QuadTileId) -> Option<TileKey> {
        if matches!(tex_id.scheme, PlanetaryScheme::PolarCap { .. })
            || self.active_source.0 == SourceId::BUNDLED_BASEMAP.0
        {
            None
        } else {
            Some(bundled_tile_key(tex_id))
        }
    }

    /// Resolve the finest cached GPU carrier tile and composed UV sub-rectangle
    /// for `(tex_id, uv_rect)`.
    pub fn resolve_carrier(
        self,
        tex_id: QuadTileId,
        uv_rect: [f32; 4],
        gpu_cache: &mut GpuTileCache,
    ) -> Option<(TileKey, [f32; 4], Arc<GpuTile>)> {
        let primary = self.primary_key(tex_id);
        if let Some((ck, c_uv, tile)) = gpu_cache.carrier(&primary) {
            return Some((ck, compose_uv_rect(c_uv, uv_rect), tile));
        }
        let fallback = self.fallback_key(tex_id)?;
        gpu_cache
            .carrier(&fallback)
            .map(|(ck, c_uv, tile)| (ck, compose_uv_rect(c_uv, uv_rect), tile))
    }
}

impl TileCarrierPolicy for BasemapCarrierPolicy {
    fn resolve_carrier(
        &self,
        tex_id: QuadTileId,
        uv_rect: [f32; 4],
        gpu_cache: &mut GpuTileCache,
    ) -> Option<(TileKey, [f32; 4], Arc<GpuTile>)> {
        BasemapCarrierPolicy::resolve_carrier(*self, tex_id, uv_rect, gpu_cache)
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
