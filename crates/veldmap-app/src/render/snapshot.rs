//! Immutable per-frame draw snapshots and shared basemap carrier-resolution
//! policy (`PolarCap` -> bundled offline cap; live -> bundled offline fallback).

use std::collections::HashSet;
use std::sync::Arc;

use veldmap_core::geo::camera::Camera;
use veldmap_core::tile::{
    PlanetaryScheme, QuadTileId, QuadtreeConfig, SourceId, TileKey, VisibleTile, compose_uv_rect,
    traverse,
};

use crate::gpu_tiles::{GpuTile, GpuTileCache, bundled_tile_key, source_tile_key};

use super::adaptive_grid_step_deg;

/// Pure planetary basemap carrier-routing policy shared by [`DrawSnapshot::build`]
/// and `BasemapManager::schedule_for_tiles`.
///
/// Enforces the explicit `PolarCap` + live-basemap contract:
/// - `PolarCap` tiles always resolve against `SourceId::BUNDLED_BASEMAP` (never
///   a live HTTP source).
/// - Non-polar live tiles fall back to cached `SourceId::BUNDLED_BASEMAP`
///   ancestors while streaming or offline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BasemapCarrierPolicy {
    pub active_source: SourceId,
}

impl BasemapCarrierPolicy {
    pub const fn new(active_source: SourceId) -> Self {
        Self { active_source }
    }

    /// Primary [`TileKey`] serving `tex_id` under this basemap policy.
    pub const fn primary_key(self, tex_id: QuadTileId) -> TileKey {
        if matches!(tex_id.scheme, PlanetaryScheme::PolarCap { .. }) {
            bundled_tile_key(tex_id)
        } else {
            source_tile_key(self.active_source, tex_id)
        }
    }

    /// Optional offline bundled fallback [`TileKey`] when `primary_key(tex_id)`
    /// has no cached carrier in VRAM yet.
    pub const fn fallback_key(self, tex_id: QuadTileId) -> Option<TileKey> {
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

/// Grouped inputs for [`DrawSnapshot::build`].
#[derive(Clone, Copy, Debug)]
pub struct SnapshotBuildParams<'a> {
    pub camera: Camera,
    pub width_px: u32,
    pub height_px: u32,
    pub pixels_per_point: f32,
    pub show_graticule: bool,
    pub textured: bool,
    pub carrier_policy: BasemapCarrierPolicy,
    pub generation: u64,
    pub config: &'a QuadtreeConfig,
}

/// Single tile draw entry inside an immutable [`DrawSnapshot`].
#[derive(Clone, Debug)]
pub struct DrawTile {
    pub visible: VisibleTile,
    /// Resolved GPU carrier texture (`None` in Ellipsoid-only mode or before
    /// any coarse ancestor tile has uploaded).
    pub texture: Option<Arc<GpuTile>>,
    /// Final UV sub-rectangle in `texture`'s coordinate space.
    pub uv_rect: [f32; 4],
}

/// Immutable per-frame draw snapshot built on `VeldmapApp` (cached by
/// `(camera_key, scene_generation)` so 120 Hz mouse hover never rebuilds tile
/// lists) and handed to `GlobePaintCallback`.
#[derive(Clone, Debug)]
pub struct DrawSnapshot {
    pub camera: Camera,
    pub tiles: Vec<DrawTile>,
    /// Exact GPU carrier tile keys referenced by `tiles`, used for active-frame
    /// `TileCache` protection so uploads never evict currently visible tiles.
    pub carrier_keys: HashSet<TileKey>,
    pub grid_cells: u32,
    pub show_graticule: bool,
    pub grid_step_deg: f64,
    pub grid_width_px: f32,
    pub generation: u64,
}

impl DrawSnapshot {
    /// Iterator over the texture tile IDs needed by this snapshot.
    pub fn visible_tex_ids(&self) -> impl Iterator<Item = QuadTileId> + '_ {
        self.tiles.iter().map(|t| t.visible.tex_id)
    }

    /// Build a draw snapshot from grouped viewport, camera, graticule, and
    /// carrier-policy parameters plus an optional [`GpuTileCache`].
    pub fn build(params: SnapshotBuildParams<'_>, cache: Option<&mut GpuTileCache>) -> Self {
        let visible = traverse(
            &params.camera,
            params.width_px,
            params.height_px,
            params.config,
        );
        let ppp = f64::from(params.pixels_per_point.max(0.5));
        let height_pt = (f64::from(params.height_px.max(1)) / ppp).max(1.0);
        let m_per_pt = params.camera.metres_per_pixel(height_pt);
        let grid_step_deg = adaptive_grid_step_deg(m_per_pt);
        let grid_width_px = (0.75 * params.pixels_per_point.max(1.0)).clamp(0.65, 3.0);

        let mut tiles = Vec::with_capacity(visible.len());
        let mut carrier_keys = HashSet::new();
        match (params.textured, cache) {
            (true, Some(gpu_cache)) => {
                for vis in visible {
                    if let Some((carrier_key, uv_rect, gpu_tile)) = params
                        .carrier_policy
                        .resolve_carrier(vis.tex_id, vis.uv_rect, gpu_cache)
                    {
                        carrier_keys.insert(carrier_key);
                        tiles.push(DrawTile {
                            visible: vis,
                            texture: Some(gpu_tile),
                            uv_rect,
                        });
                    } else {
                        tiles.push(DrawTile {
                            visible: vis,
                            texture: None,
                            uv_rect: vis.uv_rect,
                        });
                    }
                }
            }
            _ => {
                for vis in visible {
                    tiles.push(DrawTile {
                        visible: vis,
                        texture: None,
                        uv_rect: vis.uv_rect,
                    });
                }
            }
        }

        Self {
            camera: params.camera,
            tiles,
            carrier_keys,
            grid_cells: params.config.grid_cells.max(1),
            show_graticule: params.show_graticule,
            grid_step_deg,
            grid_width_px,
            generation: params.generation,
        }
    }
}
