//! Immutable per-frame draw snapshots decoupled from concrete layer/basemap
//! policy implementations via [`TileCarrierPolicy`].

use std::collections::HashSet;
use std::sync::Arc;

use veldmap_core::geo::camera::Camera;
use veldmap_core::tile::{QuadTileId, QuadtreeConfig, TileKey, VisibleTile, traverse};

use crate::gpu_tiles::{GpuTile, GpuTileCache};

use super::adaptive_grid_step_deg;

/// Strategy resolving the finest cached GPU carrier tile and composed UV
/// sub-rectangle for a visible quadtree texture tile.
pub trait TileCarrierPolicy {
    fn resolve_carrier(
        &self,
        tex_id: QuadTileId,
        uv_rect: [f32; 4],
        gpu_cache: &mut GpuTileCache,
    ) -> Option<(TileKey, [f32; 4], Arc<GpuTile>)>;
}

/// Grouped inputs for [`DrawSnapshot::build`].
#[derive(Clone, Copy)]
pub struct SnapshotBuildParams<'a> {
    pub camera: Camera,
    pub width_px: u32,
    pub height_px: u32,
    pub pixels_per_point: f32,
    pub show_graticule: bool,
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

/// Immutable per-frame draw snapshot built on the app thread (cached by
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
}

impl DrawSnapshot {
    /// Iterator over the texture tile IDs needed by this snapshot.
    pub fn visible_tex_ids(&self) -> impl Iterator<Item = QuadTileId> + '_ {
        self.tiles.iter().map(|t| t.visible.tex_id)
    }

    /// Build a draw snapshot from grouped viewport, camera, graticule, and
    /// quadtree parameters plus an optional ([`TileCarrierPolicy`],
    /// [`GpuTileCache`]) pair (`Some` for textured rendering, `None` for
    /// untextured ellipsoid rendering).
    pub fn build(
        params: SnapshotBuildParams<'_>,
        mut carrier: Option<(&dyn TileCarrierPolicy, &mut GpuTileCache)>,
    ) -> Self {
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
        for vis in visible {
            let resolved = carrier.as_mut().and_then(|(policy, gpu_cache)| {
                policy.resolve_carrier(vis.tex_id, vis.uv_rect, gpu_cache)
            });
            let (texture, uv_rect) = match resolved {
                Some((carrier_key, uv_rect, gpu_tile)) => {
                    carrier_keys.insert(carrier_key);
                    (Some(gpu_tile), uv_rect)
                }
                None => (None, vis.uv_rect),
            };
            tiles.push(DrawTile {
                visible: vis,
                texture,
                uv_rect,
            });
        }

        Self {
            camera: params.camera,
            tiles,
            carrier_keys,
            grid_cells: params.config.grid_cells.max(1),
            show_graticule: params.show_graticule,
            grid_step_deg,
            grid_width_px,
        }
    }
}
