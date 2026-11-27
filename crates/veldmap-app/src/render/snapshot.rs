//! Immutable per-frame draw snapshots decoupled from concrete layer/basemap
//! policy implementations via [`TileCarrierPolicy`].

use std::collections::HashSet;
use std::sync::Arc;

use veldmap_core::geo::{Camera, Frame, VectorMesh};
use veldmap_core::tile::{QuadTileId, QuadtreeConfig, SourceId, TileKey, VisibleTile, traverse};

use crate::colormap::{CategoricalPaletteId, ColormapId};
use crate::gpu_tiles::{GpuTile, GpuTileCache};
use crate::layers::BandMathFormula;

use super::graticule::adaptive_grid_step_deg;

/// Cache identity for an uploaded overlay cell warp-mesh vertex buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OverlayMeshKey {
    pub source: SourceId,
    pub relaid: u64,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

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

/// Single planetary basemap tile draw entry inside an immutable
/// [`DrawSnapshot`].
#[derive(Clone, Debug)]
pub struct DrawTile {
    pub visible: VisibleTile,
    /// Resolved GPU carrier texture (`None` in Ellipsoid-only mode or before
    /// any coarse ancestor tile has uploaded).
    pub texture: Option<Arc<GpuTile>>,
    /// Final UV sub-rectangle `(u_offset, v_offset, u_scale, v_scale)` in
    /// `texture`'s coordinate space.
    pub uv_rect: [f32; 4],
}

/// Single draped raster overlay warp-mesh cell inside [`DrawOverlayLayer`].
///
/// Vertex geometry is built lazily on GPU mesh-cache miss inside
/// `OverlayRenderer::prepare_resources` from the parent layer's [`Frame`] and
/// uniform per-level `segments`.
#[derive(Clone, Debug)]
pub struct DrawOverlayPatch {
    pub mesh_key: OverlayMeshKey,
    /// Pixel rectangle `[x0, y0, x1, y1]` in base-image coordinates.
    pub cell_rect: [f64; 4],
    /// Carrier UV sub-rectangle `(u_offset, v_offset, u_scale, v_scale)`.
    pub uv_rect: [f32; 4],
    pub texture: Arc<GpuTile>,
}

/// Per-layer rendering track configuration for [`DrawOverlayLayer`].
#[derive(Clone, Debug)]
pub enum DrawOverlayTrack {
    /// Pre-stretched 8-bit `Rgba8UnormSrgb` display composite.
    Display,
    /// Raw `R32Float` scientific data rendered with 8-tap nodata-aware
    /// trilinear filtering, canonical `val * scale + offset` physical scaling,
    /// live linear or `log10` dB stretch, and a 256-entry colormap LUT.
    Scientific {
        scale: f32,
        offset: f32,
        stretch_min: f32,
        stretch_max: f32,
        log10_db: bool,
        colormap: ColormapId,
        invert_colormap: bool,
    },
    /// Discrete `R8Uint` / `R16Uint` categorical class or QA bitmask layer
    /// rendered with exact nearest-neighbor sampling and a class palette LUT.
    Categorical {
        source: SourceId,
        palette: CategoricalPaletteId,
        custom_color_map: Option<Arc<[[u8; 4]]>>,
        nodata_value: Option<u32>,
    },
}

/// Single float input band rasterized into a viewport-sized `R32Float` scratch
/// target for [`DrawBandMathLayer`].
#[derive(Clone, Debug)]
pub struct DrawScratchBand {
    pub frame: Arc<Frame>,
    pub image_width: u32,
    pub image_height: u32,
    pub segments: u32,
    pub scale: f32,
    pub offset: f32,
    pub patches: Vec<DrawOverlayPatch>,
}

/// Optional discrete `R8Uint` QA / `SCL` mask rasterized nearest-neighbor into
/// the viewport-sized `R8Uint` scratch target for [`DrawBandMathLayer`].
#[derive(Clone, Debug)]
pub struct DrawScratchMask {
    pub frame: Arc<Frame>,
    pub image_width: u32,
    pub image_height: u32,
    pub segments: u32,
    pub nodata_value: Option<u32>,
    pub patches: Vec<DrawOverlayPatch>,
}

/// Multi-band screen-space float band-math & RGB composite draw payload.
#[derive(Clone, Debug)]
pub struct DrawBandMathLayer {
    pub formula: BandMathFormula,
    pub colormap: ColormapId,
    pub invert_colormap: bool,
    pub stretch_min: f32,
    pub stretch_max: f32,
    pub rgb_min: [f32; 3],
    pub rgb_max: [f32; 3],
    pub rgb_gamma: f32,
    pub scl_mask_enabled: bool,
    pub scl_mask_bits: u32,
    pub bands: Vec<DrawScratchBand>,
    pub scl: Option<DrawScratchMask>,
}

fn frames_share_exact_grid(a: &Arc<Frame>, b: &Arc<Frame>) -> bool {
    if Arc::ptr_eq(a, b) {
        return true;
    }
    const EPS_DEG: f64 = 1e-10;
    for (fx, fy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
        let (la, loa) = a.geodetic(fx, fy);
        let (lb, lob) = b.geodetic(fx, fy);
        if (la - lb).abs() > EPS_DEG || (loa - lob).abs() > EPS_DEG {
            return false;
        }
    }
    true
}

impl DrawBandMathLayer {
    /// Return whether all active float bands share one geodetic grid, image
    /// dimensions, subdivision density, and identical patch `cell_rect` list
    /// so they can be rasterized in a single MRT render pass (`2..=4` `R32Float`
    /// attachments, fitting `downlevel_defaults()` `max_color_attachments = 4`).
    pub fn can_use_mrt(&self) -> bool {
        if self.bands.len() < 2 || self.bands.len() > 4 {
            return false;
        }
        let first = &self.bands[0];
        if first.patches.is_empty() {
            return false;
        }
        self.bands[1..].iter().all(|band| {
            first.image_width == band.image_width
                && first.image_height == band.image_height
                && first.segments == band.segments
                && first.patches.len() == band.patches.len()
                && frames_share_exact_grid(&first.frame, &band.frame)
                && first
                    .patches
                    .iter()
                    .zip(&band.patches)
                    .all(|(p0, p1)| p0.cell_rect == p1.cell_rect)
        })
    }
}

/// Direct single-asset or multi-asset scratch-backed overlay draw content.
#[derive(Clone, Debug)]
pub enum DrawOverlayContent {
    Direct {
        track: DrawOverlayTrack,
        frame: Arc<Frame>,
        image_width: u32,
        image_height: u32,
        segments: u32,
        patches: Vec<DrawOverlayPatch>,
    },
    BandMath(Box<DrawBandMathLayer>),
}

/// Ordered draped raster overlay layer in painter's order (bottom-to-top).
#[derive(Clone, Debug)]
pub struct DrawOverlayLayer {
    pub opacity: f32,
    pub content: DrawOverlayContent,
}

impl DrawOverlayLayer {
    /// Return `(float_band_count, needs_u8_mask)` required in the shared
    /// viewport scratch target pool for this layer (`(0, false)` for direct
    /// overlays).
    pub(crate) fn scratch_requirements(&self) -> (usize, bool) {
        match &self.content {
            DrawOverlayContent::Direct { .. } => (0, false),
            DrawOverlayContent::BandMath(bm) => {
                let float_bands = bm.bands.len().min(4);
                let needs_mask = bm
                    .scl
                    .as_ref()
                    .is_some_and(|m| bm.scl_mask_enabled && !m.patches.is_empty());
                (float_bands, needs_mask)
            }
        }
    }
}

/// Typed GPU buffer cache key for [`DrawVectorLayer`], keeping loaded vector
/// layer IDs strictly separate from transient overlays such as interactive
/// geodesic measurements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrawVectorId {
    Layer(u64),
    Measurement,
    StacAoi,
    StacFootprints,
    StacHighlight,
}

/// Draped GeoJSON / measurement vector layer drawn in Pass 2 above raster
/// overlays and below the procedural graticule.
#[derive(Clone, Debug)]
pub struct DrawVectorLayer {
    pub id: DrawVectorId,
    pub rev: u64,
    pub opacity: f32,
    /// Polygon fill color as **sRGB RGB (`0.0..=1.0`) + linear alpha (`0.0..=1.0`)**.
    /// RGB channels are converted to linear light once per layer on CPU in
    /// `VectorRenderer::push_uniforms` before writing to the `Rgba8UnormSrgb`
    /// offscreen target.
    pub fill_color: [f32; 4],
    /// Stroke / point-marker color as **sRGB RGB (`0.0..=1.0`) + linear alpha (`0.0..=1.0`)**.
    /// RGB channels are converted to linear light once per layer on CPU in
    /// `VectorRenderer::push_uniforms` before writing to the `Rgba8UnormSrgb`
    /// offscreen target.
    pub line_color: [f32; 4],
    /// Logical stroke width in points (scaled by `pixels_per_point` on CPU in
    /// `VectorRenderer::push_uniforms` for crisp HiDPI screen-space ribbon outlines).
    pub stroke_width_px: f32,
    /// Optional diagonal screen-space hatch period in logical points (`0.0` =
    /// solid polygon fill; `> 0.0` = hatched fill for AOI / selected footprints).
    pub hatch_spacing_px: f32,
    pub mesh: Arc<VectorMesh>,
}

/// Immutable per-frame draw snapshot built on the app thread (cached by
/// `(camera_key, scene_generation)` so 120 Hz mouse hover never rebuilds tile
/// lists) and handed to `GlobePaintCallback`.
#[derive(Clone, Debug)]
pub struct DrawSnapshot {
    pub camera: Camera,
    pub tiles: Vec<DrawTile>,
    /// Ordered post-body draped raster overlays rendered bottom-to-top inside
    /// Pass 2 (`depth_stencil_attachment: None`) before the procedural
    /// graticule.
    pub overlays: Vec<DrawOverlayLayer>,
    /// Ordered draped vector overlays (GeoJSON + active geodesic measurements)
    /// rendered in Pass 2 above raster overlays and below the procedural
    /// graticule.
    pub vectors: Vec<DrawVectorLayer>,
    pub grid_cells: u32,
    pub show_graticule: bool,
    pub grid_step_deg: f64,
    pub grid_width_px: f32,
    pub pixels_per_point: f32,
}

impl DrawSnapshot {
    /// Iterator over the texture tile IDs needed by the basemap quadtree.
    pub(crate) fn visible_tex_ids(&self) -> impl Iterator<Item = QuadTileId> + '_ {
        self.tiles.iter().map(|t| t.visible.tex_id)
    }

    /// Build a draw snapshot and its referenced GPU carrier tile keys from
    /// grouped viewport, camera, graticule, and quadtree parameters plus an
    /// optional ([`TileCarrierPolicy`], [`GpuTileCache`]) pair (`Some` for
    /// textured rendering, `None` for untextured ellipsoid rendering).
    pub fn build(
        params: SnapshotBuildParams<'_>,
        mut carrier: Option<(&dyn TileCarrierPolicy, &mut GpuTileCache)>,
    ) -> (Self, HashSet<TileKey>) {
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

        let snapshot = Self {
            camera: params.camera,
            tiles,
            overlays: Vec::new(),
            vectors: Vec::new(),
            grid_cells: params.config.grid_cells.max(1),
            show_graticule: params.show_graticule,
            grid_step_deg,
            grid_width_px,
            pixels_per_point: params.pixels_per_point.max(0.5),
        };
        (snapshot, carrier_keys)
    }
}
