# 0002 — Planetary Globe Model, Reverse-Z Precision, and Camera Floor

- **Status:** Accepted (supersedes legacy `0006` and the 800 m camera floor in legacy `0007`)
- **Date:** 2026-11-21
- **Scope:** Committed v1 WGS84 ellipsoid model, double-single RTE / per-tile RTC coordinate precision, infinite Reverse-Z depth, depth-less draped lines with ellipsoidal horizon discard, exact sRGB compositing, and overlay accuracy tolerances.

## Context

VeldMap serves both planetary Earth observation (Sentinel-2, Landsat, global mosaics) and precision local imagery down to 2 cm drone orthophotos and RTK field boundaries. Two foundational questions govern the 3D globe:

1. **Should the globe mesh displace vertices with a 3D DEM or allow camera tilt?**
   Orthorectified optical satellite imagery (Sentinel-2 L1C/L2A) and local drone orthophotos already place each pixel at its true geodetic latitude and longitude. Introducing 3D terrain mesh displacement or oblique camera tilt introduces view-dependent parallax across layers, complicating orthophoto alignment, split-screen swipe, and temporal diffing.
2. **Why did the legacy renderer enforce an 800 m camera floor?**
   The legacy globe drew an inscribed 48×96 ellipsoid into a standard `0..1` depth buffer and depth-tested overlays, outlines, and the graticule against it. Every draped primitive had to float on an 80 m `SURFACE_LIFT_M` to avoid dipping beneath the ellipsoid chord sag, forcing a `10 × SURFACE_LIFT_M = 800 m` camera floor (~0.49 m/px on a 1536-row viewport—24× coarser than a 2 cm drone orthophoto).

## Decision

### 1. Smooth Nadir WGS84 Ellipsoid (`height = 0.0`)

- VeldMap v1 renders a smooth WGS84 ellipsoid directly at `height = 0.0` with a strictly nadir camera (zero pitch/tilt).
- Terrain relief is delivered as quantitative GPU layers (hillshade, slope, aspect over DEM COGs in Slice 5B-1) rather than 3D mesh displacement.
- `Geodetic.height_m`, `DMat4` infinite Reverse-Z, and tile LOD remain intact so any future v2 3D terrain mode can layer cleanly on top.

### 2. Camera Floor Lowered to `MIN_HEIGHT_M = 2.0` m with `f64` Infinite Reverse-Z

- `Camera` computes view-projection matrices in `f64` (`DMat4`) using right-handed infinite Reverse-Z (`near -> 1.0`, `infinity -> 0.0`) with `near = (height_m * 0.5) / SEMI_MAJOR_M` and `Depth32Float` (`CompareFunction::GreaterEqual`, clear `0.0`).
- Camera internals (`at`, `up`, `height_m`, `flight`) are private behind view-agnostic queries (`center()`, `height_m()`, `heading_deg()`, `eye_ecef()`, `view_projection()`, `project()`, `frustum()`, `probe()`, `metres_per_pixel()`).
- `Camera::project()`, `cull::Frame`, and `cull::Ball` operate strictly in `f64` (`DVec3`/`DMat4`). `cull::Frame` uses the exact oblate WGS84 ellipsoidal tangent horizon plane `p · (A · eye) >= 1` (`A = diag(1, 1, 1 / (1 - f)^2)`), eliminating false poleward horizon culling.
- `MIN_HEIGHT_M = 2.0` m yields ~1.87 mm/px on a 1000-row viewport at 50° vertical FOV—more than 10× finer than 2 cm drone orthophotos.

### 3. Double-Single RTE Fallback Mesh, Per-Tile RTC Quadtree, and Depth-Less Draped Lines

- The untextured WGS84 ellipsoid fallback mesh (96×192 watertight grid) and draped 15° graticule store `(pos_hi, pos_lo)` double-single ECEF coordinates, paired with `(-eye_ecef)` split into `(hi, lo)` `f32` halves in `CameraUniform`. Under non-reassociating IEEE-754 arithmetic, `(pos_hi + neg_eye_hi) + (pos_lo + neg_eye_lo)` preserves sub-millimetre precision at 2 m altitude in a single watertight draw call.
- Starting in Slice 2A, the static `96×192` fallback mesh is retired in favor of **per-tile RTC** (`tile_origin - eye` in `f64` on CPU) quadtree basemap, ellipsoid-only, and raster warp-mesh tiles—eliminating both GPU fast-math reassociation hazards across backends (e.g. Metal/DXC) and the coarse static mesh's ~1.7 km chord sag, alongside upgrading the graticule for close zoom (eliminating the static 672-segment parallel's ~35 m horizontal chord parallax).
- Draped graticule primitives (and scene outlines/ribbons/hatches when wired in Slice 4A) sit directly on `height = 0.0` and render depth-less (`depth_write_enabled: false`, `depth_compare: Always`), eliminating Z-fighting and the need for `SURFACE_LIFT_M`. Back-hemisphere fragments are rejected cleanly in the fragment shader via exact ellipsoidal-normal horizon discard: `dot(N_ellipsoid(P), normalize(E - P)) > 0.0`.

### 4. Exact 1:1 sRGB Offscreen Compositing

- The offscreen globe target uses `Rgba8UnormSrgb` with `view_formats: &[Rgba8Unorm]` and `Depth32Float`. All shader palette constants are exact IEC 61966-2-1 linear RGB values with zero manual `encode_srgb` shader hacks.
- Compositing into `egui_wgpu` uses `FilterMode::Nearest` with a texture view matching the swapchain's color space (`Rgba8UnormSrgb` for sRGB and linear float swapchains, `Rgba8Unorm` for non-sRGB 8-bit swapchains), preserving exact sRGB bytes on all backends (Vulkan, DX12, Metal).

### 5. Authoritative Overlay Accuracy Contract (Retained from Legacy ADR `0007`)

Raster warp-mesh placement (implemented in Slice 3A-1 via `a_raster_lands_where_its_binding_puts_it`) must hold across all 5 benchmark bindings in UTM zone 37 (EPSG:32637):

1. Drone orthophoto at **2 cm/px**
2. Sentinel-2 granule at **10 m/px**
3. Landsat scene at **30 m/px**
4. Sentinel-2 granule at **pyramid level 4**
5. **8000×5000** orthophoto **rotated by 30°**

Surface drift tolerances (including `f32` vertex splitting):
- `< 10⁻³` of the pixel size of the drawn pyramid level
- `< 2 mm` for any raster drawn at its native resolution
- `< 2 cm` at any pyramid level

## Consequences

- Centimetre-scale orthophotos can be inspected down to 2 m altitude with zero vertex jitter, zero depth fighting, zero horizon bleed, and zero surface-lift parallax.
- Both CPU-only `naga` validation (SPIR-V, HLSL SM 5.1, MSL 2.1) and headless `wgpu` smoke tests verify Reverse-Z close-up rendering directly above a mesh vertex at 2 m altitude and exact sRGB/UNORM compositing parity on every CI run.
