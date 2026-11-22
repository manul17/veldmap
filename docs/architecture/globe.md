# 3D WGS84 Globe Architecture

This document describes the coordinate system, camera model, culling math, offscreen rendering pipeline, and baseline GIS HUD implemented in `veldmap-core::geo` and `veldmap-app`. Authoritative precision and accuracy contracts are recorded in [ADR 0002](../decisions/0002-globe-model-and-precision.md).

## Coordinates & Projections (`veldmap-core::geo`)

- **WGS84 ECEF (`geodesy.rs`):** All Earth geometry passes through `geodesy::world`, converting geodetic `(lat_deg, lon_deg, height_m)` to right-handed ECEF (`X` at 0°N 0°E, `Y` at 0°N 90°E, `Z` at 90°N) scaled uniformly in equatorial radii (`1.0 = SEMI_MAJOR_M = 6,378,137 m`). Flattening (`1 / 298.257223563`) is preserved everywhere; polar radius is ~21.4 km shorter than equatorial radius.
- **Projections & Datums (`projection.rs`):** Pure-Rust Krueger transverse cylindrical series (4th order in third flattening `n`) with 7-parameter Helmert datum shifts (closed-form exact inverse) for UTM (`32601..=32660`, `32701..=32760`), Gauss-Krüger on Pulkovo 1942 (`28401..=28432`), MSK-61 (`6336101..=6336103`), plus spherical Web Mercator (`3857`). Only WGS84 coordinates cross module boundaries.
- **MGRS & Location Parsing (`mgrs.rs`, `coords.rs`):** Pure-Rust MGRS 100 km grid square and Sentinel-2 granule frame calculation (`tile_of`, `frame`, `parse_to_wgs84`) plus unified Go-To parsing (`parse_location`) supporting Decimal Degrees (DD), Degrees-Minutes-Seconds (DM/DMS, both prefix and suffix hemispheres), and MGRS codes.
- **Footprints & Outlines (`footprint.rs`, `outlines.rs`, `sag.rs`):** Unit-vector azimuth winding hit-testing and surface outline tessellation sharing exact parallel/meridian/great-circle edge rules and global-band seam cutting (ready for scene outline pipelines in Slice 4A).

## Nadir Camera & Culling (`camera.rs`, `cull.rs`)

- **Orthonormal Frame:** `Camera` stores private unit vectors `(at, up)` and ellipsoidal altitude `height_m` clamped to `[2.0 m, 80,000,000 m]`. Latitude/longitude and clockwise heading from North (`heading_deg`) are derived on demand so orbiting smoothly crosses the poles without gimbal lock.
- **Controls:**
  - Left-drag (`orbit`) performs a single quaternion rotation proportional to the visible vertical arc (`2 * visible_deg()`), so ground follows the cursor identically at any latitude and altitude.
  - Scroll-wheel zoom (`zoom_at`) holds the ground point under the cursor with two-pass tangent-plane residual correction and symmetric grazing-angle grip softening near the limb.
  - `focus()`, `fly_to()`, and `reset_north()` run smooth 0.6 s quaternion `slerp` + geometric-mean altitude flights (`advance(dt_s)`).
- **Infinite Reverse-Z & `f64` Culling:** `Camera::view_projection(aspect)` builds a rotation-only right-handed infinite Reverse-Z `DMat4` (`near = height_m * 0.5 / SEMI_MAJOR_M -> z = 1.0`, `infinity -> z = 0.0`). `Camera::metres_per_pixel(height_px)` computes exact center vertical GSD (`2 * height_m * tan(fov_y / 2) / height_px`). `cull::Frame` extracts the four side frustum planes and the exact oblate WGS84 ellipsoidal tangent horizon plane (`p · (A · eye) >= 1`) in `f64` (`DVec3`) for fast zero-transcendental `cull::Ball` rejection.

## Offscreen `wgpu` Renderer (`veldmap-app`)

- **Watertight Double-Single RTE Ellipsoid (`mesh.rs`):** The untextured WGS84 ellipsoid fallback is a single watertight `96 × 192` grid (`1.875°` cells). Vertices store `(pos_hi, pos_lo)` of `P_ecef`; each dirty frame uploads `view_proj` and `(-eye_ecef)` split into `(hi, lo)` in a single `CameraUniform` with zero heap allocation. In `globe.wgsl`, `(pos_hi + neg_eye_hi) + (pos_lo + neg_eye_lo)` cancels the coarse halves (Sterbenz's lemma) for sub-millimetre stability at 2 m altitude in a single draw call.
- **Depth-Less Draped Graticule (`globe.wgsl`):** The 15° graticule (672 segments per line) sits directly at `height_m = 0.0` and renders with `depth_write_enabled: false`, `depth_compare: Always`, and fragment ellipsoidal-normal horizon discard (`dot(N_ellipsoid(P), normalize(E - P)) > 0.0`) with smooth limb fade.
- **Offscreen Sizing & Exact sRGB Compositing (`render.rs`, `composite.wgsl`):**
  - `CallbackTrait::prepare()` computes physical pixel dimensions via `ViewportInPixels::from_points()`, skips `0×0`, clamps to `[1, max_texture_dimension_2d]`, and skips offscreen re-rendering whenever `(width, height, camera pose, show_graticule, scene_generation)` is unchanged and the previous offscreen render reached `composite_into()` (automatically recovering if `egui_wgpu` drops its command encoder on swapchain `Outdated`/`Lost`/`Occluded`).
  - The offscreen `Rgba8UnormSrgb` color buffer (rendered with exact IEC 61966-2-1 linear RGB palette constants, with `StoreOp::Discard` on `Depth32Float`) is composited 1:1 with `FilterMode::Nearest` onto `egui_wgpu`'s target format using `view_formats` (`Rgba8UnormSrgb` view for sRGB/linear targets, `Rgba8Unorm` view for non-sRGB 8-bit targets).

## Baseline GIS HUD (`hud.rs`, `app.rs`)

- **Top Toolbar:** Go-To input accepting DD, DMS, or MGRS (supporting standard clipboard paste and inline error feedback), Graticule toggle, and truncated active GPU adapter/backend label.
- **Viewport Overlays:** Dynamic `1–2–5` metric scale bar (`mm` to `km`, bottom-left, clamped to viewport width and hidden at extreme orbital zoom) and interactive rotating North compass dial hosted in a foreground `egui::Area` (top-right, click or press `N` without modifiers to reset North). Supports scroll-wheel and trackpad pinch / `Ctrl+scroll` zoom holding the ground point under the cursor.
- **Status Bar:** Live cursor Lat/Lon (`Camera::probe`), camera center Lat/Lon, ellipsoidal altitude, and center ground sample distance (`mm/px` to `km/px`), wrapping cleanly down to the 640 px minimum window width and updating hover readouts without triggering extra repaints.
