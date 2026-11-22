# Design Decisions

Concise record of **implemented** architectural decisions—focusing on *why alternatives were rejected* and *hard cross-slice invariants* that cannot be inferred from reading the code alone.

> **Policy:** Record a decision here only once it is implemented and proven in code (or when preserving a hard numerical/safety invariant). Unbuilt future slices live as working hypotheses in [plan.md](../plan.md), not pre-accepted contracts. Module implementation details live in Rust `//!` docstrings (`cargo doc`).

---

## 1. Native Rust Workspace & UI/GPU-Free Domain Core

- **Why not WASI / Wasmtime / protobuf IPC:** VeldMap's legacy prototype routed components through a WASI/Wasmtime event bus with protobuf serialization, adding IPC overhead, `!Send` constraints, and 32-bit guest memory limits with no desktop benefit. `wasmtime`, `prost`, and `tonic` are banned in [deny.toml](../deny.toml) to prevent regressions.
- **Why keep `veldmap-core` free of UI/GPU crates:** Keeping `egui`, `eframe`, and `wgpu` out of `veldmap-core` ensures fast headless builds/tests without graphics drivers or windowing libraries and preserves reuse in future companions.
- **Workspace crate growth:** Split workspace crates along real dependency seams (`veldmap-core` pure leaf, `veldmap-raster` landed in Slice 2A, with `veldmap-net` and `veldmap-storage` landing in Slice 2B) rather than simulating crate boundaries via feature flags.

## 2. Dependency Hygiene & Safety Defaults

- **No `memmap2` for user files (banned in [deny.toml](../deny.toml)):** Memory-mapped I/O in Rust risks `SIGBUS` UB on truncated, removable, or network-backed files. Positioned I/O (`pread`/`seek_read`) with caller buffers is safe, portable, and mirrors HTTP Range reads. (Wayland SHM wrappers `smithay-client-toolkit` and `winit` remain exempted.)
- **Pure-Rust TLS (`rustls`), no OpenSSL or heavy ORMs:** `openssl`/`openssl-sys` and `sqlx` are banned in [deny.toml](../deny.toml) to avoid cross-OS TLS linking issues and heavyweight ORM plumbing.
- **Pure-Rust default build without permanent ideological bans:** Default desktop builds avoid C geospatial dependencies (`gdal`, `proj`) and embedded C databases (`rusqlite`) for simple 3-OS compilation, without banning them permanently if real workloads (e.g. grid-shifted datums/WKT2 or high-file-count tile indexing) justify an explicit or optional integration later.

## 3. Smooth WGS84 Ellipsoid (`height = 0.0`), Reverse-Z & 2 m Floor

- **Why no 3D DEM mesh displacement in v1:** Orthorectified optical satellite imagery and cm-scale drone orthophotos place pixels at true geodetic `(lat, lon)`. Draping imagery onto a mismatched DEM (e.g., 30 m COP-DEM over a cm-scale drone orthophoto built on its own DSM) shifts pixels horizontally by `Δh × tan(θ)` (orders of magnitude above `< 2 mm`), introduces view-dependent relief parallax, conflicts with zero-resampling native-CRS warp meshes, and requires terrain-ray intersections, elevated shadow-cone horizon culling, depth-tested vector draping, and EGM2008 vertical-datum conversion. A flat, map-consistent ellipsoid is therefore the right **analytical default** (swipe, diff, readout, 2D/3D sync); terrain relief will be delivered as quantitative GPU layers (hillshade, slope, aspect; Slice 5B-1), and any future v2 3D terrain relief belongs in a separate view mode rather than replacing the flat surface.
- **Current nadir camera:** VeldMap v1 uses a nadir camera (`Camera`) for uniform center GSD and simple controls; zero camera tilt is a v1 UX simplification (not physically required to avoid relief parallax on a flat `height = 0.0` surface). Because v1 `Camera`, `cull::Frame`, `geodesy::surface_at`, and depth-less draped primitives optimize for `height_m = 0.0`, any future v2 3D terrain or oblique mode will extend or replace those modules rather than layering transparently via synchronous height-oracle traits.
- **Why `MIN_HEIGHT_M = 2.0` m with `f64` infinite Reverse-Z & depth-less overlays:** The legacy renderer floated draped lines on an 80 m `SURFACE_LIFT_M` to clear coarse mesh chord sag, forcing an 800 m camera floor (~0.75 m/px at 1000 rows, 50° FOV—37× too coarse for 2 cm drone orthophotos). Moreover, independently triangulated smooth-ellipsoid meshes (e.g., a coarse native-CRS warp mesh over a finer basemap quadtree) have different chord sagittas and clip or Z-fight if depth-tested against each other. Rendering the draped graticule and future overlay passes back-to-front **depth-less** (`depth_write: false`, `depth_compare: Always`) with fragment ellipsoidal-normal horizon discard (`dot(N_ellipsoid(P), normalize(E - P)) > 0.0`) removes `SURFACE_LIFT_M`, lowers the floor to `2.0` m (~1.87 mm/px at 1000 rows, 50° FOV), and keeps `Depth32Float` `StoreOp::Discard` memoryless on TBDR GPUs without `COPY_SRC` depth readback.
- **Exact 1:1 sRGB compositing:** Offscreen rendering targets `Rgba8UnormSrgb` with `view_formats: &[Rgba8Unorm]` and linear RGB constants, compositing 1:1 via `FilterMode::Nearest` using the view matching the swapchain color space (zero manual `encode_srgb` hacks).

## 4. Overlay Accuracy & Readout Contract (Cross-Slice Benchmark)

When raster warp-mesh placement lands in Slice 3A-1 (`a_raster_lands_where_its_binding_puts_it`), surface placement (including `f32` vertex encoding) must hold across all 5 benchmark bindings in UTM zone 37 (`EPSG:32637`):

1. Drone orthophoto at **2 cm/px**
2. Sentinel-2 granule at **10 m/px**
3. Landsat scene at **30 m/px**
4. Sentinel-2 granule at **pyramid level 4**
5. **8000×5000** orthophoto **rotated by 30°**

Tolerances (Slice 3A-1 benchmark):
- `< 10⁻³` of the pixel size of the drawn pyramid level
- `< 2 mm` for any raster drawn at its native resolution
- `< 2 cm` at any pyramid level
- Pair the 5 UTM 37 cases with a **second CRS fixture** and an end-to-end **`PixelIsArea` vs. `PixelIsPoint` (`GTRasterTypeGeoKey`)** half-pixel check.

Cross-slice numerical & readout invariants:
- **CPU placement + close-up GPU verification:** Assert `< 10⁻³` px placement on CPU exact geometry and texture coordinates (GPU bilinear filtering is typically 8-bit / `1/256` texel), paired with close-up (`2 m`) between-vertices GPU smoke tests verifying cell-midpoint sag and LOD continuity.
- **Display sampling ≠ scientific readout:** Cursor clicks identify the visible geographic location; scientific cursor readouts and time-series queries must read native source datasets on CPU with explicit resolution, interpolation, and nodata rules so values remain zoom-invariant as sharper display tiles stream in.

## 5. `f64` ECEF Metres, Geodetic Nadir Aim & Decoupled Quadtree Geometry (Slice 2A)

- **Why `f64` ECEF metres + per-tile CPU `f64` RTC (instead of normalized units and GPU double-single RTE):** Expressing ECEF directly in metres unifies `geodesy::Ellipsoid` (using Fukushima 2006 Halley-accelerated 3D inverse) with `projection::Datum` Helmert shifts and removes scale-factor conversions across `Camera`, `cull::Frame`, and tile geometry. Computing `(tile_origin - eye)` on CPU in `f64` and storing tile vertices relative to `tile_origin` in `f32` preserves ~24 µm precision at `2 m` altitude without relying on non-reassociating GPU float math across Vulkan/DX12/Metal compilers.
- **Why geodetic nadir aim (`forward = -self.at`):** Aiming geocentrically toward Earth's center introduces a `~0.192°` mid-latitude offset (~3.6 px at 1000 rows) and `~14.7 km` orbit-to-2 m center-zoom drift. Aiming along `-self.at` makes `Camera::probe(0.5, 0.5, aspect)` agree identically with `Camera::center()` across all altitudes and latitudes.
- **Why decoupled geometry vs. texture LOD (`z_geom <= 16–17`) + adaptive elliptical-disk Polar Stereographic caps:** Basemaps cap out at `max_level` (e.g. `z = 3` bundled, `z = 8–14` live), whereas `2 m` viewing requires sub-millimetre surface chord sag. Subdividing geometry up to `z_geom = 16–17` while clamping texture requests to `<= max_tex_level` and sampling ancestors via exact power-of-two sub-rect UVs guarantees zero requests above `max_tex_level`, zero `f32` UV stepping, and `< 1 mm` sag at `2 m`. Covering `|lat| >= 85.05112878°` with smooth elliptical square-to-disk Polar Stereographic quadtree caps eliminates polar triangle slivers and antimeridian UV seams while stitching seamlessly along `±85.05112878°` via per-cell sag-scaled skirts (`max(4.0 * sag_cell_diag, 0.001) m`—using unclamped span sag on polar-clamped `Geodetic` tiles—plus a slight outward tangential flare and 2:1 neighbor balancing).
- **Why post-overlay procedural `fwidth()` graticule & `GpuTile` hygiene:** Evaluating the graticule procedurally in a transparent post-overlay `grid_pipeline` reusing the quadtree surface buffers (with CPU `f64`-wrapped phase offsets, uniform-flow `fwidth()` before horizon `discard`, and small-angle Taylor polar-meridian evaluation avoiding driver `atan2` drift) removes 672-segment chord parallax and ensures future draped raster overlays never bury it. Keeping `GpuTile` strictly limited to `{ texture, view, format, width, height, byte_size, tile_id }`—with `TileKey` carrying `gutter: u8`, `TileCache<TileKey, V>` delegating ancestor touches to `TileScheme::parent` (and exposing UV-decoupled `find_carrier` for clipped/guttered `RasterPyramid` tiles), and `GlobeRenderer` caching `(Weak<GpuTile>, wgpu::BindGroup)` swept automatically on `prepare()`—lets one cache contract serve `WebMercator`/`Geodetic`/`PolarCap` basemaps (`z = 0` coarsest) and `RasterPyramid` overlays (`level = 0` finest) without inverting parent direction or leaking GPU textures across discarded `egui` passes.
