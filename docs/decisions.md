# Design Decisions

Concise record of **implemented** architectural decisions—focusing on *why alternatives were rejected* and *hard cross-slice invariants* that cannot be inferred from reading the code alone.

> **Policy:** Record a decision here only once it is implemented and proven in code (or when preserving a hard numerical/safety invariant). Unbuilt future slices live as working hypotheses in [plan.md](../plan.md), not pre-accepted contracts. Module implementation details live in Rust `//!` docstrings (`cargo doc`).

---

## 1. Native Rust Workspace & UI/GPU-Free Domain Core

- **Why not WASI / Wasmtime / protobuf IPC:** VeldMap's legacy prototype routed components through a WASI/Wasmtime event bus with protobuf serialization, adding IPC overhead, `!Send` constraints, and 32-bit guest memory limits with no desktop benefit. `wasmtime`, `prost`, and `tonic` are banned in [deny.toml](../deny.toml) to prevent regressions.
- **Why keep `veldmap-core` free of UI/GPU crates:** Keeping `egui`, `eframe`, and `wgpu` out of `veldmap-core` ensures fast headless builds/tests without graphics drivers or windowing libraries and preserves reuse in future companions.
- **Workspace crate growth:** Start with `veldmap-core` + `veldmap-app`; split additional crates along real dependency seams (`raster`, `net`, `storage`) when those subsystems land rather than simulating crate boundaries via feature flags.

## 2. Dependency Hygiene & Safety Defaults

- **No `memmap2` for user files (banned in [deny.toml](../deny.toml)):** Memory-mapped I/O in Rust risks `SIGBUS` UB on truncated, removable, or network-backed files. Positioned I/O (`pread`/`seek_read`) with caller buffers is safe, portable, and mirrors HTTP Range reads. (Wayland SHM wrappers `smithay-client-toolkit` and `winit` remain exempted.)
- **Pure-Rust TLS (`rustls`), no OpenSSL or heavy ORMs:** `openssl`/`openssl-sys` and `sqlx` are banned in [deny.toml](../deny.toml) to avoid cross-OS TLS linking issues and heavyweight ORM plumbing.
- **Pure-Rust default build without permanent ideological bans:** Default desktop builds avoid C geospatial dependencies (`gdal`, `proj`) and embedded C databases (`rusqlite`) for simple 3-OS compilation, without banning them permanently if real workloads (e.g. grid-shifted datums/WKT2 or high-file-count tile indexing) justify an explicit or optional integration later.

## 3. Smooth WGS84 Ellipsoid (`height = 0.0`), Reverse-Z & 2 m Floor

- **Why no 3D DEM mesh displacement in v1:** Orthorectified optical satellite imagery and drone orthophotos place pixels at true geodetic `(lat, lon)`. Displacing the globe mesh with a 3D DEM introduces view-dependent relief parallax and requires terrain-ray intersections and depth-tested vector draping. Terrain relief is instead delivered as quantitative GPU layers (hillshade, slope, aspect).
- **Current nadir camera:** VeldMap v1 uses a nadir camera (`Camera`) for uniform center GSD and simple controls; note that zero camera tilt is a v1 UX simplification (not physically required to avoid relief parallax on a flat `height = 0.0` surface). Because v1 `Camera`, `cull::Frame`, `geodesy::surface_at`, and depth-less `outlines` optimize for `height_m = 0.0`, any future v2 3D terrain or oblique mode will extend or replace those modules rather than layering transparently on top.
- **Why `MIN_HEIGHT_M = 2.0` m with `f64` infinite Reverse-Z:** The legacy renderer floated draped lines on an 80 m `SURFACE_LIFT_M` to clear coarse mesh chord sag, forcing an 800 m camera floor (~0.49 m/px—24× too coarse for 2 cm drone orthophotos). Rendering draped primitives depth-less (`depth_write: false`, `depth_compare: Always`) with fragment ellipsoidal-normal horizon discard (`dot(N_ellipsoid(P), normalize(E - P)) > 0.0`) removes `SURFACE_LIFT_M` and lowers the floor to `2.0` m (~1.87 mm/px at 1000 rows, 50° FOV).
- **Exact 1:1 sRGB compositing:** Offscreen rendering targets `Rgba8UnormSrgb` with `view_formats: &[Rgba8Unorm]` and linear RGB constants, compositing 1:1 via `FilterMode::Nearest` using the view matching the swapchain color space (zero manual `encode_srgb` hacks).

## 4. Overlay Accuracy Contract (Cross-Slice Benchmark)

When raster warp-mesh placement lands in Slice 3A-1 (`a_raster_lands_where_its_binding_puts_it`), surface drift (including `f32` vertex encoding) must hold across all 5 benchmark bindings in UTM zone 37 (`EPSG:32637`):

1. Drone orthophoto at **2 cm/px**
2. Sentinel-2 granule at **10 m/px**
3. Landsat scene at **30 m/px**
4. Sentinel-2 granule at **pyramid level 4**
5. **8000×5000** orthophoto **rotated by 30°**

Tolerances:
- `< 10⁻³` of the pixel size of the drawn pyramid level
- `< 2 mm` for any raster drawn at its native resolution
- `< 2 cm` at any pyramid level
