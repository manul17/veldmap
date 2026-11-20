# VeldMap Development Plan

## Goal

Build VeldMap as a fast, purely native Rust desktop workstation for:
- Interactive 3D WGS84 planetary viewing (Google Earth-style global base map + satellite overlays),
- Multi-format satellite imagery inspection (GeoTIFF/COG, Sentinel-2 JPEG 2000, NetCDF-4/HDF5, PNG/JPEG) both locally and streamed in-place over HTTP Range from Copernicus Data Space (CDSE),
- Smooth multi-temporal timeline scrubbing to observe terrain and land-cover changes over time.

A simplified browser/mobile companion is intentionally deferred to a separate project later (made possible by keeping `veldmap-core` free of UI and GPU dependencies).

---

## Architectural Guardrails

1. **Simple 2-crate native workspace:**
   - `crates/veldmap-core`: pure domain logic (geodesy, projections, tile planning, raster adapters, disk cache, CDSE/S3, HTTP Range block pool). No `egui`, `eframe`, or `wgpu`.
   - `crates/veldmap-app`: native `eframe`/`egui` + `wgpu` desktop app (renderer, VRAM tile cache, background workers, UI).
   - Split out additional crates only when compile times or concrete reuse require it.
2. **Frame-driven streaming engine (no internal transport or pub/sub ceremony):**
   - **Main thread (`eframe` + `wgpu`):** owns `AppState`, camera, timeline, and VRAM `GpuTileCache`; never blocks on disk, network, or decoding. Each frame drains completed worker results, uploads ready tiles to VRAM, draws `egui` directly over `AppState`, plans visible tiles, and dispatches prioritized/cancellable jobs.
   - **Background workers:** `tokio` multi-threaded runtime for HTTP/CDSE/downloads + bounded CPU thread pool for raster decoding and QOI cache encoding.
3. **Explicit exclusions:**
   - No WASI, Wasmtime, ConnectRPC, protobuf, browser WASM shims, Emscripten, or SQLite/ORM.
   - Use flat filesystem persistence (`.part` downloads, simple JSON metadata/layout, QOI tile files).
   - Add traits only where there is a real second implementation (e.g., `ByteSource` for local vs HTTP Range, raster decoders per format, tile providers).
4. **Reuse from `old/`, rewrite glue cleanly:**
   - Port proven algorithms from `old/code/` and only high-signal silent-bug unit tests from `old/tests/` (geodesy/projections, culling, binary parsers, windowed read counts, S3 SigV4, range boundaries, cache eviction).
   - Do not port old UI/bus/correlator plumbing. Delete each file from `old/` once ported; delete `old/` completely when migration finishes.
5. **Code hygiene (every slice):**
   - English only across code, comments, UI, docs, and commits.
   - Rust 2024; verify with `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace`.

---

## Vertical Slices

### Phase 0 — Clean Native Bootstrap `[x]`
- [x] Back up legacy repository to `F:\dev\veldmap-legacy`.
- [x] Initialize fresh `main` repository at `F:\dev\veldmap` with curated `old/` reference (`code/`, `tests/`, `docs/`) and fonts under `assets/fonts/`.
- [x] Scaffold native `veldmap-core` + `veldmap-app` (`eframe` 0.37 + `egui_wgpu` + `wgpu` 30, `Backends::PRIMARY`) passing `fmt`, `clippy -D warnings`, and `test`.

### Slice 1 — Core Geodesy & Interactive 3D WGS84 Globe `[ ]`
- [ ] Port `geodesy`, `projection`, `camera`, `cull`, `mesh`, `outlines`, `footprint`, and `wheel` from `old/code/geo/` into `veldmap-core::geo` (using `glam`).
- [ ] Restore and translate the curated unit tests from `old/tests/geo/` (WGS84 ECEF, Krüger projections, Pulkovo 1942 / MSK-61, double-float halves, culling, antimeridian seams, camera zoom/pole-crossing/flights, footprint hit-testing).
- [ ] Implement the `wgpu` WGS84 globe renderer in `veldmap-app` using `egui_wgpu::CallbackTrait`: depth-tested ellipsoid, RTE double-float precision (`globe.wgsl`), graticule, mouse orbit, zoom-at-cursor, and smooth quaternion focus flights.
- [ ] Delete ported files from `old/code/geo/` and `old/tests/geo/`.

### Slice 2 — Planetary Base Map (Global Tile Quadtree) `[ ]`
- [ ] Add a global `z/x/y` tile quadtree on WGS84 (subdivided patch meshes per tile, ready for future DEM vertex displacement + camera tilt).
- [ ] Implement a bounded VRAM `GpuTileCache` (`Arc<GpuTile>`) and a `tokio`/`reqwest` background tile fetcher with priority queue (visible tiles first) and immediate cancellation when the camera moves.
- [ ] Stream a configurable public XYZ satellite/basemap layer with simple disk caching so the full planet renders smoothly from orbit down to local field/street scale.

### Slice 3 — Local Raster Tiling, Overlays & 2D Inspector `[ ]`
- [ ] Define `ByteSource` (`read_at`, `prefetch`) with a native local file / `memmap2` implementation and BLAKE3 content identity (replacing sampled-FNV for mutable local files).
- [ ] Port `pyramid`, `resample`, `radiometry`, `cascade`, `budget`, and raster adapters (`tiff`, `jp2`/OpenJPEG, `excerpt`, `netcdf`, `jpeg`, `png`, `grid`, `table`) from `old/code/tiler/` into `veldmap-core::raster`, along with curated tests from `old/tests/tiler/`.
- [ ] Port QOI persistent LRU disk cache from `old/code/cache/` (+ `evict_tests.rs`).
- [ ] Wire a bounded multi-core CPU decode pool feeding the shared `GpuTileCache`.
- [ ] Support opening local rasters: drape georeferenced warp-mesh overlays + outlines onto the 3D globe, inspect raw rasters in a 2D viewport sharing `Arc<GpuTile>`, and switch NetCDF variables.
- [ ] Delete ported files from `old/code/tiler/`, `old/code/cache/`, `old/code/viewer/`, and their tests.

### Slice 4 — Copernicus Data Space (CDSE), Remote Streaming & Downloads `[ ]`
- [ ] Port `old/code/network/range.rs` (64 KiB block pool, readahead ramp-up, 401/403 re-sign) as an HTTP `ByteSource` over native `reqwest`/`tokio`.
- [ ] Port `provider` (`s3`, `catalogue`, `scene`, `imagery`, `manifest`, `mgrs`) and `library` (`catalog`, `storage`, `download` + `old/code/network/download.rs` resumable `.part` downloads) into `veldmap-core`, with curated tests from `old/tests/provider/`.
- [ ] Build clean `egui` panels for CDSE catalogue search, S3 path browsing, scene outlines on the globe, in-place remote raster viewing, and resumable local downloads (reading credentials from `.env`).
- [ ] Delete remaining ported files from `old/code/provider/`, `old/code/library/`, `old/code/network/`, and their tests.

### Slice 5 — Timeline Scrubbing, GPU Shaders & Workstation Polish `[ ]`
- [ ] Key GPU/disk tiles by `(SourceVersion, Variable, Level, X, Y)` and add a multi-epoch timeline scrubber with adjacent-date prefetch, instant stale-request cancellation, and smooth GPU cross-fade.
- [ ] Leverage native `wgpu` capabilities where measured useful: bindless tile arrays (`TEXTURE_BINDING_ARRAY`), raw 16-bit/float GPU textures for live WGSL band math (NDVI, NDWI, false-color, contrast stretch) and temporal diffing (`Date B − Date A`).
- [ ] Evaluate optional 3D DEM terrain displacement + camera tilt/pitch on the Slice 2 quadtree, GPU hillshade/slope/aspect layers, pane/tab layout persistence, and multi-monitor tear-off windows.
- [ ] Delete `old/` completely once migration is complete and update `README.md` with final architecture and usage.

---

## Pickup Notes for Next Session

- **Current state:** Phase 0 is complete on `main` (`F:\dev\veldmap`). Legacy backup lives at `F:\dev\veldmap-legacy`.
- **Next task:** Start **Slice 1** — add `glam` to workspace dependencies, port `old/code/geo/` + curated tests into `crates/veldmap-core/src/geo/`, and replace the gradient placeholder in `crates/veldmap-app/src/render.rs` with the interactive WGS84 globe.
