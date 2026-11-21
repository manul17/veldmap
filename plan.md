# VeldMap Development Plan

## Goal

Build VeldMap as a fast, purely native Rust desktop **Analytical Earth-Observation & Precision-Imagery Workstation** on a **Precision Nadir 3D WGS84 Globe** for:
- Interactive nadir 3D WGS84 planetary viewing (global basemap + precision local/satellite overlays down to centimetre-scale orthophotos),
- Multi-format satellite imagery inspection (GeoTIFF/BigTIFF/COG, PNG/JPEG, Sentinel-2 JPEG 2000; NetCDF-4/HDF5 in v1.1) both locally and streamed in-place over HTTP Range from zero-auth STAC/COG catalogs, public HTTP COG URLs, and Copernicus Data Space (CDSE),
- Interactive GPU band math (NDVI/NDWI/false-color with metadata-driven radiometric scaling and cloud masking), analytical terrain relief (hillshade/slope/aspect), multi-temporal timeline scrubbing, and pixel time-series inspection.

A simplified browser/mobile companion is intentionally deferred to a separate project later (enabled by keeping `veldmap-core` UI/GPU-free with minimal default-off dependencies).

> **Document lifecycle:** `plan.md` is the executable slice roadmap and exit-criteria tracker. Authoritative design contracts live in empirical single-decision ADRs under `docs/decisions/` and in `docs/architecture/`, written alongside the slice that proves them. Once a slice closes and its ADR lands, link the ADR from the slice header and treat the ADR as authoritative.

---

## Architectural Guardrails

1. **Simple 2-crate native workspace with an enforced internal module DAG:**
   - `crates/veldmap-core`: pure domain logic organized as a strict acyclic internal DAG, enforced in CI via isolated feature-matrix builds (`--no-default-features`, `--features fs`, `--features codecs`, `--features net`, `--all-features`) plus a lightweight intra-feature import check (`cache ↛ library`, `net ↛ provider`):
     - **Leaf (`--no-default-features`; `glam`, `thiserror`, `blake3` only):**
       - `geo`: WGS84 geodesy, projections/datums, `mgrs`, `DMat4`/`DVec3` camera & frustum culling, warp/georef math, geodesic measurements.
       - `tile`: `TileKey` (`Copy + Eq + Hash`), `TileScheme` (`WebMercator | Geodetic | RasterPyramid { max_level: u8 }`), `BandKey`, `PixelFormat`, `pyramid`, generic GPU-free O(1) LRU `TileCache<K, V>`, pure `TileScheduler`.
       - `io` / `mem`: strictly sync caller-buffered `ByteSource` trait, `Signer` trait, zero-dep async-waking `CancelToken`, lock-free `MemoryBudget` + `Shrinkable`, pure `StoragePaths`.
     - **Orthogonal mid-layer (depend only on leaf):**
       - `cache` (`fs`): raw-byte tile disk store, two-tier atomic writes, incremental LRU eviction.
       - `raster` (`codecs`): pure-Rust decoders (`tiff`, `png`, `jpeg`, `openjp2`, later `hdf5_pure`), `table`, `grid`, `cascade`, `resample`, `radiometry`, pure sync range-planners.
       - `net` (`net`; `tokio` + `reqwest` over `rustls`): `HttpRange` (`ByteSource` + `prefetch_pinned()` + `Shrinkable` 64 KiB LRU block pool), retry/ETag/range helpers.
       - `provider` (`net`): STAC/COG discovery, CDSE OData/S3/SigV4 (`Signer` impl), manifests.
     - **Upper layer (`fs`, optional `net`):**
       - `library`: local catalog & storage layout (`fs`) plus resumable downloader (`fs + net`) parameterized over `Signer`.
     - Default desktop features enable `codecs`, `fs`, and `net`. Zero `egui`, `eframe`, `wgpu`, or `memmap2` in `veldmap-core`.
   - `crates/veldmap-app`: native `eframe`/`egui` + `wgpu` desktop application (offscreen 3D globe renderer decoupled from `egui`, 2D inspector, `GpuTileCache`, background workers, UI).
   - Re-evaluate extracting `veldmap-raster` at **Slice 3C-1** if codec compile times degrade inner-loop speed.

2. **Committed v1 planetary model — Precision Nadir Analytical Globe:**
   - v1 renders a smooth WGS84 ellipsoid (`height = 0.0`, no 3D DEM mesh displacement or camera tilt/pitch). Relief is delivered as quantitative GPU layers (hillshade, slope, aspect) over DEM COGs in Slice 5B-1. This guarantees zero parallax for orthophoto alignment, swipe, and temporal diffing while keeping `Geodetic.height_m`, `DMat4` Reverse-Z, and tile LOD ready for any v2 3D terrain mode.

3. **Frame-driven streaming engine (no internal pub/sub bus):**
   - **Main thread (`eframe` + `wgpu`):** owns `AppState`, camera, timeline, `GpuTileCache` (directly on `VeldmapApp`, outside `CallbackResources`), and `PendingUploads`. Never blocks on disk, network, or decoding; passes immutable `Arc<GpuTile>` draw snapshots to 3D/2D `CallbackTrait` renderers.
   - **`GpuTile` hygiene:** owns only `Texture`, `TextureView`, `PixelFormat`, dimensions, true byte size (including mips), and monotonic `tile_id: u64` — never pipeline-specific `BindGroup`s or `view_formats`.
   - **Background workers & shutdown:** `tokio` multi-threaded runtime for HTTP/STAC/CDSE/downloads and Stage 1 async preflight (producing pinned memory-only `PreparedBatch`es), isolated from separate bounded **local CPU** vs **blocking remote** decode pools. Shutdown order: cancel root `CancelToken` → drop completion receiver (or use unbounded channel) & join decode pools → drop `tokio::Runtime`.
   - **Deadlock-free memory & deduplication:** Lock-free `AtomicU64` `MemoryBudget` (FIFO slow-path waiter queue; never call while holding cache locks; snapshot `Vec<Arc<dyn Shrinkable>>` before shrinking); caches only use `try_acquire()` + self-eviction; decode jobs reserve upfront private working-set `Peak` atomically; singleflight is waiter-refcounted under its map mutex with a leader `Drop` guard.

4. **Explicit exclusions, codec safety & two-tier durability:**
   - No WASI, Wasmtime, ConnectRPC, protobuf, browser WASM shims, Emscripten, SQLite/ORM, `memmap2`, GDAL/PROJ C deps, or OpenSSL C deps in `veldmap-core` (use safe positioned I/O and `reqwest` with `rustls`).
   - Pure-Rust codecs with explicit memory/IFD/padding/shift/varint limits, panic isolation, FFI soundness fixes on `openjp2`, and thread-safety audits before multi-threaded decode.
   - Injected `StoragePaths` with two-tier atomic writes (`write_atomic_durable()` with `sync_all()` + synchronous handle drop before rename vs `write_atomic_cached()` without `sync_all()` + read validation; both using per-boot `.<boot_id>.<seq>.tmp` files).
   - Add traits only where there is a real second implementation (`ByteSource`, `Signer`, raster decoders, tile providers).

5. **Reuse from `old/`, rewrite glue cleanly, retire incrementally:**
   - Tag pre-deletion state as `legacy-reference-snapshot` in Phase 0.5 and track all 78 files in `old/README.md` with `Consumer Slices` and `Delete After Slice` (always the *last* consumer slice).
   - Port proven algorithms and curated tests, stripping `!Send` `Rc`/`RefCell` assumptions and splitting monoliths (`overlay.rs`, `tiles_pipeline.rs`). Never link from `docs/` into `old/`; delete `old/` completely at Slice 6.

6. **Code hygiene (every slice):**
   - English only across `crates/`, `docs/`, `README.md`, `plan.md`, `.env.example`, `.github/`, UI, comments, and commits (CI-enforced; remove `old/` exclusion at Slice 6).
   - Rust 2024; verified by `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `cargo-deny`, `naga` SPV/HLSL/MSL codegen, headless offscreen `wgpu` smoke tests, `veldmap-core` DAG checks, and offline `docs/` link checks.

---

## Vertical Slices

### Phase 0 — Clean Native Bootstrap `[x]`
- [x] Back up legacy repository and initialize fresh `main` repository with curated `old/` reference and `assets/fonts/`.
- [x] Scaffold native `veldmap-core` + `veldmap-app` (`eframe` 0.37 + `egui_wgpu` + `wgpu` 30, `Backends::PRIMARY`) passing `fmt`, `clippy -D warnings`, and `test`.

---

### Phase 0.5 — Slim Pre-Slice 1 Hygiene, Licensing & CI `[ ]`
- [ ] **Toolchain & dev ergonomics:** Fix `rust-version = "1.95"` in `Cargo.toml` (matching `egui 0.37` / `wgpu 30`), pin `rust-toolchain.toml`, add `.gitattributes` (LF endings, `old/** linguist-vendored`), and add `[profile.dev.package."*"] opt-level = 2`.
- [ ] **Scaffold fixes in `veldmap-app`:** Pass `response.rect` guarded by `is_positive()`, drop redundant direct `egui`/`egui-wgpu` deps in favor of `eframe` re-exports, honor `WGPU_BACKEND` (excluding GL) / `WGPU_POWER_PREF`, initialize `tracing`, and hook `device.on_uncaptured_error`.
- [ ] **Remove mixed-license font bloat:** Delete unused `assets/fonts/*.ttf` (~6.3 MiB, closing the OFL + Nerd Fonts license gap) and keep `eframe`'s `default_fonts` until UI polish.
- [ ] **Repo & `old/` manifest cleanup:** Update `README.md`, `.env.example` (remove `data-provider` wording), `.gitignore` (`.env*` + `!.env.example`, drop `/runtime/*`), and `old/README.md` (remove personal Windows path, list omitted `network/download.rs` and `viewer/`, fix Cyrillic note across 27 files, add full 78-file `Consumer Slices` / `Delete After Slice` manifest). Create annotated tag `legacy-reference-snapshot`.
- [ ] **Workspace lints & lean Linux CI:**
  - Add workspace lints: `unwrap_used`/`expect_used` as `"warn"` (with `clippy.toml` test exemptions; denied in CI via `-D warnings`), `allow_attributes_without_reason`, `undocumented_unsafe_blocks`, `unimplemented`, `large_futures` (standardize on `#[expect(..., reason = "INVARIANT: ...")]`).
  - Add Linux `hygiene-and-dag` GitHub Actions job (`Swatinem/rust-cache`, `fmt`, `clippy -D warnings`, `test`, `cargo-deny` for licenses/RUSTSEC/banned crates, no-Cyrillic check, offline `lychee --offline` `docs/` link check).
- [ ] **Seed minimal docs:** Create `CONTRIBUTING.md`, `SECURITY.md`, `THIRD_PARTY_NOTICES.md`, `docs/README.md`, `docs/building.md`, `docs/decisions/README.md`, `docs/decisions/0001-native-workspace-and-threading.md` (workspace DAG, exclusions, frame-driven loop & shutdown order only), `docs/glossary.md`, and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* Green Linux workflow (`fmt`, `clippy`, `test`, `cargo-deny`, no-Cyrillic, offline `docs/` links).
  - *Manual QA:* `cargo run -p veldmap-app` launches cleanly with `tracing` logs; `legacy-reference-snapshot` tag exists; zero unlicensed font binaries remain; `plan.md` `Next Step` updated.

---

### Slice 1 — Core Geodesy & Interactive 3D WGS84 Globe `[ ]`
- [ ] **Port core geodesy & MGRS (`veldmap-core::geo`):** Port `geodesy`, `projection`, `camera`, `cull`, `outlines`, `footprint`, `wheel`, sag helpers, and **`mgrs`** (from `old/code/provider/mgrs.rs` into leaf `--no-default-features`) along with curated `old/tests/geo/` and `mgrs_tests.rs` (untextured ellipsoid mesh lives in `veldmap-app` as Slice 2's "Ellipsoid-only" fallback).
- [ ] **High-precision camera & culling:** Upgrade `Camera` to `DMat4` infinite Reverse-Z (`perspective_infinite_reverse_rh`), `height = 0.0`, `MIN_HEIGHT_M` (2–10 m), private internals behind view-agnostic queries, and `f64`/`DVec3` `Camera::project()`, `cull::Frame`, and `cull::Ball`.
- [ ] **Offscreen `wgpu` WGS84 globe renderer (`veldmap-app`):**
  - Own `Rgba8UnormSrgb` + `Depth32Float` Reverse-Z target, sized in `CallbackTrait::prepare()` by constructing `PaintCallbackInfo` from stored `rect`/`clip_rect` + `screen_descriptor` (`viewport_in_pixels()`, clamped `[1, max_texture_dimension_2d]`, skipped on `0×0`, dirty-flag redraw skipping).
  - Composite 1:1 with `FilterMode::Nearest` via `view_formats` branching on `target_format.is_srgb()` (linear RGB palette constants, no `encode_srgb` hacks).
  - Precision-safe globe geometry (octant-RTC) + depth-less draped graticule (`depth_write: false`, `depth_compare: Always`, fragment ellipsoidal-normal horizon discard `dot(N_ellipsoid(P), E - P) >= 0`, HiDPI-scaled `RIBBON_PX`/`HATCH_PX`).
  - Mouse orbit, zoom-at-cursor, and smooth quaternion flights.
- [ ] **Baseline GIS HUD:** Cursor Lat/Lon, altitude & m/px, dynamic scale bar, North compass / Reset-North (`N`), and Go-To Coordinates (DD/DMS) or MGRS jump.
- [ ] **3-OS CI, `naga` & headless GPU tests:**
  - Add CPU-only `naga` (pinned to `wgpu` major) WGSL validation + SPV/HLSL/MSL codegen test (with realistic HLSL/MSL options).
  - Add 3-OS `build-and-test` CI matrix (`ubuntu`/`windows`/`macos`) with headless offscreen `wgpu` smoke test verifying Reverse-Z depth precision and exact sRGB compositing (`VELDMAP_REQUIRE_GPU_TEST=1` on Linux Lavapipe w/ `mesa-vulkan-drivers` & Windows WARP, retrying `force_fallback_adapter: true`; graceful skip on macOS VMs).
- [ ] **Docs & `old/` cleanup:** Write `docs/decisions/0002-globe-model-and-precision.md` (recording v1 flat nadir ellipsoid from legacy `0006` and superseding `0007`'s 800 m floor with all 5 benchmark bindings & drift tolerances) and `docs/architecture/globe.md`; update `docs/glossary.md` & `docs/limitations.md`; delete ported `old/code/geo/*` (keeping `overlay.rs`), `old/code/provider/mgrs.rs`, `old/tests/geo/`, `old/tests/provider/mgrs_tests.rs`, `old/docs/globe.md`, and `old/docs/decisions/0006-terrain-on-the-globe.md` (**retain** `0007-overlay-accuracy.md` until Slice 3A-1).
- **Exit criteria:**
  - *Automated CI:* Green 3-OS CI, `naga` SPV/HLSL/MSL codegen, `veldmap-core::geo` + MGRS tests, and headless Reverse-Z/sRGB GPU smoke test.
  - *Manual QA:* Smooth 60 fps zoom from orbit to 2 m altitude with zero jitter, horizon bleed, or resize panic; Go-To Lat/Lon & MGRS lands accurately; `plan.md` `Next Step` updated.

---

### Slice 2 — Planetary Base Map (Global Tile Quadtree, Unified GPU & Disk Caches) `[ ]`
- [ ] **Global WGS84 tile quadtree:** Implement `z/x/y` quadtree over `TileScheme` (`WebMercator | Geodetic | RasterPyramid { max_level: u8 }`) with **per-tile RTC** meshes (`tile_origin - eye` in `f64` on CPU, shared static index buffer), polar cap extensions attached directly to `y = 0` / `y = max` tiles (zero T-junctions across mixed LODs), sagitta skirts, and HiDPI-aware SSE subdivision.
- [ ] **Unified `TileKey`, `TileCache` & `GpuTileCache`:**
  - Define `TileKey: Copy + Eq + Hash` (`SourceId`, `BandKey`, `TileScheme`, `PixelFormat`, `DecodeRev`, `Level`, `X`, `Y`) with deterministic `Copy` `BandKey` (`Display | Index(u16) | NamedHash(u64)`) and implement pure O(1) LRU `TileCache<K, V>` in `veldmap-core::tile` (scheme-aware ancestor touch, `<50%` byte appetite cap per pyramid).
  - Own `GpuTileCache` and `PendingUploads` directly on `VeldmapApp` (cloning `Arc`-backed `device`/`queue`, passing `Arc<GpuTile>` draw snapshots to callbacks + small per-pipeline `BindGroup` LRU keyed by `[tile_id; N]`). Drain the full completion channel each frame (unbounded channel or drop-rx before shutdown join), prune stale entries, upload coarsest-first (4–8 MiB / ~2 ms cap, ≥ 1 tile/frame), request repaint while non-empty, and trigger redraws via injected `Wakeup`.
- [ ] **Async-waking `CancelToken` & tile fetcher (`veldmap-core::io`, `veldmap-core::net`):**
  - Implement leaf `CancelToken` (`AtomicBool` + zero-dep `std::task::Waker` / listener hook so `tokio::select!` aborts immediately on pan/shutdown).
  - Implement `tokio` + `reqwest` (`rustls`) tile fetcher with per-host concurrency limits, compliant `User-Agent`, replaceable visible-first queue, and immediate cancellation.
- [ ] **Disk cache & LRU eviction (`veldmap-core::cache`):** Port `old/code/cache/` (`store`, `layout`, `tile`, `evict`) and `evict_tests.rs` using injected `StoragePaths`, two-tier atomic writes (`write_atomic_cached()` with no `sync_all()` + read validation and `write_atomic_durable()` with `sync_all()` + synchronous `into_std().await` drop before rename; both using `.<boot_id>.<seq>.tmp` and Windows sharing-violation handling), and foreign-UUID startup sweep.
- [ ] **Legally clean basemap stack & geodesic tools:**
  - Bundle a public-domain low-res global texture (z0–z4 NASA Blue Marble Next Gen or Natural Earth, ~1–3 MiB; document provenance in `THIRD_PARTY_NOTICES.md` and `assets/README.md`).
  - Wire NASA GIBS + EOX Sentinel-2 Cloudless (CC BY 4.0) with visible attribution, basemap switcher (**Satellite / Ellipsoid-only** + Custom XYZ/TMS URL & API-key settings; defer a default street "Map" layer unless a legally compliant tile source is chosen), graceful offline fallback, and WGS84 geodesic distance/area tools.
- [ ] **CI DAG check, docs & `old/` cleanup:**
  - Add `veldmap-core` feature-matrix & intra-feature import checks (`--no-default-features`, `fs`, `net`, `--all-features`) to CI.
  - Write `docs/decisions/0003-tile-identity-and-gpu-cache.md`, `docs/decisions/0004-two-tier-atomic-durability.md`, and seed `docs/architecture/viewing-pipeline.md`; delete `old/code/cache/` and `old/tests/cache/evict_tests.rs`.
- **Exit criteria:**
  - *Automated CI:* `TileCache<K, V>` (both `WebMercator` and `RasterPyramid { max_level }`), geodesic math, `evict_tests`, atomic-write/temp-sweep, and `veldmap-core` DAG checks pass.
  - *Manual QA:* Planet zooms cleanly from orbit to street scale with zero T-junctions or polar seams, bounded VRAM & disk usage, visible attribution, and offline fallback; `plan.md` `Next Step` updated.

---

### Slice 3A-1 — Local GeoTIFF/PNG/JPEG Engine, `MemoryBudget` & Warp-Mesh Draping `[ ]`
- [ ] **Sync `ByteSource` & `ContentId` (`veldmap-core::io`):**
  - Define strictly sync caller-buffered `ByteSource::read_at(&self, offset: u64, buf: &mut [u8], cancel: &CancelToken) -> Result<usize, IoError>` with safe positioned-I/O local file impl (no `memmap2`; buffered 64–256 KiB cursor; pooled `File` handles + short-read loop on Windows).
  - Key disk/VRAM tiles by sampled BLAKE3 `ContentId` (`len || head_64k || mid_64k || tail_64k`, domain-separated) paired with `BandKey`, `TileFormat`, `DecodeRev`, and gutter width in disk paths; use `(path, len, mtime_ns)` strictly as in-memory stat cache + UI "Reload / bypass cache".
- [ ] **Core raster pipeline (`veldmap-core::raster`, `codecs` feature):**
  - Port `pyramid`, `resample`, `radiometry`, `cascade`, `table`, `grid` (introducing `TilePixels` + `PixelSpec { bpp, gutter }` with `gutter = 0` so 3B extends rather than rewrites), core `tiff` (standard RGB/Gray GeoTIFF/BigTIFF/COG, `GeoDoubleParamsTag`/`GeoAsciiParamsTag`, explicit `Limits` + `MAX_IFDS`, verifying Float32 `Predictor = 3` and pure-Rust ZSTD), and `png`/`jpeg`/`full` (strip `!Send` `Rc`/`RefCell` via `OnceLock`/`Arc`/`AtomicU64`).
  - Wire 8-bit `Rgba8UnormSrgb` + QOI disk cache (`write_atomic_cached()` + header/length validation). Add `codecs` to CI feature matrix.
- [ ] **Decode pool, `MemoryBudget` & singleflight:**
  - Wire bounded local CPU decode pool, lock-free `AtomicU64` `MemoryBudget` (FIFO slow-path waiter queue to prevent large-job starvation, upfront private-working-set `Peak` without double-counting shared `DecodedChunkCache`, snapshot `Shrinkable` hooks), and atomic waiter-refcounted singleflight decoded-chunk LRU with panic Drop guard.
- [ ] **Native `TileScheduler` (`veldmap-core::tile`):** Replace legacy bus-era `Fetch`/`Passes` with `TileScheduler` (1 active pass per source for `Serve::Pass` only — allow concurrent `Serve::Pointwise`; non-cancellation of coarser passes; terminal-failure suppression; optional `coarse_start_override` ready for 4B).
- [ ] **Warp-mesh draping & `VisualLayer`:**
  - Split `old/code/geo/overlay.rs`: top-down lazy `DVec3` culling balls (preserving per-rung limb-culling margins) in `veldmap-core`; `VisualLayer` (`preview: Option<RasterSlot>` + `bands: SmallVec<[RasterSlot; 4]>`) + per-tile RTC warp meshes + horizon discard in `veldmap-app`.
  - Rebuild ADR-0007's `a_raster_lands_where_its_binding_puts_it` regression test (all 5 benchmark bindings & drift tolerances).
- [ ] **Docs & `old/` cleanup:** Write `docs/decisions/0005-raster-reading-model-and-memory-budget.md` and finish `docs/architecture/viewing-pipeline.md`; port TIFF sections of `reads_tests.rs` via `CountingByteSource`; delete `overlay.rs`, `budget.rs`, `fingerprint.rs`, `full.rs`, `tiles_pipeline.rs`, `cascade.rs`, `pyramid.rs`, `resample.rs`, `radiometry.rs`, `table.rs`, `grid.rs`, `tiff.rs`, `png.rs`, `jpeg.rs`, `pyramid_tests.rs`, `radiometry_tests.rs`, `resample_tests.rs`, `tiff_tests.rs`, `viewing-pipeline.md`, `0003-raster-reading-model.md`, and `0007-overlay-accuracy.md` (**retain** `reads_tests.rs` until Slice 6).
- **Exit criteria:**
  - *Automated CI:* `a_raster_lands_where_its_binding_puts_it`, TIFF `reads_tests`, `TileScheduler`, and `MemoryBudget`/singleflight concurrency tests pass.
  - *Manual QA:* Local GeoTIFF/BigTIFF/COG and cm-scale drone orthophotos drape accurately on the globe with responsive zooming and bounded RAM/VRAM/disk; `plan.md` `Next Step` updated.

---

### Slice 3A-2 — Local Workstation UX, Expanded Pure-Rust CRS & GeoJSON Overlay `[ ]`
- [ ] **Expanded pure-Rust CRS & world files:** Extend `veldmap-core::geo::projection` with `ETRS89/UTM` (`258xx`), `NAD83/UTM` (`269xx`), `LAEA Europe` (`3035`), Polar Stereographic (`3413`/`3031`), `GeoKey 3072 == 32767` user-defined TM parameters, wire existing `Datum::to_wgs84` into geographic bindings (preventing NAD83/Pulkovo-42 shifts), read `.tfw`/`.jgw`/`.pgw` world files in `fs`/`app` (injected as external `Option<Placement>`), and show visible UI warnings for unshifted foreign datums.
- [ ] **Local workstation UX:** Native file picker + drag-and-drop + CLI file argument, Layer Manager (visibility, reorder, opacity, zoom-to-binding, CRS/radiometry metadata inspector), searchable CRS picker with presets & custom TM parameters (unlocking all MSK-xx zones), **read-only GeoJSON overlay (stroke-only or earcut triangulation — never convex fan fills)**, and diagnostic banners (fatal vs partial error, detail-limit caption, loading shimmer).
- [ ] **Docs:** Write `docs/decisions/0006-pure-rust-georeferencing-and-crs.md` and seed `docs/architecture/georeference.md` (**retain** `old/docs/georeference.md` until NetCDF sections migrate in Slice 6).
- **Exit criteria:**
  - *Automated CI:* Expanded CRS/datum round-trip tests and GeoJSON parsing/stroke tests pass.
  - *Manual QA:* Local projected/world-file orthophotos and concave GeoJSON field polygons open cleanly via drag-and-drop or CRS override; `plan.md` `Next Step` updated.

---

### Slice 3B — Scientific `R32Float` Track, Live Stretch, Colorbar & 2D Inspector `[ ]`
- [ ] **Dual GPU track (`Display` + `Scientific`):**
  - Keep Display (`Rgba8UnormSrgb`, `512×512`) and add Scientific (`R32Float`, storing **raw numeric samples** with nodata/fill → `f32::NAN`, up/down-casting `u32`/`i32`/`f64` to `f32`, and applying `(val + offset) * scale` in WGSL uniforms + cursor readout to avoid cache aliasing).
  - Use **`528×528` (`512` interior + `8`-px apron) with `mip_level_count = 4`** (`528 → 264 → 132 → 66`) on Scientific tiles (exact divide-by-2 phase alignment, ≥1-px gutter at all 4 mips, smooth central-difference normals for 5B-1); normalize `cell_uv` against `valid_w`/`valid_h` and account for the `±8`-px halo in `Grid::footprint()`, `DecodedChunkCache`, and `Cascade` band overlap.
- [ ] **WGSL nodata-aware filtering, categorical handling & stretch:**
  - Manual 4-tap `textureLoad` bilinear filter using `(bitcast<u32>(v) & 0x7FFFFFFFu) >= 0x7F800000u` (masking both `NaN` and `±Inf`) and uniform-control-flow `dpdx`/`dpdy` LOD selection.
  - Enforce **nearest-neighbor sampling & non-averaging downsampling** for categorical bands (raw `Palette` classes / future `SCL`).
  - Live contrast stretch (linear percentile/min-max + **`log10` / dB stretch** for SAR) and colormap uniforms.
- [ ] **Multi-band GeoTIFF band selection & scientific UI:** Allow selecting individual bands from multi-band GeoTIFFs/COGs (unblocking multi-band drone orthophotos), interactive stretch histogram, and colorbar legend with physical units.
- [ ] **2D Inspector split pane:** Port `old/code/viewer/` into an internal split pane sharing `Arc<GpuTile>` with the 3D globe + cursor raw DN & physical-unit readout.
- [ ] **GPU smoke test, docs & `old/` cleanup:** Add headless `wgpu` smoke test verifying NaN/Inf bitmask bilinear math and `528×528` mip/gutter continuity; write `docs/decisions/0007-scientific-r32float-track-and-gutters.md`; delete `old/code/viewer/`.
- **Exit criteria:**
  - *Automated CI:* Headless `wgpu` nodata-bilinear/`528×528`-mip test, `naga` SPV/HLSL/MSL codegen, and `R32Float` radiometry tests pass.
  - *Manual QA:* 16-bit/float32/SAR/multi-band rasters render with exact nodata transparency, zero shimmer or tile-edge seams, live histogram/colorbar, and synchronized 2D/3D cursor readout; `plan.md` `Next Step` updated.

---

### Slice 4A — Zero-Auth Cloud EO Streaming (HTTP Range + STAC / Public COGs) `[ ]`
- [ ] **HTTP Range `ByteSource` (`veldmap-core::net`):** Port `old/code/network/range.rs` core over `reqwest` (`rustls`)/`tokio`: 64 KiB O(1) LRU `HttpBlockPool` (`Shrinkable` under `MemoryBudget`), combined open-probe + `0..64 KiB` head read + concurrent `mid_64k`/`tail_64k` `ContentId` reads (`416`-safe on 0-byte files), consumed-only readahead, atomic waiter-refcounted multi-block singleflight, auto-decompression disabled, `Content-Range` + strong-ETag `If-Match` validation, `CancelToken` async abort, and `Handle::try_current().is_err()` guard on blocking reads.
- [ ] **Two-stage pinned remote decode:** Add pure sync chunk range-planners in `raster` and `HttpRange::prefetch_pinned()` in `net` returning memory-only `PinnedByteSource` (`PreparedBatch` under upfront `MemoryPermit`) routed to the fast CPU decode lane, isolating un-prefetched blocking reads to a separate capped remote lane.
- [ ] **Zero-auth STAC & public COG provider (`veldmap-core::provider`):** Implement STAC API search (e.g., AWS Earth Search Sentinel-2 L2A COGs + COP-DEM GLO-30) and direct public HTTP COG URLs, multi-band Sentinel-2 asset discovery + STAC `raster:bands` scale/offset metadata, AOI search (viewport / drawn box / GeoJSON polygon + GeoJSON/WKT export), scene outlines on globe, and built-in zero-auth sample scenes for 15-second onboarding.
- [ ] **Mock HTTP harness & docs:** Build offline local mock HTTP Range harness (`206`, `200` rejection, `416`, strong/weak `ETag`, delivered-byte ceilings); write `docs/decisions/0008-network-streaming-and-stac.md` and `docs/configuration.md`.
- **Exit criteria:**
  - *Automated CI:* Offline mock HTTP Range/STAC/singleflight tests pass in CI.
  - *Manual QA:* Users can search an AOI and stream public Sentinel-2 L2A COGs and HTTP COG URLs with zero credentials without starving local SSD decodes; `plan.md` `Next Step` updated.

---

### Slice 5A — Interactive Band Math, Split-Screen Swipe & Session Persistence `[ ]`
- [ ] **WGSL band math & radiometric correctness:** Interactive WGSL band math (NDVI, NDWI, false-color, custom stretch) over co-registered multi-file / mixed-resolution band recipes (e.g., Sentinel-2 10 m `B04`/`B08` + 20 m `B11`/`B12`) on the `R32Float` track (working on both local multi-band GeoTIFFs and zero-auth STAC COGs).
- [ ] **Metadata-driven scaling & `SCL` masking:** Apply Sentinel-2 `BOA_ADD_OFFSET`/`RADIO_ADD_OFFSET` from STAC `raster:bands` or `MTD_MSIL2A.xml` (never hardcoded to a 2021↔2022 date cutoff), badge L1C (TOA) vs L2A (BOA), and support optional **nearest-neighbor `SCL` cloud/shadow masking** (`classes 0, 1, 3, 8, 9, 10`).
- [ ] **Split-screen swipe curtain:** Interactive swipe comparison across any two layers (local or remote, including non-co-registered swaths).
- [ ] **Session persistence & export:** Workspace/session persistence via `write_atomic_durable()` JSON (layers, CRS overrides, band math recipes, stretch params, camera bookmarks, split-pane layout), copy-coordinates, and viewport screenshot export (with optional scale bar, north arrow, colorbar, and acquisition date; GeoTIFF/CSV export deferred to post-v1).
- [ ] **Docs:** Write `docs/decisions/0009-band-math-and-analytical-relief.md`.
- **Exit criteria:**
  - *Automated CI:* Metadata-driven radiometric scaling, nearest-neighbor `SCL` mask, session JSON round-trip, and `naga` SPV/HLSL/MSL shader tests pass.
  - *Manual QA:* Multi-band Sentinel-2 NDVI/NDWI/false-color with cloud masking, swipe curtain, screenshot export, and session reload work smoothly; `plan.md` `Next Step` updated.

---

### Slice 3C-1 — Sentinel-2 JP2 (`openjp2`) & Exotic TIFFs `[ ]`
- [ ] **Compile-time check:** Re-evaluate extracting `veldmap-raster` into its own workspace crate before adding `openjp2`.
- [ ] **Audit & harden `openjp2`:** Audit crate version & `static mut` thread safety; port `codec.rs`, `jp2.rs` (`roxmltree`/`quick-xml` for GMLJP2; gate zero-as-nodata on Sentinel-2 heuristics, not all GMLJP2), and `excerpt.rs`:
  - Fix Stacked Borrows/drop-order UB (`Box::into_raw` + explicit `Drop`), `usize::MAX` EOF + `ErrorKind::Interrupted` retry, `catch_unwind` on `extern "C"` callbacks, `skip()` EOF bounds check, `COD` `levels <= 32` shift-overflow guard, PLT varint & TLM checked arithmetic + `keep <= part.len`, `MAX_PAD` cap, worker panic isolation, and 5/3 reversible decode round-trip test.
- [ ] **Exotic TIFFs:** Add `YCbCr`, `Palette` (`ColorMap` expansion on Display track; nearest-neighbor raw class indices on Scientific track), and `PlanarConfiguration = 2`.
- [ ] **Tests, docs & `old/` cleanup:** Port `jp2_tests.rs`, `excerpt_tests.rs`, JP2 sections of `reads_tests.rs`, plus malformed/truncated-input & `proptest` tests; write `docs/decisions/0010-jp2-and-netcdf-codecs.md` and `docs/architecture/raster-pipeline.md` (migrating `old/docs/imagery.md` lines 311–558); delete `codec.rs`, `jp2.rs`, `excerpt.rs`, `jp2_tests.rs`, `excerpt_tests.rs`, and `old/docs/decisions/0002` & `0004`.
- **Exit criteria:**
  - *Automated CI:* 5/3 JP2 round-trip, excerpt, exotic TIFF, and malformed/truncated-input tests pass with zero panics or UB.
  - *Manual QA:* Local Sentinel-2 JP2 granules and YCbCr/Palette/Planar-2 GeoTIFFs open and render cleanly in 2D/3D; `plan.md` `Next Step` updated.

---

### Slice 4B — Copernicus Data Space (CDSE) S3/JP2 & Resumable Downloads `[ ]`
- [ ] **CDSE provider & SigV4 re-signing (`veldmap-core::provider`, `veldmap-core::net`):** Port `s3`, `catalogue`, `scene`, `imagery`, and `manifest` implementing `Signer`; upgrade `imagery` to multi-band Sentinel-2 `.SAFE` asset discovery (`B02`–`B12`, `SCL`, `MTD_MSIL2A.xml`); add epoch-based singleflight header 401/403 re-sign (unsigned `Range` invariant) and **AOI-scoped two-phase JP2 streaming** (`PVI` quicklook placeholder + `TileScheduler` `coarse_start_override` so zoomed-in viewing skips the 121-tile coarse pass).
- [ ] **Library & hardened resumable downloads (`veldmap-core::library`):** Port `catalog`, `storage`, and unified resumable `.part` downloads (from `old/code/network/download.rs` + `old/code/library/download.rs`) for selected files/bands: S3 path sanitization, Windows `\\?\` extended-length paths (plus `longPathAware` manifest), persisted `ETag` `If-Range`, HTTP 416 recovery, `File::try_lock()` + in-process lock map + explicit `unlock()`, `downloaded == total_size`, `write_atomic_durable()`, and 401/403 re-sign.
- [ ] **Credentials & Cache Settings UI:** In-app dialog writing `ProjectDirs/.env` (`0600` on Unix), mandatory credential/SigV4 redaction in `tracing` logs & errors, optional CWD fallback, and anonymous search without keys.
- [ ] **Tests, docs & `old/` cleanup:** Port `old/tests/provider/` + prefetch section of `reads_tests.rs` + mock 401/403/`.part` crash-resume & secret-redaction tests; amend `0004-two-tier-atomic-durability.md` and `0008-network-streaming-and-stac.md`; write `docs/architecture/cdse-and-storage.md` (migrating `old/docs/download.md` & CDSE sections of `old/docs/imagery.md`); delete `old/code/{provider,library,network}/`, `old/tests/provider/`, `old/docs/download.md`, `old/docs/imagery.md`, and `old/docs/decisions/0005-network-reading.md`.
- **Exit criteria:**
  - *Automated CI:* CDSE provider, mock 401/403 & `.part` resume, and secret-redaction log tests pass.
  - *Manual QA:* Users can browse CDSE S3, stream remote Sentinel-2 JP2 granules in-place, and pause/resume downloads reliably; `plan.md` `Next Step` updated.

---

### Slice 5B — Analytical Terrain Relief & Multi-Temporal Timeline `[ ]`

#### Slice 5B-1 — Analytical Terrain Relief Layers
- [ ] **GPU hillshade, slope & aspect:** Compute derivatives over `528×528` guttered `R32Float` DEM tiles using per-vertex **2×2 inverse Jacobian `J^(-T)`** into local ENU meters (handling EPSG:4326 `cos(lat)`, COP-DEM latitude bands, and UTM/Polar rotation), scaled to the carrier tile's pyramid level, with one-sided fallback on `NaN` boundaries.
- [ ] **Relief compositing & COP-DEM GLO-30:** Add **Multiply** blend mode for relief-over-imagery compositing and out-of-the-box **Copernicus DEM GLO-30 COG** streaming. Amend `docs/decisions/0009-band-math-and-analytical-relief.md`.

#### Slice 5B-2 — Multi-Temporal Timeline, Dual-Epoch Diffing & Time-Series Plot
- [ ] **Multi-temporal timeline scrubber:** Reuse unified `TileKey` for epoch ingestion from STAC/CDSE AOI search (grouped by MGRS tile + AOI coverage, defaulting to L2A) or local dated folders, cloud-cover filtering, adjacent-date prefetch, and instant stale-request cancellation.
- [ ] **Unified offscreen dual-epoch cross-fade & diffing:** Render each epoch to an offscreen target and run fullscreen cross-fade (`mix`) or physical reflectance diffing (`Date B − Date A`) — cleanly supporting multi-band + `SCL`, mismatched carrier levels, and non-co-registered swaths without expanding `VisualLayer`.
- [ ] **1D pixel time-series plot:** Add a lightweight `egui_plot` panel showing reflectance/NDVI over time at a clicked location (skipping cloud-masked dates).
- **Exit criteria:**
  - *Automated CI:* Analytical relief (Jacobian/gutter/NaN) and temporal diffing shaders pass `naga` SPV/HLSL/MSL + headless GPU smoke tests.
  - *Manual QA:* End-to-end multi-temporal EO workflow (local + STAC/CDSE, band math, swipe/diff, pixel time-series plot, analytical DEM relief) persists cleanly across sessions; `plan.md` `Next Step` updated.

---

### Slice 6 (v1.1) — NetCDF-4 (`hdf5_pure`, Sentinel-3/5P) & Final Legacy Retirement `[ ]`
- [ ] **Audit & harden `hdf5_pure`:** Verify `Send + Sync`, HDF5 Shuffle+Deflate+Fletcher32 support, `read_f64_region`, and raw typed integer reads (`i16`/`u16`/`i32`).
- [ ] **Port `netcdf.rs`:** Read coordinates and `scale_factor`/`add_offset` in **`f64`**, check raw typed `_FillValue`/`missing_value`/`valid_range` *before* unpacking (mapping fill → `f32::NAN`), fail fast on unsupported HDF5 filters at `describe`, bound `Hdf5HandleCache` (`METADATA_CACHE` + `ROW_CACHE_CAP`) under `MemoryBudget` (`Shrinkable` LRU), support local `.SAFE` sibling geolocation, and add NetCDF variable switcher UI.
- [ ] **Tests, docs & final `old/` deletion:** Port `netcdf_tests.rs` + NetCDF section of `reads_tests.rs` + malformed-input tests; amend `docs/decisions/0010-jp2-and-netcdf-codecs.md`, `docs/architecture/raster-pipeline.md`, and `docs/architecture/georeference.md`; verify against `old/README.md` that zero unmigrated files remain, delete `old/` completely, remove the `old/` exclusion from CI no-Cyrillic check, and finalize `README.md`.
- **Exit criteria:**
  - *Automated CI:* Full 3-OS CI passes with `old/` removed, zero Cyrillic exclusions, and NetCDF `f64`/fill-value/malformed-input tests green.
  - *Manual QA:* Local and remote Sentinel-3/5P NetCDF-4 files open, switch variables, and render cleanly in 2D/3D; full `docs/` set is complete and link-checked.

---

## Next Step

- **Current state:** Phase 0 is complete on `main`; `plan.md` has been refined with deduplicated guardrails, fixed module/cache/rendering/migration contracts, PR-sized checkboxes, split mega-slices (`3A-1`/`3A-2`, `4A`/`4B`, `5B-1`/`5B-2`), zero-auth STAC/COG + band math pulled ahead of `openjp2`/CDSE, and NetCDF-4 scheduled cleanly in Slice 6 (v1.1).
- **Immediate task:** Start **Phase 0.5** (fix `rust-version = "1.95"` + `rust-toolchain.toml`, `.gitattributes`, dev profile opt-level, `veldmap-app` scaffold fixes, delete unused `assets/fonts/*.ttf`, clean up `README.md`/`.env.example`/`.gitignore`/`old/README.md` + `legacy-reference-snapshot` tag, add workspace lints + lean Linux CI with `cargo-deny`, and seed minimal foundational docs).
