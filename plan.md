# VeldMap Development Plan

## Goal

Build VeldMap as a fast, purely native Rust desktop **Analytical Earth-Observation & Precision-Imagery Workstation** on a **Precision Nadir 3D WGS84 Globe** for:
- Interactive nadir 3D WGS84 planetary viewing (global basemap + precision local/satellite overlays down to centimetre-scale orthophotos),
- Multi-format satellite imagery inspection (GeoTIFF/BigTIFF/COG, PNG/JPEG, Sentinel-2 JPEG 2000; NetCDF-4/HDF5 in v1.1) both locally and streamed in-place over HTTP Range from zero-auth STAC/COG catalogs, public HTTP COG URLs, and Copernicus Data Space (CDSE),
- Interactive GPU band math (NDVI/NDWI/false-color with metadata-driven radiometric scaling and cloud masking), analytical terrain relief (hillshade/slope/aspect), multi-temporal timeline scrubbing, and pixel time-series inspection.

A simplified browser/mobile companion is intentionally deferred to a separate project later (enabled by keeping `veldmap-core` UI/GPU-free with minimal default-off dependencies).

> **Document lifecycle:** `plan.md` is the executable slice roadmap and exit-criteria tracker (update `Next Step` as each slice closes). Authoritative design contracts live in focused ADRs under `docs/decisions/` and in `docs/architecture/`, written alongside the slice that introduces or proves them. Once a slice closes and its ADR lands, link the ADR from the slice header and treat the ADR as authoritative.

---

## Architectural Guardrails

Authoritative contracts are maintained in [ADR 0001](docs/decisions/0001-native-workspace-and-threading.md) (workspace layout, `veldmap-core` acyclic module DAG, dependency exclusions, frame-driven loop, custom bounded `std::thread` decode pools, and shutdown order) and [ADR 0002](docs/decisions/0002-globe-model-and-precision.md) (committed v1 smooth nadir WGS84 ellipsoid, `MIN_HEIGHT_M = 2.0` m, Reverse-Z `f64`, double-single RTE / per-tile RTC, exact sRGB compositing, and overlay accuracy tolerances). Key cross-slice invariants:

1. **Deadlock-free memory & deduplication:** Lock-free `AtomicU64` `MemoryBudget` (FIFO slow-path waiter queue; never call while holding cache locks; snapshot `Vec<Arc<dyn Shrinkable>>` before shrinking); caches only use `try_acquire()` + self-eviction; decode jobs reserve upfront private working-set `Peak` atomically; singleflight is waiter-refcounted under its map mutex with a leader `Drop` guard.
2. **Pure-Rust codec safety & two-tier durability:** Pure-Rust decoders with explicit memory/IFD/padding/shift/varint limits, panic isolation, FFI soundness fixes on `openjp2`, and thread-safety audits before multi-threaded decode. Injected `StoragePaths` with two-tier atomic writes (`write_atomic_durable()` with `sync_all()` + synchronous handle drop before rename vs `write_atomic_cached()` without `sync_all()` + read validation; both using per-boot `.<boot_id>.<seq>.tmp` files). Add traits only where there is a real second implementation (`ByteSource`, `Signer`, raster decoders, tile providers).
3. **`GpuTile` hygiene:** Owns only `Texture`, `TextureView`, `PixelFormat`, dimensions, true byte size (including mips), and monotonic `tile_id: u64` — never pipeline-specific `BindGroup`s or `view_formats`.
4. **Reuse from `old/`, rewrite glue cleanly, retire incrementally:** Pre-deletion state is tagged `legacy-reference-snapshot`; remaining files are tracked in `old/README.md` with `Consumer Slices` and `Delete After Slice` (always the *last* consumer slice). Never link from `docs/` into `old/`; delete `old/` completely at Slice 6.
5. **Code hygiene (every slice):** English only across repository files outside `old/` (CI-enforced; remove `old/` exclusion at Slice 6). Rust 2024; verified by `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `cargo-deny`, `naga` SPV/HLSL/MSL codegen, headless offscreen `wgpu` smoke tests, `veldmap-core` DAG checks, forbidden `docs/ -> old/` link guard, and offline `lychee` link checks.

---

## Vertical Slices

> **Execution order note:** Slices are ordered by user value rather than strict numeric sequence—zero-auth STAC COGs (`4A`) and interactive Band Math (`5A`) land before Sentinel-2 JP2 (`3C-1`) and authenticated CDSE S3 (`4B`).

### Phase 0 — Clean Native Bootstrap `[x]`
- [x] Back up legacy repository and initialize fresh `main` repository with curated `old/` reference and `assets/fonts/`.
- [x] Scaffold native `veldmap-core` + `veldmap-app` (`eframe` 0.37 + `egui_wgpu` + `wgpu` 30, `Backends::PRIMARY`) passing `fmt`, `clippy -D warnings`, and `test`.

---

### Phase 0.5 — Slim Pre-Slice 1 Hygiene, Licensing & CI `[x]` ([ADR 0001](docs/decisions/0001-native-workspace-and-threading.md))
- [x] **Toolchain & dev ergonomics:** Fix `rust-version = "1.95"` in `Cargo.toml` (matching `egui 0.37` / `wgpu 30`), pin `rust-toolchain.toml`, add `.gitattributes` (LF endings, `old/** linguist-vendored`), and add `[profile.dev.package."*"] opt-level = 2`.
- [x] **Scaffold fixes in `veldmap-app`:** Pass `response.rect` guarded by `is_positive()`, drop redundant direct `egui`/`egui-wgpu` deps in favor of `eframe` re-exports, honor `WGPU_BACKEND` (excluding GL) / `WGPU_POWER_PREF`, initialize `tracing`, and hook `device.on_uncaptured_error`.
- [x] **Remove mixed-license font bloat:** Delete unused `assets/fonts/*.ttf` (~6.3 MiB, closing the OFL + Nerd Fonts license gap) and keep `eframe`'s `default_fonts` until UI polish.
- [x] **Repo & `old/` manifest cleanup:** Update `README.md`, `.env.example` (remove `data-provider` wording), `.gitignore` (`.env*` + `!.env.example`, drop `/runtime/*`), and `old/README.md` (remove personal Windows path, list omitted `network/download.rs` and `viewer/`, fix Cyrillic note across 28 files, add full 79-file `Consumer Slices` / `Delete After Slice` manifest). Create annotated tag `legacy-reference-snapshot`.
- [x] **Workspace lints & lean Linux CI:**
  - Add workspace lints: `unwrap_used`/`expect_used` as `"warn"` (with `clippy.toml` test exemptions; denied in CI via `-D warnings`), `allow_attributes_without_reason`, `undocumented_unsafe_blocks`, `unimplemented`, `large_futures` (standardize on `#[expect(..., reason = "INVARIANT: ...")]`).
  - Add Linux `hygiene-and-dag` GitHub Actions job (`Swatinem/rust-cache`, `fmt`, `clippy -D warnings`, `test`, `cargo-deny` for licenses/RUSTSEC/banned crates, no-Cyrillic check, offline `lychee --offline` `docs/` link check).
- [x] **Seed minimal docs:** Create `CONTRIBUTING.md`, `SECURITY.md`, `THIRD_PARTY_NOTICES.md`, `docs/README.md`, `docs/building.md`, `docs/decisions/README.md`, `docs/decisions/0001-native-workspace-and-threading.md` (workspace DAG, exclusions, frame-driven loop & shutdown order only), `docs/glossary.md`, and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* Green Linux workflow (`fmt`, `clippy`, `test`, `cargo-deny`, no-Cyrillic, offline `docs/` links).
  - *Manual QA:* `cargo run -p veldmap-app` launches cleanly with `tracing` logs; `legacy-reference-snapshot` tag exists; zero unlicensed font binaries remain.

---

### Slice 1 — Core Geodesy & Interactive 3D WGS84 Globe `[x]` ([ADR 0002](docs/decisions/0002-globe-model-and-precision.md))
- [x] **Port core geodesy & MGRS (`veldmap-core::geo`):** Port `geodesy`, `projection`, `camera` (including wheel zoom constants), `cull`, `outlines`, `footprint`, sag helpers, and **`mgrs`** (from `old/code/provider/mgrs.rs` into leaf `--no-default-features`) along with curated `old/tests/geo/` and `mgrs_tests.rs`.
- [x] **High-precision camera & culling:** Upgrade `Camera` to `DMat4` infinite Reverse-Z (`perspective_infinite_reverse_rh`), `height = 0.0`, `MIN_HEIGHT_M = 2.0` m, private internals behind view-agnostic queries, and `f64`/`DVec3` `Camera::project()`, `cull::Frame`, and `cull::Ball`.
- [x] **Offscreen `wgpu` WGS84 globe renderer (`veldmap-app`):**
  - Own `Rgba8UnormSrgb` + `Depth32Float` Reverse-Z target (`StoreOp::Discard` on depth), sized in `CallbackTrait::prepare()` via `ViewportInPixels::from_points()` (clamped `[1, max_texture_dimension_2d]`, skipped on `0×0`, self-contained dirty-key redraw skipping covering viewport size, camera pose, `show_graticule`, `scene_generation`, and dropped-encoder `pending_present` recovery).
  - Composite 1:1 with `FilterMode::Nearest` via `view_formats` matching target color space (linear RGB palette constants, no `encode_srgb` hacks).
  - Watertight double-single RTE globe geometry (`(pos_hi, pos_lo)` + `neg_eye` halves in `CameraUniform` in a single draw call) + depth-less draped 15° graticule (`depth_write: false`, `depth_compare: Always`, fragment ellipsoidal-normal horizon discard `dot(N_ellipsoid(P), normalize(E - P)) > 0.0`). Scene outline/ribbon/hatch pipelines land with scene outlines in Slice 4A.
  - Mouse orbit, scroll-wheel & pinch zoom-at-cursor, and smooth quaternion flights.
- [x] **Baseline GIS HUD:** Cursor Lat/Lon, altitude & m/px, dynamic `mm`–`km` metric scale bar, North compass / Reset-North (`N`), and Go-To Coordinates (DD/DMS) or MGRS jump with clipboard support.
- [x] **3-OS CI, `naga` & headless GPU tests:**
  - Add CPU-only `naga` (pinned to `wgpu` major) WGSL validation + SPV/HLSL/MSL codegen test (with realistic HLSL/MSL options).
  - Add 3-OS `build-and-test` CI matrix (`ubuntu`/`windows`/`macos`) with headless offscreen `wgpu` smoke test verifying Reverse-Z close-up rendering, `fs_grid` horizon discard, dropped-encoder recovery, and exact sRGB compositing (`VELDMAP_REQUIRE_GPU_TEST=1` on Linux Lavapipe w/ `mesa-vulkan-drivers` & Windows WARP, retrying `force_fallback_adapter: true`; graceful skip on macOS VMs).
- [x] **Docs & `old/` cleanup:** Write `docs/decisions/0002-globe-model-and-precision.md` (recording v1 flat nadir ellipsoid from legacy `0006` and superseding `0007`'s 800 m floor with all 5 benchmark bindings & drift tolerances) and `docs/architecture/globe.md`; update `docs/glossary.md` & `docs/limitations.md`; delete ported `old/code/geo/*` (keeping `overlay.rs`), `old/code/provider/mgrs.rs`, `old/tests/geo/`, `old/tests/provider/mgrs_tests.rs`, `old/docs/globe.md`, and `old/docs/decisions/0006-terrain-on-the-globe.md` (**retain** `0007-overlay-accuracy.md` until Slice 3A-1).
- **Exit criteria:**
  - *Automated CI:* Green 3-OS CI, `naga` SPV/HLSL/MSL codegen, `veldmap-core::geo` + MGRS tests, and headless Reverse-Z/horizon-discard/sRGB GPU smoke test.
  - *Manual QA:* Smooth 60 fps zoom from orbit to 2 m altitude with zero jitter, horizon bleed, or resize panic; Go-To Lat/Lon & MGRS (including clipboard paste) lands accurately.

---

### Slice 2A — Offline Planetary Quadtree, `TileKey`/`GpuTileCache` & Bundled Basemap `[ ]`
- [ ] **Global WGS84 tile quadtree & static mesh retirement:** Implement `z/x/y` quadtree over `TileScheme` (`WebMercator | Geodetic | RasterPyramid { max_level: u8 }`) with **per-tile RTC** meshes (`tile_origin - eye` in `f64` on CPU, shared static index buffer), a **dedicated fixed-zoom polar cap mesh** (e.g., `z = 2`–`3` with skirts along `±85.051°`, avoiding long sliver triangles at high zoom), sagitta skirts across mixed-LOD edges, and HiDPI-aware SSE subdivision. Delete `mesh.rs`'s static `96×192` ellipsoid body (rendering `"Ellipsoid-only"` untextured through the quadtree) and upgrade the graticule for close zoom (procedural/shader graticule on quadtree tiles or local adaptive subdivision, eliminating the static 672-segment parallel's ~35 m horizontal chord parallax).
- [ ] **Unified `TileKey`, `TileCache` & `GpuTileCache`:**
  - Define `TileKey: Copy + Eq + Hash` (`SourceId`, `BandKey`, `TileScheme`, `PixelFormat`, `DecodeRev`, `Level`, `X`, `Y`) with deterministic `Copy` `BandKey` (`Display | Index(u16) | NamedHash(u64)`) and implement pure O(1) LRU `TileCache<K, V>` in `veldmap-core::tile` (scheme-aware ancestor touch, `<50%` byte appetite cap per pyramid).
  - Own `GpuTileCache` and `PendingUploads` directly on `VeldmapApp` (cloning `Arc`-backed `device`/`queue`, passing `Arc<GpuTile>` draw snapshots to callbacks + small per-pipeline `BindGroup` LRU keyed by `[tile_id; N]`, bumping `scene_generation` strictly on non-camera scene mutations such as uploads/evictions/basemap switches). Drain the full completion channel each frame, prune stale entries, upload coarsest-first (4–8 MiB / ~2 ms cap, ≥ 1 tile/frame), request repaint while non-empty, and trigger redraws via injected `Wakeup`.
- [ ] **Bundled offline basemap, early `codecs` PNG/JPEG decoding & basemap switcher:** Bundle a single public-domain low-res global equirectangular basemap texture (NASA Blue Marble Next Gen or Natural Earth, ~1–3 MiB; document provenance in `THIRD_PARTY_NOTICES.md` and `assets/README.md`, mark `*.png`/`*.jpg binary` in `.gitattributes`), introduce `veldmap-core`'s `codecs` feature early for lightweight pure-Rust PNG/JPEG decoding, slice/serve the bundled basemap as a virtual offline `z=0..3` tile provider through `PendingUploads` → `GpuTileCache` (exercising the full tile upload/eviction/fallback path offline), and wire the **Satellite / Ellipsoid-only** switcher.
- [ ] **CI DAG check & ADR 0003:** Add `veldmap-core` feature-matrix scaffolding (`--no-default-features`, `codecs`, `--all-features`) to CI; write `docs/decisions/0003-tile-identity-and-gpu-cache.md` and seed `docs/architecture/viewing-pipeline.md`.
- **Exit criteria:**
  - *Automated CI:* `TileCache<K, V>` (both `WebMercator` and `RasterPyramid { max_level }`), quadtree subdivision/polar-cap tests, and headless GPU tile-upload smoke test pass.
  - *Manual QA:* Offline planet zooms smoothly from orbit to street scale with zero T-junctions, polar seams, or close-zoom graticule parallax, and bounded VRAM.

---

### Slice 2B — Async Tile Streaming, Disk Cache & Live Satellite Basemaps `[ ]`
- [ ] **Async-waking `CancelToken`, tile fetcher & minimal CPU decode pool (`veldmap-core::io`, `veldmap-core::net`):**
  - Implement leaf `CancelToken` (`AtomicBool` + zero-dep `std::task::Waker` / listener hook so `tokio::select!` aborts immediately on pan/shutdown).
  - Implement `tokio` + `reqwest` (`rustls`) tile fetcher with per-host concurrency limits, compliant `User-Agent`, replaceable visible-first queue, and immediate cancellation.
  - Introduce the lightweight bounded `std::thread` CPU decode pool (reused and extended with `MemoryBudget` in Slice 3A-1a) so live PNG/JPEG basemap tiles never decode on the main thread or starve `tokio` workers, with deterministic shutdown join order per ADR 0001.
- [ ] **Disk cache & LRU eviction (`veldmap-core::cache`):** Port `old/code/cache/` (`store`, `layout`, `tile`, `evict`) and `evict_tests.rs` using injected `StoragePaths`, two-tier atomic writes (`write_atomic_cached()` with no `sync_all()` + read validation and `write_atomic_durable()` with `sync_all()` + synchronous `into_std().await` drop before rename; both using `.<boot_id>.<seq>.tmp` and Windows sharing-violation handling), and foreign-UUID startup sweep.
- [ ] **Legally clean live basemaps:** Wire NASA GIBS (static layer or explicit `{Time}`, `max_level` ~8–9) + EOX Sentinel-2 Cloudless (CC BY 4.0, `max_level` ~13–14) with visible attribution, explicit per-source `max_level` clamping + ancestor overzooming (preventing 400/404 storms at deep zoom), Custom XYZ/TMS URL & API-key settings (defer a default street "Map" layer unless a legally compliant tile source is chosen), and graceful offline fallback to the bundled Slice 2A basemap.
- [ ] **CI DAG feature matrix, 3-OS Clippy, ADR 0004 & `old/` cleanup:** Enforce `--no-default-features`, `fs`, `codecs`, `net`, `--all-features` and intra-feature import checks in CI; expand `cargo clippy --workspace --all-targets -- -D warnings` to Windows/macOS in `build-and-test` now that `#[cfg(windows)]`/`#[cfg(unix)]` code lands; write `docs/decisions/0004-two-tier-atomic-durability.md`; delete `old/code/cache/` and `old/tests/cache/evict_tests.rs`.
- **Exit criteria:**
  - *Automated CI:* `evict_tests`, atomic-write/temp-sweep, `CancelToken`, 3-OS Clippy, and `veldmap-core` DAG checks pass.
  - *Manual QA:* Live NASA GIBS / EOX tiles stream smoothly with visible attribution, bounded disk cache, and clean offline fallback.

---

### Slice 3A-1 — Local GeoTIFF/PNG/JPEG Engine, `MemoryBudget` & Warp-Mesh Draping `[ ]`

#### Slice 3A-1a — Headless Raster Engine, `ByteSource`, `MemoryBudget` & `TileScheduler`
- [ ] **Sync `ByteSource` & `ContentId` (`veldmap-core::io`):**
  - Define strictly sync caller-buffered `ByteSource::read_at(&self, offset: u64, buf: &mut [u8], cancel: &CancelToken) -> Result<usize, IoError>` with safe positioned-I/O local file impl (no `memmap2`; buffered 64–256 KiB cursor; pooled `File` handles + short-read loop on Windows).
  - Key disk/VRAM tiles by sampled BLAKE3 `ContentId` (`len || head_64k || mid_64k || tail_64k`, domain-separated) paired with `BandKey`, `TileFormat`, `DecodeRev`, and gutter width in disk paths; use `(path, len, mtime_ns)` strictly as in-memory stat cache + UI "Reload / bypass cache".
- [ ] **Core raster pipeline (`veldmap-core::raster`, `codecs` feature):**
  - Port `pyramid`, `resample`, `radiometry`, `cascade`, `table`, `grid` (introducing `TilePixels` + `PixelSpec { bpp, gutter }` with `gutter = 0` so 3B extends rather than rewrites), core `tiff` (standard RGB/Gray GeoTIFF/BigTIFF/COG, `GeoDoubleParamsTag`/`GeoAsciiParamsTag`, explicit `Limits` + `MAX_IFDS`, verifying Float32 `Predictor = 3` and pure-Rust ZSTD decoding via `ruzstd`—banning C `zstd-sys` in `deny.toml`), and `png`/`jpeg`/`full` (strip `!Send` `Rc`/`RefCell` via `OnceLock`/`Arc`/`AtomicU64`).
  - Wire 8-bit `Rgba8UnormSrgb` + QOI disk cache (`write_atomic_cached()` + header/length validation).
- [ ] **Memory budget, singleflight & `TileScheduler`:**
  - Wire lock-free `AtomicU64` `MemoryBudget` (FIFO slow-path waiter queue to prevent large-job starvation, upfront private-working-set `Peak` without double-counting shared `DecodedChunkCache`, snapshot `Shrinkable` hooks) and atomic waiter-refcounted singleflight decoded-chunk LRU with panic Drop guard into the bounded local CPU decode pool.
  - Replace legacy bus-era `Fetch`/`Passes` with native `TileScheduler` (`veldmap-core::tile`: 1 active pass per source for `Serve::Pass` only — allow concurrent `Serve::Pointwise`; non-cancellation of coarser passes; terminal-failure suppression; optional `coarse_start_override` ready for 4B).
  - Port TIFF sections of `reads_tests.rs` via `CountingByteSource` plus `MemoryBudget`/singleflight/`TileScheduler` concurrency tests.

#### Slice 3A-1b — GPU Warp-Mesh Draping, Minimal Local Opener & Accuracy Regression
- [ ] **Warp-mesh draping & `VisualLayer`:**
  - Split `old/code/geo/overlay.rs`: top-down lazy `DVec3` culling balls (preserving per-rung limb-culling margins) in `veldmap-core`; `VisualLayer` (`preview: Option<RasterSlot>` + `bands: SmallVec<[RasterSlot; 4]>`) + per-tile RTC warp meshes + horizon discard in `veldmap-app`.
  - Wire minimal CLI file argument + window drag-and-drop file opener in `veldmap-app` so local GeoTIFF/COG/orthophoto Manual QA works directly in 3A-1 (full file picker, Layer Manager, and CRS override UI land in 3A-2).
  - Rebuild ADR-0007's `a_raster_lands_where_its_binding_puts_it` regression test (all 5 benchmark bindings & drift tolerances).
- [ ] **Docs & `old/` cleanup:** Write `docs/decisions/0005-raster-reading-model-and-memory-budget.md` and finish `docs/architecture/viewing-pipeline.md`; delete `overlay.rs`, `budget.rs`, `fingerprint.rs`, `full.rs`, `tiles_pipeline.rs`, `cascade.rs`, `pyramid.rs`, `resample.rs`, `radiometry.rs`, `table.rs`, `grid.rs`, `tiff.rs`, `png.rs`, `jpeg.rs`, `pyramid_tests.rs`, `radiometry_tests.rs`, `resample_tests.rs`, `tiff_tests.rs`, `viewing-pipeline.md`, `0003-raster-reading-model.md`, and `0007-overlay-accuracy.md` (**retain** `reads_tests.rs` until Slice 6).
- **Exit criteria:**
  - *Automated CI:* `a_raster_lands_where_its_binding_puts_it`, TIFF `reads_tests`, `TileScheduler`, and `MemoryBudget`/singleflight concurrency tests pass.
  - *Manual QA:* Local GeoTIFF/BigTIFF/COG and cm-scale drone orthophotos open via CLI/drag-and-drop and drape accurately on the globe with responsive zooming and bounded RAM/VRAM/disk.

---

### Slice 3A-2 — Local Workstation UX, Expanded Pure-Rust CRS & GeoJSON Overlay `[ ]`
- [ ] **Expanded pure-Rust CRS & world files:** Extend `veldmap-core::geo::projection` with `ETRS89/UTM` (`258xx`), `NAD83/UTM` (`269xx`), `LAEA Europe` (`3035`), Polar Stereographic (`3413`/`3031`), `GeoKey 3072 == 32767` user-defined TM parameters, wire existing `Datum::to_wgs84` into geographic bindings (preventing NAD83/Pulkovo-42 shifts), read `.tfw`/`.jgw`/`.pgw` world files in `fs`/`app` (injected as external `Option<Placement>`), and show visible UI warnings for unshifted foreign datums.
- [ ] **Local workstation UX & geodesic tools:** Native file picker (extending 3A-1b's CLI + drag-and-drop opener), Layer Manager (visibility, reorder, opacity, zoom-to-binding, CRS/radiometry metadata inspector), searchable CRS picker with presets & custom TM parameters (unlocking all MSK-xx zones), WGS84 geodesic distance/area measurement tools, **read-only GeoJSON overlay (stroke-only or earcut triangulation — never convex fan fills)**, and diagnostic banners (fatal vs partial error, detail-limit caption, loading shimmer).
- [ ] **Docs:** Write `docs/decisions/0006-pure-rust-georeferencing-and-crs.md` and seed `docs/architecture/georeference.md` (**retain** `old/docs/georeference.md` until NetCDF sections migrate in Slice 6).
- **Exit criteria:**
  - *Automated CI:* Expanded CRS/datum round-trip tests and GeoJSON parsing/stroke tests pass.
  - *Manual QA:* Local projected/world-file orthophotos and concave GeoJSON field polygons open cleanly via drag-and-drop or CRS override.

---

### Slice 3B — Scientific `R32Float` Track, Live Stretch, Colorbar & 2D Inspector `[ ]`
- [ ] **Dual GPU track (`Display` + `Scientific`):**
  - Keep Display (`Rgba8UnormSrgb`, `512×512`) and add Scientific (`R32Float`, storing **raw numeric samples** with nodata/fill → `f32::NAN`, up/down-casting `u32`/`i32`/`f64` to `f32`, and applying `(val + offset) * scale` in WGSL uniforms + cursor readout to avoid cache aliasing).
  - Use **`528×528` (`512` interior + `8`-px apron) with `mip_level_count = 4`** (`528 → 264 → 132 → 66`) on Scientific tiles (exact divide-by-2 phase alignment, ≥1-px gutter at all 4 mips, smooth central-difference normals for 5B-1); normalize `cell_uv` against `valid_w`/`valid_h` and account for the `±8`-px halo in `Grid::footprint()`, `DecodedChunkCache`, and `Cascade` band overlap.
- [ ] **WGSL nodata-aware filtering, categorical handling & stretch:**
  - Manual 4-tap `textureLoad` bilinear filter using `(bitcast<u32>(v) & 0x7FFFFFFFu) >= 0x7F800000u` (masking both `NaN` and `±Inf`) and uniform-control-flow `dpdx`/`dpdy` LOD selection (**computed before any horizon `discard`** so `naga` validation stays satisfied).
  - Enforce **nearest-neighbor sampling & non-averaging downsampling** for categorical bands (raw `Palette` classes / future `SCL`).
  - Live contrast stretch (linear percentile/min-max + **`log10` / dB stretch** for SAR) and colormap uniforms.
- [ ] **Multi-band GeoTIFF band selection & scientific UI:** Allow selecting individual bands from multi-band GeoTIFFs/COGs (unblocking multi-band drone orthophotos), interactive stretch histogram, and colorbar legend with physical units.
- [ ] **2D Inspector split pane:** Port `old/code/viewer/` into an internal split pane sharing `Arc<GpuTile>` with the 3D globe + cursor raw DN & physical-unit readout.
- [ ] **GPU smoke test, docs & `old/` cleanup:** Add headless `wgpu` smoke test verifying NaN/Inf bitmask bilinear math and `528×528` mip/gutter continuity; write `docs/decisions/0007-scientific-r32float-track-and-gutters.md`; delete `old/code/viewer/`.
- **Exit criteria:**
  - *Automated CI:* Headless `wgpu` nodata-bilinear/`528×528`-mip test, `naga` SPV/HLSL/MSL codegen, and `R32Float` radiometry tests pass.
  - *Manual QA:* 16-bit/float32/SAR/multi-band rasters render with exact nodata transparency, zero shimmer or tile-edge seams, live histogram/colorbar, and synchronized 2D/3D cursor readout.

---

### Slice 4A — Zero-Auth Cloud EO Streaming (HTTP Range + STAC / Public COGs) `[ ]`
- [ ] **HTTP Range `ByteSource` (`veldmap-core::net`):** Port `old/code/network/range.rs` core over `reqwest` (`rustls`)/`tokio`: 64 KiB O(1) LRU `HttpBlockPool` (`Shrinkable` under `MemoryBudget`), combined open-probe + `0..64 KiB` head read + concurrent `mid_64k`/`tail_64k` `ContentId` reads (`416`-safe on 0-byte files), consumed-only readahead, atomic waiter-refcounted multi-block singleflight, auto-decompression disabled, `Content-Range` + strong-ETag `If-Match` validation, `CancelToken` async abort, and `Handle::try_current().is_err()` guard on blocking reads.
- [ ] **Two-stage pinned remote decode:** Add pure sync chunk range-planners in `raster` and `HttpRange::prefetch_pinned()` in `net` returning memory-only `PinnedByteSource` (`PreparedBatch` under upfront `MemoryPermit`) routed to the fast CPU decode lane, isolating un-prefetched blocking reads to a separate capped remote lane.
- [ ] **Zero-auth STAC & public COG provider (`veldmap-core::provider`):** Implement STAC API search (e.g., AWS Earth Search Sentinel-2 L2A COGs + COP-DEM GLO-30) and direct public HTTP COG URLs, multi-band Sentinel-2 asset discovery + STAC `raster:bands` scale/offset metadata, AOI search (viewport / drawn box / GeoJSON polygon + GeoJSON/WKT export), scene outlines on globe (wiring HiDPI-scaled outline/ribbon/hatch pipelines in `veldmap-app` using per-cluster RTC or guarded vertex packing in `veldmap-app`), and built-in zero-auth sample scenes for 15-second onboarding.
- [ ] **Mock HTTP harness & docs:** Build offline local mock HTTP Range harness (`206`, `200` rejection, `416`, strong/weak `ETag`, delivered-byte ceilings); write `docs/decisions/0008-network-streaming-and-stac.md` and `docs/configuration.md`.
- **Exit criteria:**
  - *Automated CI:* Offline mock HTTP Range/STAC/singleflight tests pass in CI.
  - *Manual QA:* Users can search an AOI and stream public Sentinel-2 L2A COGs and HTTP COG URLs with zero credentials without starving local SSD decodes.

---

### Slice 5A — Interactive Band Math, Split-Screen Swipe & Session Persistence `[ ]`
- [ ] **WGSL band math & radiometric correctness:** Interactive WGSL band math (NDVI, NDWI, false-color, custom stretch) over co-registered multi-file / mixed-resolution band recipes (e.g., Sentinel-2 10 m `B04`/`B08` + 20 m `B11`/`B12`) on the `R32Float` track (working on both local multi-band GeoTIFFs and zero-auth STAC COGs).
- [ ] **Metadata-driven scaling & `SCL` masking:** Apply Sentinel-2 `BOA_ADD_OFFSET`/`RADIO_ADD_OFFSET` from STAC `raster:bands` and local GeoTIFF metadata (never hardcoded to a 2021↔2022 date cutoff; `.SAFE` `MTD_MSIL2A.xml` parsing lands with CDSE `.SAFE` discovery in Slice 4B), badge L1C (TOA) vs L2A (BOA), and support optional **nearest-neighbor `SCL` cloud/shadow masking** (`classes 0, 1, 3, 8, 9, 10`).
- [ ] **Split-screen swipe curtain:** Interactive swipe comparison across any two layers (local or remote, including non-co-registered swaths).
- [ ] **Session persistence & export:** Workspace/session persistence via `write_atomic_durable()` JSON (layers, CRS overrides, band math recipes, stretch params, camera bookmarks, split-pane layout), copy-coordinates, and viewport screenshot export (with optional scale bar, north arrow, colorbar, and acquisition date; GeoTIFF/CSV export deferred to post-v1).
- [ ] **Docs:** Write `docs/decisions/0009-band-math-and-analytical-relief.md`.
- **Exit criteria:**
  - *Automated CI:* Metadata-driven radiometric scaling, nearest-neighbor `SCL` mask, session JSON round-trip, and `naga` SPV/HLSL/MSL shader tests pass.
  - *Manual QA:* Multi-band Sentinel-2 NDVI/NDWI/false-color with cloud masking, swipe curtain, screenshot export, and session reload work smoothly.

---

### Slice 3C-1 — Sentinel-2 JP2 (`openjp2`) & Exotic TIFFs `[ ]`
- [ ] **Compile-time check:** Re-evaluate extracting `veldmap-raster` into its own workspace crate before adding `openjp2`.
- [ ] **Audit & harden `openjp2`:** Audit crate version & `static mut` thread safety (note `openjp2` is C2Rust-transpiled unsafe Rust, requiring fuzzing and panic isolation); port `codec.rs`, `jp2.rs` (`roxmltree`/`quick-xml` for GMLJP2; gate zero-as-nodata on Sentinel-2 heuristics, not all GMLJP2), and `excerpt.rs`:
  - Fix Stacked Borrows/drop-order UB (`Box::into_raw` + explicit `Drop`), `usize::MAX` EOF + `ErrorKind::Interrupted` retry, `catch_unwind` on `extern "C"` callbacks, `skip()` EOF bounds check, `COD` `levels <= 32` shift-overflow guard, PLT varint & TLM checked arithmetic + `keep <= part.len`, `MAX_PAD` cap, worker panic isolation, and 5/3 reversible decode round-trip test.
- [ ] **Exotic TIFFs:** Add `YCbCr`, `Palette` (`ColorMap` expansion on Display track; nearest-neighbor raw class indices on Scientific track), and `PlanarConfiguration = 2`.
- [ ] **Tests, docs & `old/` cleanup:** Port `jp2_tests.rs`, `excerpt_tests.rs`, JP2 sections of `reads_tests.rs`, plus malformed/truncated-input & `proptest` tests; write `docs/decisions/0010-jp2-and-netcdf-codecs.md` and `docs/architecture/raster-pipeline.md` (migrating JP2/TIFF raster sections of `old/docs/imagery.md`, excluding NetCDF lines 427–535 until Slice 6); delete `codec.rs`, `jp2.rs`, `excerpt.rs`, `jp2_tests.rs`, `excerpt_tests.rs`, and `old/docs/decisions/0002` & `0004`.
- **Exit criteria:**
  - *Automated CI:* 5/3 JP2 round-trip, excerpt, exotic TIFF, and malformed/truncated-input tests pass with zero panics or UB.
  - *Manual QA:* Local Sentinel-2 JP2 granules and YCbCr/Palette/Planar-2 GeoTIFFs open and render cleanly in 2D/3D.

---

### Slice 4B — Copernicus Data Space (CDSE) S3/JP2 & Resumable Downloads `[ ]`
- [ ] **CDSE provider & SigV4 re-signing (`veldmap-core::provider`, `veldmap-core::net`):** Port `s3`, `catalogue`, `scene`, `imagery`, and `manifest` implementing `Signer`; upgrade `imagery` to multi-band Sentinel-2 `.SAFE` asset discovery (`B02`–`B12`, `SCL`, and `MTD_MSIL2A.xml` radiometric offset parsing); add epoch-based singleflight header 401/403 re-sign (unsigned `Range` invariant) and **AOI-scoped two-phase JP2 streaming** (`PVI` quicklook placeholder + `TileScheduler` `coarse_start_override` so zoomed-in viewing skips the 121-tile coarse pass).
- [ ] **Library & hardened resumable downloads (`veldmap-core::library`):** Port `catalog`, `storage`, and unified resumable `.part` downloads (from `old/code/network/download.rs` + `old/code/library/download.rs`) for selected files/bands: S3 path sanitization, Windows `\\?\` extended-length paths (plus `longPathAware` manifest), persisted `ETag` `If-Range`, HTTP 416 recovery, `File::try_lock()` + in-process lock map + explicit `unlock()`, `downloaded == total_size`, `write_atomic_durable()`, and 401/403 re-sign.
- [ ] **Credentials & Cache Settings UI:** In-app dialog writing `ProjectDirs/.env` (`0600` on Unix), mandatory credential/SigV4 redaction in `tracing` logs & errors, optional CWD fallback, and anonymous search without keys.
- [ ] **Tests, docs & `old/` cleanup:** Port `old/tests/provider/` + prefetch section of `reads_tests.rs` + mock 401/403/`.part` crash-resume & secret-redaction tests; amend `0004-two-tier-atomic-durability.md` and `0008-network-streaming-and-stac.md`; write `docs/architecture/cdse-and-storage.md` (migrating `old/docs/download.md` & CDSE sections of `old/docs/imagery.md`); delete `old/code/{provider,library,network}/`, `old/tests/provider/`, `old/docs/download.md`, and `old/docs/decisions/0005-network-reading.md` (**retain** `old/docs/imagery.md` until NetCDF sections migrate in Slice 6).
- **Exit criteria:**
  - *Automated CI:* CDSE provider, mock 401/403 & `.part` resume, and secret-redaction log tests pass.
  - *Manual QA:* Users can browse CDSE S3, stream remote Sentinel-2 JP2 granules in-place, and pause/resume downloads reliably.

---

### Slice 5B — Analytical Terrain Relief & Multi-Temporal Timeline `[ ]`

#### Slice 5B-1 — Analytical Terrain Relief Layers
- [ ] **GPU hillshade, slope & aspect:** Compute derivatives over `528×528` guttered `R32Float` DEM tiles using per-vertex **2×2 inverse Jacobian `J^(-T)`** into local ENU meters (handling EPSG:4326 `cos(lat)`, COP-DEM latitude bands, and UTM/Polar rotation), scaled to the carrier tile's pyramid level, with one-sided fallback on `NaN` boundaries.
- [ ] **Relief compositing & COP-DEM GLO-30:** Add **Multiply** blend mode for relief-over-imagery compositing and out-of-the-box **Copernicus DEM GLO-30 COG** streaming. Amend `docs/decisions/0009-band-math-and-analytical-relief.md`.

#### Slice 5B-2 — Multi-Temporal Timeline, Dual-Epoch Diffing & Time-Series Plot
- [ ] **Multi-temporal timeline scrubber:** Reuse unified `TileKey` for epoch ingestion from STAC/CDSE AOI search (grouped by MGRS tile + AOI coverage, defaulting to L2A) or local dated folders, cloud-cover filtering, adjacent-date prefetch, and instant stale-request cancellation.
- [ ] **Unified offscreen dual-epoch cross-fade & diffing:** Render each epoch to an offscreen target and run fullscreen cross-fade (`mix`) or physical reflectance diffing (`Date B − Date A`) — cleanly supporting multi-band + `SCL`, mismatched carrier levels, and non-co-registered swaths without expanding `VisualLayer`.
- [ ] **1D pixel time-series plot:** Add a lightweight 1D time-series panel (custom `egui::Painter` chart to avoid pinning bleeding-edge `egui` upgrades to `egui_plot`, or `egui_plot` if version-aligned) showing reflectance/NDVI over time at a clicked location (skipping cloud-masked dates).
- **Exit criteria:**
  - *Automated CI:* Analytical relief (Jacobian/gutter/NaN) and temporal diffing shaders pass `naga` SPV/HLSL/MSL + headless GPU smoke tests.
  - *Manual QA:* End-to-end multi-temporal EO workflow (local + STAC/CDSE, band math, swipe/diff, pixel time-series plot, analytical DEM relief) persists cleanly across sessions.

---

### Slice 6 (v1.1) — NetCDF-4 (`hdf5_pure`, Sentinel-3/5P) & Final Legacy Retirement `[ ]`
- [ ] **Audit & harden `hdf5_pure`:** Verify `Send + Sync`, HDF5 Shuffle+Deflate+Fletcher32 support, `read_f64_region`, and raw typed integer reads (`i16`/`u16`/`i32`).
- [ ] **Port `netcdf.rs`:** Read coordinates and `scale_factor`/`add_offset` in **`f64`**, check raw typed `_FillValue`/`missing_value`/`valid_range` *before* unpacking (mapping fill → `f32::NAN`), fail fast on unsupported HDF5 filters at `describe`, bound `Hdf5HandleCache` (`METADATA_CACHE` + `ROW_CACHE_CAP`) under `MemoryBudget` (`Shrinkable` LRU), support local `.SAFE` sibling geolocation, and add NetCDF variable switcher UI.
- [ ] **Tests, docs & final `old/` deletion:** Port `netcdf_tests.rs` + NetCDF section of `reads_tests.rs` + malformed-input tests; amend `docs/decisions/0010-jp2-and-netcdf-codecs.md`, `docs/architecture/raster-pipeline.md` (migrating NetCDF sections from `old/docs/imagery.md`), and `docs/architecture/georeference.md`; verify against `old/README.md` that zero unmigrated files remain, delete `old/` completely, remove the `old/` exclusion from CI no-Cyrillic check, and finalize `README.md`.
- **Exit criteria:**
  - *Automated CI:* Full 3-OS CI passes with `old/` removed, zero Cyrillic exclusions, and NetCDF `f64`/fill-value/malformed-input tests green.
  - *Manual QA:* Local and remote Sentinel-3/5P NetCDF-4 files open, switch variables, and render cleanly in 2D/3D; full `docs/` set is complete and link-checked.

---

## Next Step

- **Current state:** Phase 0.5 and Slice 1 are complete and polished on `main` (see [ADR 0001](docs/decisions/0001-native-workspace-and-threading.md), [ADR 0002](docs/decisions/0002-globe-model-and-precision.md), and [3D WGS84 Globe Architecture](docs/architecture/globe.md)).
- **Immediate task:** Start **Slice 2A — Offline Planetary Quadtree, `TileKey`/`GpuTileCache` & Bundled Basemap** (`TileKey`/`TileScheme`/`BandKey`/`TileCache<K, V>`, per-tile RTC quadtree + fixed-zoom polar cap mesh + skirts retiring the static `96×192` mesh + close-zoom graticule upgrade, `GpuTileCache` + `PendingUploads` on `VeldmapApp`, bundled public-domain basemap served as a virtual offline tile provider with early `codecs` PNG/JPEG decoding + Satellite/Ellipsoid-only switcher, ADR 0003, and CI DAG feature-matrix scaffolding).
