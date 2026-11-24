# VeldMap Development Plan

## Goal

Build VeldMap as a fast, purely native Rust desktop **Analytical Earth-Observation & Precision-Imagery Workstation** on a **Precision Nadir 3D WGS84 Globe** for:
- **v0.2–v1.0:** Interactive WGS84 planetary viewing (bundled + live basemaps and local GeoTIFF/BigTIFF/COG/PNG/JPEG overlays down to centimetre-scale orthophotos), zero-auth & custom-header STAC/COG HTTP Range streaming, GPU scientific/categorical tracks (`R32Float`/`R8Uint`/`R16Uint`) with interactive band math (NDVI/NDWI/false-color + `SCL` cloud masking), analytical terrain relief (single-file local DEM/DSM + COP-DEM GLO-30 virtual mosaic), multi-temporal timeline diffing, 1D pixel & AOI zonal statistics/time-series inspection, and Float32 GeoTIFF export.
- **v1.1:** Mission-specific codecs and authenticated archive workflows—Sentinel-2 JPEG 2000, Copernicus Data Space (CDSE) S3/SigV4 streaming & resumable downloads, and Sentinel-3/5P NetCDF-4/HDF5.

A simplified browser/mobile companion is intentionally deferred to a separate project later (enabled by keeping `veldmap-core` permanently UI/GPU/network-free).

> **Document lifecycle:** `plan.md` is the vertical-slice roadmap and exit-criteria tracker (update `Next Step` as each slice closes). Future slices (`3A-2`–`6`) are **working design hypotheses**—revise them freely as real code and benchmarks prove better designs. Settled "why not X" decisions and hard cross-slice invariants live in [docs/decisions.md](docs/decisions.md) once implemented; module details live in Rust `//!` docstrings (`cargo doc --document-private-items`).

---

## Working Cross-Slice Principles

Implemented decisions and hard cross-slice invariants are maintained in [docs/decisions.md](docs/decisions.md) (UI/GPU-free `veldmap-core`, safe positioned I/O without `memmap2`, smooth v1 WGS84 ellipsoid, `MIN_HEIGHT_M = 2.0` m, Reverse-Z `f64` ECEF metres + per-tile CPU `f64` RTC, geodetic nadir aim, decoupled quadtree/polar caps, post-overlay procedural graticule, `GpuTile` hygiene, exact sRGB compositing, and the 5 overlay accuracy benchmarks). Working principles for upcoming slices:

1. **Natural workspace crate boundaries:** Keep `veldmap-core` a permanently pure leaf (`geo`, `tile` keys/LRU, std-only `sync`/`io` traits/`MemoryBudget`, `#![forbid(unsafe_code)]`, zero OS probing or `tokio`/`reqwest`/`image`/`tiff`/`egui`/`wgpu`). Split focused workspace crates (`veldmap-raster`, `veldmap-net`, `veldmap-storage`) along real dependency seams as subsystems land—avoiding feature-gated mega-crates.
2. **Deadlock-free memory & deduplication invariants:** Bound CPU caches with startup host-RAM-scaled fixed per-cache caps (avoiding lock-reentrant cross-cache shrinking), bound decode working memory and queued upload bytes so small tiled reads never stall behind full-image decodes and large passes never starve, deduplicate concurrent chunk/block fetches with panic/cancel-safe singleflight (zero follower deadlocks or leader-cancel poisoning), and scale GPU tile VRAM budgets proportionally for multi-band EO working sets.
3. **Codec safety & durability invariants:** Decoders enforce explicit dimension/memory/IFD/filter limits, panic isolation, and thread-safety audits before multi-threaded decode. Inject `StoragePaths`; use durable fsync'd atomic writes (closing handles before rename on Windows) for user/session/catalog state and crash-safe vertically partitioned (`tile_meta`/`tile_blobs`) SQLite WAL persistence for ephemeral tile caches. Add traits only where there is a real second implementation (`ByteSource`, `Signer`, raster decoders, tile providers). Keep default builds free of external C system-library setup requirements (self-contained `cc`-compiled crates like bundled SQLite, `ring`, or `zstd-sys` are acceptable when justified; external `libgdal`/`libproj`/`libhdf5` are avoided in default builds).
4. **Legacy reference retired (`legacy-reference-snapshot` tag):** The legacy `old/` reference tree was completely retired from the working tree at the close of Slice 3A-1b, and its remaining `3A-2` / v1.1 (`3C-1`, `4B`, `6`) porting hazard checklists live below in this document. Pre-deletion state remains permanently preserved at annotated git tag `legacy-reference-snapshot` (`git show legacy-reference-snapshot:old/...`; for shallow clones: `git fetch origin tag legacy-reference-snapshot`). Design native modules cleanly rather than porting WASI-era decomposition verbatim.
5. **Code & test hygiene (every slice):** English only across the entire repository (CI-enforced). Keep cross-module, synthetic-file, mock-HTTP, and GPU/shader suites in grouped `tests/` harnesses (`tests/integration/main.rs`), short single-module unit tests (`<= ~100 LOC`) inline, and larger single-module unit suites in sibling `#[cfg(test)] #[path = "*_tests.rs"]` files. Rust 2024; verified by `cargo fmt --all -- --check`, `cargo clippy --locked --workspace --all-targets -- -D warnings`, `cargo test --locked --workspace`, isolated headless crate tests, `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps --document-private-items`, `cargo-deny`, `naga` SPV/HLSL/MSL codegen, headless offscreen `wgpu` smoke tests, and offline `lychee` link checks.

---

## Completed Foundation `[x]`

- **Phase 0 & Phase 0.5 — Native Bootstrap, Hygiene & CI `[x]`:** Initialized native `veldmap-core` + `veldmap-app` (`eframe 0.37` / `wgpu 30` / Rust `1.95`), created `legacy-reference-snapshot`, removed unlicensed font binaries, configured workspace lints (`unwrap_used`/`expect_used` warned in prod, `unsafe_op_in_unsafe_fn` denied) and Linux hygiene CI (`fmt`, `clippy --locked -D warnings`, `cargo-deny`, no-Cyrillic, offline `lychee`).
- **Slice 1 — Core Geodesy & Interactive 3D WGS84 Globe `[x]` ([Design Decisions](docs/decisions.md)):** Implemented `veldmap-core::geo` (WGS84 geodesy, Krueger projections/Helmert datums, MGRS/DD/DMS parsing, `f64` infinite Reverse-Z `Camera` with `MIN_HEIGHT_M = 2.0` m, oblate `f64` `cull::CullFrustum`, footprints), decoupled offscreen `wgpu` renderer (`Rgba8UnormSrgb` + `Depth32Float` Reverse-Z with `TRANSIENT_ATTACHMENT`, 1:1 nearest sRGB/UNORM `view_formats` compositing, `Backends::PRIMARY` minus `BROWSER_WEBGPU`), baseline GIS HUD, 3-OS CI with `naga` SPV/HLSL/MSL codegen and headless GPU smoke tests, and retired 20 legacy geo/MGRS files from `old/`.
- **Slice 2A — Offline Planetary Quadtree, `TileKey`/`GpuTileCache` & Bundled Basemap `[x]` ([Design Decisions](docs/decisions.md) §5):** Unified `f64` ECEF metres + Fukushima (2006) 3D inverse + geodetic nadir aim, implemented the decoupled geometry/texture WGS84 quadtree (`z_geom <= 16–17`, 2:1 balanced, sag-scaled outward-flared skirts) + adaptive elliptical-disk Polar Stereographic caps, per-tile CPU `f64` RTC buffers + post-overlay procedural `fwidth()` graticule, `TileKey` (`gutter: u8`), O(1) `TileCache` (`TileScheme::parent` + UV-decoupled `find_carrier`), `GpuTileCache`/`PendingUploads`, `ViewportState` (`LastRenderKey` `Arc::ptr_eq`), `Weak<GpuTile>` bind-group sweeping, and `veldmap-raster` serving bundled NASA Blue Marble (`z = 0..3`).
- **Slice 2B — Async Tile Streaming, Disk Cache & Live Basemaps + Pre-3A Architecture Hardening `[x]` ([Design Decisions](docs/decisions.md) §6):** Added `veldmap-storage` (durable private atomic writes, vertically partitioned `tile_meta`/`tile_blobs` SQLite WAL LRU cache with `16 KiB` pages, `PRAGMA user_version`, multi-band/format/gutter/encoding columns, O(1) `tile_count`) and `veldmap-net` (`tokio` + `reqwest`/`rustls` + `ring`, CPU linear-light mipmaps, NASA GIBS / EOX Sentinel-2 Cloudless / Custom XYZ/TMS with `0` HTTP requests for `PolarCap` or above `max_level`); extracted `AppStorage`, `SceneState`, `SnapshotKey` (`scene_generation`), `TileCarrierPolicy`, `ui/toolbar.rs`, and `BundledBasemapProvider` out of `app.rs`; wired host-RAM-scaled `MemoryBudget` (`256 MiB`–`1 GiB` `GpuTileCache`) + `TwoLaneDecodePool::fast()` into `SceneState`/`BasemapManager`; added multi-source enqueue-bounded `PendingUploads` and grouped `tests/` harnesses.

---

## Release Roadmap & Vertical Slices

> **Execution order:** Slices are grouped into four shippable milestones (`v0.2` → `v0.5` → `v1.0` → `v1.1`).

---

### Milestone v0.2 — Planetary Basemaps & Local Precision Workstation

#### Slice 3A-1 — Local GeoTIFF/COG Engine, `MemoryBudget` & Warp-Mesh Draping `[x]`

##### Slice 3A-1a — Core Local GeoTIFF/COG Engine & `ByteSource` `[x]`
- [x] **Sync `ByteSource` & `SourceFingerprint` (`veldmap-core` / `veldmap-storage`):**
  - Strictly sync caller-buffered `ByteSource::read_at(&self, offset: u64, buf: &mut [u8], cancel: &CancelToken) -> Result<usize, IoError>` with `LocalFileSource` (no `memmap2`; `BufferedByteReader` 64 KiB cursor; Unix `pread` / Windows pooled `seek_read` short-read loop) and OS host-RAM detection (`veldmap_storage::detect_memory_budget`).
  - Opaque `SourceFingerprint` (sampled BLAKE3 `len || head_64k || mid_64k || tail_64k` + bounded `(path, len, mtime_ns)` `LocalFingerprintCache` with `get_or_compute_with_bypass`) combined with `TileStreamSpec` (`band`, `format`, `gutter`, `decode_rev`).
- [x] **Core local GeoTIFF/BigTIFF/COG engine (`veldmap-raster`):**
  - Unified `BandInfo`, pyramid/resampling/radiometry/grid/cascade (image-rect `cell_uv` against carrier `(width, height, valid_w, valid_h, gutter)`), and `TiffSource` (standard RGB/Gray/i16/u16/f32 GeoTIFF/BigTIFF/COG, pooled `Decoder` handles, bounds-checked chunk queries, `GeoDoubleParamsTag`/`GeoAsciiParamsTag`, explicit `Limits` + `MAX_IFDS`, Float32 `Predictor = 3`, `tiff 0.11.3` native self-contained `cc`-compiled `zstd`).
  - Bounded `MemoryBudget` + `DecodeBudgetTracker` (`veldmap-core`), `DecodePeak`, panic/cancel-safe `SharedChunkCache` (lossless `RasterError` propagation + leader-cancelled follower retry; sequential passes uncached to prevent AB-BA deadlocks), `TwoLaneDecodePool`, and `PassKey`-scoped `TileScheduler`.
  - Verified TIFF read amplification via `CountingByteSource` and concurrency/memory-bound stress tests (zero deadlocks, zero HoL stalls, `peak <= cap`).

##### Slice 3A-1b — GPU Warp-Mesh Draping, Accuracy Regression, QOI Cache & `old/` Retirement `[x]`
- [x] **Multi-layer `LayerStack` & `DrawSnapshot` generalization (`veldmap-app`):**
  - Added modular `layers/` (`LayerStack`, `OverlayManager`, `OverlayRuntime`, `OverlayIoWorker`, `CursorProber`), extended `DrawSnapshot` with ordered `DrawOverlayLayer` / `DrawOverlayPatch` entries, and aggregated `wanted_tiles`, pinned roots, and proportional VRAM appetite weights across basemap + active overlays.
- [x] **Renderer modularization, post-body depth-less pass & warp-mesh draping (`veldmap-app`, `veldmap-core`):**
  - Added `render/overlay.rs` + `shaders/overlay.wgsl` with a frame-wide `DynamicUniformArena` and `depth_stencil_attachment: None` Pass 2 (`LoadOp::Load`) for overlays + procedural graticule.
  - Kept `veldmap-core::geo::binding` (`Frame`, `TieGrid`, `BindingError`) acyclic beneath `veldmap-core::tile::overlay` (`5×5`-probe lazy `OverlayCullTree`, uniform per-level `level_segments`, per-cell CPU `f64` RTC warp meshes with outward winding flip and outward vertex normals for sag-safe `dot(in.normal, -in.world) <= 0.0` horizon clipping).
- [x] **Display `RasterSource` seam, non-blocking opener/layer list, HUD & accuracy regression:**
  - Introduced the 3A display `RasterSource` trait (to be generalized for typed multi-band tracks in Slice 3B) + `frame_from_description` + chunk-guarded `sample_display_rgba` over `TiffSource`.
  - Wired non-blocking CLI/drag-and-drop/Open, Layer List (visibility, opacity, zoom-to-extent, reload/bypass cache, remove), versioned `PassKey`-scoped `TileScheduler` on `TwoLaneDecodePool`, linear-light overlay mipmaps, native Windows `GlobalMemoryStatusEx`, distinct `Viewport GSD` vs. `Source px` HUD labels, and non-blocking CPU RGB cursor readouts.
  - Verified `a_raster_lands_where_its_binding_puts_it` (5 UTM 37 benchmarks + Gauss-Krüger `EPSG:28407` + UTM `EPSG:32737` + WebMercator) plus end-to-end `PixelIsArea` vs. `PixelIsPoint` half-pixel placement.
- [x] **8-bit QOI disk cache & two-lane budget validation:**
  - Added pure-Rust QOI display tile codec helpers (`veldmap-raster`) backed by ABA-safe, cancellation-checked batched SQLite caching (`veldmap-storage`), and validated two-lane `bulk`/`fast` isolation end-to-end.
- [x] **Docs & complete `old/` retirement:**
  - Updated `docs/decisions.md` (§8) and `docs/limitations.md`, migrated remaining `3A-2` / v1.1 hazard checklists into `plan.md`, deleted `old/`, and removed all `old/` references and CI carve-outs.
- **Exit criteria:**
  - *Automated CI:* Accuracy benchmark + `PixelIsArea`/`PixelIsPoint`, south-up winding, `2 m` coarse-sag horizon facing & silhouette containment, multi-layer opacity/nodata/ancestor compositing, multi-level COG `OverlayManager` lifecycle/reload tests, `naga` uniform layout parity, and full 3-OS CI with `old/` removed pass.
  - *Manual QA:* Local GeoTIFF/BigTIFF/COG and cm-scale drone orthophotos open cleanly via CLI/drag-and-drop/Open, toggle/reload/remove in the Layer List, and drape accurately with responsive zooming and bounded RAM/VRAM/disk.

#### Slice 3A-2 — Local Workstation UX, Drone RGB Codecs, Expanded Pure-Rust CRS & GeoJSON Overlay `[ ]`
- [ ] **Workstation UI/tools, navigation & minimal session persistence (`veldmap-app`):**
  - Add focused UI/tool submodules (`layers`, `crs_picker`, `measure`, splitting `status_bar` out of `ui/toolbar.rs` if warranted) alongside native file picker (`rfd` on its pure-Rust `xdg-portal` Linux backend—never `gtk3`, verifying runtime features to avoid duplicate async executors alongside `tokio` and keeping drag-and-drop/CLI fallback for minimal WMs).
  - Add Layer Manager controls (reorder, CRS/radiometry metadata inspector, diagnostic banners for fatal vs. partial errors, detail-limit caption, loading shimmer), **manual heading rotation** (right-drag / Shift+drag / `Q`/`E`) + **keyboard pan/zoom**, and **minimal local session / recent-files persistence** via `write_json_durable_private()` (open layers, CRS overrides, camera bookmarks).
- [ ] **Drone RGB GeoTIFF codecs & georeferenced PNG/JPEG (`veldmap-raster`):**
  - Add **JPEG/4:4:4 `YCbCr`** (`YCbCr -> RGB` honoring `YCbCrCoefficients`/BT.601 + `ReferenceBlackWhite` in `tiff/tags.rs`) and enable `tiff`'s pure-Rust **`webp`** feature (`0xC351`) so standard GDAL/drone RGB COGs open out of the box in v0.2.
  - Add local full-image **PNG/JPEG adapters** over `RasterSource` (consulting `legacy-reference-snapshot:old/code/tiler/adapters/{full,png,jpeg}.rs` for streaming row-by-row PNG vs. Adam7 and `jpeg` pre-allocation budget guards) paired with external **`.tfw`/`.jgw`/`.pgw` world files** and **`.prj` sidecars** (injected as `Option<Placement>` so decoders stay pure `ByteSource`).
- [ ] **Expanded pure-Rust CRS & datum rigor (`veldmap-core` / `veldmap-raster`):**
  - Evaluate pure-Rust **`proj4rs` + `proj4wkt` + `crs-definitions`** (MIT/Apache-2.0, zero C deps) to cover WKT1/WKT2 `.prj` sidecars and broad EPSG catalogs alongside `veldmap-core::geo::projection` (retaining fast exact in-house implementations where advantageous).
  - Explicitly tag **Helmert rotation sign convention (`CoordinateFrame` EPSG:9607 vs. `PositionVector` EPSG:9606 / WKT `TOWGS84`)** on `Datum`, wire `Datum::to_wgs84` into geographic bindings (discarding phantom Helmert `Δh`), preserve `GeoKey 3072 == 32767` user-defined CRS parameters (`ProjNatOriginLatGeoKey 3081`, `ProjLinearUnitsGeoKey 3076` International / US Survey Feet) in `tiff/georef.rs`, support searchable CRS presets + custom TM parameters (unlocking all MSK-xx zones), and display visible UI warnings for unshifted/grid-shifted foreign datums.
- [ ] **Geodesic tools & read-only GeoJSON overlay (`veldmap-core` / `veldmap-app`):**
  - WGS84 geodesic distance/area measurement tools via pure-Rust MIT **`geographiclib-rs`** (Karney `Geodesic` / `PolygonArea`) with **RFC 7946 GeoJSON export** (`[lon, lat]` WGS84 + right-hand winding).
  - Read-only GeoJSON overlay with explicit `Polygon { exterior, holes }` (upgrading `footprint.rs` union semantics + 3D dihedral `turn()`, stroke-only or local tangent-plane earcut with interior edge subdivision back onto WGS84 so large polygons never sag below the ellipsoid).
- [ ] **Docs:** Update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* Expanded CRS/WKT/Helmert-convention/feet/`origin_lat_deg` round-trip tests, `YCbCr`/`WebP` TIFF + PNG/JPEG world-file tests, Karney geodesic tests, minimal session persistence tests, and GeoJSON parse/holes/subdivided-fill/RFC 7946 export round-trip tests pass.
  - *Manual QA:* Local projected, JPEG-YCbCr/WebP, and world-file/`.prj` orthophotos plus concave/holed GeoJSON polygons open cleanly, rotate/pan smoothly, persist across restart, and export measured features to GeoJSON.

---

### Milestone v0.5 — Scientific Imagery, Cloud COGs & Band Math

#### Slice 3B — Scientific `R32Float` / `R8Uint` / `R16Uint` Tracks, Extended TIFFs, Live Stretch & 2D Inspector `[ ]`
- [ ] **Multi-band GPU VRAM appetite budgeting:**
  - Allocate the `< 50%` appetite cap **proportionally by per-pyramid byte cost** rather than flat `1/N` on top of the already host-RAM-scaled `GpuTileCache` (`256 MiB`–`1 GiB`), plus optional user VRAM cap configuration, so 1440p/4K multi-band EO stacks never throttle into coarse overviews.
- [ ] **Typed `veldmap-raster` pipeline & multi-format GPU tracks (`Display`, `Scientific`, `Categorical`):**
  - Decouple `DecodedChunk`, `Grid`, `driver`, and `Cascade` from 4-byte RGBA8 display conversion so raw typed chunks/tiles flow cleanly per `PixelFormat` (`Rgba8UnormSrgb`, `R32Float`, `R8Uint`, `R16Uint`).
  - **Scientific (`R32Float`, `512×512`, `gutter = 0`, 1 mip, `1.00 MiB`/tile):** store raw numeric samples (nodata/fill → `f32::NAN`; document `f32` 24-bit precision limits for 32-bit int/`f64` readouts), apply NaN-skipping float downsampling, and evaluate canonical affine **`val * scale + offset`** (`fma`, matching STAC `raster:bands`, GDAL, and CF NetCDF) in WGSL uniforms + raw CPU cursor readout.
  - **Categorical (`R8Uint` & `R16Uint`, `512×512`, `gutter = 0`, 1 mip):** discrete class and QA bitmask bands (`Palette` classes, Sentinel-2 `SCL`, Landsat `QA_PIXEL`, MODIS QA — sharing `texture_2d<u32>` WGSL bindings) with **nearest-neighbor sampling & non-averaging downsampling**.
- [ ] **Extended GeoTIFFs & compressed scientific disk cache:**
  - Add **`PlanarConfiguration = 2`**, explicit **4-band RGB+NIR `ExtraSamples = 0`** handling (never silently dropping band 4 or treating NIR as alpha), and **`Palette`** (`ColorMap` expansion on Display; raw class indices on Categorical — decoded directly from `TileOffsets`/`StripOffsets` + `ColorMap` tags because `tiff 0.11.3` rejects `RGBPalette` inside `read_chunk*` and `deny.toml` sets `allow-git = []`).
  - Add byte-shuffled pure-Rust **`lz4_flex`** (MIT) disk caching (`TileBlobEncoding::Lz4Raw`) for `R32Float`/`R8Uint`/`R16Uint` tiles. *(Defer `gutter = 1` DEM tile production to Slice 5B-1.)*
- [ ] **WGSL nodata-aware filtering, live stretch & optional 2D Inspector (`Image2DRenderer`):**
  - Manual 4-tap `textureLoad` bilinear filter masking `NaN` and `±Inf` via raw `bitcast<u32>` exponent bits (Metal fast-math safe, keeping any derivatives strictly before horizon `discard`) with level selection biased (`<= 1.4×` texel/px) to prevent 1-mip minification shimmer.
  - Live contrast stretch (linear percentile/min-max + **`log10`/dB stretch** for SAR), colormap uniforms, multi-band GeoTIFF band selector, interactive histogram, and colorbar with physical units.
  - Dedicated orthographic **`Image2DRenderer`** in `CallbackResources` sharing immutable `Arc<GpuTile>` textures with `GlobeRenderer` (mapping composite UVs against unclamped vs. clamped `ViewportInPixels` so partially off-screen `egui::Window`s never stretch) + synchronized zoom-invariant CPU raw DN & physical-unit cursor readout. *(Can land as a small follow-on PR if needed to avoid delaying 4A/5A.)*
- [ ] **GPU smoke test & docs:** Headless `wgpu` smoke test verifying NaN/Inf bitmask bilinear math and multi-format bind-group caching; update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* Headless `wgpu` nodata-bilinear test, `naga` SPV/HLSL/MSL codegen, `R8Uint`/`R16Uint`/`R32Float` byte accounting & multi-band appetite cap, and `Planar-2`/`Palette`/4-band-RGBN TIFF tests pass.
  - *Manual QA:* 16-bit/float32/SAR/Palette/QA/multi-band rasters render with exact nodata transparency, live histogram/colorbar, and synchronized 2D/3D physical cursor readout.

#### Slice 4A — Cloud EO Streaming (HTTP Range + Zero-Auth & Custom STAC / Public COGs) `[ ]`
- [ ] **Shared `NetContext`, HTTP Range `ByteSource` & zero-extra-RTT remote `SourceFingerprint` (`veldmap-net`):**
  - Extract a shared **`NetContext`** (≥2-thread `tokio::Runtime` + shared `HostLimiters` + per-request/read timeouts instead of a global 20 s total-body `Client` timeout) owned by `VeldmapApp` and shared across `LiveTilePipeline`, `HttpRangeSource`, STAC search, and future CDSE downloads.
  - Implement range reader (consulting `legacy-reference-snapshot:old/code/network/range.rs`) over `reqwest` (`rustls`)/`tokio`: fixed host-RAM-scaled 64 KiB O(1) LRU block pool, combined open-probe + `0..64 KiB` head read (`416`-safe on 0-byte files), consumed-only readahead, and panic/cancel-safe multi-block singleflight with **refcounted waiter cancellation** (cancelling Follower A never aborts an HTTP Range GET still awaited by Follower B).
  - Explicitly disable auto-decompression (`.no_gzip().no_deflate().no_brotli().no_zstd()` / verifying absent `Content-Encoding`), reject `200` when expecting `206`, enforce `Content-Range` + **strong-ETag-only `If-Match`** (RFC 9110: never send weak `W/"..."`, isolate pool entries without strong ETags, treat `412` as non-retryable), `CancelToken` async abort, and `Handle::try_current().is_err()` guard on blocking reads.
  - Key remote tiles with `SourceFingerprint` from `(canonical_url_without_query, strong_etag, len)` (fallback `(canonical_url_without_query, len, blake3(head_64k))` when ETags are missing/weak)—avoiding wasted `mid_64k`/`tail_64k` HTTP GETs per band on COG open.
- [ ] **Single-batch pinned remote COG decode:**
  - Factor sync chunk range-planners out of `driver::direct` in `veldmap-raster` + pinned prefetch batches in `veldmap-net` (under upfront memory permits), routed to `TwoLaneDecodePool::fast()`, while isolating un-prefetched blocking reads to a separate capped remote pool so slow HTTP never starves local SSD decodes.
- [ ] **Zero-auth & custom-header STAC / public COG provider (`veldmap-net`):**
  - STAC API search (AWS Earth Search Sentinel-2 L2A COGs + COP-DEM GLO-30) plus **optional custom STAC catalog endpoint with Bearer/API-key/SAS auth headers** (reusing URL/secret redaction) and direct public HTTP COG URLs.
  - Multi-band Sentinel-2 asset discovery + STAC `raster:bands` `scale`/`offset` metadata, AOI search (viewport / drawn box / GeoJSON polygon + GeoJSON/WKT export), scene outlines on globe (HiDPI-scaled outline/ribbon/hatch pipelines with per-cluster RTC in `veldmap-app`), and built-in zero-auth sample scenes for 15-second onboarding.
- [ ] **Mock HTTP harness & docs:** Offline local mock HTTP Range harness (`206`, `200` rejection, `416`, strong/weak `ETag`, `412`, N-way singleflight deduplication + follower cancellation, request & delivered-byte ceilings); update `docs/decisions.md` and `docs/building.md`.
- **Exit criteria:**
  - *Automated CI:* Offline mock HTTP Range/STAC/singleflight tests (`0` duplicate fetches under N-way contention, bounded request/byte ceilings, zero wasted `mid`/`tail` GETs) pass.
  - *Manual QA:* Search an AOI and stream public Sentinel-2 L2A COGs and HTTP COG URLs without starving local SSD decodes.

#### Slice 5A — Interactive Band Math, Split-Screen Swipe & Full Workspace Sessions `[ ]`
- [ ] **Unified screen-space float band math & radiometric correctness:**
  - Preserve Pass 1 (opaque globe body) as the sole consumer of `Depth32Float` (`StoreOp::Discard` + `TRANSIENT_ATTACHMENT`).
  - Reuse Slice 3A-1b's `depth_stencil_attachment: None` post-body pipeline architecture and frame-wide dynamic uniform arena to interleave `[shared <=4-target scratch passes] → [depth-less color LoadOp::Load composite pass]` in painter's order before the final graticule pass, bounding transient scratch VRAM regardless of layer count:
    - Each active scientific band renders into a lazy, shared single-viewport **`R32Float`** scratch target (`blend: None`), initialized as **self-describing nodata** via a 3-vertex fullscreen quiet-`NaN` fill reading `0x7FC0_0000u` from a **runtime uniform** (never a WGSL const-expression rejected by `naga`), with nodata fragments calling `discard` (after any derivatives).
    - Reserve a dedicated **`R8Uint`** scratch target (`blend: None`, cleared to `255` uncovered sentinel) strictly for nearest-neighbor `SCL`/QA masks.
  - Fullscreen WGSL band-math (NDVI, NDWI, false-color, custom stretch) + mask + colormap composite handling mixed-GSD bands (Sentinel-2 10/20/60 m, Landsat 15/30 m half-pixel shift), independent ancestor fallbacks, and multi-sensor/rotated overlays without per-tile `[tile_id; N]` shader combinatorics.
- [ ] **Metadata-driven scaling & `SCL` masking:**
  - Apply `scale`/`offset` from STAC `raster:bands` and local GeoTIFF metadata in canonical `val * scale + offset` physical reflectance space (never hardcoded to a 2021↔2022 date cutoff; ESA `.SAFE` `MTD_MSIL2A.xml` DN-space `BOA_ADD_OFFSET` conversion lands with `4B`), badge L1C (TOA) vs. L2A (BOA), and support optional nearest-neighbor `R8Uint` `SCL` cloud/shadow masking (`classes 0, 1, 3, 8, 9, 10`).
- [ ] **Split-screen swipe curtain, full sessions & PNG screenshot export:**
  - Interactive swipe comparison across any two layers (including non-co-registered swaths), full workspace/session JSON persistence (extending 3A-2's session file with band math recipes, stretch params, and split-pane layout), copy-coordinates, and viewport PNG screenshot export (reading back `OffscreenTarget`'s `COPY_SRC` color target with optional scale bar, north arrow, colorbar, and acquisition date).
- [ ] **Docs:** Update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* Metadata-driven `val * scale + offset`, uniform-`NaN`-filled `R32Float` + nodata `discard` + nearest-neighbor `R8Uint` `SCL` mask (including valid `0.0` reflectance preservation), session JSON round-trip, and `naga` SPV/HLSL/MSL shader tests pass.
  - *Manual QA:* Multi-band Sentinel-2 NDVI/NDWI/false-color (including mixed 10 m/20 m/60 m and multi-sensor) with cloud masking, swipe curtain, screenshot export, and session reload work smoothly.

---

### Milestone v1.0 — Analytical Terrain Relief & Multi-Temporal EO Workstation

#### Slice 5B-1 — Analytical Terrain Relief Layers (Local DEM/DSM + COP-DEM Virtual Mosaic) `[ ]`
- [ ] **Single-file local DEM/DSM relief & `gutter = 1` tiles:**
  - Implement parameterized **`gutter = 1`, 1-mip (`514×514`) `R32Float`** DEM tile production (keyed via `TileKey.gutter = 1`, falling back to `gutter = 0` rather than `Serve::Pass` on oversized chunks; normalized against carrier `(valid_w, valid_h, gutter)`).
  - Compute GPU hillshade, slope, and aspect using per-vertex **2×2 inverse Jacobian `J^(-T)`** into local ENU metres (handling EPSG:4326 `cos(lat)`, COP-DEM latitude bands, and UTM/Polar rotation), scaled via exact `source_scale` `(scale_x, scale_y)` (accurate on both `ceil` and GDAL `floor` overviews), with one-sided fallback on `NaN` boundaries.
  - Ship single-file local DEM/DSM relief (for drone DSMs and local elevation GeoTIFFs) cleanly decoupled from multi-file global mosaic streaming, with explicit vertical datum labeling (`m EGM2008` / source units).
- [ ] **Multi-file virtual mosaic (`DemMosaicLayer`) & COP-DEM GLO-30:**
  - Implement `DemMosaicLayer` resolving visible `1°×1°` `PixelIsPoint` (`3601×3601`) COP-DEM GLO-30 geocell COGs under one Layer Manager entry with a bounded LRU of open remote `ByteSource`s, a coarse-zoom geocell cutoff (or coarse global overview fallback), and accounting as **1 logical pyramid** in `GpuTileCache` appetite budgeting.
  - Render shaded relief into a **`blend: None` scratch target** (with half-open `[lon, lon+1) × [lat, lat+1)` extent + sub-pixel edge closure so adjacent geocells' independent `f32` RTC rounding can neither double-darken nor leave 1-px bright seams) followed by a **single fullscreen linear-light `Multiply` composite** (`mix(vec3(1.0), relief, opacity)`) onto the globe color buffer.
- [ ] **Docs:** Update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* Synthetic analytic DEM slope/aspect error bounds across latitudes (`0°`/`60°`/`80°`) and rotated projections, `gutter = 1` continuity, `1°×1°` border non-double-multiply & zero-seam check, `DemMosaicLayer` geocell resolution/eviction, `naga` SPV/HLSL/MSL, and headless GPU smoke tests pass.
  - *Manual QA:* Both single-file local DSMs and COP-DEM GLO-30 hillshade/slope/aspect render cleanly over optical imagery with zero intra-file or `1°×1°` border creases.

#### Slice 5B-2 — Multi-Temporal Timeline, Dual-Epoch Diffing, AOI Zonal Stats & GeoTIFF Export `[ ]`
- [ ] **Multi-temporal timeline scrubber:** Reuse unified `TileKey` for epoch ingestion from **STAC AOI search** (grouped by MGRS tile + AOI coverage, defaulting to L2A) and **local dated folders**, with cloud-cover filtering, adjacent-date prefetch, and instant stale-request cancellation.
- [ ] **Unified offscreen dual-epoch cross-fade & physical reflectance diffing:** Reuse Slice 5A's lazy shared **`NaN`-filled `R32Float` + `SCL` `R8Uint`** offscreen scratch pool (pre-colormap/stretch, preserving valid `0.0` diffs and propagating `SCL`/QA validity) for physical reflectance/index diffing (`Date B − Date A`) and display targets for visual cross-fade (`mix`)—supporting multi-band + `SCL`, mismatched carrier levels, and non-co-registered swaths cleanly.
- [ ] **1D pixel & AOI polygon time-series + zonal statistics + CSV export:**
  - Lightweight 1D time-series & zonal stats panel (custom `egui::Painter` chart to avoid pinning `egui` upgrades to lagging `egui_plot`) querying **native CPU source tiles** (zoom-invariant) for both single-pixel time-series and **GeoJSON polygon AOI zonal stats** (mean, median, std, percentiles, parcel-averaged time-series, skipping cloud-masked pixels/dates), with **RFC 4180 CSV export** (ISO-8601 UTC timestamps, explicit units, documented `SCL`-masked row convention).
- [ ] **Float32 GeoTIFF export & non-Latin UI font fallback:**
  - Export computed band-math, dual-epoch diff, or AOI-clipped layers as georeferenced **Float32 GeoTIFF** via the workspace `tiff` encoder (closing the quantitative analytical loop without requiring external GDAL).
  - Load a permissively licensed fallback font (covering Cyrillic and common non-Latin scripts in file paths and metadata, verified against `deny.toml`).
- **Exit criteria:**
  - *Automated CI:* Temporal diffing shaders (`naga` SPV/HLSL/MSL + headless GPU smoke), float pre-stretch diff math (including `0.0` diff vs. nodata), zoom-invariant pixel & AOI zonal stats/time-series tests, RFC 4180 CSV export, and Float32 GeoTIFF round-trip export tests pass.
  - *Manual QA:* End-to-end v1.0 EO workflow (local + STAC COGs, band math, swipe/diff, pixel & AOI time-series + CSV/GeoTIFF export, analytical DEM relief) persists cleanly across sessions.

---

### Milestone v1.1 — Mission-Specific Codecs, Authenticated CDSE & Sentinel-3/5P NetCDF-4

#### Slice 3C-1 — Sentinel-2 JPEG 2000 `[ ]`
- [ ] **Evaluate and select JP2 decoder:** Carry out a pre-implementation evaluation and select the decoder before writing the adapter (recording the decision in `docs/decisions.md`), comparing: (1) pure safe Rust **`hayro-jpeg2000`** (preferred if single-tile + resolution-level random-access decoding is available by `v1.1`), (2) latest **`openjp2`** (`0.7+`, audited under Miri/ASan/multi-thread/fuzzing for `c2rust` soundness), and (3) **`openjpeg-sys`** (only if updated to OpenJPEG `>= 2.5.4` patching `CVE-2025-54874`, or a maintained self-contained `cc` wrapper; verify `deny.toml` `all-features = true`).
- [ ] **Implement & harden JP2 adapter (`veldmap-raster`):** Implement local `.SAFE` JP2 decoding over the selected decoder, GMLJP2 parsing (`roxmltree`/`quick-xml`), and tile-part excerpting, resolving every audited `codec.rs`/`excerpt.rs`/`jp2.rs` UB and bounds hazard (`legacy-reference-snapshot`).
- [ ] **Tests & docs:** Adapt `jp2_tests.rs`, `excerpt_tests.rs`, JP2 sections of `reads_tests.rs` (`legacy-reference-snapshot`), plus 5/3 reversible round-trip and malformed/truncated-input fuzz tests; update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* 5/3 JP2 round-trip, excerpt, and malformed/truncated-input tests pass with zero panics or UB.
  - *Manual QA:* Local Sentinel-2 JP2 granules open and render cleanly in 2D/3D.

#### Slice 4B — Copernicus Data Space (CDSE) S3/JP2 & Resumable Downloads `[ ]`
- [ ] **CDSE provider, SigV4 re-signing & timeline hook (`veldmap-net`):** Implement CDSE OData/S3/SigV4 (`Signer`), multi-band Sentinel-2 `.SAFE` asset discovery (`B02`–`B12`, `SCL`, and `MTD_MSIL2A.xml` DN-space `BOA_ADD_OFFSET` normalized to `offset = BOA_ADD_OFFSET / QUANTIFICATION_VALUE`), epoch-based singleflight 401/403 re-sign (preserving the unsigned `Range` header invariant), and timeline scrubber integration. Decouple CDSE catalog/COG/resumable download from remote HTTP JP2 packet-surgery—gate **AOI-scoped two-phase remote JP2 streaming** behind latency/ROI profiling (legacy remote JP2 excerpting measured `70–77 s` and `255` HTTP requests per band).
- [ ] **Local library, hardened resumable downloads & Credentials UI (`veldmap-storage` / `veldmap-net`):** Implement catalog, storage layout, and unified resumable `.part` downloads resolving every legacy hazard (`legacy-reference-snapshot`), plus an in-app credentials/cache dialog writing `ProjectDirs/.env` (`0600` on Unix) with mandatory credential/SigV4 redaction in logs & errors and anonymous search without keys.
- [ ] **Tests & docs:** Adapt provider + prefetch tests (`legacy-reference-snapshot`) + mock 401/403/`.part` crash-resume & secret-redaction tests; update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* CDSE provider, mock 401/403 & `.part` resume, and secret-redaction log tests pass.
  - *Manual QA:* Browse CDSE S3, download/open Sentinel-2 granules (including via the timeline), and pause/resume downloads reliably.

#### Slice 6 — NetCDF-4 (Sentinel-3/5P) `[ ]`
- [ ] **Evaluate and select HDF5/NetCDF-4 reader:** Carry out a pre-implementation spike/evaluation and select the reader before building the adapter (recording the decision in `docs/decisions.md`), benchmarking and auditing the latest **`hdf5-pure`** alongside pure-Rust alternatives (`rustyhdf5` / `netcdf-reader`) against real ESA Sentinel-3/5P NetCDF-4 files (`Send + Sync`, superblock v2/v3, Shuffle+Deflate+Fletcher32, `f64` coordinates/scaling, raw typed integer reads, and `deny.toml` `all-features = true` compliance with zero `memmap2`). Keep an optional C `libhdf5` fallback strictly as a last resort (requires NCSA license clarification and CMake build support).
- [ ] **Implement NetCDF-4 adapter (`veldmap-raster` / `veldmap-app`):** Implement the NetCDF-4 adapter over the selected reader (resolving every legacy hazard from `legacy-reference-snapshot`), local `.SAFE` sibling geolocation, `PassKey`-level `TileScheduler` reset, and NetCDF variable switcher UI.
- [ ] **Tests & docs:** Adapt NetCDF tests + malformed-input tests; update `docs/decisions.md`, `docs/limitations.md`, and `README.md`.
- **Exit criteria:**
  - *Automated CI:* Full 3-OS CI passes with NetCDF `f64`/fill-value/malformed-input tests green.
  - *Manual QA:* Local and remote Sentinel-3/5P NetCDF-4 files open, switch variables, and render cleanly in 2D/3D.

---

## Migrated Legacy Porting Hazard Checklists (`legacy-reference-snapshot`)

Audited safety, correctness, and architecture hazards preserved from retired `old/README.md` (consult `git show legacy-reference-snapshot:old/...` on demand; shallow clones: `git fetch origin tag legacy-reference-snapshot`). Do not port legacy code verbatim—resolve every item below when implementing the corresponding slice:

- **Slice 3A-2 (`old/code/tiler/adapters/{full,png,jpeg}.rs`):** Enforce streaming row-by-row non-interlaced PNG vs. whole-frame Adam7 budget guards and `jpeg` pre-allocation budget checks (no `Rc`/`RefCell`).
- **Slice 3C-1 (`old/code/tiler/adapters/{codec,excerpt,jp2}.rs`):** Audit `openjp2` for `static mut` thread safety/soundness (evaluate `hayro-jpeg2000` or `openjpeg-sys` `>= 2.5.4` if it fails); fix Stacked Borrows / drop-order UB (`Box::into_raw` + explicit `Drop`), `usize::MAX` EOF + `ErrorKind::Interrupted` retry, `catch_unwind` on `extern "C"` callbacks, `skip()` EOF bounds check, `COD` `levels <= 32` shift-overflow guard, PLT varint & TLM checked arithmetic + `keep <= part.len`, `MAX_PAD` cap, worker panic isolation, and gating zero-as-nodata on Sentinel-2 heuristics rather than all GMLJP2. Add 5/3 reversible round-trip and malformed/truncated `proptest` fuzz tests.
- **Slice 4B (`old/code/network/download.rs`, `old/code/library/*`, `old/code/provider/*`):** Drop WASI `queued_sidecars` / async rescan state machines in favor of serialized sync `std::fs` metadata; enforce S3 path sanitization, Windows `\\?\` extended-length paths (`longPathAware`), unsigned `Range` header invariant on 401/403 re-sign, persisted `ETag` `If-Range`, HTTP 416 recovery, `File::try_lock()` + in-process lock map + explicit `unlock()`, `downloaded == total_size`, and credential/SigV4 redaction in `tracing` logs/errors.
- **Slice 6 (`old/code/tiler/adapters/netcdf.rs`):** Read coordinates and `scale_factor`/`add_offset` in `f64`; check raw typed `_FillValue`/`missing_value`/`valid_range` *before* unpacking (mapping fill to `f32::NAN`); fail fast on unsupported HDF5 filters at `describe`; bound `Hdf5HandleCache` (`METADATA_CACHE` + `ROW_CACHE_CAP`) with fixed host-RAM-scaled caps.

---

## Next Step

- **Current state:** Phases 0–0.5 and Slices 1, 2A, 2B, and **3A-1 (`3A-1a` + `3A-1b`)** are complete on `main` (see [Design Decisions](docs/decisions.md) for settled invariants).
- **Immediate task:** Start **Slice 3A-2 — Local Workstation UX, Drone RGB Codecs, Expanded Pure-Rust CRS & GeoJSON Overlay** (`rfd` pure-Rust `xdg-portal` native picker, Layer Manager reorder/inspector/banners, manual heading rotation & keyboard pan/zoom, minimal local session persistence, `YCbCr`/`WebP` GeoTIFF + PNG/JPEG world-file/`.prj` adapters, expanded pure-Rust CRS & explicit Helmert sign conventions, Karney geodesic measurement + RFC 7946 GeoJSON export, and read-only GeoJSON overlay).
