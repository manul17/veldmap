# VeldMap Development Plan

## Goal

Build VeldMap as a fast, purely native Rust desktop **Analytical Earth-Observation & Precision-Imagery Workstation** on a **Precision Nadir 3D WGS84 Globe** for:
- **v0.2–v1.0:** Interactive WGS84 planetary viewing (bundled + live basemaps and local GeoTIFF/BigTIFF/COG/PNG/JPEG overlays down to centimetre-scale orthophotos), zero-auth & custom-header STAC/COG HTTP Range streaming, GPU scientific/categorical tracks (`R32Float`/`R8Uint`/`R16Uint`) with interactive band math (NDVI/NDWI/false-color + `SCL` cloud masking), split-screen swipe & cloud workspace sessions, analytical terrain relief (single-file local DEM/DSM + COP-DEM GLO-30 virtual mosaic), multi-temporal timeline diffing, 1D pixel & multi-polygon AOI zonal statistics/time-series inspection, and Float32 GeoTIFF export.
- **v1.1:** Cloud-native ESA EOPF Zarr streaming & authenticated Copernicus Data Space (CDSE) S3/resumable downloads, local Sentinel-2 `.SAFE` JPEG 2000, Sentinel-5P/3 NetCDF-4/HDF5, and an optional 1:1 native-pixel 2D orthographic inspector.

A simplified browser/mobile companion is intentionally deferred to a separate project later (enabled by keeping `veldmap-core` permanently UI/GPU/network-free).

> **Document lifecycle:** `plan.md` is the vertical-slice roadmap and exit-criteria tracker (update `Next Step` as each slice closes). Upcoming slices (`5A-2`–`6`, `3B-2`) are **working design hypotheses**—revise them freely as real code and benchmarks prove better designs. Settled "why not X" decisions and hard cross-slice invariants live in [docs/decisions.md](docs/decisions.md) once implemented; module details live in Rust `//!` docstrings (`cargo doc --document-private-items`).

---

## Working Cross-Slice Principles

Implemented decisions and hard cross-slice invariants are maintained in [docs/decisions.md](docs/decisions.md) (§1–§12: pure UI/GPU/network-free `veldmap-core`, safe positioned I/O without `memmap2`, smooth v1 WGS84 ellipsoid & `MIN_HEIGHT_M = 2.0` m, `< 2 mm` overlay accuracy benchmarks, `f64` ECEF metres + RTC, decoupled quadtree/polar caps, SQLite WAL tile cache, two-lane decode pool, depth-less post-body overlays & vectors, pure-Rust CRS, scientific/categorical GPU tracks, shared `NetContext` / HTTP Range COG streaming, and multi-asset screen-space GPU Band Math + `SCL` masking). Working principles for upcoming slices:

1. **Natural workspace crate boundaries:** Keep `veldmap-core` a permanently pure leaf (`geo`, `tile` keys/LRU, std-only `sync`/`io` traits/`MemoryBudget`, `#![forbid(unsafe_code)]`, zero OS probing or `tokio`/`reqwest`/`image`/`tiff`/`rusqlite`/`egui`/`wgpu`). Split focused workspace crates (`veldmap-raster`, `veldmap-net`, `veldmap-storage`) along real dependency seams with zero horizontal dependencies among middle crates.
2. **Deadlock-free memory & deduplication invariants:** Bound CPU caches with startup host-RAM-scaled fixed per-cache caps (avoiding lock-reentrant cross-cache shrinking), track dynamic viewport/scratch GPU VRAM in `veldmap-app` against `GpuTileCache`'s effective budget with a hard minimum floor (keeping `veldmap-core::MemoryBudget` pure/static), bound decode working memory and queued upload bytes so small tiled reads never stall behind full-image decodes and large passes never starve, deduplicate concurrent chunk/block fetches with panic/cancel-safe singleflight, and scale GPU tile VRAM budgets proportionally for multi-band EO working sets.
3. **Codec safety & durability invariants:** Decoders enforce explicit dimension/memory/IFD/filter limits, panic isolation, and thread-safety audits before multi-threaded decode. Inject `StoragePaths`; use durable fsync'd atomic writes (closing handles before rename on Windows) for user/session/project state and crash-safe vertically partitioned (`tile_meta`/`tile_blobs`) SQLite WAL persistence for ephemeral tile caches. Add traits only where there is a real second implementation (`ByteSource`, `Signer`, raster decoders, tile providers). Keep default builds free of external C system-library setup requirements (self-contained `cc`-compiled crates like bundled SQLite, `ring`, or `zstd-sys` are acceptable when justified; external `libgdal`/`libproj`/`libhdf5` are avoided in default builds).
4. **Legacy reference retired (`legacy-reference-snapshot` tag):** The legacy `old/` reference tree was retired at the close of Slice 3A-1b; remaining v1.1 (`3C-1`, `4B-1`, `6`) porting hazard checklists live at the bottom of this document. Pre-deletion state remains preserved at annotated git tag `legacy-reference-snapshot` (`git show legacy-reference-snapshot:old/...`; for shallow clones: `git fetch origin tag legacy-reference-snapshot`). Design native modules cleanly rather than porting WASI-era decomposition verbatim.
5. **Code & test hygiene (every slice):** English only across the entire repository (CI-enforced). Keep cross-module, synthetic-file, mock-HTTP, and GPU/shader suites in grouped `tests/` harnesses (`tests/integration/main.rs`), short single-module unit tests (`<= ~100 LOC`) inline, and larger single-module unit suites in sibling `#[cfg(test)] #[path = "*_tests.rs"]` files. Rust 2024; verified by `cargo fmt --all -- --check`, `cargo clippy --locked --workspace --lib --bins -- -D warnings`, `cargo clippy --locked --workspace --all-targets -- -D warnings`, `cargo test --locked --workspace`, isolated headless crate tests, `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps --all-features --document-private-items`, `cargo-deny`, `naga` SPV/HLSL/MSL codegen, headless offscreen `wgpu` smoke tests, and offline `lychee` link checks.

---

## Completed Foundation `[x]`

*(Detailed rationale and invariants for every completed slice live in [docs/decisions.md](docs/decisions.md) §1–§12.)*

- **Phase 0 & 0.5 — Bootstrap, Hygiene & CI `[x]` (§1–§2):** Native Rust 2024 workspace (`eframe 0.37` / `wgpu 30` / Rust `1.95`), `legacy-reference-snapshot` tag, workspace lints, `cargo-deny`, English-only & offline `lychee` CI.
- **Slice 1 — Core Geodesy & 3D WGS84 Globe `[x]` (§3–§4):** `veldmap-core::geo` (WGS84 geodesy, Krueger projections/Helmert datums, MGRS/DD/DMS parsing, `f64` Reverse-Z `Camera` with `MIN_HEIGHT_M = 2.0` m, oblate `CullFrustum`), offscreen `wgpu` renderer (`Rgba8UnormSrgb` + `Depth32Float` Reverse-Z, exact 1:1 sRGB compositing), baseline HUD, and 3-OS CI with `naga` + headless GPU smoke tests.
- **Slice 2A — Planetary Quadtree, `TileKey`/`GpuTileCache` & Bundled Basemap `[x]` (§5):** `f64` ECEF metres + Fukushima 3D inverse + geodetic nadir aim, decoupled geometry/texture quadtree + Polar Stereographic caps, per-tile CPU `f64` RTC + post-overlay procedural graticule, `TileKey` (`gutter: u8`), `TileCache`/`GpuTileCache`/`PendingUploads`, and bundled NASA Blue Marble (`veldmap-raster`).
- **Slice 2B — Async Tile Streaming, SQLite WAL Cache & Live Basemaps `[x]` (§6):** `veldmap-storage` (durable private atomic writes, vertically partitioned `tile_meta`/`tile_blobs` SQLite WAL LRU cache), `veldmap-net` (`tokio` + `reqwest`/`rustls` + `ring`, NASA GIBS / EOX / Custom XYZ/TMS), `TwoLaneDecodePool`, `SceneState`/`SnapshotKey`, and host-RAM-scaled `MemoryBudget`.
- **Slice 3A-1 — Local GeoTIFF/COG Engine, Warp-Mesh Draping & `old/` Retirement `[x]` (§7–§8):** Sync caller-buffered `ByteSource` (zero `memmap2`), sampled BLAKE3 `SourceFingerprint`, cross-OS `detect_memory_budget`, memory-bounded `TiffSource`, `DecodeBudgetTracker`, `SharedChunkCache`, `TileScheduler`, modular `layers/` + `CursorProber`, depth-less Pass 2 warp-mesh draping (`OverlayCullTree`, outward vertex-normal horizon clip), 5 overlay accuracy benchmarks (`< 2 mm`), QOI display tile caching, and `old/` retirement.
- **Slice 3A-2 — Workstation UX, Drone RGB Codecs, Pure-Rust CRS & GeoJSON Vectors `[x]` (Milestone v0.2 Complete, §9):** Pure-Rust `proj4rs`/`proj4wkt`/`crs-definitions` + explicit `HelmertConvention`, TIFF `YCbCr`/`WebP`, `PngSource`/`JpegSource` + world-file/`.prj` sidecars, Karney geodesic measurement (`geographiclib-rs`), pole-safe Pass-2 draped vectors, non-blocking `rfd` dialogs, heading rotation & keyboard nav, `dejavu` fallback fonts, and `session.json` persistence.
- **Slice 3B — Scientific `R32Float` / `R8Uint` / `R16Uint` Tracks, Direct Raw-Chunk GeoTIFFs & Live Stretch `[x]` (§10):** `StreamKey`-keyed uploads & multi-band VRAM appetite budgeting (`128–4096 MiB`), hoisted `TileBindGroupCache`, Display/Scientific (`R32Float` NaN-skipping mips, 8-tap nodata-trilinear WGSL, linear/`log10` dB stretch, histogram, colorbar, native-`f64` cursor readout)/Categorical (`R8Uint`/`R16Uint`) tracks, lock-free direct raw-chunk GeoTIFF decoding (`Planar-1/2`, RGB+NIR, `Palette`, `ZSTD`/`YCbCr`/`WebP`), and byte-shuffled `Lz4Raw` disk caching.
- **Slice 4A — Cloud EO Streaming (HTTP Range + Zero-Auth & Custom STAC / Public COGs) `[x]` (§11):** Shared `NetContext` (split identity/gzip clients, split timeouts, host-scoped cooldowns, 64 KiB `RangeBlockPool` & `PrefetchBudget`), `HttpRangeSource` (1-RTT `0..256 KiB` head probe, strong-ETag `If-Match`, refcounted singleflight, pinned coalesced prefetch), pure-`ByteSource` `SinglePassIfdReader` (moving `tiff` to `test-support`/dev-only), isolated `veldmap-remote-io` + offloaded `veldmap-tile-write`, AWS Earth Search Sentinel-2 L2A & COP-DEM GLO-30 + custom-auth STAC search + sample scenes, retained vector AOIs + GeoJSON/WKT export, HiDPI ribbon/hatch globe footprints, 3-tab unified Settings modal, and collapsible `Layers` | `Cloud / STAC` workbench.
- **Slice 5A-1 — Multi-Asset Layers, STAC Band Recipes & GPU Band Math + `SCL` Masking `[x]` (§12):** Tagged `LayerKind::{SingleRaster, BandMath}` + cohesive `RasterAsset` refactor & UI-decoupled `ui/settings/` + `stac/{session,aoi,auth}` projection (zero `veldmap-net` imports in `ui/`), lazy shared quiet-`NaN`-uniform-filled `R32Float` (`1..=4`) + `R8Uint` `SCL` (`255` sentinel) scratch pool with `2..=4`-attachment MRT (`@group(2)` 4-texture cache), clamped projected-AABB scissor rects, segmented Pass-2 painter's-order compositing, floored dynamic viewport/scratch VRAM accounting (`64 MiB` minimum tile-cache floor), per-band `GDAL_METADATA` + `ExternalBandMeta` `(scale, offset, nodata, unit)` with `has_embedded_scale_offset` precedence, explicit `R8Uint` + `Sentinel2Scl` loading with cloud/shadow mask toggles, 1-click STAC Scene Recipes (`True Color`, `NDVI + SCL`, `NDWI + SCL`, `False Color NIR`, `SWIR`) + sample recipes + local Band Math builder, `RdYlGn`/`BrBG` colormaps, and zoom-invariant `BandMath` CPU cursor readouts.

---

## Release Roadmap & Vertical Slices

> **Execution order:** Slices are grouped into four shippable milestones (`v0.2` → `v0.5` → `v1.0` → `v1.1`).

---

### Milestone v0.2 — Planetary Basemaps & Local Precision Workstation `[x]`

*(Complete — Slices 1, 2A, 2B, 3A-1, and 3A-2; see [Design Decisions](docs/decisions.md) §1–§9.)*

---

### Milestone v0.5 — Scientific Imagery, Cloud COGs, Band Math & Cloud Sessions

#### Slice 3B — Scientific `R32Float` / `R8Uint` / `R16Uint` Tracks, Direct Raw-Chunk GeoTIFFs & Live Stretch `[x]`
*(Complete — see **Completed Foundation** above and [Design Decisions](docs/decisions.md) §10.)*

#### Slice 4A — Cloud EO Streaming (HTTP Range + Zero-Auth & Custom STAC / Public COGs) `[x]`
*(Complete — see **Completed Foundation** above and [Design Decisions](docs/decisions.md) §11.)*

#### Slice 5A-1 — Multi-Asset Layers, STAC Band Recipes & GPU Band Math + `SCL` Masking `[x]`
*(Complete — see **Completed Foundation** above and [Design Decisions](docs/decisions.md) §12.)*

#### Slice 5A-2 — Split-Screen Swipe, Cloud Workspace Persistence & Cartographic PNG Export `[ ]`
- [ ] **Split-screen swipe curtain:** Interactive swipe comparison across any two layers (including non-co-registered swaths and `BandMath` layers), integrated into `VeldmapApp`'s unified viewport tool transitions alongside `Measure` and `DrawAoiBox`.
- [ ] **Full workspace/project persistence (local + sanitized cloud & `BandMath`):**
  - Evolve `SavedLayerEntry` into a tagged schema supporting local rasters, remote COGs, and `BandMath` recipes (plus split-pane/swipe state), and add explicit Save/Load `.veldmap.json` project files alongside `session.json`.
  - **Secret hygiene:** Strip query strings, presigned `?sig=` / `?X-Amz-Signature=` tokens, `user:pass@` userinfo, and custom auth headers before writing; persist only sanitized canonical URLs, `auth_mode` references (resolved from `settings.json` on load), `(scale, offset, nodata, unit)`, and optional `(stac_item_href, asset_key)`. Route offline/401/403 remote restore failures through `SessionRestoreTracker` so styles and recipes are never lost on transient outages.
- [ ] **Copy coordinates & cartographic PNG screenshot export:**
  - Copy cursor/center coordinates (DD / DMS / MGRS) and display compact MGRS alongside DD in the status bar.
  - Retain `OffscreenTarget`'s `color_texture` handle with `COPY_SRC` usage, read back RGBA pixels, composite optional cartographic overlays (scale bar, north arrow, colorbar, acquisition date) on CPU off the UI thread (since `egui::Painter` HUD widgets are drawn after `GlobePaintCallback`), and encode PNG asynchronously via `rfd` save dialog.
- [ ] **Docs:** Update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* Tagged `session.json` / `.veldmap.json` round-trip (including secret stripping & offline restore preservation), swipe shader/scissor tests, and offscreen PNG readback/encode tests pass.
  - *Manual QA:* Swipe curtain across arbitrary layers, save/reload of cloud + band-math workspaces without leaking credentials, copy-coordinates, and PNG export work smoothly.

---

### Milestone v1.0 — Analytical Terrain Relief & Multi-Temporal EO Workstation

*(Slices `5B-1` and `5B-2a/b` are independent after `5A-1` and may be executed in either order.)*

#### Slice 5B-1 — Analytical Terrain Relief Layers (Local DEM/DSM + COP-DEM Virtual Mosaic) `[ ]`
- [ ] **Single-file local DEM/DSM relief & `gutter = 1` tiles:**
  - Implement parameterized **`gutter = 1`, 1-mip (`514×514`) `R32Float`** DEM tile production (leveraging existing `TileKey.gutter`, `DecodedTile::try_with_valid_region`, `cell_uv_rect`, `Lz4Raw`, `GpuTile`, and `generate_tile_mipmaps` 1-mip guards): refactor `TileEmit` from a 6-tuple closure to a structured `EmittedTile` carrying `(valid_w, valid_h, gutter)`, update `RasterSource::decode_pointwise_tile` and `TiffSource::validate_stream_spec`, fall back to `gutter = 0` rather than `ServeMode::Pass` on oversized chunks, extend CPU `grid::region_of` / `grid::footprint` / `build_chunked_level_table` / `driver::plan_direct` to signed halo coordinates + `source_scale` resampling for `±1` texel halos (aligning `ceil` and GDAL `floor` overviews on `2^level` base-pixel spacing), and ensure out-of-bounds edge halo texels emit `f32::NAN` for clean one-sided boundary derivatives rather than partial in-bounds smearing.
  - Compute GPU hillshade, slope, and aspect using per-vertex **dual (contravariant) 3D ECEF tangent vectors** `[T^u, T^v] = J (J^T J)^(-1)` (`2 × vec3<f32>` in a DEM-specific vertex format with a distinct/tagged `OverlayMeshKey` variant so 32 B `OverlayVertex` and 56 B DEM buffers never collide, guarding `det(J^T J) < eps` at exact poles `lat = ±90°`) scaled by the carrier's exact `2^carrier.level` base-pixel spacing, with one-sided fallback on `NaN` boundaries.
  - Ship single-file local DEM/DSM relief cleanly decoupled from multi-file global mosaic streaming, with explicit vertical datum labeling (`m EGM2008` / source units) and **quantitative cursor readout** (`Elevation`, `Slope (°)`, `Aspect (°/compass)`).
- [ ] **Multi-file virtual mosaic (`DemMosaicLayer`) & 1-click COP-DEM GLO-30:**
  - Implement `DemMosaicLayer` with a 1-click `"COP-DEM GLO-30 Terrain Relief"` entry resolving visible `1°×1°` `PixelIsPoint` (`3601×3601`) geocell COGs deterministically under one Layer Manager entry with: a bounded LRU of open remote `ByteSource`s, negative caching of absent/404 ocean geocells, a coarse-zoom geocell cutoff (or coarse global overview fallback), bounded `pinned_head` memory across open geocells (migrating head blocks into `RangeBlockPool` or capping total pinned head bytes), and **1 logical pyramid** appetite accounting in `GpuTileCache` (`pin_root = false` — **never** adding per-geocell streams to `PendingUploads::pinned_root_streams`, exposing `PendingUploads::clear_where` as `pub(crate)` for geocell eviction).
  - Render shaded relief into a **`blend: None` scratch target** preserving the natural `PixelIsPoint` `±0.5 px` overlap across adjacent geocells (avoiding artificial half-open `[lon, lon+1)` cracks from independent `f32` RTC origins and COP-DEM's `±50°/60°/70°/75°/80°/85°` latitude-band vertex T-junctions while guaranteeing zero double-darkening via overwrite), followed by a **single fullscreen linear-light `Multiply` composite** (`mix(vec3(1.0), relief, opacity)`, calibrated for linear-light midtone contrast on `Rgba8UnormSrgb`) onto the globe color buffer.
- [ ] **Docs:** Update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* Synthetic analytic DEM slope/aspect error bounds across latitudes (`0°`/`60°`/`80°`/polar) and rotated projections, `gutter = 1` continuity, `1°×1°` border non-double-multiply & zero-crack check, `DemMosaicLayer` geocell resolution/404 negative cache/eviction (`0` per-geocell entries in `pinned_root_streams`), cursor elevation/slope/aspect readout, `naga` SPV/HLSL/MSL, and headless GPU smoke tests pass.
  - *Manual QA:* Both single-file local DSMs and 1-click COP-DEM GLO-30 hillshade/slope/aspect render cleanly over optical imagery with zero intra-file creases or `1°×1°` border gaps and accurate elevation/slope/aspect readouts.

#### Slice 5B-2a — STAC Pagination, Multi-Temporal Timeline & Dual-Epoch Diffing `[ ]`
- [ ] **STAC `links[rel="next"]` pagination & `sortby` (`veldmap-net` / `veldmap-app`):**
  - Extend `StacClient` / `StacSearchQuery` with `sortby` (newest / lowest cloud) and bounded `links[rel="next"]` pagination (handling both `GET` `href` and `POST` `href` + `body` + `merge: true` used by AWS Earth Search, enforcing HTTPS no-downgrade and same-origin auth header guards on `next` links, capped at e.g. `500` items), plus quick relative date presets (30d / 90d / 1y) and `1..=100` page-limit UI.
- [ ] **Multi-temporal timeline scrubber:** Reuse unified `TileKey` for epoch ingestion from **STAC AOI search** (grouped by MGRS tile + AOI coverage, defaulting to L2A) and **local dated folders**, with cloud-cover filtering, adjacent-date prefetch, instant stale-request cancellation, and **`pin_root = false`** (zero per-epoch entries in `PendingUploads::pinned_root_streams`).
- [ ] **Unified offscreen dual-epoch cross-fade & physical reflectance/index diffing:** Reuse Slice `5A-1`'s lazy shared **`NaN`-filled `R32Float` + `SCL` `R8Uint`** scratch pool (pre-colormap/stretch, preserving valid `0.0` diffs and propagating `SCL`/QA validity) for `"Pin as Date A / Compare Date B − Date A"` physical reflectance/index diffing (`dNDVI`, `dNBR`) and display cross-fade (`mix`)—supporting multi-band + `SCL`, mismatched carrier levels, and non-co-registered swaths cleanly.
- [ ] **Docs:** Update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* Mock STAC `GET`/`POST` `next`-link pagination + redirect/origin guards, temporal diffing shaders (`naga` SPV/HLSL/MSL + headless GPU smoke), and float pre-stretch diff math (including `0.0` diff vs. nodata) pass.
  - *Manual QA:* Multi-month STAC timeline scrubbing, adjacent-date prefetch, and dual-epoch `dNDVI` / cross-fade comparison work smoothly.

#### Slice 5B-2b — AOI Zonal Statistics, 1D Time-Series & Float32 GeoTIFF Export `[ ]`
- [ ] **1D pixel & multi-polygon AOI zonal statistics + time-series + CSV export:**
  - Add a chunk/tile-iterating region reader on `RasterSource` running off the UI thread (reusing `SharedChunkCache` / `DecodeBudgetTracker` instead of per-pixel `sample_band` lock overhead, explicitly documenting `f32` `SharedChunkCache` vs. native-`f64` reads for `U32`/`I32`/`F64` rasters) and wire multi-feature GeoJSON polygon AOIs (per-feature stats keyed by feature ID/name, not just polygon #0, plus `"Use as STAC AOI"` from Measure Area polygons).
  - Build a lightweight 1D time-series & zonal stats panel using custom `egui::Painter` charts (avoiding pinning `egui` upgrades to lagging `egui_plot`) for single-pixel time-series and per-polygon AOI zonal stats (mean, median, std, percentiles, parcel-averaged time-series skipping `SCL`-masked pixels/dates), with **RFC 4180 CSV export** (ISO-8601 UTC timestamps, explicit units, documented `SCL`-masked row convention).
- [ ] **Float32 GeoTIFF export (native striped writer preferred):**
  - Export computed band-math, dual-epoch diff, or AOI-clipped layers as georeferenced striped **Float32 GeoTIFF** via a lightweight self-contained native writer (~250–300 LOC in `veldmap-raster`) over workspace `flate2` (`zlib-rs`) / optional `zstd`, keeping `tiff` permanently out of production dependencies while supporting **`Compression = 8` (Adobe Deflate, default for universal GDAL/QGIS compatibility) + `Predictor = 3`** (TIFF TN3 big-endian 4-byte lane float horizontal diff, 2–4× smaller than `Predictor::None`) and optional `ZSTD`. Enforce strict TIFF 6.0 invariants (ascending tag IDs, `<= 4 B` inline values including single-strip offsets/counts, 2-byte aligned out-of-line payloads, NUL-terminated ASCII counts, mutually exclusive `33550`+`33922` vs. `34264`, checked Classic TIFF `u32` bounds, EPSG/CustomTM GeoKeys + optional `.prj` for arbitrary WKT CRSs). *(Fallback if scope demands: re-enable `tiff` in production with `features = ["deflate"]` and `Predictor::None`, updating `docs/decisions.md` §11.)*
- [ ] **Docs:** Update `docs/decisions.md`, `docs/limitations.md`, and `README.md`.
- **Exit criteria:**
  - *Automated CI:* Zoom-invariant pixel & multi-polygon AOI zonal stats/time-series tests, RFC 4180 CSV export, and Float32 GeoTIFF export round-trip tests decoded by **both** `TiffSource` and dev-dependency `tiff 0.11.3` pass.
  - *Manual QA:* End-to-end v1.0 EO workflow (local + STAC COGs, band math, swipe/diff, pixel & AOI time-series + CSV/GeoTIFF export, analytical DEM relief) persists cleanly across sessions.

---

### Milestone v1.1 — Cloud-Native EOPF Zarr, Authenticated CDSE & Archive Codecs

#### Slice 4B-1 — ESA EOPF Zarr Cloud Streaming & Authenticated CDSE S3 / Resumable Downloads `[ ]`
- [ ] **EOPF Zarr streaming provider (`veldmap-core` / `veldmap-raster` / `veldmap-net`):**
  - Evaluate and integrate pure-Rust **`zarrs`** (`MIT OR Apache-2.0`; audit optional features against `deny.toml`'s `all-features = true` `memmap2` ban before adopting) for ESA's cloud-native **EOPF Zarr** Sentinel-2/3 stores (replacing legacy remote HTTP JP2 packet-surgery, which measured `70–77 s` and `255` HTTP requests/band).
  - Bind Zarr at **`RasterSource`** (not forced through single-file sync `ByteSource`): add a narrow sync key-value object-store trait in `veldmap-core::io` backed by `NetContext` in `veldmap-net` (preserving zero horizontal deps between `veldmap-raster` and `veldmap-net`) and generalize remote tile prefetch in `OverlayManager` beyond single-file `Vec<ByteRange>`.
- [ ] **CDSE OData/STAC/S3 SigV4 provider, local library & resumable `.part` downloads (`veldmap-net` / `veldmap-storage`):**
  - Implement CDSE OData/S3/SigV4 (`Signer`), multi-band Sentinel-2 `.SAFE`/EOPF asset discovery (including `MTD_MSIL2A.xml` DN-space `BOA_ADD_OFFSET` normalized to `offset = BOA_ADD_OFFSET / QUANTIFICATION_VALUE`), epoch-based singleflight 401/403 re-sign, and timeline scrubber integration.
  - Implement local catalog, hardened resumable `.part` downloads (resolving every legacy hazard below), and an in-app credentials/cache dialog writing `ProjectDirs/.env` (`0600` on Unix) with mandatory credential/SigV4 redaction in logs/errors and anonymous search without keys.
- [ ] **Tests & docs:** EOPF Zarr chunk-streaming tests, CDSE provider + mock 401/403 + `.part` crash-resume + secret-redaction tests; update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* EOPF Zarr streaming, CDSE provider, mock 401/403 & `.part` resume, and secret-redaction log tests pass.
  - *Manual QA:* Stream EOPF Zarr scenes directly and browse/download/pause/resume CDSE archives reliably.

#### Slice 3C-1 — Local Sentinel-2 `.SAFE` JPEG 2000 `[ ]`
*(Scoped strictly to local offline `.SAFE` JP2 archives; cloud Sentinel-2 streams via COG and EOPF Zarr rather than remote HTTP JP2 packet-surgery.)*
- [ ] **Evaluate and select JP2 decoder:** Carry out a pre-implementation evaluation and record the decision in `docs/decisions.md`, comparing: (1) pure-Rust tile/ROI-capable decoders (`openjp2` `0.7+` audited under Miri/ASan/multi-thread/fuzzing for `c2rust` soundness, plus newer native Rust decoders such as `rust-j2k` / `jpeg2000_decoder`), (2) `hayro-jpeg2000` (consider only if single-tile + resolution-level random-access decoding is supported), and (3) `openjpeg-sys` (only if updated to OpenJPEG `>= 2.5.4` patching `CVE-2025-54874`, or a maintained self-contained `cc` wrapper; verify `deny.toml` `all-features = true`).
- [ ] **Implement & harden local JP2 adapter (`veldmap-raster`):** Implement local `.SAFE` JP2 decoding over the selected decoder and GMLJP2 parsing (`roxmltree`/`quick-xml`), resolving every applicable legacy hazard below (only keeping local tile-part excerpting if required by the selected decoder for single-tile ROI reads).
- [ ] **Tests & docs:** Adapt `jp2_tests.rs` (and `excerpt_tests.rs` if applicable) + 5/3 reversible round-trip and malformed/truncated-input fuzz tests; update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* 5/3 JP2 round-trip and malformed/truncated-input tests pass with zero panics or UB.
  - *Manual QA:* Local Sentinel-2 `.SAFE` JP2 granules open and render cleanly on the globe.

#### Slice 6 — NetCDF-4 (Sentinel-5P TROPOMI & Historical Sentinel-3) `[ ]`
- [ ] **Evaluate and select HDF5/NetCDF-4 reader:** Benchmark and audit pure-Rust candidates (`hdf5-pure` / `rustyhdf5` / `rustyhdf5-netcdf4` / `netcdf-reader`) against real ESA Sentinel-3/5P NetCDF-4 files (`Send + Sync`, superblock v2/v3, Shuffle+Deflate+Fletcher32, `f64` coordinates/scaling, raw typed integer reads, and `deny.toml` `all-features = true` compliance). Keep an optional C `libhdf5` fallback strictly as a last resort (requires NCSA license clarification, CMake build support, and scoped CI feature flags).
- [ ] **Implement NetCDF-4 adapter (`veldmap-raster` / `veldmap-app`):** Resolve every legacy hazard below, add local `.SAFE` sibling geolocation, `StreamKey`-level `TileScheduler` reset, NetCDF variable switcher UI, and upgrade `TieGrid` (`veldmap-core::geo::binding::tie_grid`) with spatial indexing, unit-sphere/ECEF polar interpolation, and **masked/`NaN`-edge cell tolerance** (eliminating `O(M×N)` UI-thread scans in `CursorProber`, pole-crossing longitude jumps, and all-or-nothing rejection of Sentinel-3 SLSTR / Sentinel-5P TROPOMI `NaN` swath edges).
- [ ] **Tests & docs:** Adapt NetCDF tests + malformed-input tests; update `docs/decisions.md`, `docs/limitations.md`, and `README.md`.
- **Exit criteria:**
  - *Automated CI:* Full 3-OS CI passes with NetCDF `f64`/fill-value/malformed-input and `TieGrid` `NaN`-edge/polar tests green.
  - *Manual QA:* Local Sentinel-3/5P NetCDF-4 files open, switch variables, and render cleanly.

#### Slice 3B-2 — Optional 1:1 Native-Pixel / Raw-Sensor 2D Inspector (`Image2DRenderer`) `[ ]`
*(Independent power-user sub-slice for ungeoreferenced / 1:1 native-pixel inspection and linked side-by-side multi-view; can land anytime after `5A-1`.)*
- [ ] Dedicated orthographic **`Image2DRenderer`** in `CallbackResources` sharing immutable `Arc<GpuTile>` textures and the hoisted `TileBindGroupCache` with `GlobeRenderer` (mapping composite UVs against unclamped vs. clamped `ViewportInPixels` so partially off-screen panels never stretch, and unioning 2D `wanted` + `carrier_keys` into `SceneState` so 3D frames never evict active 2D tiles) + synchronized zoom-invariant CPU raw DN & physical-unit cursor readout.

---

## Migrated Legacy Porting Hazard Checklists (`legacy-reference-snapshot`)

Audited safety, correctness, and architecture hazards preserved from retired `old/README.md` (consult `git show legacy-reference-snapshot:old/...` on demand; shallow clones: `git fetch origin tag legacy-reference-snapshot`). Do not port legacy code verbatim—resolve every item below when implementing the corresponding v1.1 slice:

- **Slice 4B-1 (`old/code/network/download.rs`, `old/code/library/*`, `old/code/provider/*`):** Drop WASI `queued_sidecars` / async rescan state machines in favor of serialized sync `std::fs` metadata; enforce S3 path sanitization, Windows `\\?\` extended-length paths (`longPathAware`), unsigned `Range` header invariant on 401/403 re-sign, persisted `ETag` `If-Range`, HTTP 416 recovery, `File::try_lock()` + in-process lock map + explicit `unlock()`, `downloaded == total_size`, and credential/SigV4 redaction in `tracing` logs/errors.
- **Slice 3C-1 (`old/code/tiler/adapters/{codec,excerpt,jp2}.rs`):** Audit `openjp2` for `static mut` thread safety/soundness if evaluated; fix Stacked Borrows / drop-order UB (`Box::into_raw` + explicit `Drop`), `usize::MAX` EOF + `ErrorKind::Interrupted` retry, `catch_unwind` on `extern "C"` callbacks, `skip()` EOF bounds check, `COD` `levels <= 32` shift-overflow guard, worker panic isolation, and gating zero-as-nodata on Sentinel-2 heuristics rather than all GMLJP2 (plus PLT varint & TLM checked arithmetic + `keep <= part.len` + `MAX_PAD` cap if local tile-part excerpting is retained).
- **Slice 6 (`old/code/tiler/adapters/netcdf.rs`):** Read coordinates and `scale_factor`/`add_offset` in `f64`; check raw typed `_FillValue`/`missing_value`/`valid_range` *before* unpacking (mapping fill to `f32::NAN`); fail fast on unsupported HDF5 filters at `describe`; bound `Hdf5HandleCache` (`METADATA_CACHE` + `ROW_CACHE_CAP`) with fixed host-RAM-scaled caps.

---

## Next Step

- **Current state:** Phases 0–0.5, Milestone **v0.2** (Slices 1, 2A, 2B, 3A-1, and 3A-2), and Milestone **v0.5 — Slices 3B, 4A & 5A-1** (Scientific `R32Float`/`R8Uint`/`R16Uint` Tracks, Direct Raw-Chunk GeoTIFFs, Live Stretch, Cloud EO HTTP Range / Zero-Auth & Custom STAC COG Streaming, Multi-Asset Layers, STAC Band Recipes, and Screen-Space GPU Band Math + `SCL` Masking) are complete (see [Design Decisions](docs/decisions.md) §1–§12 for settled invariants).
- **Immediate task:** Start **Milestone v0.5 — Slice 5A-2: Split-Screen Swipe, Cloud Workspace Persistence & Cartographic PNG Export** (interactive swipe curtain across any two layers including `BandMath`, tagged `session.json` / `.veldmap.json` schema with secret stripping and offline restore preservation, DD/DMS/MGRS coordinate copy, and off-UI-thread cartographic PNG screenshot export).
