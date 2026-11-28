# VeldMap Development Plan

## Goal

Build VeldMap as a fast, purely native Rust desktop **Analytical Earth-Observation & Precision-Imagery Workstation** on a **Precision Nadir 3D WGS84 Globe** for:
- **v0.2–v1.0:** Interactive WGS84 planetary viewing (bundled + live basemaps and local GeoTIFF/BigTIFF/COG/PNG/JPEG overlays down to centimetre-scale orthophotos), zero-auth & custom-header STAC/COG HTTP Range streaming, GPU scientific/categorical tracks (`R32Float`/`R8Uint`/`R16Uint`) with interactive band math (`NDVI`/`NDWI`/`NBR`/`NDSI`/false-color + `SCL` cloud masking), split-screen swipe & cloud workspace sessions, multi-temporal timeline diffing, 1D pixel & multi-polygon AOI zonal statistics/time-series inspection, analytical UX & Float32 GeoTIFF export, and analytical terrain relief (single-file local DEM/DSM + COP-DEM GLO-30 virtual mosaic).
- **v1.1:** Cloud-native ESA EOPF Zarr streaming & authenticated Copernicus Data Space (CDSE) S3/resumable downloads, local Sentinel-2 `.SAFE` JPEG 2000, Sentinel-5P/3 NetCDF-4/HDF5, and an optional 1:1 native-pixel 2D orthographic inspector.

A simplified browser/mobile companion is intentionally deferred to a separate project later (enabled by keeping `veldmap-core` permanently UI/GPU/network-free).

> **Document lifecycle:** `plan.md` is the vertical-slice roadmap and exit-criteria tracker (update `Next Step` as each slice closes). Upcoming slices (`5B-2b`–`5B-1b`, `4B-1`–`3B-2`) are **working design hypotheses**—revise them freely as real code and benchmarks prove better designs. Settled "why not X" decisions and hard cross-slice invariants live in [docs/decisions.md](docs/decisions.md) once implemented; module details live in Rust `//!` docstrings (`cargo doc --document-private-items`).

---

## Working Cross-Slice Principles

Implemented decisions and hard cross-slice invariants are maintained in [docs/decisions.md](docs/decisions.md) (§1–§14). Working principles for upcoming slices:

1. **Natural workspace crate boundaries (CI-enforced):**
   - Keep `veldmap-core` a permanently pure leaf (`geo`, `tile` keys/LRU, std-only `sync`/`io` traits/`MemoryBudget`, `#![forbid(unsafe_code)]`, zero OS probing or `tokio`/`reqwest`/`image`/`tiff`/`rusqlite`/`egui`/`wgpu`/`rfd`/`directories`).
   - Keep zero horizontal dependencies among middle crates (`veldmap-raster`, `veldmap-storage`, `veldmap-net`) and keep `veldmap-app::ui` presentation-only (zero `veldmap-net`/`veldmap-raster`/`veldmap-storage` imports).
   - Keep `veldmap-app` unified (avoid speculative crate splits while internal module boundaries and headless `test-support` GPU tests remain clean).
2. **Deadlock-free memory & deduplication invariants:**
   - Bound CPU caches with startup host-RAM-scaled fixed per-cache caps (avoiding lock-reentrant cross-cache shrinking).
   - Track dynamic viewport/scratch GPU VRAM in `veldmap-app` via typed `ScratchDemand` against `GpuTileCache`'s effective budget with a hard `64 MiB` minimum floor (keeping `veldmap-core::MemoryBudget` pure/static), and scale GPU tile VRAM appetite proportionally for multi-band EO working sets.
   - Bound decode working memory and queued upload bytes so small tiled reads never stall behind full-image decodes and large passes never starve; deduplicate concurrent chunk/block fetches with panic/cancel-safe singleflight.
3. **Codec safety & durability invariants:**
   - Decoders enforce explicit dimension/memory/IFD/filter limits, panic isolation, and thread-safety audits before multi-threaded decode.
   - Inject `StoragePaths`; use durable fsync'd atomic writes (closing handles before rename on Windows) for user/session/project state and crash-safe vertically partitioned (`tile_meta`/`tile_blobs`) SQLite WAL persistence for ephemeral tile caches.
   - Add traits only where there is a real second implementation (`ByteSource`, `Signer`, raster decoders, tile providers).
   - Keep default builds free of external C system-library setup requirements (self-contained `cc`-compiled crates like bundled SQLite, `ring`, or `zstd-sys` are acceptable when justified; external `libgdal`/`libproj`/`libhdf5` and CMake-required codecs like `blosc-src` are avoided in default builds).
4. **Legacy reference retired (pre-retirement commit `9cd029f`):** The legacy `old/` reference tree was retired in commit `0f5ff4d` at the close of Slice 3A-1b; remaining v1.1 (`4B-1`, `3C-1`, `6`) porting hazard checklists live at the bottom of this document. Pre-deletion state remains in git history at commit `9cd029f` (`0f5ff4d~1`, immediately before `0f5ff4d`; consult `git show 9cd029f:old/...` on demand). Design native modules cleanly rather than porting WASI-era decomposition verbatim.
5. **Code & test hygiene (every slice):** English only across the entire repository (CI-enforced). Follow the 3-tier Rust test placement policy:
   - **Inline `#[cfg(test)] mod tests`:** Short single-module white-box unit checks (`<= ~100 LOC`).
   - **Sibling `#[cfg(test)] #[path = "*_tests.rs"]`:** Larger single-module unit suites (`> ~100 LOC`) or unit suites needing private access/fixtures without widening `pub`.
   - **Grouped `tests/integration/main.rs`:** Cross-module, public-API contract, synthetic-file, mock-HTTP, `naga` shader codegen, and headless GPU smoke tests (never widen `pub` or `test-support` solely to relocate a white-box unit test).
   - Verified on Rust 2024 by `cargo fmt --all -- --check`, `cargo clippy --locked --workspace --lib --bins -- -D warnings`, `cargo clippy --locked --workspace --all-targets -- -D warnings`, `cargo test --locked --workspace`, isolated headless crate tests, `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps --all-features --document-private-items`, workspace DAG & `ui/` isolation checks, `cargo-deny`, `naga` SPV/HLSL/MSL codegen, headless offscreen `wgpu` smoke tests, and offline `lychee` link checks.

---

## Completed Foundation `[x]`

*(Settled rationale and cross-slice invariants live in [docs/decisions.md](docs/decisions.md) §1–§14.)*

| Milestone / Slice | Scope Delivered | Decisions |
|---|---|---|
| **Phase 0 & 0.5** `[x]` | Native Rust 2024 workspace (`eframe 0.37` / `wgpu 30` / Rust `1.95`), workspace lints, `cargo-deny`, English-only & offline `lychee` CI | §1–§2 |
| **v0.2 — Slice 1** `[x]` | `veldmap-core::geo` (WGS84 geodesy, Krueger projections/Helmert datums, MGRS/DD/DMS parsing, `f64` Reverse-Z `Camera` with `MIN_HEIGHT_M = 2.0` m, oblate `CullFrustum`), offscreen `wgpu` renderer (`Rgba8UnormSrgb` + `Depth32Float`), baseline HUD, 3-OS CI + `naga` + GPU smoke | §3–§4 |
| **v0.2 — Slice 2A** `[x]` | `f64` ECEF metres + Fukushima 3D inverse + geodetic nadir aim, decoupled geometry/texture quadtree + Polar Stereographic caps, per-tile CPU `f64` RTC + procedural graticule, `TileKey` (`gutter: u8`), `TileCache`/`GpuTileCache`/`PendingUploads`, bundled NASA Blue Marble | §5 |
| **v0.2 — Slice 2B** `[x]` | `veldmap-storage` (durable private atomic JSON, vertically partitioned `tile_meta`/`tile_blobs` SQLite WAL LRU cache), `veldmap-net` (`tokio` + `reqwest`/`rustls` + `ring`, NASA GIBS / EOX / Custom XYZ/TMS), `TwoLaneDecodePool`, `SceneState`/`SnapshotKey`, host-RAM-scaled `MemoryBudget` | §6 |
| **v0.2 — Slice 3A-1** `[x]` | Safe positioned `ByteSource` (zero `memmap2`), sampled BLAKE3 `SourceFingerprint`, cross-OS `detect_memory_budget`, memory-bounded `TiffSource`, `DecodeBudgetTracker`, `SharedChunkCache`, `TileScheduler`, `layers/` + `CursorProber`, depth-less Pass 2 warp-mesh draping (`< 2 mm` accuracy benchmarks), QOI tile cache, `old/` retirement (`0f5ff4d`) | §7–§8 |
| **v0.2 — Slice 3A-2** `[x]` | Pure-Rust `proj4rs`/`proj4wkt`/`crs-definitions` + `HelmertConvention`, TIFF `YCbCr`/`WebP`, `PngSource`/`JpegSource` + world-file/`.prj` sidecars, Karney geodesic measurement, pole-safe Pass-2 vectors, non-blocking `rfd` dialogs, heading rotation & keyboard nav, `dejavu` fallback fonts, `session.json` persistence | §9 |
| **v0.5 — Slice 3B** `[x]` | `StreamKey`-keyed uploads & multi-band VRAM appetite budgeting (`128–4096 MiB`), hoisted `TileBindGroupCache`, Display/Scientific (`R32Float` NaN-skipping mips, 8-tap nodata-trilinear WGSL, linear/`log10` dB stretch, histogram, colorbar, native-`f64` local cursor readout)/Categorical (`R8Uint`/`R16Uint`) tracks, lock-free direct raw-chunk GeoTIFF decoding (`Planar-1/2`, RGB+NIR, `Palette`, `ZSTD`/`YCbCr`/`WebP`), byte-shuffled `Lz4Raw` disk cache | §10 |
| **v0.5 — Slice 4A** `[x]` | Shared `NetContext` (split identity/gzip clients, split timeouts, host-scoped cooldowns, 64 KiB `RangeBlockPool` & `PrefetchBudget`), `HttpRangeSource` (1-RTT `0..256 KiB` head probe, strong-ETag `If-Match`, refcounted singleflight, pinned coalesced prefetch), pure-`ByteSource` `SinglePassIfdReader` (`tiff` dev-only), isolated `veldmap-remote-io` + offloaded `veldmap-tile-write`, AWS Earth Search Sentinel-2 L2A & COP-DEM GLO-30 + custom-auth STAC + sample scenes, retained vector AOIs + GeoJSON/WKT export, HiDPI ribbon/hatch globe footprints, 3-tab Settings modal, collapsible `Layers` \| `Cloud / STAC` workbench | §11 |
| **v0.5 — Slice 5A-1** `[x]` | Tagged `LayerKind::{SingleRaster, BandMath}` + `RasterAsset`, presentation-only `ui/` over STAC view DTOs, lazy shared quiet-`NaN`-filled `R32Float` + `R8Uint` `SCL` scratch pool (`ScratchDemand`) with co-grid MRT & clamped scissor bounds, floored dynamic target VRAM accounting (`64 MiB` floor), per-band GDAL/STAC 1.0 & 1.1 `(scale, offset, nodata, unit)` propagation, 1-click STAC Scene Recipes + local Band Math builder, `RdYlGn`/`BrBG` colormaps, zoom-invariant `BandMath` CPU cursor readouts | §12 |
| **v0.5 — Slice 5A-2** `[x]` | Split-screen swipe curtain (`SwipeRenderState` outside `DrawSnapshot`, physical `split_x` in `LastRenderKey`, pre-scratch `BandMath` scissor intersection, swipe-aware cursor readout), tagged v2 `session.json` & portable `.veldmap.json` project files (relative paths, sanitized URLs, same-origin imported-project credential boundary, offline-resilient slot-preserving `SessionRestoreTracker`), pure-Rust WGS84 → MGRS formatter & DD/DMS/MGRS copy, single-line fixed-height status bar, vector `"Outline only"`, Measure-to-STAC-AOI, rolling 90d UTC STAC defaults, `NBR`/`NDSI` 1-click STAC recipes, clean-frame `COPY_SRC` non-blocking cartographic PNG export via `veldmap-raster`, workspace bumped to `0.5.0` | §13 |
| **v1.0 — Pre-5B & Slice 5B-2a** `[x]` | Pre-5B capability refactor (`LayerSummaryKind`, nested `OverlayEdit`, tagged `RemoteLayerKind`, `OverlayLayer::contribute_appetite`, `compute_patch_sets_scissor_rect`, split `storage/layers.rs`), STAC `sortby` & bounded `GET`/`POST`+`merge` `links[rel="next"]` pagination (500 cap), named Custom STAC profiles + `30d`/`90d`/`1y` presets + collection override, origin-fenced singleflight `401`/`403` `stac_item_href` restore refresh, `LayerKind::Timeline` (STAC MGRS-tile L2A grouping, local dated-folder `ISO -> TIFF Tag 306 -> lexicographic` scan, bounded LRU open epochs, cloud filter, adjacent-date prefetch-wanted protection, zero-purge scrub, dual-epoch CPU cursor readout & tagged persistence), and unified offscreen dual-epoch GPU scalar diff (`band_math_reduce.wgsl`, `N = 1..=3`) + 0%-bleed-through RGB cross-fade (`rgba_targets: 0..=2` on `@group(3)`) within `downlevel_defaults()` | §14 |

---

## Upcoming Vertical Slices (`v1.0` → `v1.1`)

> **Execution order:** Slices are grouped into shippable milestones (`v1.0` → `v1.1`). Slice IDs (`5B-2a/b/c`, `5B-1a/b`, `4B-1`, `3C-1`, `6`, `3B-2`) are retained from original roadmap tracks for git-history continuity.
>
> In **v1.0**, the EO analytical loop (`5B-2a` → `5B-2b` → `5B-2c`) executes ahead of terrain relief (`5B-1a` → `5B-1b`) so `Discover → Band Math → Temporal Diff → Zonal Stats/Time-Series → CSV & Float32 GeoTIFF Export` ships first, `5B-2a` introduces the shared `Rgba8UnormSrgb` scratch target slot(s) reused by `5B-1b`, and `5B-1a`/`5B-1b` terrain layers immediately plug into Click-to-Inspect, AOI zonal stats, and GeoTIFF export without rework. (`5B-1a` remains independent if needed earlier.)

---

### Milestone v1.0 — Multi-Temporal EO Analytics, Export & Analytical Terrain Relief

#### Slice 5B-2b — Multi-Polygon AOI Zonal Statistics, 1D Time-Series & CSV Export `[ ]`
- [ ] **Off-UI-thread chunk/tile region reader (`veldmap-raster` / `veldmap-app`):**
  - Add a chunk/tile-iterating region reader on `RasterSource` running off the UI thread, reusing `SharedChunkCache` / `DecodeBudgetTracker` and async-prefetching intersecting byte ranges for remote COGs before chunk iteration.
  - Document `f32` `SharedChunkCache` vs. native-`f64` reads for `U32`/`I32`/`F64` rasters, and use `bands[0]`'s `Frame`/pixel grid as the canonical reference grid for mixed-GSD `BandMath` secondary bands and `SCL`.
- [ ] **Multi-polygon AOI zonal statistics, 1D time-series & RFC 4180 CSV export:**
  - Wire multi-feature GeoJSON polygon AOIs in `VectorManager` (per-feature stats keyed by feature ID/name, not just polygon #0) and fix `"STAC AOI"` import from multi-polygon GeoJSON layers (unioning polygons into a `MultiPolygon` or clearly reporting multi-feature bounds instead of silently using polygon #0).
  - Build a lightweight 1D time-series & zonal stats panel using custom `egui::Painter` charts (avoiding pinning `egui` upgrades to lagging `egui_plot`) for single-pixel time-series and per-polygon AOI zonal stats (mean, median, std, percentiles, parcel-averaged time-series skipping `SCL`-masked pixels/dates).
  - Export zonal stats and time-series as **RFC 4180 CSV** (ISO-8601 UTC timestamps, explicit units, documented `SCL`-masked row convention).
- [ ] **Docs:** Update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* Zoom-invariant single-pixel and multi-polygon AOI zonal stats/time-series tests (including mixed-GSD `BandMath` + `SCL` masking), multi-polygon STAC AOI import tests, and RFC 4180 CSV export tests pass.
  - *Manual QA:* Parcel-level multi-polygon zonal stats, 1D temporal charts, and CSV export work smoothly across local rasters and streamed STAC timelines.

#### Slice 5B-2c — Analytical UX Essentials & Float32 GeoTIFF Export `[ ]`
- [ ] **Pragmatic v1.0 analytical UX essentials (zero QGIS bloat):**
  - Expose an unlink / **per-channel R/G/B stretch** toggle on `BandMath` cards (unlocking per-channel `rgb_min`/`rgb_max` already implemented in `band_math_composite.wgsl`).
  - Add lightweight per-layer **user NoData / black-collar override** persisted in the tagged layer schema: apply at **decode time on commit** (checkbox/Enter/blur, never per-frame drag) folded deterministically into `TileStreamSpec::decode_rev` (avoiding mipmap/bilinear dark fringes on `Display` & `Scientific` while keeping rendering, cursor readouts, histograms, zonal stats, and GeoTIFF export 100% consistent).
  - Add a lean **Click-to-Inspect** popover displaying multi-layer raster stack values + GeoJSON feature `properties` (via a formatted key-value accessor in `veldmap-core`, keeping `serde_json` out of `veldmap-app`), performing a bounded async single-chunk Range fetch + native `f64` probe on remote COGs on explicit click (resolving zoomed-out `n/a` and `u32`/`i32`/`f64` precision on demand while keeping hover strictly cache-only).
  - Upgrade cartographic PNG export (`export.rs`) text rendering from the 5×7 ASCII bitmap font to bundled `dejavu::sans::regular()` so non-ASCII layer captions and colorbar titles (European diacritics, Cyrillic, Greek, etc.) render crisply.
  - *(Arbitrary custom RPN Band Math expressions remain intentionally deferred beyond v1.0 so WGSL, CPU readout, zonal stats, and GeoTIFF export stay simple and lockstep.)*
- [ ] **Float32 GeoTIFF export (native striped writer preferred):**
  - Default remote/computed exports to **Active AOI or Current Viewport bounds** (with explicit opt-in for full extent and an upfront `W × H px` / estimated-byte guardrail preventing accidental 480 MiB full-Sentinel-2 transfers).
  - Retain `source_epsg`/`source_wkt` on `Proj4Projection` and `epsg` on `CrsSpec::Geographic` / `FrameKind::Grid`, exposing `epsg_code()`, `source_wkt()`, and `pixel_affine()` on `Projection` and `Frame` so exported GeoKeys and optional `.prj` sidecars round-trip arbitrary CRSs accurately.
  - Export computed band-math, dual-epoch diff, or AOI-clipped layers as georeferenced striped **Float32 GeoTIFF** via a compact self-contained native writer in `veldmap-raster` over workspace `flate2` (`zlib-rs`) / optional `zstd` (`Compression = 8` Adobe Deflate + `Predictor = 3` float horizontal diff by default, keeping `tiff` dev-only, with checked Classic TIFF `u32` bounds). *(Fallback if scope demands: re-enable `tiff` in production with `deflate` + `Predictor::None`, updating `docs/decisions.md` §11.)*
- [ ] **Docs:** Update `docs/decisions.md`, `docs/limitations.md`, and `README.md`.
- **Exit criteria:**
  - *Automated CI:* Per-channel RGB stretch, decode-time `decode_rev` NoData override, Click-to-Inspect (multi-layer + remote single-chunk `f64` + GeoJSON `properties`), DejaVu non-ASCII PNG export text, CRS provenance round-trip, and Float32 GeoTIFF export round-trip tests decoded by **both** `TiffSource` and dev-dependency `tiff 0.11.3` pass.
  - *Manual QA:* End-to-end EO analytical workflow (local + STAC COGs, band math, swipe/diff, pixel & AOI time-series + CSV/GeoTIFF/PNG export) persists cleanly across sessions.

#### Slice 5B-1a — Single-File Local DEM/DSM Terrain Relief & `gutter = 1` Tiles `[ ]`
- [ ] **`gutter = 1`, 1-mip (`514×514`) `R32Float` tile pipeline (`veldmap-raster`):**
  - Replace the 6-tuple `TileEmit` closure with a structured `EmittedTile` carrying `(valid_w, valid_h, gutter)`.
  - Enable `gutter = 1` in `RasterSource` / `TiffSource` (widening `Grid::footprint`, `direct_peak_for_footprint`, and `plan_direct` by `+2*gutter`; falling back to `gutter = 0` **stream-wide** rather than per-level on oversized chunks so `PyramidMeta` and `TileKey::parent` ancestor carrier walks stay consistent; `ChunkKey` is already gutter-agnostic so scientific and relief streams share cached raw chunks).
  - Handle stripped / non-overviewed local DEM GeoTIFFs in `ServeMode::Pass` explicitly: either add 1-row top/bottom overlap buffering to `Cascade` for `gutter = 1` emission or document stream-wide `gutter = 0` fallback for `Pass`-mode DEMs.
  - Extend CPU tiling & resampling (`region_of` in signed `i128`) to signed `±1` halo coordinates (aligning `ceil` and GDAL `floor` overviews on `2^level` base-pixel spacing) and explicitly guard out-of-bounds halo texels in `resample_f32_window` to emit `f32::NAN` rather than normalizing partial edge slivers (clean one-sided boundary derivatives).
- [ ] **Analytical GPU hillshade, slope & aspect + cursor readout (`veldmap-core` / `veldmap-app`):**
  - Compute per-vertex **dual (contravariant) 3D ECEF tangent vectors** `[T^x, T^y] = J_base (J_base^T J_base)^(-1)` with respect to **base-image pixels** (keeping 56 B DEM vertex buffers invariant across clipped edges and carrier LOD upgrades):
    - Tag vertex layout distinctly in `OverlayMeshKey` (32 B vs. 56 B), update `OverlayMeshPool::account_patch_set` to account `56 B/vertex` for DEM meshes, and add `carrier_level: u8` to `DrawOverlayPatch` so WGSL multiplies finite-difference derivatives by `2^(-carrier.level)` with one-sided fallback on `NaN` boundaries.
    - Enforce a **scale-invariant & polar-safe** degeneracy guard: `!det.is_finite() || g00 <= min_g || g11 <= min_g || det <= eps_rel * g00 * g11` (e.g., `min_g = 1e-12` `(m/px)^2`, preventing false-zeroing of cm-scale drone DSMs where `det ~ 1e-8..1e-11` while still clamping polar `lat = ±90°` orthogonal `g00 → 0` collapse).
  - Ship single-file local DEM/DSM relief with interactive **sun azimuth/altitude, Z-factor**, linear-light **Multiply blend** option (for draping drone DSM hillshades over orthophotos), explicit vertical datum labeling (`m EGM2008` / source units), and quantitative cursor / Click-to-Inspect readout (`Elevation`, `Slope (°)`, `Aspect (°/compass)`).
- **Exit criteria:**
  - *Automated CI:* Synthetic analytic DEM slope/aspect error bounds across latitudes (`0°`/`60°`/`80°`/polar), cm-scale drone DSMs, and rotated projections; `gutter = 1` continuity & `floor`-overview out-of-bounds `NaN` halo checks; cursor elevation/slope/aspect readout; `naga` SPV/HLSL/MSL; and headless GPU smoke tests pass.
  - *Manual QA:* Single-file local drone DSMs and GeoTIFF DEMs render hillshade (including Multiply over orthophoto), slope, and aspect cleanly with zero intra-file tile creases and accurate readouts.

#### Slice 5B-1b — Global COP-DEM GLO-30 Virtual Mosaic (`DemMosaicLayer`) `[ ]`
- [ ] **Multi-file virtual mosaic & 1-click COP-DEM GLO-30:**
  - Implement `DemMosaicLayer` resolving visible `1°×1°` `PixelIsPoint` (`3601×3601`) geocell COGs deterministically under one Layer Manager entry with:
    - Bounded LRU of open remote `ByteSource`s and negative caching of absent/404 ocean geocells;
    - Coarse-zoom geocell cutoff (or coarse global overview fallback);
    - **1 logical pyramid** appetite accounting via `OverlayLayer::appetite_weight_bytes()` in `GpuTileCache`, and `pin_root = false` (zero per-geocell entries in `PendingUploads::pinned_root_streams`, exposing `PendingUploads::clear_where` as `pub(crate)` for geocell eviction).
  - Add `HttpRangeSource::release_pinned_head()` migrating complete `64 KiB` blocks (plus a shorter final block strictly when `start + len == file_len`, never partial non-EOF tails) into the bounded `RangeBlockPool` and dropping the permanent head pin once `TiffSource::open` finishes, bounding open-geocell head RAM globally for all remote COGs.
- [ ] **Crack-free & double-multiply-free relief scratch compositing:**
  - Reuse `ScratchTargetPool`'s `blend: None` `Rgba8UnormSrgb` scratch target (`+4 B/px`, keeping `DemMosaic` excluded from the Band Math builder) preserving natural `PixelIsPoint` `±0.5 px` overlap across adjacent geocells (avoiding artificial cracks at COP-DEM's `±50°/60°/70°/75°/80°/85°` latitude-band T-junctions while guaranteeing zero double-darkening via overwrite).
  - Composite via a single fullscreen pass (intersecting `ResolvedSwipe::scissor_for_layer` before scratch): linear-light `Multiply` (`mix(vec3(1.0), relief, opacity)` with flat-terrain illumination calibrated near `1.0` so `[0, 1]` `Rgba8UnormSrgb` does not over-darken flat areas) for hillshade, and standard alpha blend for colormapped slope/aspect.
- [ ] **Docs:** Update `docs/decisions.md`, `docs/limitations.md`, and `README.md`.
- **Exit criteria:**
  - *Automated CI:* `1°×1°` border non-double-multiply & zero-crack check, `DemMosaicLayer` geocell resolution / 404 negative cache / bounded pinned-head release / eviction (`0` per-geocell entries in `pinned_root_streams`), `naga`, and headless GPU smoke tests pass.
  - *Manual QA:* 1-click COP-DEM GLO-30 hillshade/slope/aspect renders cleanly over optical imagery across latitude-band transitions and ocean coastlines with accurate elevation/slope/aspect readouts and AOI zonal elevation/slope stats.

---

### Milestone v1.1 — Cloud-Native EOPF Zarr, Authenticated CDSE & Archive Codecs

#### Slice 4B-1 — ESA EOPF Zarr Cloud Streaming & Authenticated CDSE S3 / Resumable Downloads `[ ]`
- [ ] **EOPF Zarr streaming provider (`veldmap-core` / `veldmap-raster` / `veldmap-net`):**
  - Evaluate and integrate pure-Rust **`zarrs`** (`MIT OR Apache-2.0`; audit optional features against `deny.toml`'s `all-features = true` `memmap2` ban). Because ESA EOPF Zarr Sentinel-2 stores default to `blosc` (`zstd` + bitshuffle) while `zarrs`'s `blosc` feature pulls CMake-dependent `blosc-src` on desktop platforms, keep `zarrs` with `default-features = false` and evaluate a pure-Rust Blosc1 codec plugin over workspace `zstd` / `lz4_flex` / `flate2` (or `blusc`) so default builds stay CMake-free (replacing legacy remote HTTP JP2 packet-surgery, which measured `70–77 s` and `255` HTTP requests/band).
  - Bind Zarr at **`RasterSource`** (not forced through single-file sync `ByteSource`): add a narrow sync key-value object-store trait in `veldmap-core::io` backed by `NetContext` in `veldmap-net` (preserving zero horizontal deps between `veldmap-raster` and `veldmap-net`) and generalize remote tile prefetch in `OverlayManager` / `LayerOrigin::Remote` beyond single-file `HttpRangeSource` + `Vec<ByteRange>`.
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
- [ ] **Evaluate and select HDF5/NetCDF-4 reader:** Run an early spike against real ESA Sentinel-3/5P NetCDF-4 files auditing pure-Rust candidates (`hdf5-pure` / `rustyhdf5` / `rustyhdf5-netcdf4` / `netcdf-reader`) for `Send + Sync`, superblock v2/v3, extensible/fixed-array chunk indices, Shuffle+Deflate+Fletcher32, `f64` coordinates/scaling, raw typed integer reads, and `deny.toml` `all-features = true` compliance. Keep an optional C `libhdf5` fallback strictly as a last resort (requires NCSA license clarification, CMake build support, and scoped CI feature flags).
- [ ] **Implement NetCDF-4 adapter (`veldmap-raster` / `veldmap-app`):**
  - Resolve every legacy hazard below, add local `.SAFE` sibling geolocation, `StreamKey`-level `TileScheduler` reset, and NetCDF variable switcher UI (with named-variable persistence beyond `band_index: Option<u16>`).
  - Upgrade `TieGrid` (`veldmap-core::geo::binding::tie_grid`) with spatial indexing, unit-sphere/ECEF polar interpolation, and **masked/`NaN`-edge cell tolerance** (eliminating `O(M×N)` UI-thread scans in `CursorProber`, pole-crossing longitude jumps, and all-or-nothing rejection of Sentinel-3 SLSTR / Sentinel-5P TROPOMI `NaN` swath edges).
- [ ] **Tests & docs:** Adapt NetCDF tests + malformed-input tests; update `docs/decisions.md`, `docs/limitations.md`, and `README.md`.
- **Exit criteria:**
  - *Automated CI:* Full 3-OS CI passes with NetCDF `f64`/fill-value/malformed-input and `TieGrid` `NaN`-edge/polar tests green.
  - *Manual QA:* Local Sentinel-3/5P NetCDF-4 files open, switch variables, and render cleanly.

#### Slice 3B-2 — Optional 1:1 Native-Pixel / Raw-Sensor 2D Inspector (`Image2DRenderer`) `[ ]`
*(Independent power-user sub-slice for ungeoreferenced / 1:1 native-pixel inspection and linked side-by-side multi-view; can land anytime after `5A-1`.)*
- [ ] Dedicated orthographic **`Image2DRenderer`** in `CallbackResources` sharing immutable `Arc<GpuTile>` textures and the hoisted `TileBindGroupCache` with `GlobeRenderer`:
  - Hoist `TileBindGroupCache::begin_mrt_frame(frame_id)` to once per UI frame / monotonic frame ID so dual viewports never prematurely evict each other's MRT bind groups.
  - Map composite UVs against unclamped vs. clamped `ViewportInPixels` so partially off-screen panels never stretch.
  - Union 2D `wanted` + `carrier_keys` into `SceneState` so 3D frames never evict active 2D tiles, and add synchronized zoom-invariant CPU raw DN & physical-unit cursor readout.
- [ ] **Docs:** Update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* `Image2DRenderer` UV clamping / partial-offscreen tests, dual-viewport `TileBindGroupCache` MRT-epoch isolation tests, `SceneState` unioned `wanted`/`carrier_keys` protection tests, `naga` SPV/HLSL/MSL, and headless GPU smoke tests pass.
  - *Manual QA:* Ungeoreferenced and georeferenced rasters open in 1:1 native-pixel view alongside the 3D globe with zero texture duplication, zero cross-viewport tile eviction, and accurate synchronized DN/physical readouts.

---

## Migrated Legacy Porting Hazard Checklists (Pre-Retirement Commit `9cd029f`)

Audited safety, correctness, and architecture hazards preserved from retired `old/README.md` (consult `git show 9cd029f:old/...` on demand; `9cd029f` is `0f5ff4d~1`, immediately before commit `0f5ff4d` retired `old/`). Do not port legacy code verbatim—resolve every item below when implementing the corresponding v1.1 slice:

- **Slice 4B-1 (`old/code/network/download.rs`, `old/code/library/*`, `old/code/provider/*`):** Drop WASI `queued_sidecars` / async rescan state machines in favor of serialized sync `std::fs` metadata; enforce S3 path sanitization, Windows `\\?\` extended-length paths (`longPathAware`), unsigned `Range` header invariant on 401/403 re-sign, persisted `ETag` `If-Range`, HTTP 416 recovery, `File::try_lock()` + in-process lock map + explicit `unlock()`, `downloaded == total_size`, and credential/SigV4 redaction in `tracing` logs/errors.
- **Slice 3C-1 (`old/code/tiler/adapters/{codec,excerpt,jp2}.rs`):** Audit `openjp2` for `static mut` thread safety/soundness if evaluated; fix Stacked Borrows / drop-order UB (`Box::into_raw` + explicit `Drop`), `usize::MAX` EOF + `ErrorKind::Interrupted` retry, `catch_unwind` on `extern "C"` callbacks, `skip()` EOF bounds check, `COD` `levels <= 32` shift-overflow guard, worker panic isolation, and gating zero-as-nodata on Sentinel-2 heuristics rather than all GMLJP2 (plus PLT varint & TLM checked arithmetic + `keep <= part.len` + `MAX_PAD` cap if local tile-part excerpting is retained).
- **Slice 6 (`old/code/tiler/adapters/netcdf.rs`):** Read coordinates and `scale_factor`/`add_offset` in `f64`; check raw typed `_FillValue`/`missing_value`/`valid_range` *before* unpacking (mapping fill to `f32::NAN`); fail fast on unsupported HDF5 filters at `describe`; bound `Hdf5HandleCache` (`METADATA_CACHE` + `ROW_CACHE_CAP`) with fixed host-RAM-scaled caps.

---

## Next Step

- **Current state:** Milestones **v0.2** (`1`–`3A-2`), **v0.5** (`3B`, `4A`, `5A-1`, `5A-2`), and **v1.0 Pre-5B + Slice 5B-2a** (STAC pagination/profiles/refresh, multi-temporal `Timeline` layer, and dual-epoch GPU scalar diff & RGB cross-fade) are complete (see **Completed Foundation** above and [docs/decisions.md](docs/decisions.md) §1–§14).
- **Immediate task:** Start **Slice 5B-2b** (Off-UI-thread chunk/tile region reader, multi-polygon AOI zonal statistics, 1D pixel & parcel time-series charts via custom `egui::Painter`, and RFC 4180 CSV export).
