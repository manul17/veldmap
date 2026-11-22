# VeldMap Development Plan

## Goal

Build VeldMap as a fast, purely native Rust desktop **Analytical Earth-Observation & Precision-Imagery Workstation** on a **Precision Nadir 3D WGS84 Globe** for:
- **v0.2–v1.0:** Interactive WGS84 planetary viewing (bundled + live basemaps and local GeoTIFF/BigTIFF/COG/PNG/JPEG overlays down to centimetre-scale orthophotos), zero-auth STAC/COG HTTP Range streaming, GPU scientific/categorical tracks (`R32Float`/`R8Uint`/`R16Uint`) with interactive band math (NDVI/NDWI/false-color + `SCL` cloud masking), analytical terrain relief (hillshade/slope/aspect + COP-DEM GLO-30), multi-temporal timeline diffing, and 1D pixel time-series inspection.
- **v1.1:** Mission-specific codecs and authenticated archive workflows—Sentinel-2 JPEG 2000, Copernicus Data Space (CDSE) S3/SigV4 streaming & resumable downloads, and Sentinel-3/5P NetCDF-4/HDF5.

A simplified browser/mobile companion is intentionally deferred to a separate project later (enabled by keeping `veldmap-core` permanently UI/GPU/network-free).

> **Document lifecycle:** `plan.md` is the vertical-slice roadmap and exit-criteria tracker (update `Next Step` as each slice closes). Future slices (`2A`–`6`) are **working design hypotheses**—revise them freely as real code and benchmarks prove better designs. Settled "why not X" decisions and hard cross-slice invariants live in [docs/decisions.md](docs/decisions.md) once implemented; module details live in Rust `//!` docstrings (`cargo doc`).

---

## Working Cross-Slice Principles

Implemented decisions and hard cross-slice invariants are maintained in [docs/decisions.md](docs/decisions.md) (UI/GPU-free `veldmap-core`, safe positioned I/O without `memmap2`, smooth v1 WGS84 ellipsoid, `MIN_HEIGHT_M = 2.0` m, Reverse-Z `f64`, exact sRGB compositing, and the 5 overlay accuracy benchmarks). Working principles for upcoming slices:

1. **Natural workspace crate boundaries:** Keep `veldmap-core` a permanently pure leaf (`geo`, `tile` keys/LRU, std-only `io` traits/`CancelToken`/`MemoryBudget`, zero `tokio`/`reqwest`/`image`/`tiff`/`egui`/`wgpu`). Introduce focused workspace crates (`veldmap-raster`, `veldmap-net`, `veldmap-storage`) as their dependencies actually land so Cargo enforces dependency direction and compiles crates in parallel—avoiding feature-gated mega-crates.
2. **Deadlock-free memory & deduplication:** Startup host-RAM-scaled **fixed per-cache caps** (clamped `[min, max]`, preserving deterministic `Pointwise` vs. `Pass` classification and avoiding `Shrinkable` lock-reentrancy) paired with a lightweight **two-lane decode budget** (dedicated 1-slot large-pass lane + bounded pointwise-tile lane with `CancelToken`-aware wakeups, preventing both FIFO head-of-line blocking and large-pass starvation); decode jobs reserve upfront private working-set `Peak` without double-counting shared `DecodedChunkCache`; singleflight is waiter-refcounted under its map mutex with a leader `Drop` guard (preventing follower deadlock on panic/cancel).
3. **Codec safety & two-tier durability:** Decoders enforce explicit memory/IFD/padding/shift/varint limits, panic isolation, and thread-safety audits before multi-threaded decode. Injected `StoragePaths` with two-tier atomic writes (`write_atomic_durable()` with `sync_all()` + synchronous handle drop before rename on Windows vs. `write_atomic_cached()` without `sync_all()` + read validation; both using per-boot `.<boot_id>.<seq>.tmp` files + foreign-UUID startup sweep). Add traits only where there is a real second implementation (`ByteSource`, `Signer`, raster decoders, tile providers).
4. **`GpuTile` hygiene:** Owns only `Texture`, `TextureView`, `PixelFormat`, dimensions, true byte size (including mips), and monotonic `tile_id: u64`—never pipeline-specific `BindGroup`s or `view_formats`.
5. **Treat `old/` as optional reference, retire cleanly:** Pre-deletion state is tagged `legacy-reference-snapshot`; remaining files are tracked in [old/README.md](old/README.md) with `Consumer Slices` and `Delete After Slice` (always the *last* consumer slice). Design native modules cleanly rather than porting WASI-era decomposition verbatim. Never link from `docs/` into `old/`; delete `old/` completely at Slice 6 (v1.1).
6. **Code hygiene (every slice):** English only outside `old/` (CI-enforced; remove `old/` exclusion at Slice 6). Rust 2024; verified by `cargo fmt --all -- --check`, `cargo clippy --locked --workspace --all-targets -- -D warnings`, `cargo test --locked --workspace`, `cargo-deny`, `naga` SPV/HLSL/MSL codegen, headless offscreen `wgpu` smoke tests, forbidden `docs/ -> old/` link guard, and offline `lychee` link checks.

---

## Completed Foundation `[x]`

- **Phase 0 & Phase 0.5 — Native Bootstrap, Hygiene & CI `[x]`:** Initialized clean native `veldmap-core` + `veldmap-app` (`eframe 0.37` / `wgpu 30` / Rust `1.95`), created `legacy-reference-snapshot`, removed unlicensed font binaries, configured workspace lints (`unwrap_used`/`expect_used` warned in prod, `unsafe_op_in_unsafe_fn` denied) and Linux hygiene CI (`fmt`, `clippy --locked -D warnings`, `cargo-deny`, no-Cyrillic, offline `lychee`).
- **Slice 1 — Core Geodesy & Interactive 3D WGS84 Globe `[x]` ([Design Decisions](docs/decisions.md)):** Implemented `veldmap-core::geo` (WGS84 geodesy, Krueger projections/Helmert datums, MGRS/DD/DMS parsing, `f64` infinite Reverse-Z nadir `Camera` with `MIN_HEIGHT_M = 2.0` m, oblate `f64` `cull::Frame`, footprints/outlines), decoupled offscreen `wgpu` renderer (`Rgba8UnormSrgb` + `Depth32Float` Reverse-Z, 1:1 nearest sRGB/UNORM `view_formats` compositing, depth-less draped graticule with ellipsoidal horizon discard, `Backends::PRIMARY` minus `BROWSER_WEBGPU`), baseline GIS HUD, 3-OS CI with `naga` SPV/HLSL/MSL codegen and headless GPU smoke tests, and retired 20 legacy geo/MGRS files from `old/`.

---

## Release Roadmap & Vertical Slices

> **Execution order:** Slices are grouped into four shippable milestones (`v0.2` → `v0.5` → `v1.0` → `v1.1`). Slice IDs (`2A`–`6`) match [old/README.md](old/README.md) retirement tracking verbatim.

---

### Milestone v0.2 — Planetary Basemaps & Local Precision Workstation

#### Slice 2A — Offline Planetary Quadtree, `TileKey`/`GpuTileCache` & Bundled Basemap `[x]`
- [x] **Core geodesy, camera & dead-module cleanup:**
  - Unify `Ellipsoid` + `f64` ECEF metres + Fukushima (2006) 3D inverse across `geodesy.rs` and `projection.rs` (keeping 2D `Projection::to_geodetic` discarding phantom Helmert `Δh`).
  - Update `cull::Frame` horizon constant to `a² / len`, add the Reverse-Z near plane (`forward = row3.truncate().normalize()`, `near = row2.w / row3.truncate().length()`), and flip `Camera::basis()` to geodetic nadir aim (`forward = -self.at`, eliminating the `~3.6 px` HUD offset and `14.7 km` orbit-to-2 m center-zoom drift).
  - Delete obsolete `outlines.rs` (+ dead `geodesy::offset`/`geodesy::encircles`), retain `sag.rs` for 2A and `footprint.rs` (+ its `geodesy.rs` ring-tracing helpers) for 3A-2/4A, and retire `geodesy::parts`, `World = [f32; 3]`, and `neg_eye_{hi,lo}` once per-tile RTC lands.
- [x] **Scheme-matched WGS84 quadtree, adaptive polar cap & post-overlay graticule:**
  - Implement a `z/x/y` quadtree over `TileScheme` (`WebMercator` 1×1 | `Geodetic` 2×1 | `RasterPyramid { max_level: u8 }`) with **decoupled geometry vs. texture LOD** (`z_geom` subdividing past basemap `max_level` up to **`z_geom = 16–17`** to avoid `f32` UV stepping while keeping sub-millimetre sag at `2 m`, sampling ancestor `z_tex <= max_level` via sub-rect UVs with `0` requests above `max_level`), per-tile CPU `f64` RTC vertex buffers + shared static index buffer, HiDPI-aware SSE subdivision, and per-cell sag-scaled skirts (`max(4.0 * sag_cell_diag, 0.001) m` with 2:1 LOD margin).
  - Add an **adaptive Polar Stereographic (azimuthal) square quadtree cap** over `±85.051°..±90°` driven by the same HiDPI SSE + `sag` criterion (zero pole slivers, `< 1 cm` sag at `2 m`, skirt-stitched at `±85.051°`).
  - Delete `mesh.rs`'s static `96×192` ellipsoid body (rendering `"Ellipsoid-only"` untextured through the quadtree) and replace the static 672-segment graticule with a procedural fragment-shader `fwidth()` graticule in a lightweight transparent **post-overlay `grid_pipeline`** reusing the **same quadtree vertex/index buffers** (so 3A-1b overlays never bury it), using CPU `f64` tile-relative `(dlat, dlon)` + `f64`-wrapped phase offset uniforms, `fwidth()` unconditionally *before* horizon `discard` (`naga`), and `smoothstep` polar meridian fade.
- [x] **Unified `TileKey`, `TileCache`, `GpuTileCache` & renderer prep:**
  - Define `Copy + Eq + Hash` `TileKey` (`SourceId`, `BandKey`, `TileScheme`, `PixelFormat`, `DecodeRev`, `Level`, `X`, `Y`) with deterministic `Copy` `BandKey` (`Display | Index(u16) | NamedHash(u64)`) and pure O(1) LRU `TileCache<K, V>` in `veldmap-core::tile` (delegating ancestor-touch to **`TileScheme::parent(level, x, y)`** so `z = 0` coarsest XYZ/Geodetic and `level = 0` finest `RasterPyramid` never invert, plus `<50%` byte appetite cap per pyramid).
  - Own `GpuTileCache` and `PendingUploads` directly on `VeldmapApp` (cloning `Arc`-backed `device`/`queue`, passing immutable `Arc<GpuTile>` draw snapshots to callbacks + lightweight `HashMap<u64, (Weak<GpuTile>, wgpu::BindGroup)>` swept automatically on `prepare()`). Cache quadtree traversal & draw snapshots by `(camera_key, scene_generation)` so 120 Hz hover never rebuilds tile lists, and group per-viewport state (`camera_buffer`, `offscreen`, `last_render`, `pending_present`) into a `ViewportState` wrapper to ease 3B multi-viewport without recreating `camera_buffer` on resize. Drain the full completion channel each frame, prune stale entries, upload coarsest-first (`4–8 MiB` / `~2 ms` cap with `>= 1` tile/frame anti-starvation), request repaint while non-empty, and wake the UI via injected `Wakeup`.
- [x] **Bundled offline basemap & `veldmap-raster` seed:** Bundle a public-domain low-res global equirectangular basemap (NASA Blue Marble Next Gen or Natural Earth, ~1–3 MiB; document provenance in `THIRD_PARTY_NOTICES.md` and `assets/README.md`, mark `*.png`/`*.jpg binary` in `.gitattributes`), decode raw PNG/JPEG map tiles in a new `veldmap-raster` crate (keeping `veldmap-core` leaf dependency-free), serve the bundled basemap as a virtual offline `z = 0..3` provider (`z = 0` PolarCap, compact `Rgb8` linear-light mip pyramid) through `PendingUploads` → `GpuTileCache`, and wire the **Satellite / Ellipsoid-only** switcher.
- [x] **Docs:** Update [docs/decisions.md](docs/decisions.md) (settled tradeoffs only) and [docs/limitations.md](docs/limitations.md) as static mesh/graticule and geocentric-aim limits retire.
- **Exit criteria:**
  - *Automated CI:* `TileCache<K, V>` (`TileScheme::parent` for `WebMercator`/`Geodetic` & `RasterPyramid`, synthetic byte-cap eviction, `<50%` appetite cap), quadtree/adaptive polar-cap tests, and headless GPU tile-upload + **close-up (`2 m`) between-vertices accuracy test** (midpoint sag, LOD seam, coarse parent texture overzoom, and polar cap) + `0` idle re-render smoke tests pass.
  - *Manual QA:* Offline planet zooms smoothly from orbit to 2 m (including polar regions) with zero T-junctions, polar seams, or close-zoom graticule parallax, and VRAM stays strictly within budget.

#### Slice 2B — Async Tile Streaming, Disk Cache & Live Satellite Basemaps `[ ]`
- [ ] **Workspace crates (`veldmap-net`, `veldmap-storage`), `CancelToken` & decode pool:**
  - Implement zero-dep async-waking `CancelToken` (`AtomicBool` + `std::task::Waker` / listener hook so `tokio::select!` aborts immediately on pan/shutdown) in `veldmap-core`.
  - Introduce `veldmap-net` (`tokio` + `reqwest` over `rustls`): tile fetcher with per-host concurrency limits, compliant `User-Agent`, replaceable visible-first queue, and immediate cancellation.
  - Introduce the bounded `std::thread` CPU decode pool (extended with the two-lane decode budget in `3A-1a`) so live PNG/JPEG tiles never decode on the UI thread or starve `tokio`, with deterministic shutdown join order (`CancelToken` → drop completion rx / join decode pools → drop `tokio::Runtime`).
- [ ] **Disk cache & bucket/tile LRU eviction (`veldmap-storage`):** Implement the native tile cache (referencing `old/code/cache/` & `evict_tests.rs`), upgraded from legacy whole-source eviction to **per-tile or per-`(source, z)` bucket LRU eviction with sharded subdirectories** (required so single-source live basemaps neither grow unboundedly nor wipe at once; benchmark `redb` or `rusqlite` only if small-file counts actually bottleneck Windows NTFS, accounting for `redb` single-instance exclusive locks and `compact()` stalls). Implement two-tier atomic writes (`write_atomic_cached` + read validation vs. `write_atomic_durable` + `sync_all()` + synchronous handle drop before rename; `.<boot_id>.<seq>.tmp` + Windows sharing-violation handling + foreign-UUID startup sweep).
- [ ] **Legally clean live basemaps:** Wire NASA GIBS (static layer or explicit `{Time}`, `max_level` ~8–9) + EOX Sentinel-2 Cloudless (CC BY 4.0, `max_level` ~13–14) with visible attribution, per-source `max_level` clamping + ancestor overzooming (`0` requests above `max_level`), Custom XYZ/TMS URL & API-key settings, `Rgba8UnormSrgb` tile mipmaps + `MipmapFilterMode::Linear` (eliminating limb minification shimmer on sharp live tiles), active-frame `TileCache` protection + `remove_where(source)` / shared `DecodedTile` in `veldmap-core::tile`, and lazy-loaded offline fallback to Slice 2A's bundled basemap.
- [ ] **3-OS Clippy, docs & `old/` cleanup:** Expand `cargo clippy --locked --workspace --all-targets -- -D warnings` to Windows/macOS in CI now that platform `#[cfg]` code lands; update `docs/decisions.md` and `docs/limitations.md`; delete `old/code/cache/` and `old/tests/cache/evict_tests.rs`.
- **Exit criteria:**
  - *Automated CI:* Eviction (disk bytes `<= cap`), atomic-write/temp-sweep, `CancelToken`, `0` requests above `max_level`, and 3-OS Clippy pass.
  - *Manual QA:* Live NASA GIBS / EOX tiles stream with visible attribution, bounded disk cache, and clean offline fallback.

#### Slice 3A-1 — Local GeoTIFF/PNG/JPEG Engine, `MemoryBudget` & Warp-Mesh Draping `[ ]`

##### Slice 3A-1a — Headless Raster Engine, `ByteSource`, Two-Lane `MemoryBudget` & `TileScheduler`
- [ ] **Sync `ByteSource` & `ContentId` (`veldmap-core` / `veldmap-storage`):**
  - Strictly sync caller-buffered `ByteSource::read_at(&self, offset: u64, buf: &mut [u8], cancel: &CancelToken) -> Result<usize, IoError>` with safe positioned-I/O local file impl (no `memmap2`; buffered 64–256 KiB cursor; pooled `File` handles + short-read loop on Windows).
  - Key disk and GPU tiles by sampled BLAKE3 `ContentId` (`len || head_64k || mid_64k || tail_64k`, domain-separated) plus band, pixel format, decode revision, and gutter width so incompatible tiles never alias; keep `(path, len, mtime_ns)` strictly as an in-memory stat cache + UI "Reload / bypass cache".
- [ ] **Local raster engine (`veldmap-raster`):**
  - Implement a unified `BandInfo` model (band metadata, scale/offset, nodata, pixel format) and pyramid/resampling/radiometry/grid/cascade (referencing `old/code/tiler/`, parameterizing pixel format and gutter width with default `0` so `3B`/`5B-1` extend without rewriting `3A-1a`), core `tiff` (standard RGB/Gray GeoTIFF/BigTIFF/COG, `GeoDoubleParamsTag`/`GeoAsciiParamsTag`, explicit `Limits` + `MAX_IFDS`, verifying Float32 `Predictor = 3`, benchmarking `ruzstd` vs. `zstd` on real ZSTD COGs before committing), and georeferenced full-image local PNG/JPEG adapters (stripping `!Send` `Rc`/`RefCell`).
  - Wire 8-bit `Rgba8UnormSrgb` + QOI disk cache (`write_atomic_cached` + header/length validation).
- [ ] **Two-lane `MemoryBudget`, singleflight & `TileScheduler`:**
  - Wire startup host-RAM-scaled **fixed per-cache caps** (clamped `[min, max]`, no `Shrinkable` lock-reentrancy; evaluate `quick_cache` for multi-threaded CPU caches) + lightweight **two-lane decode budget** (dedicated 1-slot large-pass lane + bounded pointwise lane, `CancelToken`-aware wakeups, upfront `Peak` OOM check fixing legacy `direct_peak()` so pointwise workers never double-count shared `DecodedChunkCache`) and waiter-refcounted singleflight decoded-chunk LRU with panic `Drop` guard into the bounded local CPU decode pool.
  - Implement native `TileScheduler` (`veldmap-core::tile`): 1 active pass per source for `Serve::Pass` *only* (concurrent `Serve::Pointwise` allowed), non-cancellation of coarser passes, and terminal-failure suppression.
  - Verify TIFF read amplification via `CountingByteSource` (referencing `old/tests/tiler/reads_tests.rs`) plus two-lane `MemoryBudget`/singleflight/`TileScheduler` concurrency stress tests (verifying zero HoL blocking and zero large-pass starvation).

##### Slice 3A-1b — GPU Warp-Mesh Draping, Minimal Local Opener & Accuracy Regression
- [ ] **Depth-less warp-mesh draping & layer model:** Implement top-down lazy `DVec3` culling balls (preserving per-rung limb-culling margins) in `veldmap-core`, and multi-band-ready layer state + per-tile CPU `f64` RTC warp meshes (using `16` segments at coarse UTM zone edges where needed for `< 2 cm`) rendered **depth-less (`depth_write: false`, `depth_compare: Always`) with fragment ellipsoidal-normal horizon discard** in painter's order back-to-front in `veldmap-app` (referencing `old/code/geo/overlay.rs`).
- [ ] **Minimal local opener, HUD/readout separation & accuracy regression:** Wire CLI file argument + drag-and-drop opener in `veldmap-app`; label HUD **viewport GSD** separately from **source pixel size** and keep scientific cursor readouts querying native CPU tiles (zoom-invariant); rebuild `a_raster_lands_where_its_binding_puts_it` (CPU exact geometry + UV placement) verifying all 5 UTM 37 benchmark bindings & drift tolerances in [docs/decisions.md](docs/decisions.md) §4 (`< 10⁻³ px`, `< 2 mm` native, `< 2 cm` pyramid) plus a **second CRS fixture** and an end-to-end **`PixelIsArea` vs. `PixelIsPoint` (`GTRasterTypeGeoKey`)** half-pixel check.
- [ ] **Docs & `old/` cleanup:** Update `docs/decisions.md` and `docs/limitations.md`; delete `old/code/geo/overlay.rs`, `old/code/tiler/{budget,fingerprint,tiles_pipeline,cascade,pyramid,resample}.rs`, `old/code/tiler/adapters/{full,radiometry,table,grid,tiff,png,jpeg}.rs`, `old/tests/tiler/{pyramid,radiometry,resample,tiff}_tests.rs`, `old/docs/viewing-pipeline.md`, `old/docs/decisions/0003-raster-reading-model.md`, and `old/docs/decisions/0007-overlay-accuracy.md` (**retain** `reads_tests.rs` until Slice 6).
- **Exit criteria:**
  - *Automated CI:* `a_raster_lands_where_its_binding_puts_it` (all 5 UTM 37 benchmarks + 2nd CRS + `PixelIsArea`/`PixelIsPoint`), COG `CountingByteSource` tests (touching only IFD + intersected chunks), two-lane `MemoryBudget` stress (`peak <= cap`, 0 deadlocks, 0 HoL stalls), `TileScheduler`, and `0` idle GPU re-render checks pass.
  - *Manual QA:* Local GeoTIFF/BigTIFF/COG and cm-scale drone orthophotos open via CLI/drag-and-drop and drape accurately with responsive zooming and bounded RAM/VRAM/disk.

#### Slice 3A-2 — Local Workstation UX, Expanded Pure-Rust CRS & GeoJSON Overlay `[ ]`
- [ ] **`app.rs` UI/state decomposition:** Before piling on workstation panels, split `veldmap-app/src/app.rs` into a thin frame/input shell, separated workspace/layer state, and focused UI/tool submodules (`toolbar`, `layers`, `status_bar`, `crs_picker`, `measure`).
- [ ] **Expanded pure-Rust CRS & world files:** Keep `veldmap-core::geo::projection` in-house (deferring `proj4rs` to a future wrapped long-tail fallback only if arbitrary WKT2/EPSG rasters demand it); extend with `ETRS89/UTM` (`258xx`), `NAD83/UTM` (`269xx`), `LAEA Europe` (`3035`), Polar Stereographic (`3413`/`3031`), `GeoKey 3072 == 32767` user-defined TM parameters (**including `ProjNatOriginLatGeoKey (3081)` `origin_lat_deg` and `ProjLinearUnitsGeoKey (3076)` International / US Survey Feet**); wire `Datum::to_wgs84` into geographic bindings (discarding phantom Helmert `Δh` on 2D output); read `.tfw`/`.jgw`/`.pgw` world files externally (injected as `Option<Placement>` so decoders stay pure `ByteSource`); show visible UI warnings for unshifted foreign datums.
- [ ] **Local workstation UX & diagnostics:** Native file picker, Layer Manager (visibility, reorder, opacity, zoom-to-binding, CRS/radiometry metadata inspector), searchable CRS picker with presets & custom TM parameters (unlocking all MSK-xx zones), and diagnostic banners (fatal vs. partial error, detail-limit caption, loading shimmer).
- [ ] **Geodesic tools & GeoJSON overlay:** WGS84 geodesic distance/area measurement tools (via pure-Rust MIT **`geographiclib-rs`** Karney `Geodesic` / `PolygonArea`) with **RFC 7946 GeoJSON export** (`[lon, lat]` WGS84 + right-hand winding), plus **read-only GeoJSON overlay with explicit `Polygon { exterior, holes }`** (upgrading `footprint.rs` union semantics + 3D dihedral `turn()`, stroke-only or local tangent-plane earcut with interior edge subdivision back onto WGS84 so large polygons never sag below the ellipsoid).
- [ ] **Docs:** Update `docs/decisions.md` and `docs/limitations.md` (**retain** `old/docs/georeference.md` until Slice 6).
- **Exit criteria:**
  - *Automated CI:* Expanded CRS/datum/feet/`origin_lat_deg` round-trip tests, Karney geodesic tests, and GeoJSON parse/holes/subdivided-fill/RFC 7946 export round-trip tests pass.
  - *Manual QA:* Local projected/world-file orthophotos and concave/holed GeoJSON field polygons open cleanly via picker/drag-and-drop or CRS override, and measured features export cleanly to GeoJSON.

---

### Milestone v0.5 — Scientific Imagery, Zero-Auth Cloud COGs & Band Math

#### Slice 3B — Scientific `R32Float` / `R8Uint` / `R16Uint` Tracks, Extended TIFFs, Live Stretch & 2D Inspector `[ ]`
- [ ] **Multi-viewport renderer prep:** Refactor singleton `GlobeRenderer` in `CallbackResources` into shared GPU tile resources + `ViewportId`-keyed per-viewport offscreen targets and camera uniform buffers (eliminating single-`camera_buffer` overwrite hazards for the 2D Inspector and future dual-epoch diffing).
- [ ] **Multi-format GPU tracks (`Display`, `Scientific`, `Categorical`) & extended GeoTIFFs:**
  - Keep Display (`Rgba8UnormSrgb`, `512×512`) and add:
    - **Scientific (`R32Float`, `512×512`, `gutter = 0`, 1 mip, `1.00 MiB`/tile for optical/SAR):** stores **raw numeric samples** (nodata/fill → `f32::NAN`; document `f32` 24-bit precision limits for 32-bit int/`f64` readouts) and applies canonical affine **`val * scale + offset`** (`fma`, matching STAC `raster:bands`, GDAL, and CF NetCDF) in WGSL uniforms + cursor readout to avoid cache aliasing.
    - **Categorical (`R8Uint` & `R16Uint`, `512×512`, `gutter = 0`, 1 mip):** discrete class and QA bitmask bands (raw `Palette` classes, Sentinel-2 `SCL`, Landsat `QA_PIXEL`, MODIS QA — `R8Uint` and `R16Uint` share the same `texture_2d<u32>` WGSL binding) with **nearest-neighbor sampling & non-averaging downsampling** (`1×` COG chunk alignment, `256 KiB` / `512 KiB` per tile).
  - Add **`YCbCr`**, **`Palette`** (`ColorMap` expansion on Display; raw class indices on `Categorical`), and **`PlanarConfiguration = 2`** to `veldmap-raster`'s GeoTIFF decoder so standard v1.0 COGs/land-cover maps work out of the box.
  - Keep optical/SAR `R32Float` **unguttered (`gutter = 0`, 1 mip; 2 mips max only if QA sees minification shimmer)** for exact `1×` COG chunk alignment (avoiding `9×` chunk straddling and `Serve::Pass` downgrades); support parameterized **`gutter = 1`, 1 mip** (`514×514`, `1.008 MiB`) strictly for Slice 5B-1 DEM relief (with host-RAM-scaled `CHUNK_CACHE` and `gutter = 0` fallback on oversized chunks) and normalize `cell_uv` against `valid_w`/`valid_h`.
  - Add compressed disk caching for `R32Float`/`R8Uint`/`R16Uint` tiles (e.g. header + byte-shuffled `lz4_flex`; QOI stays 8-bit Display only).
- [ ] **WGSL nodata-aware filtering, stretch & 2D Inspector:**
  - Manual 4-tap `textureLoad` bilinear filter masking both `NaN` and `±Inf` via `(bitcast<u32>(v) & 0x7FFFFFFFu) >= 0x7F800000u` (computing any `dpdx`/`dpdy` derivatives—e.g., if 2-mip LOD is enabled—in uniform control flow **before any horizon `discard`** so `naga` validation stays satisfied).
  - Live contrast stretch (linear percentile/min-max + **`log10`/dB stretch** for SAR), colormap uniforms, multi-band GeoTIFF band selector, interactive histogram, and colorbar with physical units.
  - 2D Inspector split pane (referencing `old/code/viewer/`) sharing `Arc<GpuTile>` with the 3D globe + synchronized zoom-invariant CPU cursor raw DN & physical-unit readout.
- [ ] **GPU smoke test, docs & `old/` cleanup:** Add headless `wgpu` smoke test verifying NaN/Inf bitmask bilinear math and gutter continuity; update `docs/decisions.md` and `docs/limitations.md`; delete `old/code/viewer/`.
- **Exit criteria:**
  - *Automated CI:* Headless `wgpu` nodata-bilinear + gutter continuity test, `naga` SPV/HLSL/MSL codegen, `R8Uint`/`R16Uint`/`R32Float` byte accounting, and `YCbCr`/`Palette`/`Planar-2` TIFF tests pass.
  - *Manual QA:* 16-bit/float32/SAR/Palette/QA/multi-band rasters render with exact nodata transparency, live histogram/colorbar, and synchronized 2D/3D cursor readout.

#### Slice 4A — Zero-Auth Cloud EO Streaming (HTTP Range + STAC / Public COGs) `[ ]`
- [ ] **HTTP Range `ByteSource` (`veldmap-net`):** Implement range reader (referencing `old/code/network/range.rs`) over `reqwest` (`rustls`)/`tokio`: 64 KiB O(1) LRU `HttpBlockPool` (startup host-RAM-scaled fixed cap), combined open-probe + `0..64 KiB` head read + concurrent `mid_64k`/`tail_64k` `ContentId` reads (`416`-safe on 0-byte files), consumed-only readahead, waiter-refcounted multi-block singleflight, **auto-decompression disabled**, reject `200` when expecting `206`, `Content-Range` + **strong-ETag-only `If-Match`** (RFC 9110: never send weak `W/"..."` in `If-Match`, isolate `HttpBlockPool` entries without strong ETags, treat `412` as non-retryable), `CancelToken` async abort, and `Handle::try_current().is_err()` guard on blocking reads.
- [ ] **Single-batch pinned remote COG decode:** Sync chunk range-planners in `veldmap-raster` + `HttpRange::prefetch_pinned()` in `veldmap-net` producing memory-only `PinnedByteSource` batches (under upfront `MemoryPermit`) routed to the fast CPU decode pool—keeping 4A focused cleanly on single-batch COG prefetch without unused multi-round JP2 probe hooks, while isolating un-prefetched blocking reads to a separate capped remote pool so slow HTTP never starves local SSD decodes.
- [ ] **Zero-auth STAC & public COG provider (`veldmap-net`):** STAC API search (e.g. AWS Earth Search Sentinel-2 L2A COGs + COP-DEM GLO-30) and direct public HTTP COG URLs, multi-band Sentinel-2 asset discovery + STAC `raster:bands` `scale`/`offset` metadata, AOI search (viewport / drawn box / GeoJSON polygon + GeoJSON/WKT export), scene outlines on globe (HiDPI-scaled outline/ribbon/hatch pipelines with per-cluster RTC in `veldmap-app`), and built-in zero-auth sample scenes for 15-second onboarding.
- [ ] **Mock HTTP harness & docs:** Offline local mock HTTP Range harness (`206`, `200` rejection, `416`, strong/weak `ETag`, `412`, N-way singleflight deduplication, request & delivered-byte ceilings); update `docs/decisions.md` and `docs/building.md`.
- **Exit criteria:**
  - *Automated CI:* Offline mock HTTP Range/STAC/singleflight tests (`0` duplicate fetches under N-way contention, bounded request/byte ceilings) pass.
  - *Manual QA:* Search an AOI and stream public Sentinel-2 L2A COGs and HTTP COG URLs with zero credentials without starving local SSD decodes.

#### Slice 5A — Interactive Band Math, Split-Screen Swipe & Session Persistence `[ ]`
- [ ] **Unified screen-space float band math & radiometric correctness:**
  - Render each active band with the existing single-band `R32Float` / `R8Uint` / `R16Uint` warp-mesh pipeline into a **lazy, shared single-viewport offscreen scratch pool**: **`R32Float`** per required band (avoiding `Rgba16Float` quantization of Sentinel-2 DN and 300 K thermal; cleared to `0.0` with **`blend: None`** — never `BlendState::REPLACE`, which fails `wgpu` validation on unblendable float/uint formats, and never non-portable `LoadOp::Clear(NaN)`) paired with a dedicated **`R8Uint` validity/mask target** (from nearest-neighbor `SCL`/QA masks, `blend: None`, `discard` on nodata).
  - Run a fullscreen WGSL band-math (NDVI, NDWI, false-color, custom stretch) + mask + colormap pass—unifying 5A with 5B-2 to handle Sentinel-2 60 m (`6:1`), Landsat 15 m/30 m (`7.5 m` half-pixel shift), independent per-band ancestor fallbacks, and multi-sensor/rotated overlays without per-tile `[tile_id; N]` shader combinatorics (cursor readouts continue querying native CPU tiles).
- [ ] **Metadata-driven scaling & `SCL` masking:** Apply Sentinel-2 `scale`/`offset` from STAC `raster:bands` and local GeoTIFF metadata in canonical `val * scale + offset` physical reflectance space (never hardcoded to a 2021↔2022 date cutoff; ESA `.SAFE` `MTD_MSIL2A.xml` DN-space `BOA_ADD_OFFSET` conversion lands with `4B`), badge L1C (TOA) vs. L2A (BOA), and support optional **nearest-neighbor `R8Uint` `SCL` cloud/shadow masking** (`classes 0, 1, 3, 8, 9, 10`).
- [ ] **Split-screen swipe curtain & session persistence:** Interactive swipe comparison across any two layers (including non-co-registered swaths); workspace/session persistence via `write_atomic_durable()` JSON (layers, CRS overrides, band math recipes, stretch params, camera bookmarks, split-pane layout), copy-coordinates, and viewport PNG screenshot export (with optional scale bar, north arrow, colorbar, and acquisition date; heavy GeoTIFF raster re-export deferred to post-v1, while GeoJSON measurements land in `3A-2` and CSV time-series lands in `5B-2`).
- [ ] **Docs:** Update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* Metadata-driven `val * scale + offset`, screen-space `R32Float` + nearest-neighbor `R8Uint` `SCL` mask, session JSON round-trip, and `naga` SPV/HLSL/MSL shader tests pass.
  - *Manual QA:* Multi-band Sentinel-2 NDVI/NDWI/false-color (including mixed 10 m/20 m/60 m and multi-sensor) with cloud masking, swipe curtain, screenshot export, and session reload work smoothly.

---

### Milestone v1.0 — Analytical Terrain Relief & Multi-Temporal EO Workstation

#### Slice 5B-1 — Analytical Terrain Relief Layers `[ ]`
- [ ] **GPU hillshade, slope & aspect:** Compute derivatives over **`gutter = 1`, 1-mip (`514×514`) `R32Float`** DEM tiles (falling back to `gutter = 0` rather than `Serve::Pass` on oversized chunks) using per-vertex **2×2 inverse Jacobian `J^(-T)`** into local ENU metres (handling EPSG:4326 `cos(lat)`, COP-DEM latitude bands, and UTM/Polar rotation), scaled to the carrier tile's pyramid level, with one-sided fallback on `NaN` boundaries and across `1°×1°` COP-DEM file borders (or neighbor-file border stitching).
- [ ] **Relief compositing & COP-DEM GLO-30:** Add **Multiply** blend mode for relief-over-imagery compositing, label elevation readouts explicitly in **`m EGM2008`**, and wire out-of-the-box **Copernicus DEM GLO-30 COG** streaming. Update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* Synthetic analytic DEM slope/aspect error bounds across latitudes (`0°`/`60°`/`80°`) and rotated projections + `naga` SPV/HLSL/MSL + headless GPU smoke tests pass.
  - *Manual QA:* COP-DEM GLO-30 hillshade/slope/aspect renders cleanly over optical imagery with zero intra-file tile-edge creases.

#### Slice 5B-2 — Multi-Temporal Timeline, Dual-Epoch Diffing & 1D Time-Series Plot `[ ]`
- [ ] **Multi-temporal timeline scrubber:** Reuse unified `TileKey` for epoch ingestion from **STAC AOI search** (grouped by MGRS tile + AOI coverage, defaulting to L2A) and **local dated folders**, with cloud-cover filtering, adjacent-date prefetch, and instant stale-request cancellation.
- [ ] **Unified offscreen dual-epoch cross-fade & physical reflectance diffing:** Reuse Slice 5A's lazy shared **`R32Float` + `R8Uint` validity/mask** offscreen scratch pool (`blend: None`, clear `0.0`, pre-colormap/stretch, propagating `SCL`/QA validity) for physical reflectance/index diffing (`Date B − Date A`) and display targets for visual cross-fade (`mix`)—cleanly supporting multi-band + `SCL`, mismatched carrier levels, and non-co-registered swaths without complicating per-tile shaders.
- [ ] **1D pixel time-series plot & CSV export:** Lightweight 1D time-series panel (custom `egui::Painter` chart to avoid pinning `egui` upgrades to lagging `egui_plot`, or `egui_plot` if version-aligned) querying **native CPU source tiles** (zoom-invariant) to plot reflectance/NDVI over time at a clicked location (skipping cloud-masked dates), with **RFC 4180 CSV export** (ISO-8601 UTC timestamps, explicit units, documented `SCL`-masked row convention).
- **Exit criteria:**
  - *Automated CI:* Temporal diffing shaders (`naga` SPV/HLSL/MSL + headless GPU smoke), float pre-stretch diff math, zoom-invariant time-series readout, and RFC 4180 CSV export tests pass.
  - *Manual QA:* End-to-end v1.0 EO workflow (local + STAC COGs, band math, swipe/diff, pixel time-series + CSV export, analytical DEM relief) persists cleanly across sessions.

---

### Milestone v1.1 — Mission-Specific Codecs, Authenticated CDSE & Final Legacy Retirement

#### Slice 3C-1 — Sentinel-2 JPEG 2000 `[ ]`
- [ ] **Audit & harden `openjp2` (or evaluate native alternative):** Audit the latest `openjp2` release for `static mut` thread safety and soundness (or evaluate `openjpeg-sys` as an optional feature if `openjp2` fails fuzzing/soundness/thread-safety audit; note `hayro-jpeg2000` lacks single-tile random access and 12/16-bit scientific planes); implement local `.SAFE` JP2 decoding, GMLJP2 parsing (`roxmltree`/`quick-xml`; gate zero-as-nodata on Sentinel-2 heuristics, not all GMLJP2), and tile-part excerpting (referencing `old/code/tiler/adapters/{codec,jp2,excerpt}.rs`):
  - Resolve all known `codec.rs`/`excerpt.rs` hazards: Stacked Borrows/drop-order UB (`Box::into_raw` + explicit `Drop`), `usize::MAX` EOF + `ErrorKind::Interrupted` retry, `catch_unwind` on `extern "C"` callbacks, `skip()` EOF bounds check, `COD` `levels <= 32` shift-overflow guard, PLT varint & TLM checked arithmetic + `keep <= part.len`, `MAX_PAD` cap, worker panic isolation, and 5/3 reversible decode round-trip test.
- [ ] **Tests, docs & `old/` cleanup:** Port/adapt `jp2_tests.rs`, `excerpt_tests.rs`, JP2 sections of `reads_tests.rs`, plus malformed/truncated-input & `proptest` fuzz tests; update `docs/decisions.md` and `docs/limitations.md`; delete `old/code/tiler/adapters/{codec,jp2,excerpt}.rs`, `old/tests/tiler/{jp2,excerpt}_tests.rs`, and `old/docs/decisions/{0002-jpeg2000-decoder,0004-jp2-excerpt}.md`.
- **Exit criteria:**
  - *Automated CI:* 5/3 JP2 round-trip, excerpt, and malformed/truncated-input tests pass with zero panics or UB.
  - *Manual QA:* Local Sentinel-2 JP2 granules open and render cleanly in 2D/3D.

#### Slice 4B — Copernicus Data Space (CDSE) S3/JP2 & Resumable Downloads `[ ]`
- [ ] **CDSE provider, SigV4 re-signing & timeline hook (`veldmap-net`):** Implement CDSE OData/S3/SigV4 (`Signer`) and multi-band Sentinel-2 `.SAFE` asset discovery (`B02`–`B12`, `SCL`, and `MTD_MSIL2A.xml` DN-space `BOA_ADD_OFFSET` normalized to `offset = BOA_ADD_OFFSET / QUANTIFICATION_VALUE`) referencing `old/code/provider/`; add epoch-based singleflight 401/403 re-sign (**unsigned `Range` header invariant**) and plug CDSE AOI search into v1.0's multi-temporal timeline scrubber. Decouple CDSE catalog/COG/resumable download from remote HTTP JP2 packet-surgery—gate **AOI-scoped two-phase remote JP2 streaming** (`PVI` quicklook + `TileScheduler` `coarse_start_override`) behind latency/ROI profiling (legacy remote JP2 excerpting measured `70–77 s` and `255` HTTP requests per band).
- [ ] **Local library & hardened resumable downloads (`veldmap-storage` / `veldmap-net`):** Implement catalog, storage layout, and unified resumable `.part` downloads (referencing `old/code/library/` & `old/code/network/download.rs`, dropping WASI `queued_sidecars` / async rescan state machines in favor of serialized sync `std::fs` metadata): S3 path sanitization, Windows `\\?\` extended-length paths (`longPathAware` manifest), persisted `ETag` `If-Range`, HTTP 416 recovery, `File::try_lock()` + in-process lock map + explicit `unlock()`, `downloaded == total_size`, `write_atomic_durable()`, and 401/403 re-sign.
- [ ] **Credentials & Cache Settings UI:** In-app dialog writing `ProjectDirs/.env` (`0600` on Unix), mandatory credential/SigV4 redaction in `tracing` logs & errors, optional CWD fallback, and anonymous search without keys.
- [ ] **Tests, docs & `old/` cleanup:** Adapt `old/tests/provider/` + prefetch section of `reads_tests.rs` + mock 401/403/`.part` crash-resume & secret-redaction tests; update `docs/decisions.md` and `docs/limitations.md`; delete `old/code/{provider,library,network}/`, `old/tests/provider/`, `old/docs/download.md`, and `old/docs/decisions/0005-network-reading.md` (**retain** `old/docs/imagery.md` until Slice 6).
- **Exit criteria:**
  - *Automated CI:* CDSE provider, mock 401/403 & `.part` resume, secret-redaction log tests (and AOI-scoped JP2 coarse-pass skip if remote JP2 streaming is enabled) pass.
  - *Manual QA:* Browse CDSE S3, download/open Sentinel-2 granules (including via the timeline), and pause/resume downloads reliably.

#### Slice 6 — NetCDF-4 (`hdf5_pure`, Sentinel-3/5P) & Final Legacy Retirement `[ ]`
- [ ] **Audit & harden `hdf5_pure`:** Verify `Send + Sync`, HDF5 Shuffle+Deflate+Fletcher32 support, `read_f64_region`, and raw typed integer reads (`i16`/`u16`/`i32`).
- [ ] **NetCDF-4 adapter (`veldmap-raster`):** Implement NetCDF-4 reading (referencing `old/code/tiler/adapters/netcdf.rs`): read coordinates and `scale_factor`/`add_offset` in **`f64`**, check raw typed `_FillValue`/`missing_value`/`valid_range` *before* unpacking (mapping fill → `f32::NAN`), fail fast on unsupported HDF5 filters at `describe`, bound `Hdf5HandleCache` (`METADATA_CACHE` + `ROW_CACHE_CAP`) with startup host-RAM-scaled fixed LRU caps, support local `.SAFE` sibling geolocation, and add NetCDF variable switcher UI.
- [ ] **Tests, docs & final `old/` deletion:** Adapt `netcdf_tests.rs` + NetCDF section of `reads_tests.rs` + malformed-input tests; update `docs/decisions.md` and `docs/limitations.md`; verify against `old/README.md` that zero unmigrated files remain, delete `old/` completely, remove the `old/` exclusion from CI no-Cyrillic check, and finalize `README.md`.
- **Exit criteria:**
  - *Automated CI:* Full 3-OS CI passes with `old/` removed, zero Cyrillic exclusions, and NetCDF `f64`/fill-value/malformed-input tests green.
  - *Manual QA:* Local and remote Sentinel-3/5P NetCDF-4 files open, switch variables, and render cleanly in 2D/3D.

---

## Next Step

- **Current state:** Phases 0–0.5, Slice 1, and Slice 2A are complete on `main` (see [Design Decisions](docs/decisions.md)).
- **Immediate task:** Start **Slice 2B — Async Tile Streaming, Disk Cache & Live Satellite Basemaps**: zero-dep async-waking `CancelToken` in `veldmap-core`, `veldmap-net` (`tokio` + `reqwest`/`rustls` with per-host concurrency limits and visible-first cancellation), bounded CPU decode pool, `veldmap-storage` per-tile/bucket LRU disk cache with two-tier atomic writes (`write_atomic_cached` / `write_atomic_durable`), live NASA GIBS + EOX Sentinel-2 Cloudless basemaps with attribution & offline fallback, 3-OS Clippy, and `old/code/cache/` retirement.
