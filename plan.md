# VeldMap Development Plan

## Goal

Build VeldMap as a fast, purely native Rust desktop **Analytical Earth-Observation & Precision-Imagery Workstation** on a **Precision Nadir 3D WGS84 Globe** for:
- **v0.2–v1.0:** Interactive WGS84 planetary viewing (bundled + live basemaps and local GeoTIFF/BigTIFF/COG/PNG/JPEG overlays down to centimetre-scale orthophotos), zero-auth STAC/COG HTTP Range streaming, GPU scientific/categorical tracks (`R32Float`/`R8Uint`/`R16Uint`) with interactive band math (NDVI/NDWI/false-color + `SCL` cloud masking), analytical terrain relief (hillshade/slope/aspect + COP-DEM GLO-30), multi-temporal timeline diffing, and 1D pixel time-series inspection.
- **v1.1:** Mission-specific codecs and authenticated archive workflows—Sentinel-2 JPEG 2000, Copernicus Data Space (CDSE) S3/SigV4 streaming & resumable downloads, and Sentinel-3/5P NetCDF-4/HDF5.

A simplified browser/mobile companion is intentionally deferred to a separate project later (enabled by keeping `veldmap-core` permanently UI/GPU/network-free).

> **Document lifecycle:** `plan.md` is the vertical-slice roadmap and exit-criteria tracker (update `Next Step` as each slice closes). Future slices (`2B`–`6`) are **working design hypotheses**—revise them freely as real code and benchmarks prove better designs. Settled "why not X" decisions and hard cross-slice invariants live in [docs/decisions.md](docs/decisions.md) once implemented; module details live in Rust `//!` docstrings (`cargo doc`).

---

## Working Cross-Slice Principles

Implemented decisions and hard cross-slice invariants are maintained in [docs/decisions.md](docs/decisions.md) (UI/GPU-free `veldmap-core`, safe positioned I/O without `memmap2`, smooth v1 WGS84 ellipsoid, `MIN_HEIGHT_M = 2.0` m, Reverse-Z `f64` ECEF metres + per-tile CPU `f64` RTC, geodetic nadir aim, decoupled quadtree/polar caps, post-overlay procedural graticule, `GpuTile` hygiene, exact sRGB compositing, and the 5 overlay accuracy benchmarks). Working principles for upcoming slices:

1. **Natural workspace crate boundaries:** Keep `veldmap-core` a permanently pure leaf (`geo`, `tile` keys/LRU, std-only `io` traits/`CancelToken`/`MemoryBudget`, zero `tokio`/`reqwest`/`image`/`tiff`/`egui`/`wgpu`). Split focused workspace crates (`veldmap-raster`, `veldmap-net`, `veldmap-storage`) along real dependency seams as subsystems land—avoiding feature-gated mega-crates.
2. **Deadlock-free memory & deduplication invariants:** Bound CPU caches with startup host-RAM-scaled fixed per-cache caps (avoiding lock-reentrant cross-cache shrinking), bound decode working memory so small tiled reads never stall behind full-image decodes and large passes never starve, deduplicate concurrent chunk/block fetches with panic/cancel-safe singleflight (zero follower deadlocks), and scale GPU tile VRAM budgets for multi-band EO working sets.
3. **Codec safety & durability invariants:** Decoders enforce explicit dimension/memory/IFD/filter limits, panic isolation, and thread-safety audits before multi-threaded decode. Inject `StoragePaths`; use durable fsync'd atomic writes (closing handles before rename on Windows) for user/session/catalog state and crash-safe validated persistence for ephemeral tile caches. Add traits only where there is a real second implementation (`ByteSource`, `Signer`, raster decoders, tile providers).
4. **Treat `old/` as optional reference, retire cleanly:** Pre-deletion state is tagged `legacy-reference-snapshot`; remaining files and known legacy UB/security porting hazards are tracked in [old/README.md](old/README.md) with `Consumer Slices` and `Delete After Slice` (always the *last* consumer slice). Design native modules cleanly rather than porting WASI-era decomposition verbatim. Never link from `docs/` into `old/`; delete `old/` completely at Slice 6 (v1.1).
5. **Code hygiene (every slice):** English only outside `old/` (CI-enforced; remove `old/` exclusion at Slice 6). Rust 2024; verified by `cargo fmt --all -- --check`, `cargo clippy --locked --workspace --all-targets -- -D warnings`, `cargo test --locked --workspace`, `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps`, `cargo-deny`, `naga` SPV/HLSL/MSL codegen, headless offscreen `wgpu` smoke tests, forbidden `docs/ -> old/` link guard, and offline `lychee` link checks.

---

## Completed Foundation `[x]`

- **Phase 0 & Phase 0.5 — Native Bootstrap, Hygiene & CI `[x]`:** Initialized clean native `veldmap-core` + `veldmap-app` (`eframe 0.37` / `wgpu 30` / Rust `1.95`), created `legacy-reference-snapshot`, removed unlicensed font binaries, configured workspace lints (`unwrap_used`/`expect_used` warned in prod, `unsafe_op_in_unsafe_fn` denied) and Linux hygiene CI (`fmt`, `clippy --locked -D warnings`, `cargo-deny`, no-Cyrillic, offline `lychee`).
- **Slice 1 — Core Geodesy & Interactive 3D WGS84 Globe `[x]` ([Design Decisions](docs/decisions.md)):** Implemented `veldmap-core::geo` (WGS84 geodesy, Krueger projections/Helmert datums, MGRS/DD/DMS parsing, `f64` infinite Reverse-Z `Camera` with `MIN_HEIGHT_M = 2.0` m, oblate `f64` `cull::Frame`, footprints), decoupled offscreen `wgpu` renderer (`Rgba8UnormSrgb` + `Depth32Float` Reverse-Z, 1:1 nearest sRGB/UNORM `view_formats` compositing, `Backends::PRIMARY` minus `BROWSER_WEBGPU`), baseline GIS HUD, 3-OS CI with `naga` SPV/HLSL/MSL codegen and headless GPU smoke tests, and retired 20 legacy geo/MGRS files from `old/`.
- **Slice 2A — Offline Planetary Quadtree, `TileKey`/`GpuTileCache` & Bundled Basemap `[x]` ([Design Decisions](docs/decisions.md) §5):** Unified `f64` ECEF metres + Fukushima (2006) 3D inverse + geodetic nadir aim, deleted legacy `outlines.rs`, implemented the decoupled geometry/texture WGS84 quadtree (`z_geom <= 16–17`, 2:1 balanced, sag-scaled outward-flared skirts) + adaptive elliptical-disk Polar Stereographic caps, replaced static meshes with per-tile CPU `f64` RTC buffers + post-overlay procedural `fwidth()` graticule, added `TileKey` (with `gutter: u8`), O(1) `TileCache` (`TileScheme::parent` + UV-decoupled `find_carrier`), `GpuTileCache`/`PendingUploads`, `ViewportState`, `Weak<GpuTile>` bind-group sweeping, and `veldmap-raster` serving the public-domain NASA Blue Marble `z = 0..3` basemap (`Satellite / Ellipsoid-only`).

---

## Release Roadmap & Vertical Slices

> **Execution order:** Slices are grouped into four shippable milestones (`v0.2` → `v0.5` → `v1.0` → `v1.1`). Slice IDs (`2B`–`6`) match [old/README.md](old/README.md) retirement tracking verbatim.

---

### Milestone v0.2 — Planetary Basemaps & Local Precision Workstation

#### Slice 2B — Async Tile Streaming, Disk Cache & Live Satellite Basemaps `[ ]`
- [ ] **Workspace crates (`veldmap-net`, `veldmap-storage`), `CancelToken` & multi-producer decode pipeline:**
  - Move shared `DecodedTile` (with optional mip chain) and zero-dep async-waking `CancelToken` (`AtomicBool` + `Waker` / listener hook for immediate `tokio::select!` abort on pan/shutdown) into `veldmap-core`.
  - Introduce `veldmap-net` (`tokio` + `reqwest` over `rustls` — explicitly select `ring` vs. `aws-lc-rs` upfront and verify `deny.toml` license compliance + 3-OS builds): HTTP tile fetcher with per-host concurrency limits, compliant `User-Agent`, replaceable visible-first queue, and immediate cancellation.
  - Introduce the bounded `std::thread` CPU decode pool (generating **CPU linear-light mipmaps** on decoded `Rgba8UnormSrgb` tiles reusing `veldmap-raster`'s LUT math so `GpuTile` stays `TEXTURE_BINDING | COPY_DST`, paired with `MipmapFilterMode::Linear` on the tile sampler) and deterministic shutdown join order (`CancelToken` → drop completion rx / join decode pool → drop `tokio::Runtime`).
  - Refactor `PendingUploads` to own its channel and expose cloneable senders, parameterize `DrawSnapshot` by active `SourceId`, replace `is_coarsest_zero() && level == 0` root pinning with active-source `key.parent().is_none()`, add active-frame `TileCache` protection + `remove_where(source)`, and extract live basemap orchestration & settings UI out of `app.rs`.
- [ ] **Disk cache & durable storage (`veldmap-storage`):**
  - Implement durable atomic writes (`write_atomic_durable()` with `sync_all()` + synchronous handle drop before rename on Windows) for user settings/catalog state, plus crash-safe bounded tile disk caching (referencing `old/code/cache/` & `evict_tests.rs`).
  - Avoid whole-source or coarse `(source, z)` bucket eviction (which thrashes single-source live basemaps at `z = 12..14`) and avoid loose-file `relatime`/Windows NTFS-Defender storms: prefer **SQLite WAL (`rusqlite` bundled)** for single-file multi-process-safe O(log N) LRU + `incremental_vacuum`, or **spatial macro-buckets** `(source, z, x >> 4, y >> 4)` if keeping 2B strictly pure-Rust.
- [ ] **Legally clean live basemaps & `PolarCap` contract:**
  - Wire NASA GIBS via **`GoogleMapsCompatible_Level9` (`WebMercator`)** (avoiding GIBS's non-standard `288°/tile` `EPSG4326` matrix; static layer or explicit `{Time}`) + EOX Sentinel-2 Cloudless (`WebMercator`, CC BY 4.0, `max_level` ~13–14) with visible attribution, per-source `max_level` clamping + ancestor overzooming (`0` requests above `max_level`), and Custom XYZ/TMS URL & API-key settings.
  - Define the `PolarCap` + live-basemap contract explicitly: guarantee **`0` live HTTP requests for `PolarCap` keys**, and route `PolarCap` textures either to a tiny pre-baked polar asset or to lazy-loaded `SourceId::BUNDLED_BASEMAP` offline fallback (keeping Frame-1 orbital views consistent).
- [ ] **3-OS Clippy, docs & `old/` cleanup:** Expand `cargo clippy --locked --workspace --all-targets -- -D warnings` to Windows/macOS in CI now that platform `#[cfg]` code lands; update `docs/decisions.md` and `docs/limitations.md`; delete `old/code/cache/` and `old/tests/cache/evict_tests.rs`.
- **Exit criteria:**
  - *Automated CI:* Eviction (`disk_bytes <= cap`), durable atomic-write/crash-recovery, `CancelToken`, `0` requests above `max_level`, `0` HTTP requests for `PolarCap`, and 3-OS Clippy pass.
  - *Manual QA:* Live NASA GIBS / EOX tiles stream with mipmapped limb smoothness, visible attribution, bounded disk cache, and clean offline fallback.

#### Slice 3A-1 — Local GeoTIFF/PNG/JPEG Engine, `MemoryBudget` & Warp-Mesh Draping `[ ]`

##### Slice 3A-1a — Core Local GeoTIFF/COG Engine & `ByteSource`
- [ ] **Sync `ByteSource` & `SourceFingerprint` (`veldmap-core` / `veldmap-storage`):**
  - Strictly sync caller-buffered `ByteSource::read_at(&self, offset: u64, buf: &mut [u8], cancel: &CancelToken) -> Result<usize, IoError>` with safe positioned-I/O local file impl (no `memmap2`; buffered cursor; pooled `File` handles + short-read loop on Windows).
  - Define opaque `SourceFingerprint` (sampled BLAKE3 `len || head_64k || mid_64k || tail_64k` for local files; in-memory `(path, len, mtime_ns)` stat cache + UI "Reload / bypass cache") combined with band, pixel format, decode revision, and `TileKey.gutter` so incompatible tiles never alias.
- [ ] **Core local GeoTIFF/BigTIFF/COG engine (`veldmap-raster`):**
  - Implement unified `BandInfo` (band metadata, scale/offset, nodata, pixel format), pyramid/resampling/radiometry/grid/cascade (referencing `old/code/tiler/`, using image-rect `cell_uv` against carrier `(width, height, valid_w, valid_h, gutter)` rather than planetary power-of-two `sub_rect_in_ancestor`), and core `tiff` (standard RGB/Gray GeoTIFF/BigTIFF/COG, `GeoDoubleParamsTag`/`GeoAsciiParamsTag`, explicit `Limits` + `MAX_IFDS`, verifying Float32 `Predictor = 3`, benchmarking `ruzstd` vs. `zstd` on real ZSTD COGs).
  - Implement bounded local decode memory budget, panic/cancel-safe singleflight decoded-chunk cache, and native `TileScheduler` (`veldmap-core::tile`: 1 active pass per source for `Serve::Pass` *only*, concurrent `Serve::Pointwise`, non-cancellation of coarser passes, terminal-failure suppression).
  - Verify TIFF read amplification via `CountingByteSource` (referencing `old/tests/tiler/reads_tests.rs`) and concurrency/memory-bound stress tests (zero deadlocks, zero HoL stalls, `peak <= cap`).

##### Slice 3A-1b — GPU Warp-Mesh Draping, Accuracy Regression, PNG/JPEG & QOI Cache
- [ ] **Renderer modularization & depth-less warp-mesh draping:** Split `veldmap-app/src/render.rs` (~1,080 LOC) into focused submodules (`globe`, `overlay`, `snapshot`) as warp-mesh draping lands. Implement top-down lazy `DVec3` culling balls (preserving per-rung limb-culling margins) in `veldmap-core`, and multi-band-ready layer state + per-tile CPU `f64` RTC warp meshes (using `16` segments at coarse UTM zone edges where needed for `< 2 cm`) rendered **depth-less (`depth_write: false`, `depth_compare: Always`) with fragment ellipsoidal-normal horizon discard** between Pass 1 (globe body) and Pass 2 (post-overlay `grid_pipeline`), referencing `old/code/geo/overlay.rs`.
- [ ] **Minimal local opener, HUD/readout separation & accuracy regression:** Wire CLI file argument + drag-and-drop opener; label HUD **viewport GSD** separately from **source pixel size** and keep cursor readouts querying native CPU tiles (zoom-invariant); rebuild `a_raster_lands_where_its_binding_puts_it` (CPU exact geometry + UV placement) verifying all 5 UTM 37 benchmark bindings & tolerances in [docs/decisions.md](docs/decisions.md) §4 (`< 10⁻³ px`, `< 2 mm` native, `< 2 cm` pyramid) plus a **second CRS fixture** and an end-to-end **`PixelIsArea` vs. `PixelIsPoint` (`GTRasterTypeGeoKey`)** half-pixel check.
- [ ] **Georeferenced PNG/JPEG & 8-bit disk cache:** Add local full-image PNG/JPEG adapters (resolving legacy `!Send` `Rc`/`RefCell` per [old/README.md](old/README.md)) and QOI disk caching for 8-bit `Rgba8UnormSrgb` tiles; validate two-lane decode budgeting (preventing full-image decodes from head-of-line blocking tiled COG reads).
- [ ] **Docs & `old/` cleanup:** Update `docs/decisions.md` and `docs/limitations.md`; delete `old/code/geo/overlay.rs`, `old/code/tiler/{budget,fingerprint,tiles_pipeline,cascade,pyramid,resample}.rs`, `old/code/tiler/adapters/{full,radiometry,table,grid,tiff,png,jpeg}.rs`, `old/tests/tiler/{pyramid,radiometry,resample,tiff}_tests.rs`, `old/docs/viewing-pipeline.md`, `old/docs/decisions/0003-raster-reading-model.md`, and `old/docs/decisions/0007-overlay-accuracy.md` (**retain** `reads_tests.rs` until Slice 6).
- **Exit criteria:**
  - *Automated CI:* `a_raster_lands_where_its_binding_puts_it` (all 5 UTM 37 benchmarks + 2nd CRS + `PixelIsArea`/`PixelIsPoint`), COG `CountingByteSource` tests (touching only IFD + intersected chunks), memory-budget/singleflight/`TileScheduler` concurrency stress tests, and `0` idle GPU re-render checks pass.
  - *Manual QA:* Local GeoTIFF/BigTIFF/COG and cm-scale drone orthophotos open via CLI/drag-and-drop and drape accurately with responsive zooming and bounded RAM/VRAM/disk.

#### Slice 3A-2 — Local Workstation UX, Expanded Pure-Rust CRS & GeoJSON Overlay `[ ]`
- [ ] **`app.rs` workstation UI/state decomposition:** Split `veldmap-app/src/app.rs` into a thin frame/input shell, separated workspace/layer state, and focused UI/tool submodules (`toolbar`, `layers`, `status_bar`, `crs_picker`, `measure`).
- [ ] **Expanded pure-Rust CRS & world files:** Keep `veldmap-core::geo::projection` in-house (deferring wrapped `proj4rs`/optional `PROJ` only if arbitrary WKT2/grid-shifted rasters demand it); extend with `ETRS89/UTM` (`258xx`), `NAD83/UTM` (`269xx`), `LAEA Europe` (`3035`), Polar Stereographic (`3413`/`3031`), `GeoKey 3072 == 32767` user-defined TM parameters (**including `ProjNatOriginLatGeoKey (3081)` `origin_lat_deg` and `ProjLinearUnitsGeoKey (3076)` International / US Survey Feet**); wire `Datum::to_wgs84` into geographic bindings (discarding phantom Helmert `Δh` on 2D output); read `.tfw`/`.jgw`/`.pgw` world files externally (injected as `Option<Placement>` so decoders stay pure `ByteSource`); show visible UI warnings for unshifted foreign datums.
- [ ] **Local workstation UX & diagnostics:** Native file picker, Layer Manager (visibility, reorder, opacity, zoom-to-binding, CRS/radiometry metadata inspector), searchable CRS picker with presets & custom TM parameters (unlocking all MSK-xx zones), and diagnostic banners (fatal vs. partial error, detail-limit caption, loading shimmer).
- [ ] **Geodesic tools & GeoJSON overlay:** WGS84 geodesic distance/area measurement tools (via pure-Rust MIT **`geographiclib-rs`** Karney `Geodesic` / `PolygonArea`) with **RFC 7946 GeoJSON export** (`[lon, lat]` WGS84 + right-hand winding), plus **read-only GeoJSON overlay with explicit `Polygon { exterior, holes }`** (upgrading `footprint.rs` union semantics + 3D dihedral `turn()`, stroke-only or local tangent-plane earcut with interior edge subdivision back onto WGS84 so large polygons never sag below the ellipsoid).
- [ ] **Docs:** Update `docs/decisions.md` and `docs/limitations.md` (**retain** `old/docs/georeference.md` until Slice 6).
- **Exit criteria:**
  - *Automated CI:* Expanded CRS/datum/feet/`origin_lat_deg` round-trip tests, Karney geodesic tests, and GeoJSON parse/holes/subdivided-fill/RFC 7946 export round-trip tests pass.
  - *Manual QA:* Local projected/world-file orthophotos and concave/holed GeoJSON field polygons open cleanly via picker/drag-and-drop or CRS override, and measured features export cleanly to GeoJSON.

---

### Milestone v0.5 — Scientific Imagery, Zero-Auth Cloud COGs & Band Math

#### Slice 3B — Scientific `R32Float` / `R8Uint` / `R16Uint` Tracks, Extended TIFFs, Live Stretch & 2D Inspector `[ ]`
- [ ] **Multi-band GPU VRAM budget scaling:** Scale `GpuTileCache` above the initial `256 MiB` baseline (configurable / host-RAM-scaled `512 MiB`–`1 GiB` for desktop GPUs) and allocate the `< 50%` appetite cap **proportionally by per-pyramid byte cost** rather than flat `1/N`, so 1440p/4K multi-band EO (basemap + 3–4 `R32Float` bands + `SCL` + DEM) never throttles into coarse overviews.
- [ ] **Multi-format GPU tracks (`Display`, `Scientific`, `Categorical`) & extended GeoTIFFs:**
  - Keep Display (`Rgba8UnormSrgb`, `512×512`) and add:
    - **Scientific (`R32Float`, `512×512`, `gutter = 0`, 1 mip, `1.00 MiB`/tile for optical/SAR):** stores **raw numeric samples** (nodata/fill → `f32::NAN`; document `f32` 24-bit precision limits for 32-bit int/`f64` readouts) and applies canonical affine **`val * scale + offset`** (`fma`, matching STAC `raster:bands`, GDAL, and CF NetCDF) in WGSL uniforms + cursor readout to avoid cache aliasing.
    - **Categorical (`R8Uint` & `R16Uint`, `512×512`, `gutter = 0`, 1 mip):** discrete class and QA bitmask bands (raw `Palette` classes, Sentinel-2 `SCL`, Landsat `QA_PIXEL`, MODIS QA — sharing `texture_2d<u32>` WGSL bindings) with **nearest-neighbor sampling & non-averaging downsampling**.
  - Add **`YCbCr`**, **`Palette`** (`ColorMap` expansion on Display; raw class indices on `Categorical`), and **`PlanarConfiguration = 2`** to `veldmap-raster`'s GeoTIFF decoder so standard v1.0 COGs/land-cover maps work out of the box.
  - Add compressed disk caching for `R32Float`/`R8Uint`/`R16Uint` tiles (e.g. byte-shuffled LZ4; QOI stays 8-bit Display only). *(Defer `gutter = 1` DEM tile production to Slice 5B-1, its actual consumer; `TileKey.gutter` is already in place.)*
- [ ] **WGSL nodata-aware filtering, live stretch & optional 2D Inspector (`Image2DRenderer`):**
  - Manual 4-tap `textureLoad` bilinear filter masking both `NaN` and `±Inf` via raw `bitcast<u32>` exponent bits (Metal fast-math safe, keeping any derivatives strictly before horizon `discard`).
  - Live contrast stretch (linear percentile/min-max + **`log10`/dB stretch** for SAR), colormap uniforms, multi-band GeoTIFF band selector, interactive histogram, and colorbar with physical units.
  - Add a dedicated orthographic **`Image2DRenderer`** (referencing `old/code/viewer/`) in `CallbackResources` sharing immutable `Arc<GpuTile>` textures with `GlobeRenderer` (without prematurely converting `GlobeRenderer` itself into a multi-viewport 3D map) + synchronized zoom-invariant CPU cursor raw DN & physical-unit readout. *(Can land as a small follow-on PR if needed to avoid delaying 4A/5A.)*
- [ ] **GPU smoke test, docs & `old/` cleanup:** Add headless `wgpu` smoke test verifying NaN/Inf bitmask bilinear math and multi-format bind-group caching; update `docs/decisions.md` and `docs/limitations.md`; delete `old/code/viewer/`.
- **Exit criteria:**
  - *Automated CI:* Headless `wgpu` nodata-bilinear test, `naga` SPV/HLSL/MSL codegen, `R8Uint`/`R16Uint`/`R32Float` byte accounting & multi-band appetite cap, and `YCbCr`/`Palette`/`Planar-2` TIFF tests pass.
  - *Manual QA:* 16-bit/float32/SAR/Palette/QA/multi-band rasters render with exact nodata transparency, live histogram/colorbar, and synchronized 2D/3D cursor readout.

#### Slice 4A — Zero-Auth Cloud EO Streaming (HTTP Range + STAC / Public COGs) `[ ]`
- [ ] **HTTP Range `ByteSource` & zero-extra-RTT remote `SourceFingerprint` (`veldmap-net`):**
  - Implement range reader (referencing `old/code/network/range.rs`) over `reqwest` (`rustls`)/`tokio`: fixed host-RAM-scaled 64 KiB O(1) LRU block pool, combined open-probe + `0..64 KiB` head read (`416`-safe on 0-byte files), consumed-only readahead, panic/cancel-safe multi-block singleflight, **auto-decompression disabled**, reject `200` when expecting `206`, `Content-Range` + **strong-ETag-only `If-Match`** (RFC 9110: never send weak `W/"..."` in `If-Match`, isolate block-pool entries without strong ETags, treat `412` as non-retryable), `CancelToken` async abort, and `Handle::try_current().is_err()` guard on blocking reads.
  - Key remote tiles with `SourceFingerprint` from `(canonical_url_without_query, strong_etag, len)` (falling back to `(canonical_url_without_query, len, blake3(head_64k))` when ETags are missing/weak)—avoiding wasted `mid_64k`/`tail_64k` HTTP GETs per band on COG open.
- [ ] **Single-batch pinned remote COG decode:** Sync chunk range-planners in `veldmap-raster` + pinned prefetch batches in `veldmap-net` (under upfront memory permits) routed to the fast CPU decode pool, while isolating un-prefetched blocking reads to a separate capped remote pool so slow HTTP never starves local SSD decodes.
- [ ] **Zero-auth STAC & public COG provider (`veldmap-net`):** STAC API search (e.g. AWS Earth Search Sentinel-2 L2A COGs + COP-DEM GLO-30) and direct public HTTP COG URLs, multi-band Sentinel-2 asset discovery + STAC `raster:bands` `scale`/`offset` metadata, AOI search (viewport / drawn box / GeoJSON polygon + GeoJSON/WKT export), scene outlines on globe (HiDPI-scaled outline/ribbon/hatch pipelines with per-cluster RTC in `veldmap-app`), and built-in zero-auth sample scenes for 15-second onboarding.
- [ ] **Mock HTTP harness & docs:** Offline local mock HTTP Range harness (`206`, `200` rejection, `416`, strong/weak `ETag`, `412`, N-way singleflight deduplication, request & delivered-byte ceilings); update `docs/decisions.md` and `docs/building.md`.
- **Exit criteria:**
  - *Automated CI:* Offline mock HTTP Range/STAC/singleflight tests (`0` duplicate fetches under N-way contention, bounded request/byte ceilings, zero wasted `mid`/`tail` GETs) pass.
  - *Manual QA:* Search an AOI and stream public Sentinel-2 L2A COGs and HTTP COG URLs with zero credentials without starving local SSD decodes.

#### Slice 5A — Interactive Band Math, Split-Screen Swipe & Session Persistence `[ ]`
- [ ] **Unified screen-space float band math & radiometric correctness:**
  - Pre-record per-band scratch passes **before** `record_offscreen_pass` opens the main `Rgba8UnormSrgb` + `Depth32Float` pass (keeping `Depth32Float` `StoreOp::Discard` intact on TBDR GPUs):
    - Each active scientific band renders into a lazy, shared single-viewport **`R32Float`** scratch target (`blend: None`, avoiding `Rgba16Float` quantization of Sentinel-2 DN and 300 K thermal), initialized as **self-describing nodata** via a 3-vertex fullscreen quiet-`NaN` fill as the first draw of that band's pass (supplying quiet `NaN` via uniform or runtime `bitcast<f32>(0x7FC00000u)` so `naga` const-eval passes, avoiding non-portable `LoadOp::Clear(NaN)`, and avoiding `0.0` vs. nodata ambiguity or unblendable multi-band `R8Uint` overwrites).
    - Reserve a dedicated **`R8Uint`** scratch target (`blend: None`, cleared to `255` uncovered sentinel) strictly for nearest-neighbor `SCL`/QA masks.
  - Run a fullscreen WGSL band-math (NDVI, NDWI, false-color, custom stretch) + mask + colormap composite draw in painter's order inside the main globe pass—handling Sentinel-2 60 m (`6:1`), Landsat 15 m/30 m (`7.5 m` half-pixel shift), independent per-band ancestor fallbacks, and multi-sensor/rotated overlays without per-tile `[tile_id; N]` shader combinatorics (cursor readouts continue querying native CPU tiles).
- [ ] **Metadata-driven scaling & `SCL` masking:** Apply Sentinel-2 `scale`/`offset` from STAC `raster:bands` and local GeoTIFF metadata in canonical `val * scale + offset` physical reflectance space (never hardcoded to a 2021↔2022 date cutoff; ESA `.SAFE` `MTD_MSIL2A.xml` DN-space `BOA_ADD_OFFSET` conversion lands with `4B`), badge L1C (TOA) vs. L2A (BOA), and support optional **nearest-neighbor `R8Uint` `SCL` cloud/shadow masking** (`classes 0, 1, 3, 8, 9, 10`).
- [ ] **Split-screen swipe curtain & session persistence:** Interactive swipe comparison across any two layers (including non-co-registered swaths); workspace/session persistence via `write_atomic_durable()` JSON (layers, CRS overrides, band math recipes, stretch params, camera bookmarks, split-pane layout), copy-coordinates, and viewport PNG screenshot export (with optional scale bar, north arrow, colorbar, and acquisition date; heavy GeoTIFF raster re-export deferred to post-v1).
- [ ] **Docs:** Update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* Metadata-driven `val * scale + offset`, self-describing `NaN`-filled `R32Float` + nearest-neighbor `R8Uint` `SCL` mask (including valid `0.0` reflectance preservation), session JSON round-trip, and `naga` SPV/HLSL/MSL shader tests pass.
  - *Manual QA:* Multi-band Sentinel-2 NDVI/NDWI/false-color (including mixed 10 m/20 m/60 m and multi-sensor) with cloud masking, swipe curtain, screenshot export, and session reload work smoothly.

---

### Milestone v1.0 — Analytical Terrain Relief & Multi-Temporal EO Workstation

#### Slice 5B-1 — Analytical Terrain Relief Layers & COP-DEM Virtual Mosaic `[ ]`
- [ ] **`gutter = 1` DEM tiles & GPU hillshade, slope, aspect:** Implement parameterized **`gutter = 1`, 1-mip (`514×514`) `R32Float`** DEM tile production (keyed via `TileKey.gutter = 1`, falling back to `gutter = 0` rather than `Serve::Pass` on oversized chunks; normalized against carrier `(valid_w, valid_h, gutter)`). Compute derivatives using per-vertex **2×2 inverse Jacobian `J^(-T)`** into local ENU metres (handling EPSG:4326 `cos(lat)`, COP-DEM latitude bands, and UTM/Polar rotation), scaled to the carrier tile's pyramid level, with one-sided fallback on `NaN` boundaries and across `1°×1°` COP-DEM file borders.
- [ ] **Multi-file virtual mosaic (`DemMosaicLayer`) & COP-DEM GLO-30:** Because COP-DEM GLO-30 is distributed as ~26,450 separate `1°×1°` COGs with latitude-dependent horizontal spacing, implement a **`DemMosaicLayer`** abstraction resolving visible `1°×1°` geocell COGs under one Layer Manager entry with a bounded LRU of open remote `ByteSource`s, a coarse-zoom geocell cutoff (or coarse global overview fallback), and accounting as **1 logical pyramid** in `GpuTileCache` appetite budgeting. Add **Multiply** blend mode for relief-over-imagery compositing and label elevation readouts explicitly in **`m EGM2008`**. Update `docs/decisions.md` and `docs/limitations.md`.
- **Exit criteria:**
  - *Automated CI:* Synthetic analytic DEM slope/aspect error bounds across latitudes (`0°`/`60°`/`80°`) and rotated projections, `gutter = 1` continuity, `DemMosaicLayer` geocell resolution/eviction, `naga` SPV/HLSL/MSL, and headless GPU smoke tests pass.
  - *Manual QA:* COP-DEM GLO-30 hillshade/slope/aspect renders cleanly over optical imagery with zero intra-file tile-edge creases.

#### Slice 5B-2 — Multi-Temporal Timeline, Dual-Epoch Diffing & 1D Time-Series Plot `[ ]`
- [ ] **Multi-temporal timeline scrubber:** Reuse unified `TileKey` for epoch ingestion from **STAC AOI search** (grouped by MGRS tile + AOI coverage, defaulting to L2A) and **local dated folders**, with cloud-cover filtering, adjacent-date prefetch, and instant stale-request cancellation.
- [ ] **Unified offscreen dual-epoch cross-fade & physical reflectance diffing:** Reuse Slice 5A's lazy shared **`NaN`-filled `R32Float` + `SCL` `R8Uint`** offscreen scratch pool (pre-colormap/stretch, preserving valid `0.0` diffs and propagating `SCL`/QA validity) for physical reflectance/index diffing (`Date B − Date A`) and display targets for visual cross-fade (`mix`)—supporting multi-band + `SCL`, mismatched carrier levels, and non-co-registered swaths cleanly.
- [ ] **1D pixel time-series plot & CSV export:** Lightweight 1D time-series panel (custom `egui::Painter` chart to avoid pinning `egui` upgrades to lagging `egui_plot`) querying **native CPU source tiles** (zoom-invariant) to plot reflectance/NDVI over time at a clicked location (skipping cloud-masked dates), with **RFC 4180 CSV export** (ISO-8601 UTC timestamps, explicit units, documented `SCL`-masked row convention).
- **Exit criteria:**
  - *Automated CI:* Temporal diffing shaders (`naga` SPV/HLSL/MSL + headless GPU smoke), float pre-stretch diff math (including `0.0` diff vs. nodata), zoom-invariant time-series readout, and RFC 4180 CSV export tests pass.
  - *Manual QA:* End-to-end v1.0 EO workflow (local + STAC COGs, band math, swipe/diff, pixel time-series + CSV export, analytical DEM relief) persists cleanly across sessions.

---

### Milestone v1.1 — Mission-Specific Codecs, Authenticated CDSE & Final Legacy Retirement

#### Slice 3C-1 — Sentinel-2 JPEG 2000 `[ ]`
- [ ] **Audit & harden JP2 decoder:** Audit the latest `openjp2` (or evaluate optional C `openjpeg-sys` / mature pure-Rust alternatives if `openjp2` fails soundness/thread-safety/fuzzing audit) and implement local `.SAFE` JP2 decoding, GMLJP2 parsing (`roxmltree`/`quick-xml`), and tile-part excerpting, resolving every audited `codec.rs`/`excerpt.rs`/`jp2.rs` UB and bounds hazard listed in [old/README.md](old/README.md).
- [ ] **Tests, docs & `old/` cleanup:** Adapt `jp2_tests.rs`, `excerpt_tests.rs`, JP2 sections of `reads_tests.rs`, plus 5/3 reversible round-trip and malformed/truncated-input fuzz tests; update `docs/decisions.md` and `docs/limitations.md`; delete `old/code/tiler/adapters/{codec,jp2,excerpt}.rs`, `old/tests/tiler/{jp2,excerpt}_tests.rs`, and `old/docs/decisions/{0002-jpeg2000-decoder,0004-jp2-excerpt}.md`.
- **Exit criteria:**
  - *Automated CI:* 5/3 JP2 round-trip, excerpt, and malformed/truncated-input tests pass with zero panics or UB.
  - *Manual QA:* Local Sentinel-2 JP2 granules open and render cleanly in 2D/3D.

#### Slice 4B — Copernicus Data Space (CDSE) S3/JP2 & Resumable Downloads `[ ]`
- [ ] **CDSE provider, SigV4 re-signing & timeline hook (`veldmap-net`):** Implement CDSE OData/S3/SigV4 (`Signer`), multi-band Sentinel-2 `.SAFE` asset discovery (`B02`–`B12`, `SCL`, and `MTD_MSIL2A.xml` DN-space `BOA_ADD_OFFSET` normalized to `offset = BOA_ADD_OFFSET / QUANTIFICATION_VALUE`), epoch-based singleflight 401/403 re-sign (preserving the unsigned `Range` header invariant), and timeline scrubber integration. Decouple CDSE catalog/COG/resumable download from remote HTTP JP2 packet-surgery—gate **AOI-scoped two-phase remote JP2 streaming** behind latency/ROI profiling (legacy remote JP2 excerpting measured `70–77 s` and `255` HTTP requests per band).
- [ ] **Local library, hardened resumable downloads & Credentials UI (`veldmap-storage` / `veldmap-net`):** Implement catalog, storage layout, and unified resumable `.part` downloads resolving every legacy hazard in [old/README.md](old/README.md), plus an in-app credentials/cache dialog writing `ProjectDirs/.env` (`0600` on Unix) with mandatory credential/SigV4 redaction in logs & errors and anonymous search without keys.
- [ ] **Tests, docs & `old/` cleanup:** Adapt `old/tests/provider/` + prefetch section of `reads_tests.rs` + mock 401/403/`.part` crash-resume & secret-redaction tests; update `docs/decisions.md` and `docs/limitations.md`; delete `old/code/{provider,library,network}/`, `old/tests/provider/`, `old/docs/download.md`, and `old/docs/decisions/0005-network-reading.md` (**retain** `old/docs/imagery.md` until Slice 6).
- **Exit criteria:**
  - *Automated CI:* CDSE provider, mock 401/403 & `.part` resume, and secret-redaction log tests pass.
  - *Manual QA:* Browse CDSE S3, download/open Sentinel-2 granules (including via the timeline), and pause/resume downloads reliably.

#### Slice 6 — NetCDF-4 (Sentinel-3/5P) & Final Legacy Retirement `[ ]`
- [ ] **Audit & implement NetCDF-4/HDF5 reader (`veldmap-raster`):** Spike and audit `hdf5_pure` against real ESA Sentinel-3/5P NetCDF-4 files (`Send + Sync`, superblock v2/v3, Shuffle+Deflate+Fletcher32, `f64` coordinates/scaling, raw typed integer reads), keeping an optional C `libhdf5` feature as an explicit fallback if pure-Rust HDF5 fails on real mission granules. Implement the NetCDF-4 adapter (resolving every legacy hazard in [old/README.md](old/README.md)), local `.SAFE` sibling geolocation, and NetCDF variable switcher UI.
- [ ] **Tests, docs & final `old/` deletion:** Adapt `netcdf_tests.rs` + NetCDF section of `reads_tests.rs` + malformed-input tests; update `docs/decisions.md` and `docs/limitations.md`; verify against `old/README.md` that zero unmigrated files remain, delete `old/` completely, remove the `old/` exclusion from CI no-Cyrillic check, and finalize `README.md`.
- **Exit criteria:**
  - *Automated CI:* Full 3-OS CI passes with `old/` removed, zero Cyrillic exclusions, and NetCDF `f64`/fill-value/malformed-input tests green.
  - *Manual QA:* Local and remote Sentinel-3/5P NetCDF-4 files open, switch variables, and render cleanly in 2D/3D.

---

## Next Step

- **Current state:** Phases 0–0.5, Slice 1, and Slice 2A are complete on `main` (see [Design Decisions](docs/decisions.md)).
- **Immediate task:** Start **Slice 2B — Async Tile Streaming, Disk Cache & Live Satellite Basemaps**: move shared `DecodedTile` + zero-dep async-waking `CancelToken` into `veldmap-core`, introduce `veldmap-net` (`tokio` + `reqwest`/`rustls`, verifying `ring` vs. `aws-lc-rs` against `deny.toml` and 3-OS builds), bounded CPU decode pool with linear-light mipmap generation, `veldmap-storage` durable atomic writes + SQLite WAL (or spatial macro-bucket) LRU disk cache, multi-producer `PendingUploads` + active-source `DrawSnapshot` + basemap UI extraction from `app.rs`, live NASA GIBS (`GoogleMapsCompatible_Level9`) + EOX Sentinel-2 Cloudless with explicit `PolarCap` contract & offline fallback, 3-OS Clippy, and `old/code/cache/` retirement.
