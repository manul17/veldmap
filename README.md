# VeldMap

Fast, purely native Rust desktop **Analytical Earth-Observation & Precision-Imagery Workstation** built on a **Precision Nadir 3D WGS84 Globe**.

## Current Status & Roadmap

VeldMap is built in vertical slices tracked in [plan.md](plan.md):

- **Implemented (Slices 1–5A-2 / Milestone v0.5 Complete):**
  - *Globe & Basemaps:* Interactive 3D WGS84 globe (`2 m` altitude floor, geodetic-nadir camera with heading rotation and keyboard nav, bidirectional MGRS/DD/DMS Go-To & coordinate copy, camera bookmarks, adaptive quadtree with Polar Stereographic caps, post-overlay procedural graticule), plus offline NASA Blue Marble and async live HTTP basemaps (NASA GIBS, EOX Sentinel-2 Cloudless, Custom XYZ/TMS) backed by SQLite WAL LRU disk caching.
  - *Local & Cloud Rasters:* Local and HTTP Range Cloud-Optimized GeoTIFF/BigTIFF (`SinglePassIfdReader`, lock-free direct raw-chunk decoding for chunky & `PlanarConfiguration = 2`, `ZSTD`/`YCbCr`/`WebP`, 4-band RGB+NIR, `Palette`), local PNG/JPEG with world-file/`.prj` sidecars, and expanded pure-Rust CRS (`proj4rs`/`proj4wkt`/EPSG presets/custom TM).
  - *Scientific, Categorical, Band Math & Swipe:* Display (`Rgba8UnormSrgb`), Scientific (`R32Float` with 8-tap nodata-aware trilinear filtering, live linear/`log10` dB stretch, interactive histogram, colorbar, and zoom-invariant native-`f64` cursor readouts), Categorical (`R8Uint`/`R16Uint`), multi-asset GPU Band Math + `SCL` cloud/shadow masking (`NDVI`, `NDWI`, `True Color`, `False Color NIR`, `SWIR`, local Band Math builder, `RdYlGn`/`BrBG` colormaps), and interactive split-screen swipe comparison across any two raster layers (including non-co-registered swaths and `BandMath`) with offloaded byte-shuffled `Lz4Raw` and lossless `Qoi` tile disk caching.
  - *Cloud STAC & Vectors:* Zero-auth AWS Earth Search (Sentinel-2 L2A & COP-DEM GLO-30) and custom-auth STAC discovery (rolling last-90-days UTC defaults) with 1-click scene recipes and built-in sample scenes/recipes, AOI drawing & Measure-to-STAC-AOI / GeoJSON / WKT export, HiDPI ribbon/hatch globe footprints, retained GeoJSON vector overlays (with `"Outline only"` styling), and Karney WGS84 geodesic measurement with GeoJSON export.
  - *Workstation UX, Workspaces & Export:* Non-blocking OS file dialogs (`rfd`) with drag-and-drop/path fallback, collapsible `Layers` | `Cloud / STAC` workbench, 3-tab Settings modal (`Basemap`, `Cache & GPU`, `STAC / Cloud`), configurable GPU VRAM cap (`128–4096 MiB`), durable `session.json` & portable `.veldmap.json` workspaces (local + sanitized remote COGs & `BandMath` recipes with offline resilience), and non-blocking offscreen cartographic PNG export (scale bar, North arrow, colorbar, caption).
- **Planned (v1.0):** Analytical local & COP-DEM terrain relief (`5B-1`); STAC pagination & multi-temporal timeline diffing (`5B-2a`); and AOI zonal stats/time-series & Float32 GeoTIFF export (`5B-2b`).
- **Planned (v1.1):** Cloud-native ESA EOPF Zarr streaming & authenticated CDSE S3/resumable downloads (`4B-1`), local Sentinel-2 `.SAFE` JPEG 2000 (`3C-1`), Sentinel-5P/3 NetCDF-4/HDF5 (`6`), and optional 1:1 native-pixel 2D orthographic inspector (`3B-2`).

## Workspace Layout

- `crates/veldmap-core` — Pure `#![forbid(unsafe_code)]` domain logic (`geo` geodesy/projections/pure-Rust CRS/Karney geodesic tools/GeoJSON & WKT export/tessellation/encapsulated `Frame` bindings, `tile` keys/LRU/quadtree/overlay culling & RTC warp meshes, `budget`, `sync::{CancelToken, Wakeup}`, `io::{ByteSource, BufferedByteReader}`; zero OS probing, UI, GPU, image, or network dependencies).
- `crates/veldmap-raster` — Memory-bounded pure-`ByteSource` `SinglePassIfdReader` + direct raw-chunk GeoTIFF/BigTIFF/COG (`Planar-1/2`, `ZSTD`/`YCbCr`/`WebP`, RGB+NIR, `Palette`), PNG, and JPEG decoding with world-file/`.prj` sidecar loaders (`RasterSource`, `TiffSource`, `PngSource`, `JpegSource`, `SharedChunkCache`, versioned `TileScheduler`, typed Display/Scientific/Categorical pyramid/resampling/radiometry pipeline, remote COG byte-range planning), lossless `Qoi`, byte-shuffled `Lz4Raw`, and `encode_png_rgba8` codec helpers, bounded CPU `DecodePool` / `TwoLaneDecodePool`, CPU linear-light & NaN-skipping mipmap generation, and virtual offline equirectangular basemap tiling.
- `crates/veldmap-storage` — Injectable `StoragePaths`, cross-OS host-RAM budget detection (`detect_memory_budget`), durable atomic file & JSON persistence (`write_atomic_durable`, `write_json_durable`, `read_json_limited`, `read_json_or_default_with_backup`, `write_json_durable_private`), positioned `LocalFileSource`, bounded sampled BLAKE3 `LocalFingerprintCache`, and codec-free crash-safe vertically partitioned SQLite WAL LRU tile disk cache (`rusqlite` `prepare_cached`).
- `crates/veldmap-net` — Shared async `NetContext` (`tokio` + `reqwest`/`rustls` + `ring`), live HTTP tile streaming (`LiveTilePipeline`), HTTP Range COG random-access `ByteSource` (`HttpRangeSource` with 64 KiB LRU `RangeBlockPool`, `PrefetchBudget`, refcounted singleflight, and strong-ETag validation), zero-auth & custom-auth STAC API client (`StacClient`), per-host concurrency/cooldown limiting, pluggable `TileDecodeExecutor`/`TileBlobStore` seams, and NASA GIBS / EOX / Custom XYZ/TMS specs.
- `crates/veldmap-app` — Native `eframe`/`egui` + `wgpu` desktop application (`[lib]` + `[[bin]]` `veldmap` so rendering and shader tests run headlessly), including depth-less post-body display/scientific/categorical raster, multi-asset screen-space GPU Band Math + `SCL` masking, split-screen swipe scissors, cartographic PNG export (`export.rs`), and HiDPI ribbon/hatch vector overlay draping, `LayerStack` / `OverlayManager`, `StacSession` / Cloud workbench, CRS picker & measurement tools, non-blocking OS file dialogs, unified user settings, session & `.veldmap.json` project persistence (`storage/{settings,session,project}.rs`), and runtime storage/decode/network wiring.
- `assets/` — Public-domain bundled assets and provenance notes ([assets/README.md](assets/README.md)).
- `docs/` — Build guide, design decisions, glossary, and current limitations ([docs/README.md](docs/README.md)).
- `plan.md` — Executable vertical-slice roadmap and exit-criteria tracker ([plan.md](plan.md)).

## Quick Start

Requires Rust `1.95` or newer (pinned via `rust-toolchain.toml`) and a Vulkan, DX12, or Metal GPU driver. See [docs/building.md](docs/building.md) for platform prerequisites, verification commands, and environment overrides (`WGPU_BACKEND`, `WGPU_POWER_PREF`, `RUST_LOG`).

```bash
cargo run -p veldmap-app
cargo test --workspace
```

## Contributing & Security

- Contribution workflow and code hygiene rules: [CONTRIBUTING.md](CONTRIBUTING.md)
- Vulnerability reporting and security posture: [SECURITY.md](SECURITY.md)
- License and third-party attributions: [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
