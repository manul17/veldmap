# VeldMap

Fast, purely native Rust desktop **Analytical Earth-Observation & Precision-Imagery Workstation** built on a **Precision Nadir 3D WGS84 Globe**.

## Current Status & Roadmap

VeldMap is built in vertical slices tracked in [plan.md](plan.md):

- **Implemented (Slices 1–3B / Milestone v0.2 Complete, v0.5 In Progress):** Interactive 3D WGS84 globe (`2 m` altitude floor, geodetic-nadir camera with manual heading rotation and keyboard nav, MGRS/DD/DMS Go-To, camera bookmarks, adaptive quadtree with Polar Stereographic caps, post-overlay procedural graticule), offline NASA Blue Marble and async live HTTP basemaps (NASA GIBS, EOX Sentinel-2 Cloudless, Custom XYZ/TMS) backed by SQLite WAL LRU disk caching, local GeoTIFF/BigTIFF/COG (lock-free direct raw-chunk decoding for chunky & `PlanarConfiguration = 2`, `ZSTD`/`YCbCr`/`WebP`, 4-band RGB+NIR, `Palette`) and PNG/JPEG (world-file + `.prj` sidecar) raster overlays across Display (`Rgba8UnormSrgb`), Scientific (`R32Float` with 8-tap nodata-aware trilinear filtering, live linear/`log10` dB stretch, interactive histogram, colorbar, and native-`f64` cursor readouts), and Categorical (`R8Uint`/`R16Uint`) tracks with byte-shuffled `Lz4Raw` and lossless `Qoi` tile disk caching, expanded pure-Rust CRS (`proj4rs`/`proj4wkt`/EPSG presets/custom TM), read-only GeoJSON vector overlays, Karney WGS84 geodesic measurement with GeoJSON export, native OS file dialogs (`rfd`) with drag-and-drop/path fallback, Layer Manager, configurable GPU VRAM cap (`128–4096 MiB`), and durable `session.json` persistence.
- **In Progress / Planned (v0.5–v1.0):** Optional orthographic 2D inspector (3B-2); zero-auth & custom-header STAC/COG HTTP Range streaming (4A); interactive band math (NDVI/NDWI/false-color + `SCL` cloud masking) & split-screen swipe (5A); analytical local & COP-DEM terrain relief (5B-1); and multi-temporal timeline diffing, AOI zonal stats/time-series & Float32 GeoTIFF export (5B-2).
- **Planned (v1.1):** Sentinel-2 JPEG 2000, authenticated CDSE S3 streaming & resumable downloads, and Sentinel-3/5P NetCDF-4/HDF5 (with EOPF Zarr checkpoint).

## Workspace Layout

- `crates/veldmap-core` — Pure `#![forbid(unsafe_code)]` domain logic (`geo` geodesy/projections/pure-Rust CRS/Karney geodesic tools/GeoJSON tessellation/encapsulated `Frame` bindings, `tile` keys/LRU/quadtree/overlay culling & RTC warp meshes, `budget`, `sync::{CancelToken, Wakeup}`, `io::{ByteSource, BufferedByteReader}`; zero OS probing, UI, GPU, image, or network dependencies).
- `crates/veldmap-raster` — Memory-bounded direct raw-chunk GeoTIFF/BigTIFF/COG (`Planar-1/2`, `ZSTD`/`YCbCr`/`WebP`, RGB+NIR, `Palette`), PNG, and JPEG decoding with world-file/`.prj` sidecar loaders (`RasterSource`, `TiffSource`, `PngSource`, `JpegSource`, `SharedChunkCache`, versioned `TileScheduler`, typed Display/Scientific/Categorical pyramid/resampling/radiometry pipeline), lossless `Qoi` and byte-shuffled `Lz4Raw` tile codec helpers, bounded CPU `DecodePool` / `TwoLaneDecodePool`, CPU linear-light & NaN-skipping mipmap generation, and virtual offline equirectangular basemap tiling.
- `crates/veldmap-storage` — Injectable `StoragePaths`, cross-OS host-RAM budget detection (`detect_memory_budget`), durable atomic file & JSON persistence (`write_atomic_durable` for raw bytes; `read_json_or_default_with_backup` / `write_json_durable_private` for durable JSON), positioned `LocalFileSource`, bounded sampled BLAKE3 `LocalFingerprintCache`, and codec-free crash-safe vertically partitioned SQLite WAL LRU tile disk cache.
- `crates/veldmap-net` — Async `tokio` + `reqwest`/`rustls` (`ring`) live HTTP tile streaming, per-host concurrency limiting, pluggable `TileDecodeExecutor`/`TileBlobStore` seams, and NASA GIBS / EOX / Custom XYZ/TMS specs.
- `crates/veldmap-app` — Native `eframe`/`egui` + `wgpu` desktop application (`[lib]` + `[[bin]]` `veldmap` so rendering and shader tests run headlessly), including depth-less post-body display/scientific/categorical raster & vector overlay draping, `LayerStack` / `OverlayManager`, CRS picker & measurement tools, non-blocking OS file dialogs, user settings & session persistence (`settings.rs`, `session.rs`), and runtime storage/decode wiring.
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
