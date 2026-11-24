# VeldMap

Fast, purely native Rust desktop **Analytical Earth-Observation & Precision-Imagery Workstation** built on a **Precision Nadir 3D WGS84 Globe**.

## Current Status & Roadmap

VeldMap is built in vertical slices tracked in [plan.md](plan.md):

- **Implemented (Slices 1–3A-1):** Interactive 3D WGS84 globe (`2 m` altitude floor, geodetic-nadir camera, MGRS/DD/DMS Go-To navigation, baseline GIS HUD) backed by pure-Rust `f64` geodesy/projections, an adaptive planetary quadtree with Polar Stereographic caps, a post-overlay procedural graticule, bundled offline NASA Blue Marble (`veldmap-raster`), durable settings + SQLite WAL LRU disk caching with lossless QOI display tile compression (`veldmap-storage`), async live HTTP basemaps (NASA GIBS, EOX Sentinel-2 Cloudless, Custom XYZ/TMS) with CPU linear-light mipmaps and offline/polar fallback (`veldmap-net`), plus local GeoTIFF/BigTIFF/ZSTD-COG opening (CLI, drag-and-drop, Open trigger), Layer List, depth-less CPU `f64` RTC warp-mesh draping, zoom-invariant RGB cursor readouts, two-lane (`fast`/`bulk`) `TileScheduler` decoding, and full overlay accuracy regression (`veldmap-core` / `veldmap-raster` / `veldmap-app`).
- **In Progress / Planned (v0.2–v1.0):** Slice 3A-2 local workstation UX (`rfd` native picker, manual heading rotation, keyboard nav, minimal session persistence), PNG/JPEG world-file overlays, `YCbCr`/`WebP` TIFFs, expanded pure-Rust CRS, geodesic measurement tools & read-only GeoJSON overlay; `R32Float`/`R8Uint`/`R16Uint` scientific & categorical tracks (3B); zero-auth & custom-header STAC/COG HTTP Range streaming (4A); interactive band math (NDVI/NDWI/false-color + `SCL` cloud masking; 5A); analytical local & COP-DEM terrain relief (5B-1); and multi-temporal timeline diffing, AOI zonal stats/time-series & Float32 GeoTIFF export (5B-2).
- **Planned (v1.1):** Sentinel-2 JPEG 2000, authenticated CDSE S3 streaming & resumable downloads, and Sentinel-3/5P NetCDF-4/HDF5.

## Workspace Layout

- `crates/veldmap-core` — Pure `#![forbid(unsafe_code)]` domain logic (`geo` including overlay `Frame`/lazy culling balls/RTC warp meshes, `tile`, `budget`, `sync::{CancelToken, Wakeup}`, `io::{ByteSource, BufferedByteReader}`; zero OS probing, `egui`, `eframe`, `wgpu`, `image`, or network dependencies).
- `crates/veldmap-raster` — Memory-bounded PNG/JPEG & GeoTIFF/BigTIFF/COG decoding (`RasterSource`, `TiffSource`, `SharedChunkCache`, `TileScheduler`, `DecodePeak`, pyramid/resampling/radiometry pipeline), bounded CPU `DecodePool` / `TwoLaneDecodePool`, CPU linear-light mipmap generation, and virtual offline equirectangular basemap tiling.
- `crates/veldmap-storage` — Injectable `StoragePaths`, cross-OS host-RAM budget detection (`detect_memory_budget`), durable atomic JSON persistence (`write_atomic_durable` / `write_atomic_durable_private`), positioned `LocalFileSource`, bounded sampled BLAKE3 `LocalFingerprintCache`, and crash-safe vertically partitioned SQLite WAL LRU tile disk cache (with pure-Rust QOI display tile encoding).
- `crates/veldmap-net` — Async `tokio` + `reqwest`/`rustls` (`ring`) live HTTP tile streaming, per-host concurrency limiting, pluggable `TileDecodeExecutor`/`TileBlobStore` seams, and NASA GIBS / EOX / Custom XYZ/TMS specs.
- `crates/veldmap-app` — Native `eframe`/`egui` + `wgpu` desktop application (`[lib]` + `[[bin]]` so rendering and shader tests run headlessly), including depth-less post-body overlay draping, `LayerStack` / `OverlayManager`, user settings (`settings.rs`), and runtime storage/decode wiring.
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
