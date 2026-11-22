# VeldMap

Fast, purely native Rust desktop **Analytical Earth-Observation & Precision-Imagery Workstation** built on a **Precision Nadir 3D WGS84 Globe**.

## Current Status & Roadmap

VeldMap is built in vertical slices tracked in [plan.md](plan.md):

- **Implemented (Slices 1–2A):** Pure-Rust WGS84 geodesy (`f64` ECEF metres + Fukushima 2006 3D inverse), map projections, MGRS/DD/DMS coordinate navigation, `f64` infinite Reverse-Z geodetic-nadir camera & oblate horizon culling, decoupled offscreen `wgpu` WGS84 planetary quadtree with adaptive Polar Stereographic caps, per-tile CPU `f64` RTC geometry, procedural post-overlay `fwidth()` graticule, `TileKey`/`TileCache`/`GpuTileCache`, bundled offline NASA Blue Marble basemap (`veldmap-raster`), exact linear-to-sRGB compositing, and baseline GIS HUD.
- **Planned (v0.2–v1.0):** Async live satellite basemaps (NASA GIBS / EOX Sentinel-2 Cloudless) & disk cache, local GeoTIFF/BigTIFF/COG/PNG/JPEG overlays & CRS/geodesic tools, zero-auth STAC/COG HTTP Range streaming, `R32Float`/`R8Uint`/`R16Uint` scientific & categorical band math (NDVI/NDWI/false-color + `SCL` cloud masking), analytical DEM relief, and multi-temporal timeline diffing & time-series export.
- **Planned (v1.1):** Sentinel-2 JPEG 2000, authenticated CDSE S3 streaming & resumable downloads, and Sentinel-3/5P NetCDF-4/HDF5.

## Workspace Layout

- `crates/veldmap-core` — Pure domain logic (`geo`, `tile`; zero `egui`, `eframe`, `wgpu`, `image`, or network dependencies).
- `crates/veldmap-raster` — Memory-bounded PNG/JPEG tile decoding and virtual offline equirectangular basemap tiling.
- `crates/veldmap-app` — Native `eframe`/`egui` + `wgpu` desktop application (`[lib]` + `[[bin]]` so rendering and shader tests run headlessly).
- `assets/` — Public-domain bundled assets and provenance notes ([assets/README.md](assets/README.md)).
- `docs/` — Build guide, design decisions, glossary, and current limitations ([docs/README.md](docs/README.md)).
- `plan.md` — Executable vertical-slice roadmap and exit-criteria tracker ([plan.md](plan.md)).
- `old/` — Temporary legacy migration reference, retired incrementally by slice ([old/README.md](old/README.md)).

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
