# VeldMap

Fast, purely native Rust desktop **Analytical Earth-Observation & Precision-Imagery Workstation** built on a **Precision Nadir 3D WGS84 Globe**.

## Overview

VeldMap is designed for:

- Interactive nadir 3D WGS84 planetary viewing from global basemaps down to centimetre-scale local orthophotos.
- Multi-format satellite imagery inspection (GeoTIFF/BigTIFF/COG, PNG/JPEG, Sentinel-2 JPEG 2000; NetCDF-4/HDF5 in v1.1), both locally and streamed in-place over HTTP Range from zero-auth STAC/COG catalogs, public COG URLs, and Copernicus Data Space Ecosystem (CDSE).
- Interactive GPU band math (NDVI, NDWI, false-color with metadata-driven radiometric scaling and cloud masking), analytical terrain relief (hillshade, slope, aspect), multi-temporal timeline scrubbing, and pixel time-series inspection.

## Workspace Layout

- `crates/veldmap-core` — Pure, UI-free, GPU-free domain logic with a strict acyclic internal module DAG (`geo`, `tile`, `io`/`mem`, plus optional `fs`, `codecs`, and `net` features).
- `crates/veldmap-app` — Native `eframe`/`egui` + `wgpu` desktop workstation (offscreen 3D globe renderer, 2D inspector, GPU tile cache, background workers, and UI).
- `docs/` — Architecture notes, build guide, glossary, limitations, and Architectural Decision Records ([docs/README.md](docs/README.md)).
- `plan.md` — Executable vertical-slice roadmap and exit-criteria tracker ([plan.md](plan.md)).
- `old/` — Temporary legacy migration reference, retired incrementally by slice ([old/README.md](old/README.md)).

## Quick Start

Requires Rust `1.95` or newer (pinned via `rust-toolchain.toml`) and a Vulkan, DX12, or Metal GPU driver. See [docs/building.md](docs/building.md) for platform prerequisites and environment overrides (`WGPU_BACKEND`, `WGPU_POWER_PREF`, `RUST_LOG`).

```bash
cargo run -p veldmap-app
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

## Credentials

Zero-auth STAC/COG catalogs and local files work without credentials. For authenticated Copernicus Data Space Ecosystem (CDSE) S3 access, copy `.env.example` to `.env` (git-ignored) and set `COPERNICUS_ACCESS_KEY` and `COPERNICUS_ACCESS_SECRET`.

## Contributing & Security

- Contribution workflow and code hygiene rules: [CONTRIBUTING.md](CONTRIBUTING.md)
- Vulnerability reporting and credential handling: [SECURITY.md](SECURITY.md)
- License and third-party attributions: [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
