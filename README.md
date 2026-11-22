# VeldMap

Fast, purely native Rust desktop **Analytical Earth-Observation & Precision-Imagery Workstation** built on a **Precision Nadir 3D WGS84 Globe**.

## Current Status & Roadmap

VeldMap is built in vertical slices tracked in [plan.md](plan.md):

- **Implemented (Slice 1):** Pure-Rust WGS84 geodesy, map projections, MGRS/DD/DMS coordinate navigation, `f64` infinite Reverse-Z nadir camera & oblate horizon culling, decoupled offscreen `wgpu` WGS84 globe renderer with exact linear-to-sRGB compositing, and baseline GIS HUD (cursor/center readout, altitude, GSD, dynamic scale bar, North compass, Go-To bar).
- **Planned (Slices 2–6):** Global WGS84 tile quadtree with bundled and live satellite basemaps, local GeoTIFF/BigTIFF/COG/PNG/JPEG and Sentinel-2 JP2 streaming, `R32Float` scientific band math (NDVI/NDWI/false-color + cloud masking), analytical DEM relief, multi-temporal timeline diffing, and NetCDF-4 (v1.1).

## Workspace Layout

- `crates/veldmap-core` — Pure domain logic (zero `egui`, `eframe`, or `wgpu` dependencies). Currently provides `geo`; `tile`, `io`/`mem`, and feature-gated `fs`/`codecs`/`net` modules land in upcoming slices ([ADR 0001](docs/decisions/0001-native-workspace-and-threading.md)).
- `crates/veldmap-app` — Native `eframe`/`egui` + `wgpu` desktop application (`[lib]` + `[[bin]]` so rendering and shader tests run headlessly).
- `docs/` — Architecture notes, build guide, glossary, limitations, and Architectural Decision Records ([docs/README.md](docs/README.md)).
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
