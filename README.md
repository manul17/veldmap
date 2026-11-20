# VeldMap

Native Rust desktop application for planetary 3D Earth visualization, satellite
imagery (Copernicus Data Space, GeoTIFF, JPEG 2000, NetCDF-4), and temporal
change exploration.

## Workspace

- `crates/veldmap-core` — UI-free and GPU-free domain logic (geodesy, projections,
  raster adapters, tile cache, providers, range streaming).
- `crates/veldmap-app` — native `eframe`/`egui` + `wgpu` desktop application.
- `old/` — temporary migration reference (English-translated domain code,
  selected tests, and domain docs from the legacy repository).

## Build and run

```bash
cargo run -p veldmap-app
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Copernicus credentials

Copy `.env.example` to `.env` and fill in `COPERNICUS_ACCESS_KEY` and
`COPERNICUS_ACCESS_SECRET` for authenticated CDSE S3 access.
