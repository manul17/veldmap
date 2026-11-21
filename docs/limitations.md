# Known Limitations

Explicit scope boundaries and current implementation limits of VeldMap. Entries are updated as slices close and capabilities land.

## Planetary Model (Committed v1 Scope)

- **Flat WGS84 nadir ellipsoid (no 3D DEM mesh displacement or camera tilt):** VeldMap v1 renders a smooth WGS84 ellipsoid (`height = 0.0`) with a strictly nadir camera. Terrain relief is delivered as quantitative GPU layers (hillshade, slope, aspect) rather than 3D vertex displacement, guaranteeing zero parallax for orthophoto alignment, split-screen swipe, and temporal diffing.
- **Desktop native only:** VeldMap targets native desktop GPUs via `wgpu::Backends::PRIMARY` (Vulkan, DirectX 12, Metal). OpenGL/GLES and browser WASM builds are intentionally out of scope for this repository.

## Current Milestone Limits (Bootstrap / Pre-Slice 1)

- **Placeholder viewport:** The desktop application currently renders a minimal `wgpu` viewport scaffold; the interactive WGS84 3D globe, geodesy math, MGRS navigation, basemaps, local/remote raster decoders, and analytical tools are delivered incrementally across Slices 1–6 (see [plan.md](../plan.md)).
- **Default UI fonts only:** Bundled third-party font binaries were removed to avoid mixed-license bloat; `eframe`'s built-in default fonts are used until UI polish.
- **No persistent workspace or credentials UI yet:** Runtime configuration is currently controlled via environment variables (`RUST_LOG`, `WGPU_BACKEND`, `WGPU_POWER_PREF`).
