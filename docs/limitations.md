# Known Limitations

Explicit scope boundaries and current implementation limits of VeldMap. Entries are updated as slices close and capabilities land.

## Planetary Model (Committed v1 Scope)

- **Flat WGS84 nadir ellipsoid (no 3D DEM mesh displacement or camera tilt):** VeldMap v1 renders a smooth WGS84 ellipsoid (`height = 0.0`) with a strictly nadir camera ([ADR 0002](decisions/0002-globe-model-and-precision.md)). Terrain relief is delivered as quantitative GPU layers (hillshade, slope, aspect) in Slice 5B-1 rather than 3D vertex displacement, guaranteeing zero parallax for orthophoto alignment, split-screen swipe, and temporal diffing.
- **Desktop native only:** VeldMap targets native desktop GPUs via `wgpu::Backends::PRIMARY` (Vulkan, DirectX 12, Metal). OpenGL/GLES and browser WASM builds are intentionally out of scope for this repository.

## Current Milestone Limits (Post-Slice 1)

- **Untextured ellipsoid + graticule only:** Slice 1 delivers the precision WGS84 ellipsoid, draped 15° graticule, camera controls, MGRS/DD/DMS Go-To navigation, and GIS HUD. Global satellite basemaps and quadtree streaming land in Slices 2A–2B; local/cloud raster imagery and geodesic measurement tools land in Slices 3–4 (see [plan.md](../plan.md)).
- **CRS, datum shifts & MGRS coverage:** `veldmap-core::geo::projection` currently supports WGS84 UTM (`326xx`/`327xx`), Gauss-Krüger on Pulkovo 1942 (`284xx`), MSK-61 (`6336101..=6336103`), and Web Mercator (`3857`) using static 7-parameter Helmert datum shifts (horizontal only; no grid-based shifts such as NADCON5/NTv2/OSTN15 or WKT2/LCC/Albers projections yet—datasets on tectonic/grid-shifted datums may retain decimetre-to-metre offsets relative to WGS84/ITRF). ETRS89/NAD83 UTM, LAEA Europe (`3035`), Polar Stereographic (`3413`/`3031`), user-defined TM parameters, and world files land in Slice 3A-2. MGRS parsing covers UTM bands `C`–`X` (including Norway `31V`/`32V` and Svalbard `31X`/`33X`/`35X`/`37X` bounds), excluding UPS polar zones (`A`, `B`, `Y`, `Z`).
- **Polar basemap detail:** `WebMercator` basemaps terminate at `±85.051°` (covered by a fixed-zoom polar cap in Slice 2A); full polar detail for Arctic/Antarctic imagery requires a `Geodetic` (`EPSG:4326`) basemap provider.
- **Default UI fonts only:** `eframe`'s built-in default fonts are used until UI polish.
- **No persistent workspace or credentials UI yet:** Runtime configuration is currently controlled via environment variables (`RUST_LOG`, `WGPU_BACKEND`, `WGPU_POWER_PREF`).
