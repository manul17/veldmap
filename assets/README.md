# Bundled Assets

## `basemap/blue_marble_ng.jpg`

- **Dataset:** NASA Blue Marble Next Generation + Topography + Bathymetry (global equirectangular Plate Carrée composite, `-180°..+180°`, `+90°..-90°`).
- **Author / Source:** Reto Stöckli, NASA Earth Observatory (distributed via Wikimedia Commons: `File:Blue_Marble_Next_Generation_+_topography_+_bathymetry.jpg`).
- **License:** Public Domain (work of the United States Federal Government / NASA).
- **Dimensions:** `3600 x 1800` pixels (JPEG, ~1.93 MiB).
- **SHA-256:** `062582f953750c05c12514b9ff498b8db5b2c68e1e0563b0bea2f3385be9f5db`
- **Usage in VeldMap:** Embedded via `include_bytes!` as the zero-network offline fallback basemap, served as virtual `z = 0..3` `Geodetic` tiles and `z = 0..1` adaptive Polar Stereographic cap tiles through `veldmap-raster::BundledBasemap` → `PendingUploads` → `GpuTileCache`.
