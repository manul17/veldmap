# Glossary

Canonical English domain terms implemented across VeldMap's code, UI, and documentation. New terms are added alongside the slice that introduces them.

| Term | Meaning |
|---|---|
| **ECEF** | Earth-Centered, Earth-Fixed Cartesian coordinate system on the WGS84 ellipsoid. |
| **Footprint** | Geographic boundary polygon of a satellite scene or local raster product. |
| **Graticule** | Latitude/longitude reference grid draped over the WGS84 ellipsoid. |
| **GSD** | Ground Sample Distance—surface metres per pixel at viewport center. |
| **MGRS** | Military Grid Reference System (e.g., `37UDQ`), widely used for Sentinel-2 tile indexing and coordinate jumps. |
| **Nadir globe** | VeldMap v1's smooth `height = 0.0` WGS84 ellipsoid camera model with zero camera pitch/tilt, avoiding renderer-induced relief parallax and keeping uniform center GSD. |
| **Precision RTC** | Relative-to-Center rendering (per-tile `f64` origin subtraction on CPU before casting to `f32`) so GPU vertex precision scales with tile size and distance to the camera rather than Earth's radius. |
| **Reverse-Z** | Depth-buffer convention mapping the near plane to `1.0` and infinity to `0.0` in `Depth32Float`, providing uniform precision from orbit down to metre-scale altitude. |
| **Sag / Sagitta** | Maximum distance a flat chord or triangle dips below the curved ellipsoid surface between vertices. |
