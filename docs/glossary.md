# Glossary

Canonical English domain terms used across VeldMap's code, UI, and documentation.

## Implemented Terms (Slice 1)

| Term | Meaning |
|---|---|
| **ECEF** | Earth-Centered, Earth-Fixed Cartesian coordinate system on the WGS84 ellipsoid. |
| **Footprint** | Geographic boundary polygon of a satellite scene or local raster product. |
| **Graticule** | Latitude/longitude reference grid draped over the WGS84 ellipsoid. |
| **GSD** | Ground Sample Distance—surface metres per pixel at viewport center. |
| **MGRS** | Military Grid Reference System (e.g., `37UDQ`), widely used for Sentinel-2 tile indexing and coordinate jumps. |
| **Nadir globe** | VeldMap v1's flat WGS84 ellipsoid camera model with zero camera pitch/tilt, preventing orthophoto parallax. |
| **Precision RTC / RTE** | Relative-to-Center (per-tile `f64` subtraction on CPU) or Relative-to-Eye (double-single high/low split) coordinate rendering so GPU `f32` precision scales with distance to the camera rather than Earth's radius. |
| **Reverse-Z** | Depth-buffer convention mapping the near plane to `1.0` and infinity to `0.0` in `Depth32Float`, providing uniform precision from orbit down to metre-scale altitude. |
| **Sag / Sagitta** | Maximum distance a flat chord or triangle dips below the curved ellipsoid surface between vertices. |

## Planned Architecture Terms (Slices 2–5)

| Term | Meaning |
|---|---|
| **Appetite cap** | Maximum fraction of GPU/tile cache budget a single raster pyramid may occupy at once, preventing one layer from starving others. |
| **BandKey** | Compact `Copy` identifier distinguishing a display render (`Display`), a numeric band index (`Index(u16)`), or a hashed named band (`NamedHash(u64)`). |
| **Binding** | Georeferencing mapping between raster pixel coordinates and Earth coordinates (projected CRS transform, geographic tie lattice, or footprint quad). |
| **ByteSource** | Strictly synchronous, caller-buffered positioned-read interface (`read_at`) implemented by local files, HTTP Range streams, and pinned memory buffers. |
| **CancelToken** | Lightweight cooperative cancellation token (`AtomicBool` + async waker hook) used to abort stale network fetches, decodes, and shutdown tasks immediately. |
| **Carrier** | The tile actually drawn for a requested cell—either the cell's own tile or a sub-rectangle of its nearest available coarser ancestor. |
| **ContentId** | Sampled BLAKE3 content fingerprint (`len || head_64k || mid_64k || tail_64k`, domain-separated) used to key disk and VRAM caches. |
| **Detail limit** | Finest pyramid level a raster source can serve within the configured memory budget. |
| **MemoryBudget** | Lock-free global memory accounting governor coordinating upfront decode reservations and `Shrinkable` cache eviction. |
| **TileKey** | Deterministic `Copy + Eq + Hash` identity for cached tiles across basemaps, local rasters, scientific bands, and temporal epochs. |
| **Warp mesh** | Subdivided surface quad mesh carrying projection distortion on the GPU so raster pixels are never CPU-resampled before draping. |
