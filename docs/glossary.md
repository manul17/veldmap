# Glossary

Canonical English domain terms implemented across VeldMap's code, UI, and documentation. New terms are added alongside the slice that introduces them.

| Term | Meaning |
|---|---|
| **ByteSource** | Synchronous, caller-buffered, positioned random-access byte reader (`read_at`) shared across local files (`LocalFileSource` via `pread`/`seek_read`, never `memmap2`) and HTTP Range readers (`HttpRangeSource`). |
| **Carrier / Overzoom** | Finest cached ancestor tile used to texture a subdivided geometry tile via a sub-rectangle UV mapping when finer imagery is unavailable or exceeds `max_level`. |
| **COG** | Cloud-Optimized GeoTIFF: a tiled GeoTIFF with embedded reduced-resolution overview IFDs enabling bounded random-access tile reads without decoding the full raster. |
| **Display / Scientific / Categorical tracks** | The three typed GPU raster pipelines: `Display` (`Rgba8UnormSrgb` with linear-light mipmaps), `Scientific` (`R32Float` with NaN-skipping mipmaps, 8-tap nodata-aware trilinear filtering, `val * scale + offset` physical scaling, and live linear/`10·log10` dB stretch), and `Categorical` (`R8Uint`/`R16Uint`, 1-mip nearest-neighbor class/QA masks). |
| **ECEF** | Earth-Centered, Earth-Fixed Cartesian coordinate system on the WGS84 ellipsoid. |
| **Footprint** | Geographic boundary polygon of a satellite scene or local raster product. |
| **Graticule** | Procedural latitude/longitude reference grid draped post-overlay over the WGS84 ellipsoid. |
| **GSD** | Ground Sample Distance—surface metres per pixel at viewport center (distinct from a dataset's native source pixel size). |
| **Gutter** | Extra border ring of texels reserved in `TileKey` / `DecodedTile` / `Lz4Raw` (e.g., `gutter = 1` on `514x514` DEM tiles, landing in Slice 5B-1) providing neighbor samples for crease-free finite-difference derivatives across tile edges. |
| **Linear-light mipmap** | CPU-downsampled `Rgba8UnormSrgb` mip chain averaged in linear light (with alpha weighting) so grazing limb angles stay anti-aliased without gamma darkening. |
| **Lz4Raw** | Byte-shuffled, `lz4_flex`-compressed mip-0 tile blob encoding (`VML4` 24-byte self-describing header + pre-allocation bounds guard) used to cache `R32Float`/`R8Uint`/`R16Uint` tiles in SQLite. |
| **MGRS** | Military Grid Reference System (e.g., `37UDQ`), widely used for Sentinel-2 tile indexing and coordinate jumps. |
| **Nadir globe** | VeldMap v1's smooth `height = 0.0` WGS84 ellipsoid camera model with zero camera pitch/tilt, avoiding renderer-induced relief parallax and keeping uniform center GSD. |
| **Pointwise / Pass / Cascade** | Raster pyramid execution modes: `Pointwise` decodes individual tiles on demand from intersecting chunks/overviews; `Pass` streams a full-file top-to-bottom scan through a streaming `Cascade` emitting every pyramid level in one pass. |
| **Polar Cap** | Adaptive elliptical square-to-disk Polar Stereographic quadtree cap covering `|lat| >= 85.05112878°` with zero live HTTP requests and no polar triangle slivers. |
| **Precision RTC** | Relative-to-Center rendering (per-tile `f64` origin subtraction on CPU before casting to `f32`) so GPU vertex precision scales with tile size and distance to the camera rather than Earth's radius. |
| **Quadtree** | Scheme-matched hierarchical WGS84 surface mesh with decoupled geometry (`z_geom <= 16–17`) vs. texture LOD, 2:1 neighbor balancing, and sag-scaled skirts. |
| **Reverse-Z** | Depth-buffer convention mapping the near plane to `1.0` and infinity to `0.0` in `Depth32Float`, providing uniform precision from orbit down to metre-scale altitude. |
| **Sag / Sagitta** | Maximum distance a flat chord or triangle dips below the curved ellipsoid surface between vertices. |
| **Singleflight** | In-flight request deduplication ensuring concurrent callers requesting the same chunk, stretch computation, or HTTP Range block span share one leader execution while honoring independent follower cancellation and panic recovery. |
| **SourceFingerprint** | Opaque 256-bit content identity (sampled BLAKE3 over `len || head_64k || mid_64k || tail_64k` for local files; `veldmap-remote-v1` over canonical URL, length, and strong ETag or `blake3(head_64k)` for HTTP objects) deterministically mapped to `SourceId` and disk-cache keys. |
| **STAC** | SpatioTemporal Asset Catalog: JSON/GeoJSON REST API specification used to search Earth-Observation imagery by AOI, datetime, and cloud cover and resolve Cloud-Optimized GeoTIFF asset URLs and `raster:bands` metadata. |
| **StreamKey** | Per-stream identity `(source, band, format, gutter)` scoping `TileScheduler` passes, multi-band VRAM appetite weighting, and `UploadBatchReport` byte tracking independently across tracks sharing one `SourceId`. |
| **TileKey / TileScheme** | Unified `Copy` cache key (`source`, `band`, `scheme`, `format`, `gutter`, `decode_rev`, `level`, `x`, `y`) serving both planetary basemaps (`z = 0` coarsest) and `RasterPyramid` overlays (`level = 0` finest). |
| **WAL / LRU** | SQLite Write-Ahead Logging combined with Least-Recently-Used per-tile eviction across vertically partitioned `tile_meta` and `tile_blobs` tables. |
| **WMTS / XYZ / TMS** | Web tiled basemap addressing conventions (`WebMercator` slippy-map `{z}/{x}/{y}`, NASA GIBS `GoogleMapsCompatible` WMTS matrices, and south-up `{-y}` TMS flips). |
