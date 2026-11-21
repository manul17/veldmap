# Legacy Reference (`old/`)

Curated migration input preserved from the legacy VeldMap repository at annotated tag `legacy-reference-snapshot`. This directory is **reference-only** (marked `linguist-vendored` in `.gitattributes`), is never linked from `docs/`, and is deleted incrementally slice-by-slice until complete removal in Slice 6.

## Provenance & Language Note

- `code/geo`, `code/tiler` (except `tiles_pipeline.rs`), `code/provider`, `code/cache`, `code/library`, `code/viewer`: English-translated domain logic extracted from the legacy repository (`2eb1bc5` and `master`).
- `code/network/range.rs`, `code/network/download.rs`, `code/tiler/tiles_pipeline.rs`: extracted from `master` (HTTP Range 64 KiB block pool, readahead, 401/403 re-signing, resumable downloads, and legacy tile pipeline).
- `code/viewer/`: legacy 2D raster canvas camera and WGSL shader reference (`camera.rs`, `image_view.wgsl`).
- `tests/`: selected `#[cfg(test)]` suites from `master` covering geodesy, projections, culling, JP2/TIFF/NetCDF parsing, radiometry, S3/MGRS, and cache eviction.
- `docs/`: legacy domain architecture notes and ADRs `0002`–`0007`.
- **Cyrillic note:** 28 files under `old/` still contain legacy Cyrillic comments, test strings, or glossary entries (`code/network/download.rs`, `code/network/range.rs`, `code/tiler/tiles_pipeline.rs`, 4 files in `docs/`, and all 21 files in `tests/`). Everything ported into `crates/` and `docs/` must be strictly English; the `old/` exclusion in CI's no-Cyrillic check is removed in Slice 6 once `old/` is deleted.

## Retirement Manifest (79 Reference Files)

Each reference file is deleted only after its **last** consumer slice closes and its tests/ADRs land.

| File | Consumer Slices | Delete After Slice |
|---|---|---|
| `code/cache/evict.rs` | Slice 2 | Slice 2 |
| `code/cache/layout.rs` | Slice 2 | Slice 2 |
| `code/cache/store.rs` | Slice 2 | Slice 2 |
| `code/cache/tile.rs` | Slice 2 | Slice 2 |
| `code/geo/camera.rs` | Slice 1 | Slice 1 |
| `code/geo/cull.rs` | Slice 1 | Slice 1 |
| `code/geo/footprint.rs` | Slice 1 | Slice 1 |
| `code/geo/geodesy.rs` | Slice 1 | Slice 1 |
| `code/geo/globe.wgsl` | Slice 1 | Slice 1 |
| `code/geo/mesh.rs` | Slice 1 | Slice 1 |
| `code/geo/outlines.rs` | Slice 1 | Slice 1 |
| `code/geo/overlay.rs` | Slice 3A-1 | Slice 3A-1 |
| `code/geo/projection.rs` | Slice 1 | Slice 1 |
| `code/geo/wheel.rs` | Slice 1 | Slice 1 |
| `code/library/catalog.rs` | Slice 4B | Slice 4B |
| `code/library/download.rs` | Slice 4B | Slice 4B |
| `code/library/storage.rs` | Slice 4B | Slice 4B |
| `code/network/download.rs` | Slice 4B | Slice 4B |
| `code/network/range.rs` | Slice 4A, Slice 4B | Slice 4B |
| `code/provider/catalogue.rs` | Slice 4B | Slice 4B |
| `code/provider/imagery.rs` | Slice 4B | Slice 4B |
| `code/provider/manifest.rs` | Slice 4B | Slice 4B |
| `code/provider/mgrs.rs` | Slice 1 | Slice 1 |
| `code/provider/s3.rs` | Slice 4B | Slice 4B |
| `code/provider/scene.rs` | Slice 4B | Slice 4B |
| `code/tiler/adapters/codec.rs` | Slice 3C-1 | Slice 3C-1 |
| `code/tiler/adapters/excerpt.rs` | Slice 3C-1 | Slice 3C-1 |
| `code/tiler/adapters/full.rs` | Slice 3A-1 | Slice 3A-1 |
| `code/tiler/adapters/grid.rs` | Slice 3A-1 | Slice 3A-1 |
| `code/tiler/adapters/jp2.rs` | Slice 3C-1 | Slice 3C-1 |
| `code/tiler/adapters/jpeg.rs` | Slice 3A-1 | Slice 3A-1 |
| `code/tiler/adapters/mod.rs` | Slice 3A-1, Slice 3C-1, Slice 6 | Slice 6 |
| `code/tiler/adapters/netcdf.rs` | Slice 6 | Slice 6 |
| `code/tiler/adapters/png.rs` | Slice 3A-1 | Slice 3A-1 |
| `code/tiler/adapters/radiometry.rs` | Slice 3A-1 | Slice 3A-1 |
| `code/tiler/adapters/table.rs` | Slice 3A-1 | Slice 3A-1 |
| `code/tiler/adapters/tiff.rs` | Slice 3A-1 | Slice 3A-1 |
| `code/tiler/budget.rs` | Slice 3A-1 | Slice 3A-1 |
| `code/tiler/cascade.rs` | Slice 3A-1 | Slice 3A-1 |
| `code/tiler/fingerprint.rs` | Slice 3A-1 | Slice 3A-1 |
| `code/tiler/pyramid.rs` | Slice 3A-1 | Slice 3A-1 |
| `code/tiler/resample.rs` | Slice 3A-1 | Slice 3A-1 |
| `code/tiler/tiles_pipeline.rs` | Slice 3A-1 | Slice 3A-1 |
| `code/viewer/camera.rs` | Slice 3B | Slice 3B |
| `code/viewer/image_view.wgsl` | Slice 3B | Slice 3B |
| `docs/decisions/0002-jpeg2000-decoder.md` | Slice 3C-1 | Slice 3C-1 |
| `docs/decisions/0003-raster-reading-model.md` | Slice 3A-1 | Slice 3A-1 |
| `docs/decisions/0004-jp2-excerpt.md` | Slice 3C-1 | Slice 3C-1 |
| `docs/decisions/0005-network-reading.md` | Slice 4A, Slice 4B | Slice 4B |
| `docs/decisions/0006-terrain-on-the-globe.md` | Slice 1 | Slice 1 |
| `docs/decisions/0007-overlay-accuracy.md` | Slice 1, Slice 3A-1 | Slice 3A-1 |
| `docs/download.md` | Slice 4B | Slice 4B |
| `docs/georeference.md` | Slice 3A-2, Slice 6 | Slice 6 |
| `docs/globe.md` | Slice 1 | Slice 1 |
| `docs/glossary.md` | Phase 0.5, Slice 1–6 | Slice 6 |
| `docs/imagery.md` | Slice 3C-1, Slice 4B | Slice 4B |
| `docs/limitations.md` | Phase 0.5, Slice 1–6 | Slice 6 |
| `docs/viewing-pipeline.md` | Slice 2, Slice 3A-1 | Slice 3A-1 |
| `tests/cache/evict_tests.rs` | Slice 2 | Slice 2 |
| `tests/geo/camera_tests.rs` | Slice 1 | Slice 1 |
| `tests/geo/cull_tests.rs` | Slice 1 | Slice 1 |
| `tests/geo/footprint_tests.rs` | Slice 1 | Slice 1 |
| `tests/geo/geodesy_tests.rs` | Slice 1 | Slice 1 |
| `tests/geo/mesh_tests.rs` | Slice 1 | Slice 1 |
| `tests/geo/outlines_tests.rs` | Slice 1 | Slice 1 |
| `tests/geo/projection_tests.rs` | Slice 1 | Slice 1 |
| `tests/provider/imagery_tests.rs` | Slice 4B | Slice 4B |
| `tests/provider/manifest_tests.rs` | Slice 4B | Slice 4B |
| `tests/provider/mgrs_tests.rs` | Slice 1 | Slice 1 |
| `tests/provider/s3_tests.rs` | Slice 4B | Slice 4B |
| `tests/provider/scene_tests.rs` | Slice 4B | Slice 4B |
| `tests/tiler/excerpt_tests.rs` | Slice 3C-1 | Slice 3C-1 |
| `tests/tiler/jp2_tests.rs` | Slice 3C-1 | Slice 3C-1 |
| `tests/tiler/netcdf_tests.rs` | Slice 6 | Slice 6 |
| `tests/tiler/pyramid_tests.rs` | Slice 3A-1 | Slice 3A-1 |
| `tests/tiler/radiometry_tests.rs` | Slice 3A-1 | Slice 3A-1 |
| `tests/tiler/reads_tests.rs` | Slice 3A-1, Slice 3C-1, Slice 4B, Slice 6 | Slice 6 |
| `tests/tiler/resample_tests.rs` | Slice 3A-1 | Slice 3A-1 |
| `tests/tiler/tiff_tests.rs` | Slice 3A-1 | Slice 3A-1 |
