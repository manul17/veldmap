# Legacy Reference (`old/`)

Curated migration input preserved from the legacy VeldMap repository at annotated tag `legacy-reference-snapshot`. This directory is **reference-only** (marked `linguist-vendored` in `.gitattributes`), is never linked from `docs/`, and is scheduled for complete working-tree retirement right after **Slice 3A-1b** (migrating the remaining v1.1 porting hazard checklists into `plan.md` first; later slices consult `git show legacy-reference-snapshot:old/...` on demand).

- **Original reference snapshot (`legacy-reference-snapshot`):** 79 reference files.
- **Retired in Slice 1:** 20 files (`code/geo/*` except `overlay.rs`, `code/provider/mgrs.rs`, `tests/geo/*`, `tests/provider/mgrs_tests.rs`, `docs/globe.md`, `docs/decisions/0006-terrain-on-the-globe.md`).
- **Retired in Slice 2B:** 5 files (`code/cache/{evict,layout,store,tile}.rs`, `tests/cache/evict_tests.rs`).
- **Remaining reference files:** 54 files (everything ported into `crates/` and `docs/` is strictly English, and both `old/` and its CI exclusion are removed right after Slice 3A-1b).

## Remaining Retirement Manifest (54 Reference Files)

The table below tracks which future slice each legacy reference file informs. The working-tree copy of `old/` is scheduled for complete retirement right after **Slice 3A-1b** (migrating the remaining v1.1 porting hazard checklists into `plan.md` first); slices after 3A-1b consult `git show legacy-reference-snapshot:old/...` on demand.

| File | Consumer Slices |
|---|---|
| `code/geo/overlay.rs` | Slice 3A-1 |
| `code/library/catalog.rs` | Slice 4B |
| `code/library/download.rs` | Slice 4B |
| `code/library/storage.rs` | Slice 4B |
| `code/network/download.rs` | Slice 4B |
| `code/network/range.rs` | Slice 4A, Slice 4B |
| `code/provider/catalogue.rs` | Slice 4B |
| `code/provider/imagery.rs` | Slice 4B |
| `code/provider/manifest.rs` | Slice 4B |
| `code/provider/s3.rs` | Slice 4B |
| `code/provider/scene.rs` | Slice 4B |
| `code/tiler/adapters/codec.rs` | Slice 3C-1 |
| `code/tiler/adapters/excerpt.rs` | Slice 3C-1 |
| `code/tiler/adapters/full.rs` | Slice 3A-1 |
| `code/tiler/adapters/grid.rs` | Slice 3A-1 |
| `code/tiler/adapters/jp2.rs` | Slice 3C-1 |
| `code/tiler/adapters/jpeg.rs` | Slice 3A-1 |
| `code/tiler/adapters/mod.rs` | Slice 3A-1, Slice 3C-1, Slice 6 |
| `code/tiler/adapters/netcdf.rs` | Slice 6 |
| `code/tiler/adapters/png.rs` | Slice 3A-1 |
| `code/tiler/adapters/radiometry.rs` | Slice 3A-1 |
| `code/tiler/adapters/table.rs` | Slice 3A-1 |
| `code/tiler/adapters/tiff.rs` | Slice 3A-1 |
| `code/tiler/budget.rs` | Slice 3A-1 |
| `code/tiler/cascade.rs` | Slice 3A-1 |
| `code/tiler/fingerprint.rs` | Slice 3A-1 |
| `code/tiler/pyramid.rs` | Slice 3A-1 |
| `code/tiler/resample.rs` | Slice 3A-1 |
| `code/tiler/tiles_pipeline.rs` | Slice 3A-1 |
| `code/viewer/camera.rs` | Slice 3B |
| `code/viewer/image_view.wgsl` | Slice 3B |
| `docs/decisions/0002-jpeg2000-decoder.md` | Slice 3C-1 |
| `docs/decisions/0003-raster-reading-model.md` | Slice 3A-1 |
| `docs/decisions/0004-jp2-excerpt.md` | Slice 3C-1 |
| `docs/decisions/0005-network-reading.md` | Slice 4A, Slice 4B |
| `docs/decisions/0007-overlay-accuracy.md` | Slice 1, Slice 3A-1 |
| `docs/download.md` | Slice 4B |
| `docs/georeference.md` | Slice 3A-2, Slice 6 |
| `docs/glossary.md` | Phase 0.5, Slice 1–6 |
| `docs/imagery.md` | Slice 3C-1, Slice 4B, Slice 6 |
| `docs/limitations.md` | Phase 0.5, Slice 1–6 |
| `docs/viewing-pipeline.md` | Slice 2A, Slice 3A-1 |
| `tests/provider/imagery_tests.rs` | Slice 4B |
| `tests/provider/manifest_tests.rs` | Slice 4B |
| `tests/provider/s3_tests.rs` | Slice 4B |
| `tests/provider/scene_tests.rs` | Slice 4B |
| `tests/tiler/excerpt_tests.rs` | Slice 3C-1 |
| `tests/tiler/jp2_tests.rs` | Slice 3C-1 |
| `tests/tiler/netcdf_tests.rs` | Slice 6 |
| `tests/tiler/pyramid_tests.rs` | Slice 3A-1 |
| `tests/tiler/radiometry_tests.rs` | Slice 3A-1 |
| `tests/tiler/reads_tests.rs` | Slice 3A-1, Slice 3C-1, Slice 4B, Slice 6 |
| `tests/tiler/resample_tests.rs` | Slice 3A-1 |
| `tests/tiler/tiff_tests.rs` | Slice 3A-1 |

## Known Legacy Porting Hazards (Do Not Port Verbatim)

Concrete safety, correctness, and architecture bugs audited in `old/` that must be resolved whenever the corresponding native slice consumes these files (migrated into `plan.md` right before `old/` is retired after Slice 3A-1b):

- **Slice 3A-1 (`code/tiler/budget.rs`, `code/tiler/adapters/png.rs`, `jpeg.rs`):** Strip WASI `!Send` `Rc`/`RefCell`; fix legacy `direct_peak()` double-counting shared decoded chunks; avoid `Shrinkable` lock-reentrancy.
- **Slice 3C-1 (`code/tiler/adapters/codec.rs`, `excerpt.rs`, `jp2.rs`):** Audit `openjp2` for `static mut` thread safety/soundness (evaluate `openjpeg-sys` optional feature if it fails); fix Stacked Borrows / drop-order UB (`Box::into_raw` + explicit `Drop`), `usize::MAX` EOF + `ErrorKind::Interrupted` retry, `catch_unwind` on `extern "C"` callbacks, `skip()` EOF bounds check, `COD` `levels <= 32` shift-overflow guard, PLT varint & TLM checked arithmetic + `keep <= part.len`, `MAX_PAD` cap, worker panic isolation, and gating zero-as-nodata on Sentinel-2 heuristics rather than all GMLJP2. Add 5/3 reversible round-trip and malformed/truncated `proptest` fuzz tests.
- **Slice 4B (`code/network/download.rs`, `code/library/*`, `code/provider/*`):** Drop WASI `queued_sidecars` / async rescan state machines in favor of serialized sync `std::fs` metadata; enforce S3 path sanitization, Windows `\\?\` extended-length paths (`longPathAware`), unsigned `Range` header invariant on 401/403 re-sign, persisted `ETag` `If-Range`, HTTP 416 recovery, `File::try_lock()` + in-process lock map + explicit `unlock()`, `downloaded == total_size`, and credential/SigV4 redaction in `tracing` logs/errors.
- **Slice 6 (`code/tiler/adapters/netcdf.rs`):** Read coordinates and `scale_factor`/`add_offset` in `f64`; check raw typed `_FillValue`/`missing_value`/`valid_range` *before* unpacking (mapping fill to `f32::NAN`); fail fast on unsupported HDF5 filters at `describe`; bound `Hdf5HandleCache` (`METADATA_CACHE` + `ROW_CACHE_CAP`) with fixed host-RAM-scaled caps.
