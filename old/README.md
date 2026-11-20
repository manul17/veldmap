# Legacy Reference (`old/`)

Curated migration input extracted from `F:\dev\veldmap-legacy`. Remove this
directory once all required functionality has been ported into `crates/`.

- `code/geo`, `code/tiler`, `code/provider`, `code/cache`, `code/library`:
  English-translated domain logic from commit `2eb1bc5` (before comments were
  stripped in `b0b8124`).
- `code/network/range.rs`: 64 KiB HTTP Range block pool, readahead, and
  401/403 re-signing from `master`.
- `tests/`: Selected high-signal `#[cfg(test)]` blocks from `master` covering
  geodesy, projections, culling, JP2/TIFF/NetCDF parsing, radiometry, S3/MGRS,
  and cache eviction.
- `docs/`: Domain architecture notes and ADRs `0002`–`0007` from `master`.
