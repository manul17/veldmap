# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in VeldMap—including credential leakage, codec memory-safety/DoS bugs (malformed GeoTIFF/JPEG 2000/NetCDF files), path traversal in downloads/caches, or TLS/signing issues—please report it responsibly rather than opening a public issue.

- Use GitHub Security Advisories ("Report a vulnerability") on the repository.
- Include reproduction steps, affected commit/version, and any crafted test input needed to verify the fix.

## Active Protections (Enforced Today)

- **Native Memory & Dependency Hygiene:** `unsafe_op_in_unsafe_fn` is denied workspace-wide, and `undocumented_unsafe_blocks`, `unwrap_used`, and `expect_used` are enforced via Clippy `-D warnings` in CI. `memmap2` (outside Wayland SHM wrappers) and `openssl`/`openssl-sys` are banned in [deny.toml](deny.toml).
- **TLS, Network Caps & Input Allowlists:** Live HTTP tile requests use `rustls` with the `ring` crypto provider, cap response bodies at `16 MiB` (and initial `Content-Length` pre-allocation at `256 KiB`), validate `DiskTileKey` `source_key` values against a strict ASCII allowlist (`<= 128` bytes, no path separators), and reject out-of-range / `PolarCap` live URLs.
- **Private Settings & Secret Redaction:** User settings (`settings.json`, which may contain Custom XYZ/TMS `api_key` values) are written atomically with owner-only `0600` permissions on Unix via `write_json_durable_private()` (backed by `write_atomic_durable_private()`); `api_key` values and URL query strings are redacted in `Debug` output and error messages, and `.env*` files (except `.env.example`) are excluded by `.gitignore`.
- **Bounded Local GeoTIFF/COG Engine & QOI Cache (Slice 3A-1):** `veldmap-raster` enforces explicit `tiff::decoder::Limits`, `MAX_IFDS`, `MAX_SOURCE_SIDE` (for untiled TIFFs), symmetric `16 MiB` QOI tile encode/decode bounds (`MAX_QOI_DECODED_BYTES`), `MemoryBudget`/`DecodeBudgetTracker` caps, positioned `ByteSource` I/O without `memmap2`, and `TwoLaneDecodePool` worker panic isolation for GeoTIFF/BigTIFF/COG sources.

## Planned Security Controls (Upcoming Slices)

- **Credentials & Redaction (Slice 4B):** CDSE S3 keys will live in user-local `.env` files (`0600` permissions on Unix when written via the settings UI), with mandatory redaction of credential headers and SigV4 signatures in `tracing` logs and errors.
- **Untrusted Inputs & Downloads (Slices 3A-2, 3C-1, 4A, 4B, 6):** Remaining local PNG/JPEG full-image adapters (`3A-2`) and future JP2/NetCDF decoders will enforce explicit dimension, memory, and filter limits with panic isolation; remote range readers and downloads will validate S3 path safety, HTTP `Content-Range`, and strong `ETag` integrity.
