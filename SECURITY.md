# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in VeldMap—including credential leakage, codec memory-safety/DoS bugs (malformed GeoTIFF/JPEG 2000/NetCDF files), path traversal in downloads/caches, or TLS/signing issues—please report it responsibly rather than opening a public issue.

- Use GitHub Security Advisories ("Report a vulnerability") on the repository.
- Include reproduction steps, affected commit/version, and any crafted test input needed to verify the fix.

## Active Protections (Enforced Today)

- **Native Memory & Dependency Hygiene:** `veldmap-core`, `veldmap-raster`, and `veldmap-net` enforce `#![forbid(unsafe_code)]`; `unsafe_op_in_unsafe_fn` is denied workspace-wide, and `undocumented_unsafe_blocks`, `unwrap_used`, and `expect_used` are enforced via Clippy `-D warnings` in CI. `memmap2` (outside Wayland SHM wrappers) and `openssl`/`openssl-sys` are banned in [deny.toml](deny.toml).
- **TLS, Network Caps & Input Allowlists:** Live HTTP tile requests use `rustls` with the `ring` crypto provider, cap response bodies at `16 MiB` (and initial `Content-Length` pre-allocation at `256 KiB`), validate `DiskTileKey` `source_key` values against a strict ASCII allowlist (`<= 128` bytes, no path separators), and reject out-of-range / `PolarCap` live URLs.
- **Private Settings, Session & Secret Redaction:** User settings (`settings.json`, which may contain Custom XYZ/TMS `api_key` values) and workspace sessions (`session.json`) are written atomically with owner-only `0600` permissions on Unix via `write_json_durable_private()` (backed by `write_atomic_durable_private()`); `api_key` values and URL query strings are redacted in `Debug` output and error messages, and `.env*` files (except `.env.example`) are excluded by `.gitignore`.
- **Bounded Local Raster, Vector & QOI Cache Engine (Slices 3A-1 & 3A-2):** `veldmap-raster` enforces explicit `tiff::decoder::Limits`, `MAX_IFDS`, `MAX_SOURCE_SIDE` (for untiled TIFFs, PNG, and JPEG), `png::Limits` (ignoring text/iCCP chunks and checking Adam7 whole-frame peaks), `jpeg-decoder` `set_max_decoding_buffer_size` + progressive/lossless peak checks, pure-Rust TIFF `YCbCr`/`WebP` decoding, symmetric `16 MiB` QOI tile encode/decode bounds (`MAX_QOI_DECODED_BYTES`), `MemoryBudget`/`DecodeBudgetTracker` caps, positioned `ByteSource` I/O without `memmap2`, and `TwoLaneDecodePool` worker panic isolation; read-only GeoJSON vector loads enforce regular-file validation and a `64 MiB` file-size cap.

## Planned Security Controls (Upcoming Slices)

- **Credentials & Redaction (Slice 4B):** CDSE S3 keys will live in user-local `.env` files (`0600` permissions on Unix when written via the settings UI), with mandatory redaction of credential headers and SigV4 signatures in `tracing` logs and errors.
- **Untrusted Inputs & Downloads (Slices 3B, 3C-1, 4A, 4B, 6):** Extended direct raw-chunk TIFF decoding (`3B`), `Lz4Raw` disk cache deserialization (`3B`), and future JP2/NetCDF decoders (`3C-1`, `6`) will enforce explicit dimension, memory, and filter limits with panic isolation; remote range readers and downloads (`4A`, `4B`) will validate S3 path safety, HTTP `Content-Range`, and strong `ETag` integrity.
