# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in VeldMap—including credential leakage, codec memory-safety/DoS bugs (malformed GeoTIFF/JPEG 2000/NetCDF files), path traversal in downloads/caches, or TLS/signing issues—please report it responsibly rather than opening a public issue.

- Use GitHub Security Advisories ("Report a vulnerability") on the repository.
- Include reproduction steps, affected commit/version, and any crafted test input needed to verify the fix.

## Active Protections (Enforced Today)

- **Native Memory & Dependency Hygiene:** `unsafe_op_in_unsafe_fn` is denied workspace-wide, and `undocumented_unsafe_blocks`, `unwrap_used`, and `expect_used` are enforced via Clippy `-D warnings` in CI. `memmap2` (outside Wayland SHM wrappers) and C codec/TLS libraries (`openssl`, `gdal`, `proj`) are banned in [deny.toml](deny.toml).
- **Secret Exclusion:** `.env*` files (except `.env.example`) are excluded by `.gitignore` so credentials are never accidentally committed.

## Planned Security Controls (Upcoming Slices)

- **Credentials & Redaction (Slice 4B):** CDSE S3 keys will live in user-local `.env` files (`0600` permissions on Unix when written via the settings UI), with mandatory redaction of credential headers and SigV4 signatures in `tracing` logs and errors.
- **Untrusted Inputs & Downloads (Slices 3A-1, 3C-1, 4A, 4B, 6):** Pure-Rust raster decoders will enforce explicit dimension, IFD, padding, and `MemoryBudget` limits with worker panic isolation; remote range readers and downloads will validate S3 path safety, HTTP `Content-Range`, and strong `ETag` integrity.
