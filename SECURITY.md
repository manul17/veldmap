# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in VeldMap—including credential leakage, codec memory-safety/DoS bugs (malformed GeoTIFF/JPEG 2000/NetCDF files), path traversal in downloads/caches, or TLS/signing issues—please report it responsibly rather than opening a public issue.

- Use GitHub Security Advisories ("Report a vulnerability") on the repository, or contact the maintainer directly.
- Include reproduction steps, affected commit/version, and any crafted test input needed to verify the fix.

## Security & Privacy Practices

- **Credentials:** CDSE S3 keys live in user-local `.env` files (git-ignored; `0600` permissions on Unix when created via settings) and are never committed or logged. Credential headers and SigV4 signatures are redacted from logs and error messages.
- **Native Memory Safety:** `unsafe_op_in_unsafe_fn` is denied workspace-wide and `undocumented_unsafe_blocks` is enforced in CI. `memmap2` and C codec/TLS libraries (`openssl`, `gdal`, `proj`) are banned via `deny.toml`.
- **Untrusted Inputs:** Raster decoders enforce explicit dimension, IFD, padding, and memory-budget limits; remote downloads validate S3 path safety, HTTP `Content-Range`, and `ETag` integrity.
