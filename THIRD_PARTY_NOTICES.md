# Third-Party Notices

VeldMap itself is licensed under the MIT License ([LICENSE](LICENSE)).

## Rust Dependencies

Third-party Rust crate licenses are verified on every CI run using `cargo-deny` against the allow-list in [deny.toml](deny.toml).

## UI Fonts

No standalone font binaries are stored in this repository. The desktop UI layers two permissively licensed Rust-crate font sources:

- **`epaint_default_fonts`** (`eframe`'s `default_fonts` feature):
  - **Ubuntu-Light** — licensed under the Ubuntu Font License 1.0 (`Ubuntu-font-1.0`).
  - **Hack** / **NotoEmoji** / **emoji-icon-font** / **egui-icons** — licensed under the SIL Open Font License 1.1 (`OFL-1.1`) / MIT / Apache-2.0 as packaged by `epaint_default_fonts`.
- **`dejavu` (`2.37.0`)** (non-Latin proportional & monospace fallback — **DejaVu Sans** and **DejaVu Sans Mono**):
  - Rust wrapper crate licensed under `MIT OR Apache-2.0`.
  - Bitstream Vera Fonts Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved (permissive Bitstream Vera license); DejaVu changes are in the Public Domain.

## Geospatial Definitions

- **`crs-definitions` (`0.6.0`)** — EPSG CRS definition table licensed under `CC0-1.0`.

## Bundled Imagery Assets

- **`assets/basemap/blue_marble_ng.jpg`** — NASA *Blue Marble Next Generation + Topography + Bathymetry* by Reto Stöckli / NASA Earth Observatory (**Public Domain**, U.S. Government work). See [assets/README.md](assets/README.md) for source provenance, dimensions, and SHA-256 checksum.
