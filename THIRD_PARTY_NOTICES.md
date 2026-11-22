# Third-Party Notices

VeldMap itself is licensed under the MIT License ([LICENSE](LICENSE)).

## Rust Dependencies

Third-party Rust crate licenses are verified on every CI run using `cargo-deny` against the allow-list in [deny.toml](deny.toml).

## UI Fonts

No standalone font binaries are stored in this repository. The desktop UI currently uses `eframe`'s `default_fonts` feature (`epaint_default_fonts`), which embeds:

- **Ubuntu-Light** — licensed under the Ubuntu Font License 1.0 (`Ubuntu-font-1.0`).
- **Hack** / **NotoEmoji** / **emoji-icon-font** / **egui-icons** — licensed under the SIL Open Font License 1.1 (`OFL-1.1`) / MIT / Apache-2.0 as packaged by `epaint_default_fonts`.

## Bundled Imagery Assets

- **`assets/basemap/blue_marble_ng.jpg`** — NASA *Blue Marble Next Generation + Topography + Bathymetry* by Reto Stöckli / NASA Earth Observatory (**Public Domain**, U.S. Government work). See [assets/README.md](assets/README.md) for source provenance, dimensions, and SHA-256 checksum.
