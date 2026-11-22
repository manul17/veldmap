# Third-Party Notices

VeldMap itself is licensed under the MIT License ([LICENSE](LICENSE)).

## Rust Dependencies

Third-party Rust crate licenses are verified on every CI run using `cargo-deny` against the allow-list in [deny.toml](deny.toml).

## UI Fonts

No standalone font binaries are stored in this repository. The desktop UI currently uses `eframe`'s `default_fonts` feature (`epaint_default_fonts`), which embeds:

- **Ubuntu-Light** — licensed under the Ubuntu Font License 1.0 (`Ubuntu-font-1.0`).
- **Hack** / **NotoEmoji** / **emoji-icon-font** / **egui-icons** — licensed under the SIL Open Font License 1.1 (`OFL-1.1`) / MIT / Apache-2.0 as packaged by `epaint_default_fonts`.

## Bundled Imagery Assets

- **`assets/basemap/blue_marble_ng.jpg`** — NASA *Blue Marble Next Generation + Topography + Bathymetry* (`3600 x 1800` global equirectangular composite by Reto Stöckli, NASA Earth Observatory, sourced unmodified from Wikimedia Commons `File:Blue_Marble_Next_Generation_+_topography_+_bathymetry.jpg`, SHA-256 `062582f953750c05c12514b9ff498b8db5b2c68e1e0563b0bea2f3385be9f5db`). **Public Domain** (U.S. Government work / NASA). See [assets/README.md](assets/README.md).
