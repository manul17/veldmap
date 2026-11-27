# Third-Party Notices

VeldMap itself is licensed under the MIT License ([LICENSE](LICENSE)).

## Rust Dependencies

Third-party Rust crate licenses are verified on every CI run using `cargo-deny` against the allow-list in [deny.toml](deny.toml). Notable bundled native, cryptographic, root-certificate, and non-MIT/Apache-only crates compiled into desktop builds include:

- **`ring` (`0.17.14`)** — `Apache-2.0 AND ISC` (TLS cryptography via `rustls`).
- **`zstd-sys` (`2.1.3+zstd.1.5.7`)** — `BSD-3-Clause AND BSD-2-Clause` (bundled Zstandard decoder via `zstd`).
- **`libsqlite3-sys` (`0.40.0`)** — `MIT` Rust wrapper bundling the **Public Domain** SQLite C amalgamation (via `rusqlite` `bundled`).
- **`webpki-root-certs` (`1.0.12`)** — `CDLA-Permissive-2.0` (Mozilla root CA bundle for `rustls`).
- **`option-ext` (`0.2.0`)** — `MPL-2.0` (file-level copyleft transitive dependency of `directories`).

## UI Fonts

No standalone font binaries are stored in this repository. The desktop UI layers two permissively licensed Rust-crate font sources:

- **`epaint_default_fonts` (`0.37.0`)** (`eframe`'s `default_fonts` feature; `monochrome_emoji_fonts` — `NotoEmoji` under `OFL-1.1` and full `emoji-icon-font` — is **not** enabled in VeldMap builds):
  - **Ubuntu-Light** (`Ubuntu-Light.ttf`) — licensed under the **Ubuntu Font Licence 1.0** (`Ubuntu-font-1.0`; compiled binary distributions embedding `Ubuntu-Light.ttf` must accompany the binary with the UFL 1.0 copyright notice and licence text from `epaint_default_fonts/fonts/UFL.txt`).
  - **Hack** (`Hack-Regular.ttf`) — licensed under the **MIT License** (Copyright 2018 Source Foundry Authors) + **DejaVu** (Public Domain) + **Bitstream Vera Fonts License** (reproduced below).
  - **egui-icons** (`egui-icons.ttf`, 5-glyph subset of `emoji-icon-font.ttf`) — licensed under the **MIT License** (Copyright 2014 John Slegers).
- **`dejavu` (`2.37.0`)** (non-Latin proportional & monospace fallback — **DejaVu Sans** and **DejaVu Sans Mono**):
  - Rust wrapper crate licensed under `MIT OR Apache-2.0`.
  - DejaVu modifications are in the Public Domain; underlying glyphs are governed by the Bitstream Vera and Arev font licenses reproduced below.

### Bitstream Vera Fonts License

Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. Bitstream Vera is a trademark of Bitstream, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of the fonts accompanying this license ("Fonts") and associated documentation files (the "Font Software"), to reproduce and distribute the Font Software, including without limitation the rights to use, copy, merge, publish, distribute, and/or sell copies of the Font Software, and to permit persons to whom the Font Software is furnished to do so, subject to the following conditions:

The above copyright and trademark notices and this permission notice shall be included in all copies of one or more of the Font Software typefaces.

The Font Software may be modified, altered, or added to, and in particular the designs of glyphs or characters in the Fonts may be modified and additional glyphs or characters may be added to the Fonts, only if the fonts are renamed to names not containing either the words "Bitstream" or the word "Vera".

This License becomes null and void to the extent applicable to Fonts or Font Software that has been modified and is distributed under the "Bitstream Vera" names.

The Font Software may be sold as part of a larger software package but no copy of one or more of the Font Software typefaces may be sold by itself.

THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL BITSTREAM OR THE GNOME FOUNDATION BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM OTHER DEALINGS IN THE FONT SOFTWARE.

Except as contained in this notice, the names of Gnome, the Gnome Foundation, and Bitstream Inc., shall not be used in advertising or otherwise to promote the sale, use or other dealings in this Font Software without prior written authorization from the Gnome Foundation or Bitstream Inc., respectively.

### Arev Fonts License

Copyright (c) 2006 by Tavmjong Bah. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of the fonts accompanying this license ("Fonts") and associated documentation files (the "Font Software"), to reproduce and distribute the modifications to the Bitstream Vera Font Software, including without limitation the rights to use, copy, merge, publish, distribute, and/or sell copies of the Font Software, and to permit persons to whom the Font Software is furnished to do so, subject to the following conditions:

The above copyright and trademark notices and this permission notice shall be included in all copies of one or more of the Font Software typefaces.

The Font Software may be modified, altered, or added to, and in particular the designs of glyphs or characters in the Fonts may be modified and additional glyphs or characters may be added to the Fonts, only if the fonts are renamed to names not containing either the words "Tavmjong Bah" or the word "Arev".

This License becomes null and void to the extent applicable to Fonts or Font Software that has been modified and is distributed under the "Tavmjong Bah Arev" names.

The Font Software may be sold as part of a larger software package but no copy of one or more of the Font Software typefaces may be sold by itself.

THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL TAVMJONG BAH BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM OTHER DEALINGS IN THE FONT SOFTWARE.

Except as contained in this notice, the name of Tavmjong Bah shall not be used in advertising or otherwise to promote the sale, use or other dealings in this Font Software without prior written authorization from Tavmjong Bah.

## Geospatial Definitions

- **`crs-definitions` (`0.6.0`)** — EPSG CRS definition table licensed under `CC0-1.0`.

## Bundled Imagery Assets

- **`assets/basemap/blue_marble_ng.jpg`** — NASA *Blue Marble Next Generation + Topography + Bathymetry* by Reto Stöckli / NASA Earth Observatory (**Public Domain**, U.S. Government work). See [assets/README.md](assets/README.md) for source provenance, dimensions, and SHA-256 checksum.
