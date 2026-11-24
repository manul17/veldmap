//! Permissively licensed non-Latin UI font fallback (`dejavu` crate:
//! DejaVu Sans & DejaVu Sans Mono) layered behind `egui`'s default proportional
//! and monospace fonts so international / Cyrillic / Pulkovo / MSK-xx file
//! paths and metadata render without missing-glyph tofu boxes across all OSes.

use std::sync::Arc;

use eframe::egui;

/// Font registry key for the bundled DejaVu Sans proportional fallback.
pub const FALLBACK_PROPORTIONAL_FONT: &str = "dejavu_sans";

/// Font registry key for the bundled DejaVu Sans Mono monospace fallback.
pub const FALLBACK_MONOSPACE_FONT: &str = "dejavu_sans_mono";

/// Build `egui::FontDefinitions` keeping `eframe`'s built-in default fonts
/// first and appending DejaVu Sans / DejaVu Sans Mono as non-Latin fallbacks.
pub fn build_font_definitions() -> egui::FontDefinitions {
    let mut fonts = egui::FontDefinitions::default();

    fonts.font_data.insert(
        FALLBACK_PROPORTIONAL_FONT.to_owned(),
        Arc::new(egui::FontData::from_static(dejavu::sans::regular())),
    );
    fonts.font_data.insert(
        FALLBACK_MONOSPACE_FONT.to_owned(),
        Arc::new(egui::FontData::from_static(dejavu::sans_mono::regular())),
    );

    // Insert right after the primary default UI font (index 1 if present) so
    // non-Latin glyphs resolve via DejaVu before emoji/icon fallback fonts.
    if let Some(prop) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
        let pos = usize::from(!prop.is_empty());
        prop.insert(pos, FALLBACK_PROPORTIONAL_FONT.to_owned());
    }
    if let Some(mono) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
        let pos = usize::from(!mono.is_empty());
        mono.insert(pos, FALLBACK_MONOSPACE_FONT.to_owned());
    }

    fonts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse TrueType `cmap` format 4 / 12 tables directly in pure Rust to
    /// verify glyph coverage without extra test dependencies.
    fn ttf_has_codepoint(ttf: &[u8], cp: u32) -> bool {
        if ttf.len() < 12 {
            return false;
        }
        let num_tables = u16::from_be_bytes([ttf[4], ttf[5]]) as usize;
        let mut cmap_off = None;
        for i in 0..num_tables {
            let rec = 12 + i * 16;
            if rec + 16 > ttf.len() {
                break;
            }
            if &ttf[rec..rec + 4] == b"cmap" {
                cmap_off = Some(u32::from_be_bytes([
                    ttf[rec + 8],
                    ttf[rec + 9],
                    ttf[rec + 10],
                    ttf[rec + 11],
                ]) as usize);
                break;
            }
        }
        let Some(cmap) = cmap_off else {
            return false;
        };
        if cmap + 4 > ttf.len() {
            return false;
        }
        let num_sub = u16::from_be_bytes([ttf[cmap + 2], ttf[cmap + 3]]) as usize;
        for i in 0..num_sub {
            let rec = cmap + 4 + i * 8;
            if rec + 8 > ttf.len() {
                break;
            }
            let sub_off =
                u32::from_be_bytes([ttf[rec + 4], ttf[rec + 5], ttf[rec + 6], ttf[rec + 7]])
                    as usize;
            let off = cmap + sub_off;
            if off + 2 > ttf.len() {
                continue;
            }
            let fmt = u16::from_be_bytes([ttf[off], ttf[off + 1]]);
            if fmt == 4 && cp <= 0xFFFF && off + 14 <= ttf.len() {
                let seg_count = u16::from_be_bytes([ttf[off + 6], ttf[off + 7]]) as usize / 2;
                let end_base = off + 14;
                let start_base = end_base + seg_count * 2 + 2;
                if start_base + seg_count * 2 <= ttf.len() {
                    for s in 0..seg_count {
                        let e =
                            u16::from_be_bytes([ttf[end_base + s * 2], ttf[end_base + s * 2 + 1]])
                                as u32;
                        let st = u16::from_be_bytes([
                            ttf[start_base + s * 2],
                            ttf[start_base + s * 2 + 1],
                        ]) as u32;
                        if st != 0xFFFF && (st..=e).contains(&cp) {
                            return true;
                        }
                    }
                }
            } else if fmt == 12 && off + 16 <= ttf.len() {
                let n_groups = u32::from_be_bytes([
                    ttf[off + 12],
                    ttf[off + 13],
                    ttf[off + 14],
                    ttf[off + 15],
                ]) as usize;
                for g in 0..n_groups {
                    let g_off = off + 16 + g * 12;
                    if g_off + 12 > ttf.len() {
                        break;
                    }
                    let st = u32::from_be_bytes([
                        ttf[g_off],
                        ttf[g_off + 1],
                        ttf[g_off + 2],
                        ttf[g_off + 3],
                    ]);
                    let e = u32::from_be_bytes([
                        ttf[g_off + 4],
                        ttf[g_off + 5],
                        ttf[g_off + 6],
                        ttf[g_off + 7],
                    ]);
                    if (st..=e).contains(&cp) {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[test]
    fn fallback_fonts_are_registered_and_cover_cyrillic_and_international_scripts() {
        let defs = build_font_definitions();
        assert!(defs.font_data.contains_key(FALLBACK_PROPORTIONAL_FONT));
        assert!(defs.font_data.contains_key(FALLBACK_MONOSPACE_FONT));
        assert!(
            defs.families[&egui::FontFamily::Proportional]
                .contains(&FALLBACK_PROPORTIONAL_FONT.to_owned())
        );
        assert!(
            defs.families[&egui::FontFamily::Monospace]
                .contains(&FALLBACK_MONOSPACE_FONT.to_owned())
        );

        // Strictly ASCII-escaped codepoints (English-only CI check compliance):
        // Cyrillic, Extended Cyrillic (Kazakh/Ukrainian), Greek, Arabic, Hebrew,
        // Armenian, Georgian, and GIS/math symbols (deg, micro, squared, Delta).
        let required_codepoints: &[u32] = &[
            0x0410, 0x042F, 0x0430, 0x044F, 0x0401, 0x0451, // Basic Cyrillic
            0x049A, 0x0407, 0x0457, // Extended Cyrillic (Kazakh/Ukrainian)
            0x03A9, 0x03B1, // Greek
            0x0639, // Arabic
            0x05D0, // Hebrew
            0x0531, // Armenian
            0x10D0, // Georgian
            0x00B0, 0x00B5, 0x00B2, 0x0394, // GIS symbols
        ];
        let sans = dejavu::sans::regular();
        for &cp in required_codepoints {
            assert!(
                ttf_has_codepoint(sans, cp),
                "DejaVu Sans missing expected codepoint U+{cp:04X}"
            );
        }
    }
}
