//! Permissively licensed non-Latin UI font fallback (`dejavu` crate:
//! DejaVu Sans & DejaVu Sans Mono) layered behind `egui`'s default proportional
//! and monospace fonts so Cyrillic, Greek, Arabic, Hebrew, Armenian, Georgian,
//! and GIS symbols render without missing-glyph tofu boxes across all OSes.
//! (CJK and Indic scripts are not bundled.)

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

    /// Parse Unicode TrueType `cmap` (platform 0 / 3) format 4 / 12 tables in
    /// pure Rust to verify non-Latin glyph coverage in both bundled fonts.
    fn ttf_has_codepoint(ttf: &[u8], cp: u32) -> bool {
        if ttf.len() < 12 {
            return false;
        }
        let num_tables = u16::from_be_bytes([ttf[4], ttf[5]]) as usize;
        let cmap = (0..num_tables).find_map(|i| {
            let rec = 12 + i * 16;
            if rec + 16 <= ttf.len() && &ttf[rec..rec + 4] == b"cmap" {
                Some(
                    u32::from_be_bytes([ttf[rec + 8], ttf[rec + 9], ttf[rec + 10], ttf[rec + 11]])
                        as usize,
                )
            } else {
                None
            }
        });
        let Some(cmap) = cmap else {
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
            let platform_id = u16::from_be_bytes([ttf[rec], ttf[rec + 1]]);
            // Inspect only Unicode (0) or Windows Unicode (3) subtables.
            if platform_id != 0 && platform_id != 3 {
                continue;
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
                let delta_base = start_base + seg_count * 2;
                let range_base = delta_base + seg_count * 2;
                if range_base + seg_count * 2 <= ttf.len() {
                    for s in 0..seg_count {
                        let e =
                            u16::from_be_bytes([ttf[end_base + s * 2], ttf[end_base + s * 2 + 1]])
                                as u32;
                        let st = u16::from_be_bytes([
                            ttf[start_base + s * 2],
                            ttf[start_base + s * 2 + 1],
                        ]) as u32;
                        if st == 0xFFFF || !(st..=e).contains(&cp) {
                            continue;
                        }
                        let range_off = u16::from_be_bytes([
                            ttf[range_base + s * 2],
                            ttf[range_base + s * 2 + 1],
                        ]) as usize;
                        let glyph_id = if range_off == 0 {
                            let delta = i16::from_be_bytes([
                                ttf[delta_base + s * 2],
                                ttf[delta_base + s * 2 + 1],
                            ]);
                            (cp as i32 + i32::from(delta)) as u16
                        } else {
                            let idx_off = range_base + s * 2 + range_off + ((cp - st) as usize) * 2;
                            if idx_off + 2 > ttf.len() {
                                continue;
                            }
                            u16::from_be_bytes([ttf[idx_off], ttf[idx_off + 1]])
                        };
                        if glyph_id != 0 {
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
    fn fallback_fonts_are_registered_and_cover_non_latin_scripts_and_gis_symbols() {
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

        let common_codepoints: &[u32] = &[
            0x0410, 0x042F, 0x0430, 0x044F, 0x0401, 0x0451, // Basic Cyrillic
            0x049A, 0x0407, 0x0457, // Extended Cyrillic (Kazakh/Ukrainian)
            0x03A9, 0x03B1, // Greek
            0x00B0, 0x00B5, 0x00B2, 0x0394, // GIS symbols (deg, micro, squared, Delta)
        ];
        let sans = dejavu::sans::regular();
        let mono = dejavu::sans_mono::regular();
        for &cp in common_codepoints {
            assert!(
                ttf_has_codepoint(sans, cp),
                "DejaVu Sans missing expected codepoint U+{cp:04X}"
            );
            assert!(
                ttf_has_codepoint(mono, cp),
                "DejaVu Sans Mono missing expected codepoint U+{cp:04X}"
            );
        }
        for &cp in &[0x0639, 0x05D0, 0x0531, 0x10D0] {
            assert!(
                ttf_has_codepoint(sans, cp),
                "DejaVu Sans missing expected codepoint U+{cp:04X}"
            );
        }
    }
}
