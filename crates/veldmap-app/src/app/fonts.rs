//! Permissively licensed non-Latin UI font fallback (`dejavu` crate:
//! DejaVu Sans & DejaVu Sans Mono) layered behind `egui`'s default proportional
//! and monospace fonts so Cyrillic, Greek, Arabic, Hebrew, Armenian, Georgian,
//! and GIS symbols render without missing-glyph tofu boxes across all OSes.
//! (CJK and Indic scripts are not bundled.)

use std::sync::Arc;

use eframe::egui;

/// Font registry key for the bundled DejaVu Sans proportional fallback.
const FALLBACK_PROPORTIONAL_FONT: &str = "dejavu_sans";

/// Font registry key for the bundled DejaVu Sans Mono monospace fallback.
const FALLBACK_MONOSPACE_FONT: &str = "dejavu_sans_mono";

/// Build `egui::FontDefinitions` keeping `eframe`'s built-in default fonts
/// first and appending DejaVu Sans / DejaVu Sans Mono as non-Latin fallbacks.
pub(super) fn build_font_definitions() -> egui::FontDefinitions {
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
#[path = "fonts_tests.rs"]
mod tests;
