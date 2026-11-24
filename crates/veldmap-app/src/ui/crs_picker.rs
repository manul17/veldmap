//! Floating searchable CRS Picker & Custom Transverse Mercator (MSK-xx)
//! window for overriding or assigning raster coordinate reference systems.

use std::path::PathBuf;

use eframe::egui;
use veldmap_core::geo::{
    CrsPreset, CrsSpec, CrsTextKind, CustomTmParams, DatumPreset, LinearUnit, builtin_crs_presets,
    search_crs_presets,
};

use crate::layers::LayerId;
use crate::session::{SavedCrsOverride, SavedCustomTmParams};

use super::ERROR_TEXT_COLOR;

const MAX_VISIBLE_PRESETS: usize = 48;

/// Target that the CRS Picker is configuring.
#[derive(Clone, Debug)]
enum CrsPickerTarget {
    /// Rebind an already loaded raster layer.
    Layer { id: LayerId, name: String },
    /// Re-open a file (e.g. world-file PNG/JPEG without `.prj`) using the
    /// selected CRS.
    PendingPath(PathBuf),
}

/// Action emitted by the CRS Picker window and applied by `VeldmapApp`.
#[derive(Clone, Debug)]
pub(crate) enum CrsPickerAction {
    ApplyToLayer(LayerId, Option<SavedCrsOverride>),
    OpenPathWithCrs(PathBuf, SavedCrsOverride),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum CrsTab {
    #[default]
    Presets,
    CustomTm,
    WktProj,
}

/// Transient UI state for the CRS Picker window.
#[derive(Clone, Debug, Default)]
pub(crate) struct CrsPickerState {
    open: bool,
    target: Option<CrsPickerTarget>,
    tab: CrsTab,
    search_query: String,
    cached_query: Option<String>,
    cached_presets: Vec<CrsPreset>,
    custom_tm: CustomTmParams,
    wkt_input: String,
    error: Option<String>,
}

impl CrsPickerState {
    fn reset_form_from(&mut self, existing: Option<&SavedCrsOverride>) {
        self.error = None;
        self.search_query.clear();
        self.cached_query = None;
        self.custom_tm = CustomTmParams::default();
        self.wkt_input.clear();
        self.tab = CrsTab::Presets;

        if let Some(ov) = existing {
            match ov {
                SavedCrsOverride::Epsg { code } => {
                    self.search_query = code.to_string();
                    self.tab = CrsTab::Presets;
                }
                SavedCrsOverride::Wkt { wkt } => {
                    self.wkt_input = wkt.clone();
                    self.tab = CrsTab::WktProj;
                }
                SavedCrsOverride::Proj { proj } => {
                    self.wkt_input = proj.clone();
                    self.tab = CrsTab::WktProj;
                }
                SavedCrsOverride::CustomTm(saved) => {
                    self.custom_tm = CustomTmParams::from(saved);
                    self.tab = CrsTab::CustomTm;
                }
            }
        }
    }

    pub(crate) fn open_for_layer(
        &mut self,
        id: LayerId,
        name: String,
        existing: Option<SavedCrsOverride>,
    ) {
        self.target = Some(CrsPickerTarget::Layer { id, name });
        self.reset_form_from(existing.as_ref());
        self.open = true;
    }

    pub(crate) fn open_for_pending_path(&mut self, path: PathBuf) {
        self.target = Some(CrsPickerTarget::PendingPath(path));
        self.reset_form_from(None);
        self.open = true;
    }

    fn ensure_presets_cached(&mut self) {
        if self.cached_query.as_deref() != Some(self.search_query.as_str()) {
            self.cached_presets = search_crs_presets(&self.search_query, MAX_VISIBLE_PRESETS);
            self.cached_query = Some(self.search_query.clone());
        }
    }

    fn finish_selection(&mut self, selection: SavedCrsOverride) -> Option<CrsPickerAction> {
        self.error = None;
        let target = self.target.clone()?;
        self.open = false;
        match target {
            CrsPickerTarget::Layer { id, .. } => {
                Some(CrsPickerAction::ApplyToLayer(id, Some(selection)))
            }
            CrsPickerTarget::PendingPath(path) => {
                Some(CrsPickerAction::OpenPathWithCrs(path, selection))
            }
        }
    }

    fn emit_selection(&mut self, selection: SavedCrsOverride) -> Option<CrsPickerAction> {
        if let Err(err) = selection.to_crs_spec() {
            self.error = Some(err);
            return None;
        }
        self.finish_selection(selection)
    }
}

fn show_presets_tab(ui: &mut egui::Ui, state: &mut CrsPickerState) -> Option<CrsPickerAction> {
    ui.horizontal(|ui| {
        ui.label("Search:");
        ui.add(
            egui::TextEdit::singleline(&mut state.search_query)
                .hint_text("EPSG code or name (e.g. 32637, 28407, MSK, 27700)"),
        );
    });
    state.ensure_presets_cached();
    let total_presets = builtin_crs_presets().len();
    if state.search_query.trim().is_empty() && state.cached_presets.len() < total_presets {
        ui.weak(
            egui::RichText::new(format!(
                "Showing {} of {} curated presets — type to filter or enter any EPSG code.",
                state.cached_presets.len(),
                total_presets
            ))
            .small(),
        );
    }
    ui.add_space(4.0);
    let mut picked_epsg: Option<u32> = None;
    egui::ScrollArea::vertical()
        .max_height(240.0)
        .show(ui, |ui| {
            if state.cached_presets.is_empty() {
                ui.weak("No matching EPSG presets.");
            }
            for preset in &state.cached_presets {
                ui.horizontal(|ui| {
                    if ui.button(preset.code_label()).clicked() {
                        picked_epsg = Some(preset.epsg);
                    }
                    ui.label(&preset.name);
                });
            }
        });
    picked_epsg.and_then(|code| state.emit_selection(SavedCrsOverride::Epsg { code }))
}

fn show_custom_tm_tab(ui: &mut egui::Ui, state: &mut CrsPickerState) -> Option<CrsPickerAction> {
    egui::Grid::new("custom_tm_grid")
        .num_columns(2)
        .spacing([10.0, 6.0])
        .show(ui, |ui| {
            ui.label("Name:");
            ui.text_edit_singleline(&mut state.custom_tm.name);
            ui.end_row();

            ui.label("Datum:");
            egui::ComboBox::from_id_salt("custom_tm_datum")
                .selected_text(state.custom_tm.datum.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.custom_tm.datum,
                        DatumPreset::Wgs84,
                        DatumPreset::Wgs84.label(),
                    );
                    ui.selectable_value(
                        &mut state.custom_tm.datum,
                        DatumPreset::Pulkovo1942,
                        DatumPreset::Pulkovo1942.label(),
                    );
                });
            ui.end_row();

            ui.label("Central Meridian (deg):");
            ui.add(
                egui::DragValue::new(&mut state.custom_tm.central_meridian_deg)
                    .speed(0.05)
                    .range(-180.0..=180.0)
                    .max_decimals(8),
            );
            ui.end_row();

            ui.label("Origin Latitude (deg):");
            ui.add(
                egui::DragValue::new(&mut state.custom_tm.origin_lat_deg)
                    .speed(0.05)
                    .range(-90.0..=90.0)
                    .max_decimals(8),
            );
            ui.end_row();

            ui.label("Scale Factor (k0):");
            ui.add(
                egui::DragValue::new(&mut state.custom_tm.scale)
                    .speed(0.0001)
                    .range(0.0001..=2.0)
                    .max_decimals(6),
            );
            ui.end_row();

            ui.label("False Easting (m):")
                .on_hover_text("Always specified in metres, even when Linear Unit is feet.");
            ui.add(
                egui::DragValue::new(&mut state.custom_tm.false_easting_m)
                    .speed(100.0)
                    .max_decimals(3),
            );
            ui.end_row();

            ui.label("False Northing (m):")
                .on_hover_text("Always specified in metres, even when Linear Unit is feet.");
            ui.add(
                egui::DragValue::new(&mut state.custom_tm.false_northing_m)
                    .speed(100.0)
                    .max_decimals(3),
            );
            ui.end_row();

            ui.label("Linear Unit:");
            egui::ComboBox::from_id_salt("custom_tm_unit")
                .selected_text(state.custom_tm.unit.label())
                .show_ui(ui, |ui| {
                    for unit in [
                        LinearUnit::Metre,
                        LinearUnit::InternationalFoot,
                        LinearUnit::UsSurveyFoot,
                    ] {
                        ui.selectable_value(&mut state.custom_tm.unit, unit, unit.label());
                    }
                });
            ui.end_row();
        });
    ui.add_space(8.0);
    if ui.button("Apply Custom TM").clicked() {
        let sel = SavedCrsOverride::CustomTm(SavedCustomTmParams::from(&state.custom_tm));
        return state.emit_selection(sel);
    }
    None
}

fn show_wkt_proj_tab(ui: &mut egui::Ui, state: &mut CrsPickerState) -> Option<CrsPickerAction> {
    ui.label("Paste WKT1 / WKT2 or PROJ.4 string:");
    ui.add(
        egui::TextEdit::multiline(&mut state.wkt_input)
            .desired_rows(7)
            .desired_width(f32::INFINITY)
            .hint_text(
                "+proj=tmerc +lat_0=0 +lon_0=37.98333333333 +k=1 +x_0=1300000 +y_0=-4811057.628 ...",
            ),
    );
    ui.add_space(6.0);
    if ui.button("Apply WKT / PROJ").clicked() {
        let raw = state
            .wkt_input
            .trim()
            .trim_start_matches('\u{FEFF}')
            .trim()
            .to_owned();
        match CrsSpec::parse_text(&raw) {
            Ok((_, kind)) => {
                let sel = match kind {
                    CrsTextKind::Wkt => SavedCrsOverride::Wkt { wkt: raw },
                    CrsTextKind::UserString => SavedCrsOverride::Proj { proj: raw },
                };
                return state.finish_selection(sel);
            }
            Err(err) => {
                state.error = Some(err.to_string());
            }
        }
    }
    None
}

/// Render the CRS Picker window if open and return any resulting action.
pub(crate) fn show_crs_picker_window(
    ctx: &egui::Context,
    state: &mut CrsPickerState,
) -> Option<CrsPickerAction> {
    if !state.open || state.target.is_none() {
        return None;
    }
    let mut action: Option<CrsPickerAction> = None;
    let mut open = state.open;
    let title = match &state.target {
        Some(CrsPickerTarget::Layer { name, .. }) => format!("CRS Override — {name}"),
        Some(CrsPickerTarget::PendingPath(p)) => {
            let fname = p.file_name().and_then(|n| n.to_str()).unwrap_or("raster");
            format!("Assign CRS — {fname}")
        }
        None => "CRS Picker".to_owned(),
    };

    egui::Window::new(title)
        .id(egui::Id::unique("crs_picker_window"))
        .open(&mut open)
        .resizable(true)
        .default_width(440.0)
        .default_height(390.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut state.tab, CrsTab::Presets, "EPSG Presets");
                ui.selectable_value(&mut state.tab, CrsTab::CustomTm, "Custom TM (MSK-xx)");
                ui.selectable_value(&mut state.tab, CrsTab::WktProj, "WKT / PROJ");
            });
            ui.separator();

            action = match state.tab {
                CrsTab::Presets => show_presets_tab(ui, state),
                CrsTab::CustomTm => show_custom_tm_tab(ui, state),
                CrsTab::WktProj => show_wkt_proj_tab(ui, state),
            };

            if let Some(err) = &state.error {
                ui.separator();
                ui.colored_label(ERROR_TEXT_COLOR, err);
            }

            if let Some(CrsPickerTarget::Layer { id, .. }) = &state.target {
                let id = *id;
                ui.separator();
                if ui.button("Reset to Embedded / Sidecar CRS").clicked() {
                    state.open = false;
                    action = Some(CrsPickerAction::ApplyToLayer(id, None));
                }
            }
        });

    state.open = open && state.open;
    action
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{SavedCustomTmDatum, SavedLinearUnit};

    #[test]
    fn reset_form_from_selects_matching_tab_and_populates_fields() {
        let mut state = CrsPickerState::default();

        state.reset_form_from(Some(&SavedCrsOverride::Epsg { code: 32637 }));
        assert_eq!(state.tab, CrsTab::Presets);
        assert_eq!(state.search_query, "32637");

        state.reset_form_from(Some(&SavedCrsOverride::Wkt {
            wkt: "GEOGCS[\"WGS 84\"]".to_owned(),
        }));
        assert_eq!(state.tab, CrsTab::WktProj);
        assert_eq!(state.wkt_input, "GEOGCS[\"WGS 84\"]");

        state.reset_form_from(Some(&SavedCrsOverride::Proj {
            proj: "+proj=utm +zone=37 +datum=WGS84".to_owned(),
        }));
        assert_eq!(state.tab, CrsTab::WktProj);
        assert_eq!(state.wkt_input, "+proj=utm +zone=37 +datum=WGS84");

        state.reset_form_from(Some(&SavedCrsOverride::CustomTm(SavedCustomTmParams {
            name: "MSK-61 Z1".to_owned(),
            datum: SavedCustomTmDatum::Pulkovo1942,
            central_meridian_deg: 39.0,
            origin_lat_deg: 0.0,
            scale: 1.0,
            false_easting_m: 1_300_000.0,
            false_northing_m: 0.0,
            unit: SavedLinearUnit::Metre,
        })));
        assert_eq!(state.tab, CrsTab::CustomTm);
        assert_eq!(state.custom_tm.name, "MSK-61 Z1");
        assert_eq!(state.custom_tm.datum, DatumPreset::Pulkovo1942);
    }
}
