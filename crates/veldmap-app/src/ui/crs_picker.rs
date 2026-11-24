//! Floating searchable CRS Picker & Custom Transverse Mercator (MSK-xx)
//! window for overriding or assigning raster coordinate reference systems.

use std::path::PathBuf;

use eframe::egui;
use veldmap_core::geo::{CrsSpec, CustomTmParams, DatumPreset, LinearUnit, search_crs_presets};

use crate::layers::LayerId;
use crate::session::{SavedCrsOverride, SavedLinearUnit};

use super::ERROR_TEXT_COLOR;

/// Target that the CRS Picker is configuring.
#[derive(Clone, Debug)]
pub(crate) enum CrsPickerTarget {
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
#[derive(Clone, Debug)]
pub(crate) struct CrsPickerState {
    pub(crate) open: bool,
    target: Option<CrsPickerTarget>,
    tab: CrsTab,
    search_query: String,
    custom_tm: CustomTmParams,
    wkt_input: String,
    error: Option<String>,
}

impl Default for CrsPickerState {
    fn default() -> Self {
        Self {
            open: false,
            target: None,
            tab: CrsTab::Presets,
            search_query: String::new(),
            custom_tm: CustomTmParams::default(),
            wkt_input: String::new(),
            error: None,
        }
    }
}

impl CrsPickerState {
    pub(crate) fn open_for_layer(&mut self, id: LayerId, name: String) {
        self.target = Some(CrsPickerTarget::Layer { id, name });
        self.error = None;
        self.open = true;
    }

    pub(crate) fn open_for_pending_path(&mut self, path: PathBuf) {
        self.target = Some(CrsPickerTarget::PendingPath(path));
        self.error = None;
        self.open = true;
    }

    fn emit_selection(&mut self, selection: SavedCrsOverride) -> Option<CrsPickerAction> {
        if let Err(err) = selection.to_crs_spec() {
            self.error = Some(err);
            return None;
        }
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
        .open(&mut open)
        .resizable(true)
        .default_width(430.0)
        .default_height(380.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut state.tab, CrsTab::Presets, "EPSG Presets");
                ui.selectable_value(&mut state.tab, CrsTab::CustomTm, "Custom TM (MSK-xx)");
                ui.selectable_value(&mut state.tab, CrsTab::WktProj, "WKT / PROJ");
            });
            ui.separator();

            match state.tab {
                CrsTab::Presets => {
                    ui.horizontal(|ui| {
                        ui.label("Search:");
                        ui.add(
                            egui::TextEdit::singleline(&mut state.search_query)
                                .hint_text("EPSG code or name (e.g. 32637, 28407, MSK, 27700)"),
                        );
                    });
                    ui.add_space(4.0);
                    let presets = search_crs_presets(&state.search_query, 32);
                    egui::ScrollArea::vertical()
                        .max_height(240.0)
                        .show(ui, |ui| {
                            if presets.is_empty() {
                                ui.weak("No matching EPSG presets.");
                            }
                            for preset in presets {
                                ui.horizontal(|ui| {
                                    if ui
                                        .button(format!("EPSG:{}", preset.epsg))
                                        .clicked()
                                    {
                                        action = state.emit_selection(SavedCrsOverride::Epsg {
                                            code: preset.epsg,
                                        });
                                    }
                                    ui.label(&preset.name);
                                });
                            }
                        });
                }
                CrsTab::CustomTm => {
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
                                    .range(0.5..=1.5)
                                    .max_decimals(6),
                            );
                            ui.end_row();

                            ui.label("False Easting (m):");
                            ui.add(
                                egui::DragValue::new(&mut state.custom_tm.false_easting_m)
                                    .speed(100.0)
                                    .max_decimals(3),
                            );
                            ui.end_row();

                            ui.label("False Northing (m):");
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
                                        ui.selectable_value(
                                            &mut state.custom_tm.unit,
                                            unit,
                                            unit.label(),
                                        );
                                    }
                                });
                            ui.end_row();
                        });
                    ui.add_space(8.0);
                    if ui.button("Apply Custom TM").clicked() {
                        let p = &state.custom_tm;
                        let sel = SavedCrsOverride::CustomTm {
                            name: p.name.clone(),
                            pulkovo42: p.datum == DatumPreset::Pulkovo1942,
                            central_meridian_deg: p.central_meridian_deg,
                            origin_lat_deg: p.origin_lat_deg,
                            scale: p.scale,
                            false_easting_m: p.false_easting_m,
                            false_northing_m: p.false_northing_m,
                            unit: SavedLinearUnit::from(p.unit),
                        };
                        action = state.emit_selection(sel);
                    }
                }
                CrsTab::WktProj => {
                    ui.label("Paste WKT1 / WKT2 or PROJ.4 string:");
                    ui.add(
                        egui::TextEdit::multiline(&mut state.wkt_input)
                            .desired_rows(7)
                            .desired_width(f32::INFINITY)
                            .hint_text("+proj=tmerc +lat_0=0 +lon_0=39 +k=1 +x_0=1300000 +y_0=-4811057.628 ..."),
                    );
                    ui.add_space(6.0);
                    if ui.button("Apply WKT / PROJ").clicked() {
                        let raw = state.wkt_input.trim().to_owned();
                        let sel = if CrsSpec::from_wkt(&raw).is_ok() {
                            SavedCrsOverride::Wkt { wkt: raw }
                        } else {
                            SavedCrsOverride::Proj { proj: raw }
                        };
                        action = state.emit_selection(sel);
                    }
                }
            }

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
