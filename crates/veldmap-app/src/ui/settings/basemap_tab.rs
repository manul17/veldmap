//! Basemap tab inside the unified Settings modal (NASA GIBS daily date and
//! Custom XYZ / TMS tile source configuration).

use eframe::egui;

use crate::storage::settings::{
    AppSettings, BasemapSelection, CUSTOM_MAX_LEVEL_RANGE, validate_gibs_date,
};
use crate::ui::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR};

use super::SettingsWindowState;

fn show_gibs_section(ui: &mut egui::Ui, state: &mut SettingsWindowState) {
    ui.heading("NASA GIBS Daily Imagery");
    ui.horizontal(|ui| {
        ui.label("VIIRS Date (UTC YYYY-MM-DD):");
        ui.add(
            egui::TextEdit::singleline(&mut state.draft.gibs_date)
                .desired_width(110.0)
                .hint_text("2024-07-01"),
        );
    });
    if !validate_gibs_date(&state.draft.gibs_date) {
        ui.label(
            egui::RichText::new("Enter a valid YYYY-MM-DD date (e.g. 2024-07-01).")
                .color(WARN_TEXT_COLOR)
                .small(),
        );
    }
}

fn show_custom_xyz_section(
    ui: &mut egui::Ui,
    settings: &AppSettings,
    state: &mut SettingsWindowState,
) {
    ui.heading("Custom XYZ / TMS Tile Source");
    if state.draft.active_basemap == BasemapSelection::CustomXyz
        && settings.active_basemap != BasemapSelection::CustomXyz
    {
        ui.label(
            egui::RichText::new(
                "Enter a valid URL template below, then click Apply & Save or Use Custom XYZ.",
            )
            .weak()
            .small(),
        );
    }
    egui::Grid::new("custom_xyz_grid")
        .num_columns(2)
        .spacing([8.0, 6.0])
        .show(ui, |ui| {
            ui.label("Name:");
            ui.add(
                egui::TextEdit::singleline(&mut state.draft.custom_xyz.name).desired_width(260.0),
            );
            ui.end_row();

            ui.label("URL template:");
            ui.add(
                egui::TextEdit::singleline(&mut state.draft.custom_xyz.url_template)
                    .desired_width(260.0)
                    .hint_text("https://{s}.host/{z}/{x}/{y}.png?key={key}"),
            );
            ui.end_row();

            ui.label("API key (optional):");
            ui.add(
                egui::TextEdit::singleline(&mut state.draft.custom_xyz.api_key)
                    .password(true)
                    .desired_width(260.0)
                    .hint_text("Substituted for {key} / {api_key}"),
            );
            ui.end_row();

            ui.label("Attribution:");
            ui.add(
                egui::TextEdit::singleline(&mut state.draft.custom_xyz.attribution)
                    .desired_width(260.0)
                    .hint_text("Data provider credit"),
            );
            ui.end_row();

            ui.label("Max zoom:");
            ui.add(egui::Slider::new(
                &mut state.draft.custom_xyz.max_level,
                CUSTOM_MAX_LEVEL_RANGE,
            ));
            ui.end_row();
        });

    ui.checkbox(
        &mut state.draft.custom_xyz.tms_flip_y,
        "TMS Y-axis flip (invert {y} from south-up)",
    );

    let custom_err = state.custom_validation().err().map(str::to_owned);
    if let Some(err) = &custom_err
        && (state.draft.active_basemap == BasemapSelection::CustomXyz
            || !state.draft.custom_xyz.url_template.trim().is_empty())
    {
        ui.label(egui::RichText::new(err).color(ERROR_TEXT_COLOR).small());
    }
}

pub(super) fn show_basemap_tab(
    ui: &mut egui::Ui,
    settings: &AppSettings,
    state: &mut SettingsWindowState,
) {
    show_gibs_section(ui, state);
    ui.separator();
    show_custom_xyz_section(ui, settings, state);
}
