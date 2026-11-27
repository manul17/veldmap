//! STAC catalog preset selector, temporal/cloud/limit filters, search trigger,
//! and error banner inside the Cloud / STAC panel.

use eframe::egui;

use crate::stac::{StacCatalogPreset, StacSearchRequest, StacViewSnapshot};
use crate::ui::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR};

use super::{StacUiAction, StacUiState};

pub(super) fn show_search_form(
    ui: &mut egui::Ui,
    stac: &StacViewSnapshot<'_>,
    state: &mut StacUiState,
    actions: &mut Vec<StacUiAction>,
) {
    ui.strong("STAC Catalog Search");
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("stac_catalog_preset")
            .selected_text(state.preset.label())
            .width(ui.available_width() - 60.0)
            .show_ui(ui, |ui| {
                for preset in StacCatalogPreset::ALL {
                    ui.selectable_value(&mut state.preset, preset, preset.label());
                }
            });
        if state.preset == StacCatalogPreset::Custom && ui.small_button("Edit…").clicked() {
            actions.push(StacUiAction::OpenStacSettings);
        }
    });

    if state.preset == StacCatalogPreset::Custom && !stac.custom_endpoint_configured {
        ui.colored_label(
            WARN_TEXT_COLOR,
            egui::RichText::new("Configure a Custom STAC endpoint in Settings → STAC / Cloud.")
                .small(),
        );
    }

    if state.preset.supports_temporal_and_cloud_filters() {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Dates:").small());
            ui.add(
                egui::TextEdit::singleline(&mut state.datetime_input)
                    .desired_width(ui.available_width())
                    .hint_text("YYYY-MM-DD/YYYY-MM-DD (optional)"),
            );
        });

        ui.horizontal(|ui| {
            ui.checkbox(&mut state.filter_clouds, "Max cloud %:");
            ui.add_enabled(
                state.filter_clouds,
                egui::Slider::new(&mut state.max_cloud_cover, 0.0..=100.0).fixed_decimals(0),
            );
        });
    }

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Limit:").small());
        ui.add(egui::Slider::new(&mut state.limit, 1..=50));
    });

    ui.horizontal(|ui| {
        if stac.is_searching {
            if ui.button("Cancel Search").clicked() {
                actions.push(StacUiAction::CancelSearch);
            }
            ui.spinner();
        } else {
            let can_search =
                state.preset != StacCatalogPreset::Custom || stac.custom_endpoint_configured;
            if ui
                .add_enabled(can_search, egui::Button::new("Search Catalog"))
                .clicked()
            {
                actions.push(StacUiAction::Search(StacSearchRequest {
                    preset: state.preset,
                    datetime_input: state.datetime_input.clone(),
                    filter_clouds: state.filter_clouds,
                    max_cloud_cover: state.max_cloud_cover,
                    limit: state.limit,
                }));
            }
        }
        if !stac.items.is_empty() && ui.small_button("Clear Results").clicked() {
            actions.push(StacUiAction::ClearResults);
        }
    });

    if let Some(err) = stac.last_error {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(ERROR_TEXT_COLOR, egui::RichText::new(err).small());
            if ui.small_button("Dismiss").clicked() {
                actions.push(StacUiAction::DismissError);
            }
        });
    }
}
