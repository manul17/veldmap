//! STAC catalog preset / named profile selector, inline collection override,
//! quick date presets (`30d` / `90d` / `1y`), `sortby`, temporal/cloud/limit
//! filters, search trigger, and error banner inside the Cloud / STAC panel.

use eframe::egui;

use crate::stac::{StacCatalogPreset, StacSearchRequest, StacSortOption, StacViewSnapshot};
use crate::ui::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR};

use super::{StacUiAction, StacUiState, rolling_days_utc_interval};

fn selected_preset_label(preset: StacCatalogPreset, stac: &StacViewSnapshot<'_>) -> String {
    match preset {
        StacCatalogPreset::AwsSentinel2L2a | StacCatalogPreset::AwsCopDemGlo30 => {
            preset.builtin_label().to_owned()
        }
        StacCatalogPreset::CustomProfile(idx) => stac
            .custom_profiles
            .get(idx)
            .map_or_else(|| "Custom STAC Profile".to_owned(), |p| p.name.clone()),
    }
}

fn collection_hint(preset: StacCatalogPreset, stac: &StacViewSnapshot<'_>) -> String {
    let default_col = match preset {
        StacCatalogPreset::AwsSentinel2L2a | StacCatalogPreset::AwsCopDemGlo30 => {
            preset.default_collection()
        }
        StacCatalogPreset::CustomProfile(idx) => stac
            .custom_profiles
            .get(idx)
            .map_or("", |p| p.default_collection.as_str()),
    };
    if default_col.is_empty() {
        "Collection ID (optional)".to_owned()
    } else {
        format!("{default_col} (default)")
    }
}

pub(super) fn show_search_form(
    ui: &mut egui::Ui,
    stac: &StacViewSnapshot<'_>,
    state: &mut StacUiState,
    actions: &mut Vec<StacUiAction>,
) {
    // Reconcile deleted custom profile index gracefully.
    if let StacCatalogPreset::CustomProfile(idx) = state.preset
        && idx >= stac.custom_profiles.len()
        && !stac.custom_profiles.is_empty()
    {
        state.preset = StacCatalogPreset::CustomProfile(0);
    }

    ui.strong("STAC Catalog Search");
    ui.horizontal(|ui| {
        let selected = selected_preset_label(state.preset, stac);
        egui::ComboBox::from_id_salt("stac_catalog_preset")
            .selected_text(selected)
            .width((ui.available_width() - 62.0).max(100.0))
            .show_ui(ui, |ui| {
                for preset in StacCatalogPreset::BUILTINS {
                    ui.selectable_value(&mut state.preset, preset, preset.builtin_label());
                }
                if stac.custom_profiles.is_empty() {
                    ui.selectable_value(
                        &mut state.preset,
                        StacCatalogPreset::CustomProfile(0),
                        "Custom STAC Profile…",
                    );
                } else {
                    for profile in &stac.custom_profiles {
                        ui.selectable_value(
                            &mut state.preset,
                            StacCatalogPreset::CustomProfile(profile.index),
                            format!("Custom: {}", profile.name),
                        );
                    }
                }
            });
        if ui
            .small_button("Profiles…")
            .on_hover_text("Manage named Custom STAC profiles and credentials in Settings")
            .clicked()
        {
            actions.push(StacUiAction::OpenStacSettings);
        }
    });

    let custom_ready = match state.preset {
        StacCatalogPreset::CustomProfile(idx) => {
            stac.custom_profiles.get(idx).is_some_and(|p| p.configured)
        }
        _ => true,
    };

    if state.preset.is_custom() && !custom_ready {
        ui.colored_label(
            WARN_TEXT_COLOR,
            egui::RichText::new("Configure a Custom STAC profile in Settings → STAC / Cloud.")
                .small(),
        );
    }

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Collection:").small());
        ui.add(
            egui::TextEdit::singleline(&mut state.collection_override)
                .desired_width(ui.available_width())
                .hint_text(collection_hint(state.preset, stac)),
        );
    });

    if state.preset.supports_temporal_and_cloud_filters() {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Dates:").small());
            for (label, days) in [("30d", 30), ("90d", 90), ("1y", 365)] {
                if ui.small_button(label).clicked() {
                    state.datetime_input =
                        rolling_days_utc_interval(std::time::SystemTime::now(), days);
                }
            }
            if ui
                .small_button("Any")
                .on_hover_text("Clear datetime filter")
                .clicked()
            {
                state.datetime_input.clear();
            }
        });
        ui.add(
            egui::TextEdit::singleline(&mut state.datetime_input)
                .desired_width(ui.available_width())
                .hint_text("YYYY-MM-DD/YYYY-MM-DD (optional)"),
        );

        ui.horizontal(|ui| {
            ui.checkbox(&mut state.filter_clouds, "Max cloud %:");
            ui.add_enabled(
                state.filter_clouds,
                egui::Slider::new(&mut state.max_cloud_cover, 0.0..=100.0).fixed_decimals(0),
            );
        });

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Sort:").small());
            egui::ComboBox::from_id_salt("stac_sort_by")
                .selected_text(egui::RichText::new(state.sort_by.label()).small())
                .width(142.0)
                .show_ui(ui, |ui| {
                    for opt in StacSortOption::ALL {
                        ui.selectable_value(&mut state.sort_by, opt, opt.label());
                    }
                });
        });
    }

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Page size:").small());
        ui.add(egui::Slider::new(&mut state.limit, 1..=100));
    });

    ui.horizontal(|ui| {
        if stac.is_searching {
            if ui.button("Cancel Search").clicked() {
                actions.push(StacUiAction::CancelSearch);
            }
            ui.spinner();
        } else {
            if ui
                .add_enabled(custom_ready, egui::Button::new("Search Catalog"))
                .clicked()
            {
                actions.push(StacUiAction::Search(StacSearchRequest {
                    preset: state.preset,
                    collection_override: state.collection_override.clone(),
                    datetime_input: state.datetime_input.clone(),
                    filter_clouds: state.filter_clouds,
                    max_cloud_cover: state.max_cloud_cover,
                    sort_by: state.sort_by,
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
