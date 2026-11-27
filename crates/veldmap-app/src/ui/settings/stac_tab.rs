//! STAC / Cloud tab inside the unified Settings modal (custom STAC API
//! endpoint, collection ID, and authentication credentials).

use eframe::egui;

use crate::storage::settings::{StacAuthKind, StacSettings};
use crate::ui::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR};

pub(super) fn show_stac_tab(ui: &mut egui::Ui, stac: &mut StacSettings) {
    ui.heading("Custom STAC API Endpoint & Auth");
    ui.label(
        egui::RichText::new(
            "Built-in AWS Earth Search Sentinel-2 L2A and COP-DEM GLO-30 work with zero auth. \
             Configure a custom STAC API endpoint and optional credentials below.",
        )
        .weak()
        .small(),
    );
    ui.add_space(4.0);

    egui::Grid::new("custom_stac_grid")
        .num_columns(2)
        .spacing([8.0, 6.0])
        .show(ui, |ui| {
            ui.label("STAC API URL:");
            ui.add(
                egui::TextEdit::singleline(&mut stac.custom_endpoint)
                    .desired_width(270.0)
                    .hint_text("https://earth-search.aws.element84.com/v1"),
            );
            ui.end_row();

            ui.label("Collection ID:");
            ui.add(
                egui::TextEdit::singleline(&mut stac.custom_collection)
                    .desired_width(270.0)
                    .hint_text("sentinel-2-l2a (optional)"),
            );
            ui.end_row();

            ui.label("Auth mode:");
            let prev_auth_kind = stac.auth_kind;
            egui::ComboBox::from_id_salt("stac_auth_kind")
                .selected_text(stac.auth_kind.label())
                .width(170.0)
                .show_ui(ui, |ui| {
                    for kind in StacAuthKind::ALL {
                        ui.selectable_value(&mut stac.auth_kind, kind, kind.label());
                    }
                });
            if stac.auth_kind != prev_auth_kind {
                // Scrub the hidden secret whenever auth mode changes so a Bearer
                // token can never leak into a SAS query string or linger on disk
                // after switching to `None`.
                stac.auth_secret.clear();
            }
            ui.end_row();

            if stac.auth_kind == StacAuthKind::ApiKey {
                ui.label("Header name:");
                ui.add(
                    egui::TextEdit::singleline(&mut stac.auth_header_name)
                        .desired_width(270.0)
                        .hint_text("x-api-key"),
                );
                ui.end_row();
            }

            if stac.auth_kind != StacAuthKind::None {
                let label = match stac.auth_kind {
                    StacAuthKind::Bearer => "Bearer token:",
                    StacAuthKind::ApiKey => "API key value:",
                    StacAuthKind::SasQuery => "SAS query:",
                    StacAuthKind::None => "Secret:",
                };
                let hint = match stac.auth_kind {
                    StacAuthKind::Bearer => "Paste bearer token",
                    StacAuthKind::ApiKey => "Paste API key",
                    StacAuthKind::SasQuery => "sv=2022-...&sig=...",
                    StacAuthKind::None => "",
                };
                ui.label(label);
                ui.add(
                    egui::TextEdit::singleline(&mut stac.auth_secret)
                        .password(true)
                        .desired_width(270.0)
                        .hint_text(hint),
                );
                ui.end_row();
            }
        });

    if let Some(err) = stac.validation_error() {
        ui.colored_label(ERROR_TEXT_COLOR, egui::RichText::new(err).small());
    } else if stac.auth_kind != StacAuthKind::None && stac.auth_secret.trim().is_empty() {
        ui.colored_label(
            WARN_TEXT_COLOR,
            egui::RichText::new("Secret is empty — requests will be sent without auth.").small(),
        );
    }
}
