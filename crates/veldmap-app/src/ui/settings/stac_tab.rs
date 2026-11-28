//! STAC / Cloud tab inside the unified Settings modal (named Custom STAC API
//! profiles with endpoint, default collection ID, and authentication
//! credentials).

use eframe::egui;

use crate::storage::settings::{StacAuthKind, StacSettings};
use crate::ui::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR};

pub(super) fn show_stac_tab(ui: &mut egui::Ui, stac: &mut StacSettings) {
    ui.heading("Custom STAC API Profiles & Auth");
    ui.label(
        egui::RichText::new(
            "Built-in AWS Earth Search Sentinel-2 L2A and COP-DEM GLO-30 work with zero auth. \
             Configure named custom STAC API profiles and optional credentials below.",
        )
        .weak()
        .small(),
    );
    ui.add_space(4.0);

    ui.horizontal(|ui| {
        ui.label("Profile:");
        if stac.profiles.is_empty() {
            ui.weak("None configured");
        } else {
            let active_idx = stac.active_profile.min(stac.profiles.len() - 1);
            stac.active_profile = active_idx;
            let selected_label = stac.profiles[active_idx].name.clone();
            egui::ComboBox::from_id_salt("stac_profile_select")
                .selected_text(selected_label)
                .width(175.0)
                .show_ui(ui, |ui| {
                    for (idx, p) in stac.profiles.iter().enumerate() {
                        ui.selectable_value(&mut stac.active_profile, idx, &p.name);
                    }
                });
        }
        if ui
            .small_button("+ Add")
            .on_hover_text("Create a new named Custom STAC profile")
            .clicked()
        {
            stac.add_profile();
        }
        if !stac.profiles.is_empty()
            && ui
                .small_button("Delete")
                .on_hover_text("Delete the selected Custom STAC profile")
                .clicked()
        {
            stac.remove_active_profile();
        }
    });

    if stac.profiles.is_empty() {
        ui.add_space(6.0);
        ui.weak(
            egui::RichText::new(
                "Click '+ Add' to configure a custom STAC endpoint (e.g. Planetary Computer, \
                 self-hosted PgSTAC, or private enterprise catalog).",
            )
            .small(),
        );
        return;
    }

    ui.add_space(4.0);
    let profile = stac.ensure_active_profile_mut();

    egui::Grid::new("custom_stac_grid")
        .num_columns(2)
        .spacing([8.0, 6.0])
        .show(ui, |ui| {
            ui.label("Profile name:");
            ui.add(
                egui::TextEdit::singleline(&mut profile.name)
                    .desired_width(270.0)
                    .hint_text("My STAC Catalog"),
            );
            ui.end_row();

            ui.label("STAC API URL:");
            ui.add(
                egui::TextEdit::singleline(&mut profile.endpoint)
                    .desired_width(270.0)
                    .hint_text("https://earth-search.aws.element84.com/v1"),
            );
            ui.end_row();

            ui.label("Default collection:");
            ui.add(
                egui::TextEdit::singleline(&mut profile.collection)
                    .desired_width(270.0)
                    .hint_text("sentinel-2-l2a (optional)"),
            );
            ui.end_row();

            ui.label("Auth mode:");
            let mut auth_kind = profile.auth_kind;
            egui::ComboBox::from_id_salt("stac_auth_kind")
                .selected_text(auth_kind.label())
                .width(170.0)
                .show_ui(ui, |ui| {
                    for kind in StacAuthKind::ALL {
                        ui.selectable_value(&mut auth_kind, kind, kind.label());
                    }
                });
            profile.set_auth_kind(auth_kind);
            ui.end_row();

            if profile.auth_kind == StacAuthKind::ApiKey {
                ui.label("Header name:");
                ui.add(
                    egui::TextEdit::singleline(&mut profile.auth_header_name)
                        .desired_width(270.0)
                        .hint_text("x-api-key"),
                );
                ui.end_row();
            }

            if profile.auth_kind != StacAuthKind::None {
                let label = match profile.auth_kind {
                    StacAuthKind::Bearer => "Bearer token:",
                    StacAuthKind::ApiKey => "API key value:",
                    StacAuthKind::SasQuery => "SAS query:",
                    StacAuthKind::None => "Secret:",
                };
                let hint = match profile.auth_kind {
                    StacAuthKind::Bearer => "Paste bearer token",
                    StacAuthKind::ApiKey => "Paste API key",
                    StacAuthKind::SasQuery => "sv=2022-...&sig=...",
                    StacAuthKind::None => "",
                };
                ui.label(label);
                ui.add(
                    egui::TextEdit::singleline(&mut profile.auth_secret)
                        .password(true)
                        .desired_width(270.0)
                        .hint_text(hint),
                );
                ui.end_row();
            }
        });

    if let Some(err) = stac.validation_error() {
        ui.colored_label(ERROR_TEXT_COLOR, egui::RichText::new(err).small());
    } else if let Some(active) = stac.active_profile()
        && active.auth_kind != StacAuthKind::None
        && active.auth_secret.trim().is_empty()
    {
        ui.colored_label(
            WARN_TEXT_COLOR,
            egui::RichText::new("Secret is empty — requests will be sent without auth.").small(),
        );
    }
}
