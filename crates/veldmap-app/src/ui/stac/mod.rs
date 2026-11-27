//! Cloud EO & STAC Catalog workbench panel (zero-auth AWS Earth Search
//! Sentinel-2 L2A & COP-DEM GLO-30, custom authenticated STAC, direct HTTP COG
//! URL streaming, built-in sample scenes, AOI tools, and scene asset browser).
//!
//! Strictly presentation- and action-only: reads [`StacViewSnapshot`] and emits
//! [`StacUiAction`] back to `VeldmapApp` with zero direct `veldmap-net` or
//! `veldmap-raster` imports.

mod aoi_section;
mod item_card;
mod search_form;

use eframe::egui;
use veldmap_core::geo::FootprintBounds;

use crate::layers::BandMathPreset;
use crate::stac::{
    RedactedUrl, SampleRecipeView, SampleSceneView, StacCatalogPreset, StacCogAuthMode,
    StacCogStreamSpec, StacRecipeAssetView, StacSearchRequest, StacViewSnapshot,
};
use crate::vectors::{VectorLayerId, VectorLayerSummary};

use super::BADGE_RECIPE_COLOR;
use super::widgets::submitted_with_enter;

use aoi_section::show_aoi_section;
use item_card::show_item_card;
use search_form::show_search_form;

/// User intent emitted by the Cloud / STAC panel, applied by `VeldmapApp`.
#[derive(Clone, Debug)]
pub(crate) enum StacUiAction {
    Search(StacSearchRequest),
    CancelSearch,
    ClearResults,
    DismissError,
    CaptureViewportAoi,
    ToggleDrawAoiBox,
    UseVectorAoi(VectorLayerId),
    ClearAoi,
    CopyAoiGeoJson,
    CopyAoiWkt,
    SelectItem(usize),
    HoverItem(Option<usize>),
    ZoomToExtent(FootprintBounds),
    StreamCog(StacCogStreamSpec),
    OpenRecipe {
        title: String,
        preset: BandMathPreset,
        auth_mode: StacCogAuthMode,
        bands: Vec<StacRecipeAssetView>,
        scl: Option<StacRecipeAssetView>,
    },
    OpenStacSettings,
}

/// Transient form/filter state for the Cloud / STAC workbench tab.
#[derive(Clone, Debug)]
pub(crate) struct StacUiState {
    preset: StacCatalogPreset,
    datetime_input: String,
    filter_clouds: bool,
    max_cloud_cover: f32,
    limit: usize,
    direct_cog_url: RedactedUrl,
    direct_cog_title: String,
    direct_cog_use_custom_auth: bool,
    copy_notice: Option<String>,
}

impl Default for StacUiState {
    fn default() -> Self {
        Self {
            preset: StacCatalogPreset::AwsSentinel2L2a,
            datetime_input: "2024-06-01T00:00:00Z/2024-09-01T23:59:59Z".to_owned(),
            filter_clouds: true,
            max_cloud_cover: 20.0,
            limit: 15,
            direct_cog_url: RedactedUrl::default(),
            direct_cog_title: String::new(),
            direct_cog_use_custom_auth: false,
            copy_notice: None,
        }
    }
}

impl StacUiState {
    pub(crate) fn set_copy_notice(&mut self, notice: String) {
        self.copy_notice = Some(notice);
    }

    pub(crate) fn clear_copy_notice(&mut self) {
        self.copy_notice = None;
    }
}

fn show_sample_scenes_section(
    ui: &mut egui::Ui,
    samples: &[SampleSceneView],
    recipes: &[SampleRecipeView],
    actions: &mut Vec<StacUiAction>,
) {
    egui::CollapsingHeader::new(egui::RichText::new("Quick-Open Sample Scenes").strong())
        .id_salt("stac_sample_scenes")
        .default_open(true)
        .show(ui, |ui| {
            for recipe in recipes {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.colored_label(
                            BADGE_RECIPE_COLOR,
                            egui::RichText::new("Recipe").small().strong(),
                        );
                        ui.strong(recipe.title);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .small_button("Open")
                                .on_hover_text(
                                    "Stream this public multi-band Sentinel-2 recipe with SCL masking",
                                )
                                .clicked()
                            {
                                actions.push(StacUiAction::OpenRecipe {
                                    title: recipe.title.to_owned(),
                                    preset: recipe.preset,
                                    auth_mode: StacCogAuthMode::None,
                                    bands: recipe.bands.clone(),
                                    scl: recipe.scl.clone(),
                                });
                            }
                        });
                    });
                    ui.weak(egui::RichText::new(recipe.description).small());
                });
            }
            for sample in samples {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.strong(sample.title);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .small_button("Stream")
                                .on_hover_text("Stream this public zero-auth COG over HTTP Range")
                                .clicked()
                            {
                                actions.push(StacUiAction::StreamCog(sample.to_stream_spec()));
                            }
                        });
                    });
                    ui.weak(egui::RichText::new(sample.description).small());
                });
            }
        });
}

fn show_direct_cog_section(
    ui: &mut egui::Ui,
    state: &mut StacUiState,
    actions: &mut Vec<StacUiAction>,
) {
    egui::CollapsingHeader::new(egui::RichText::new("Open Direct COG URL").strong())
        .id_salt("stac_direct_cog")
        .default_open(false)
        .show(ui, |ui| {
            let resp = ui.add(
                egui::TextEdit::singleline(state.direct_cog_url.as_mut_string())
                    .desired_width(ui.available_width())
                    .hint_text("https://.../scene.tif"),
            );
            let enter = submitted_with_enter(ui, &resp);
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut state.direct_cog_title)
                        .desired_width(140.0)
                        .hint_text("Layer title (optional)"),
                );
                let url_ready = !state.direct_cog_url.as_str().trim().is_empty();
                if ui
                    .add_enabled(url_ready, egui::Button::new("Stream COG").small())
                    .clicked()
                    || (enter && url_ready)
                {
                    let url = RedactedUrl::new(state.direct_cog_url.as_str().trim());
                    let title = state.direct_cog_title.trim().to_owned();
                    let auth_mode = if state.direct_cog_use_custom_auth {
                        StacCogAuthMode::DirectCustom
                    } else {
                        StacCogAuthMode::None
                    };
                    actions.push(StacUiAction::StreamCog(StacCogStreamSpec::direct(
                        url, title, auth_mode,
                    )));
                }
            });
            ui.checkbox(&mut state.direct_cog_use_custom_auth, "Use Custom STAC auth")
                .on_hover_text(
                    "Attach credentials configured in Settings → STAC / Cloud (leave off for public URLs)",
                );
        });
}

/// Render the Cloud / STAC tab contents inside the left workbench panel.
pub(crate) fn show_stac_tab_contents(
    ui: &mut egui::Ui,
    stac: &StacViewSnapshot<'_>,
    vectors: &[VectorLayerSummary],
    draw_aoi_active: bool,
    state: &mut StacUiState,
) -> Vec<StacUiAction> {
    let mut actions = Vec::new();

    egui::ScrollArea::vertical()
        .id_salt("stac_workbench_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            show_sample_scenes_section(ui, stac.sample_scenes, stac.sample_recipes, &mut actions);
            ui.add_space(4.0);
            show_direct_cog_section(ui, state, &mut actions);
            ui.separator();
            show_aoi_section(ui, stac, vectors, draw_aoi_active, state, &mut actions);
            ui.separator();
            show_search_form(ui, stac, state, &mut actions);

            if !stac.items.is_empty() {
                ui.separator();
                let header = match stac.matched {
                    Some(total) => format!("Results ({} of {total})", stac.items.len()),
                    None => format!("Results ({})", stac.items.len()),
                };
                ui.strong(header);
                let mut hovered_any = None;
                for (idx, item) in stac.items.iter().enumerate() {
                    show_item_card(
                        ui,
                        idx,
                        item,
                        stac.selected_idx == Some(idx),
                        &mut hovered_any,
                        &mut actions,
                    );
                }
                if hovered_any != stac.hovered_idx {
                    actions.push(StacUiAction::HoverItem(hovered_any));
                }
            }
        });

    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stac_ui_debug_redacts_url_userinfo_and_query_tokens() {
        let state = StacUiState {
            direct_cog_url: RedactedUrl::new("https://user:pw@blob.example/a.tif?sig=top_secret"),
            ..StacUiState::default()
        };
        let state_dbg = format!("{state:?}");
        assert!(
            !state_dbg.contains("top_secret"),
            "leaked query: {state_dbg}"
        );
        assert!(!state_dbg.contains("pw"), "leaked userinfo: {state_dbg}");

        let action = StacUiAction::StreamCog(StacCogStreamSpec::direct(
            RedactedUrl::new("https://user:pw@blob.example/a.tif?sig=top_secret"),
            "Test".to_owned(),
            StacCogAuthMode::DirectCustom,
        ));
        let action_dbg = format!("{action:?}");
        assert!(
            !action_dbg.contains("top_secret"),
            "leaked query: {action_dbg}"
        );
        assert!(!action_dbg.contains("pw"), "leaked userinfo: {action_dbg}");
    }
}
