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
    StacCogStreamSpec, StacRecipeAssetView, StacSearchRequest, StacSortOption,
    StacTimelineGroupView, StacTimelineMode, StacViewSnapshot,
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
    LoadMore,
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
    CreateTimeline {
        group_key: String,
        mode: StacTimelineMode,
    },
    OpenStacSettings,
}

/// Transient form/filter state for the Cloud / STAC workbench tab.
#[derive(Clone, Debug)]
pub(crate) struct StacUiState {
    preset: StacCatalogPreset,
    collection_override: String,
    datetime_input: String,
    filter_clouds: bool,
    max_cloud_cover: f32,
    sort_by: StacSortOption,
    limit: usize,
    direct_cog_url: RedactedUrl,
    direct_cog_title: String,
    direct_cog_use_custom_auth: bool,
    copy_notice: Option<String>,
    timeline_group_idx: usize,
    timeline_mode_idx: usize,
}

/// Convert days since Unix epoch (`1970-01-01`) into proleptic Gregorian
/// `(year, month, day)` using Howard Hinnant's exact civil-from-days algorithm.
fn civil_from_unix_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = y + i64::from(m <= 2);
    (year, m as u32, d as u32)
}

/// Build a rolling last-`window_days` UTC ISO-8601 STAC datetime interval from `now`.
pub(super) fn rolling_days_utc_interval(now: std::time::SystemTime, window_days: i64) -> String {
    const SECS_PER_DAY: u64 = 86_400;
    let end_days = now
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.as_secs() / SECS_PER_DAY) as i64)
        .unwrap_or(20_000);
    let start_days = end_days.saturating_sub(window_days.max(1));
    let (sy, sm, sd) = civil_from_unix_days(start_days);
    let (ey, em, ed) = civil_from_unix_days(end_days);
    format!("{sy:04}-{sm:02}-{sd:02}T00:00:00Z/{ey:04}-{em:02}-{ed:02}T23:59:59Z")
}

impl Default for StacUiState {
    fn default() -> Self {
        Self {
            preset: StacCatalogPreset::AwsSentinel2L2a,
            collection_override: String::new(),
            datetime_input: rolling_days_utc_interval(std::time::SystemTime::now(), 90),
            filter_clouds: true,
            max_cloud_cover: 20.0,
            sort_by: StacSortOption::Newest,
            limit: 15,
            direct_cog_url: RedactedUrl::default(),
            direct_cog_title: String::new(),
            direct_cog_use_custom_auth: false,
            copy_notice: None,
            timeline_group_idx: 0,
            timeline_mode_idx: 0,
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

fn show_timeline_builder_bar(
    ui: &mut egui::Ui,
    groups: &[StacTimelineGroupView],
    state: &mut StacUiState,
    actions: &mut Vec<StacUiAction>,
) {
    if groups.is_empty() {
        return;
    }
    if state.timeline_group_idx >= groups.len() {
        state.timeline_group_idx = 0;
        state.timeline_mode_idx = 0;
    }
    let group = &groups[state.timeline_group_idx];
    if state.timeline_mode_idx >= group.available_modes.len() {
        state.timeline_mode_idx = 0;
    }

    ui.group(|ui| {
        ui.horizontal(|ui| {
            ui.colored_label(
                BADGE_RECIPE_COLOR,
                egui::RichText::new("Timeline").small().strong(),
            );
            ui.weak(
                egui::RichText::new(format!("{} · {}", group.scene_count, group.date_span)).small(),
            );
        });
        ui.horizontal(|ui| {
            if groups.len() > 1 {
                egui::ComboBox::from_id_salt("stac_tl_group")
                    .selected_text(egui::RichText::new(&group.group_key).small())
                    .width(76.0)
                    .show_ui(ui, |ui| {
                        for (idx, g) in groups.iter().enumerate() {
                            let label = format!("{} ({})", g.group_key, g.scene_count);
                            if ui
                                .selectable_label(state.timeline_group_idx == idx, label)
                                .clicked()
                            {
                                state.timeline_group_idx = idx;
                                state.timeline_mode_idx = 0;
                            }
                        }
                    });
            }
            let group = &groups[state.timeline_group_idx];
            if state.timeline_mode_idx >= group.available_modes.len() {
                state.timeline_mode_idx = 0;
            }
            let selected_mode = group.available_modes[state.timeline_mode_idx];
            egui::ComboBox::from_id_salt("stac_tl_mode")
                .selected_text(egui::RichText::new(selected_mode.label()).small())
                .width(126.0)
                .show_ui(ui, |ui| {
                    for (idx, &mode) in group.available_modes.iter().enumerate() {
                        if ui
                            .selectable_label(state.timeline_mode_idx == idx, mode.label())
                            .clicked()
                        {
                            state.timeline_mode_idx = idx;
                        }
                    }
                });
            if ui
                .small_button("Build")
                .on_hover_text("Create a multi-temporal Timeline layer from these STAC scenes")
                .clicked()
            {
                let g = &groups[state.timeline_group_idx];
                let m = g.available_modes[state.timeline_mode_idx.min(g.available_modes.len() - 1)];
                actions.push(StacUiAction::CreateTimeline {
                    group_key: g.group_key.clone(),
                    mode: m,
                });
            }
        });
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
                show_timeline_builder_bar(ui, stac.timeline_groups, state, &mut actions);
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
                if stac.is_loading_more {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.weak(egui::RichText::new("Loading next page…").small());
                    });
                } else if stac.can_load_more {
                    if ui
                        .button("Load More")
                        .on_hover_text("Fetch next STAC page via links[rel=\"next\"]")
                        .clicked()
                    {
                        actions.push(StacUiAction::LoadMore);
                    }
                } else if stac.reached_page_cap {
                    ui.weak(
                        egui::RichText::new(
                            "Page cap reached (500 items). Narrow AOI or date range.",
                        )
                        .small(),
                    );
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

    #[test]
    fn rolling_days_utc_interval_handles_leap_years_and_exact_span() {
        // 2024-03-01T12:00:00Z = day 19783 since 1970-01-01 (leap year 2024).
        // 90 days earlier = 2023-12-02; 30 days earlier = 2024-01-31.
        let t = std::time::UNIX_EPOCH + std::time::Duration::from_secs(19_783 * 86_400 + 43_200);
        assert_eq!(
            rolling_days_utc_interval(t, 90),
            "2023-12-02T00:00:00Z/2024-03-01T23:59:59Z"
        );
        assert_eq!(
            rolling_days_utc_interval(t, 30),
            "2024-01-31T00:00:00Z/2024-03-01T23:59:59Z"
        );
    }
}
