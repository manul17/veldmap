//! Cloud EO & STAC Catalog workbench panel (zero-auth AWS Earth Search
//! Sentinel-2 L2A & COP-DEM GLO-30, custom authenticated STAC, direct HTTP COG
//! URL streaming, built-in sample scenes, AOI tools, and scene asset browser).
//!
//! Strictly action-only: emits [`StacUiAction`] back to `VeldmapApp`.

use std::fmt;

use eframe::egui;
use veldmap_core::geo::FootprintBounds;
use veldmap_net::{
    SampleScene, StacItem, StacSearchQuery, builtin_sample_scenes, sanitize_url_for_display,
};

use crate::stac::{
    StacCatalogPreset, StacCogAuthMode, StacSession, footprint_bounds_from_bbox, geometry_bbox,
};
use crate::storage::settings::AppSettings;
use crate::vectors::{VectorLayerId, VectorLayerSummary};

use super::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR, submitted_with_enter};

/// User intent emitted by the Cloud / STAC panel, applied by `VeldmapApp`.
#[derive(Clone)]
pub(crate) enum StacUiAction {
    Search {
        preset: StacCatalogPreset,
        query: StacSearchQuery,
    },
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
    StreamCog {
        url: String,
        title: String,
        auth_mode: StacCogAuthMode,
        scale_offset: Option<(f64, f64)>,
    },
    OpenStacSettings,
}

impl fmt::Debug for StacUiAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Search { preset, query } => f
                .debug_struct("Search")
                .field("preset", preset)
                .field("query", query)
                .finish(),
            Self::CancelSearch => f.write_str("CancelSearch"),
            Self::ClearResults => f.write_str("ClearResults"),
            Self::DismissError => f.write_str("DismissError"),
            Self::CaptureViewportAoi => f.write_str("CaptureViewportAoi"),
            Self::ToggleDrawAoiBox => f.write_str("ToggleDrawAoiBox"),
            Self::UseVectorAoi(id) => f.debug_tuple("UseVectorAoi").field(id).finish(),
            Self::ClearAoi => f.write_str("ClearAoi"),
            Self::CopyAoiGeoJson => f.write_str("CopyAoiGeoJson"),
            Self::CopyAoiWkt => f.write_str("CopyAoiWkt"),
            Self::SelectItem(idx) => f.debug_tuple("SelectItem").field(idx).finish(),
            Self::HoverItem(idx) => f.debug_tuple("HoverItem").field(idx).finish(),
            Self::ZoomToExtent(bounds) => f.debug_tuple("ZoomToExtent").field(bounds).finish(),
            Self::StreamCog {
                url,
                title,
                auth_mode,
                scale_offset,
            } => f
                .debug_struct("StreamCog")
                .field("url", &sanitize_url_for_display(url))
                .field("title", title)
                .field("auth_mode", auth_mode)
                .field("scale_offset", scale_offset)
                .finish(),
            Self::OpenStacSettings => f.write_str("OpenStacSettings"),
        }
    }
}

/// Transient form/filter state for the Cloud / STAC workbench tab.
#[derive(Clone)]
pub(crate) struct StacUiState {
    pub(crate) preset: StacCatalogPreset,
    pub(crate) datetime_input: String,
    pub(crate) filter_clouds: bool,
    pub(crate) max_cloud_cover: f32,
    pub(crate) limit: usize,
    pub(crate) direct_cog_url: String,
    pub(crate) direct_cog_title: String,
    pub(crate) direct_cog_use_custom_auth: bool,
    pub(crate) copy_notice: Option<String>,
    pub(crate) local_error: Option<String>,
}

impl fmt::Debug for StacUiState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StacUiState")
            .field("preset", &self.preset)
            .field("datetime_input", &self.datetime_input)
            .field("filter_clouds", &self.filter_clouds)
            .field("max_cloud_cover", &self.max_cloud_cover)
            .field("limit", &self.limit)
            .field(
                "direct_cog_url",
                &sanitize_url_for_display(&self.direct_cog_url),
            )
            .field("direct_cog_title", &self.direct_cog_title)
            .field(
                "direct_cog_use_custom_auth",
                &self.direct_cog_use_custom_auth,
            )
            .finish()
    }
}

impl Default for StacUiState {
    fn default() -> Self {
        Self {
            preset: StacCatalogPreset::AwsSentinel2L2a,
            datetime_input: "2024-06-01T00:00:00Z/2024-09-01T23:59:59Z".to_owned(),
            filter_clouds: true,
            max_cloud_cover: 20.0,
            limit: 15,
            direct_cog_url: String::new(),
            direct_cog_title: String::new(),
            direct_cog_use_custom_auth: false,
            copy_notice: None,
            local_error: None,
        }
    }
}

fn show_sample_scenes_section(ui: &mut egui::Ui, actions: &mut Vec<StacUiAction>) {
    egui::CollapsingHeader::new(egui::RichText::new("Quick-Open Sample Scenes").strong())
        .id_salt("stac_sample_scenes")
        .default_open(true)
        .show(ui, |ui| {
            let samples: &[SampleScene] = builtin_sample_scenes();
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
                                actions.push(StacUiAction::StreamCog {
                                    url: sample.cog_url.to_owned(),
                                    title: sample.title.to_owned(),
                                    auth_mode: StacCogAuthMode::None,
                                    scale_offset: sample.scale_offset(),
                                });
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
                egui::TextEdit::singleline(&mut state.direct_cog_url)
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
                let url_ready = !state.direct_cog_url.trim().is_empty();
                if ui
                    .add_enabled(url_ready, egui::Button::new("Stream COG").small())
                    .clicked()
                    || (enter && url_ready)
                {
                    let url = state.direct_cog_url.trim().to_owned();
                    let title = state.direct_cog_title.trim().to_owned();
                    let auth_mode = if state.direct_cog_use_custom_auth {
                        StacCogAuthMode::DirectCustom
                    } else {
                        StacCogAuthMode::None
                    };
                    actions.push(StacUiAction::StreamCog {
                        url,
                        title,
                        auth_mode,
                        scale_offset: None,
                    });
                }
            });
            ui.checkbox(&mut state.direct_cog_use_custom_auth, "Use Custom STAC auth")
                .on_hover_text(
                    "Attach credentials configured in Settings → STAC / Cloud (leave off for public URLs)",
                );
        });
}

fn show_aoi_section(
    ui: &mut egui::Ui,
    stac: &StacSession,
    vectors: &[VectorLayerSummary],
    draw_aoi_active: bool,
    state: &mut StacUiState,
    actions: &mut Vec<StacUiAction>,
) {
    ui.strong("Area of Interest (AOI)");
    ui.horizontal_wrapped(|ui| {
        if ui
            .small_button("Viewport")
            .on_hover_text("Use current visible globe extent as STAC search AOI")
            .clicked()
        {
            state.copy_notice = None;
            actions.push(StacUiAction::CaptureViewportAoi);
        }
        if ui
            .selectable_label(draw_aoi_active, "Draw Box")
            .on_hover_text(
                "Click and drag on the globe to draw a search AOI rectangle (Esc cancels)",
            )
            .clicked()
        {
            state.copy_notice = None;
            actions.push(StacUiAction::ToggleDrawAoiBox);
        }
        let poly_vectors: Vec<&VectorLayerSummary> =
            vectors.iter().filter(|v| v.has_polygons).collect();
        if !poly_vectors.is_empty() {
            ui.menu_button("From GeoJSON", |ui| {
                for vec_layer in poly_vectors {
                    if ui.button(&vec_layer.name).clicked() {
                        state.copy_notice = None;
                        actions.push(StacUiAction::UseVectorAoi(vec_layer.id));
                        ui.close();
                    }
                }
            });
        }
        if stac.aoi_geometry().is_some()
            && ui
                .small_button("Clear")
                .on_hover_text("Clear active search AOI")
                .clicked()
        {
            state.copy_notice = None;
            actions.push(StacUiAction::ClearAoi);
        }
    });

    if let Some(bbox) = stac.aoi_bbox() {
        let label = stac.aoi_label().unwrap_or("Custom AOI");
        ui.small(format!(
            "{label}: [{:.3}, {:.3}] → [{:.3}, {:.3}]",
            bbox[0], bbox[1], bbox[2], bbox[3]
        ));
        ui.horizontal(|ui| {
            if ui
                .small_button("Copy GeoJSON")
                .on_hover_text("Copy active AOI polygon as RFC 7946 GeoJSON")
                .clicked()
            {
                actions.push(StacUiAction::CopyAoiGeoJson);
            }
            if ui
                .small_button("Copy WKT")
                .on_hover_text("Copy active AOI polygon as Well-Known Text (WKT)")
                .clicked()
            {
                actions.push(StacUiAction::CopyAoiWkt);
            }
        });
        if let Some(notice) = &state.copy_notice {
            ui.weak(egui::RichText::new(notice).small());
        }
    } else if draw_aoi_active {
        ui.colored_label(
            WARN_TEXT_COLOR,
            egui::RichText::new("Drag on the globe to draw an AOI box (Esc to cancel).").small(),
        );
    } else {
        ui.weak(
            egui::RichText::new("No AOI set — click Viewport or Draw Box to bound search.").small(),
        );
    }
}

fn show_search_form(
    ui: &mut egui::Ui,
    stac: &StacSession,
    settings: &AppSettings,
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

    if state.preset == StacCatalogPreset::Custom && settings.stac.custom_endpoint.trim().is_empty()
    {
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
        if stac.is_searching() {
            if ui.button("Cancel Search").clicked() {
                actions.push(StacUiAction::CancelSearch);
            }
            ui.spinner();
        } else {
            let can_search = state.preset != StacCatalogPreset::Custom
                || !settings.stac.custom_endpoint.trim().is_empty();
            if ui
                .add_enabled(can_search, egui::Button::new("Search Catalog"))
                .clicked()
            {
                state.local_error = None;
                let query = state.preset.build_search_query(
                    stac.aoi_geometry().cloned(),
                    &state.datetime_input,
                    state.filter_clouds,
                    state.max_cloud_cover,
                    state.limit,
                );
                actions.push(StacUiAction::Search {
                    preset: state.preset,
                    query,
                });
            }
        }
        if !stac.items().is_empty() && ui.small_button("Clear Results").clicked() {
            actions.push(StacUiAction::ClearResults);
        }
    });

    if let Some(err) = state.local_error.as_deref().or(stac.last_error()) {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(ERROR_TEXT_COLOR, egui::RichText::new(err).small());
            if ui.small_button("Dismiss").clicked() {
                actions.push(StacUiAction::DismissError);
            }
        });
    }
}

fn show_item_card(
    ui: &mut egui::Ui,
    idx: usize,
    item: &StacItem,
    selected: bool,
    hovered_any: &mut Option<usize>,
    actions: &mut Vec<StacUiAction>,
) {
    let frame = egui::Frame::group(ui.style()).stroke(if selected {
        egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 200, 64))
    } else {
        ui.style().visuals.widgets.noninteractive.bg_stroke
    });
    let resp = frame
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(selected, egui::RichText::new(&item.id).strong().small())
                    .clicked()
                {
                    actions.push(StacUiAction::SelectItem(idx));
                }
            });
            let mut meta_parts = Vec::new();
            if let Some(dt) = &item.datetime {
                let short = dt.split('T').next().unwrap_or(dt);
                meta_parts.push(short.to_owned());
            }
            if let Some(cc) = item.cloud_cover {
                meta_parts.push(format!("{cc:.1}% cloud"));
            }
            if let Some(tile) = &item.mgrs_tile {
                meta_parts.push(tile.clone());
            }
            if !meta_parts.is_empty() {
                ui.weak(egui::RichText::new(meta_parts.join(" · ")).small());
            }

            ui.horizontal_wrapped(|ui| {
                if let Some(asset) = item.preferred_cog_asset()
                    && ui
                        .small_button(format!("Stream {}", asset.key))
                        .on_hover_text("Stream primary Cloud-Optimized GeoTIFF asset")
                        .clicked()
                {
                    actions.push(StacUiAction::StreamCog {
                        url: asset.href.clone(),
                        title: format!("{} ({})", item.id, asset.key),
                        auth_mode: StacCogAuthMode::FromSearchResult,
                        scale_offset: asset.primary_scale_offset(),
                    });
                }
                let zoom_bbox = item
                    .bbox
                    .or_else(|| item.geometry.as_ref().and_then(geometry_bbox));
                if let Some(bbox) = zoom_bbox
                    && let Some(bounds) = footprint_bounds_from_bbox(bbox)
                    && ui
                        .small_button("Zoom")
                        .on_hover_text("Fly camera to scene footprint")
                        .clicked()
                {
                    actions.push(StacUiAction::ZoomToExtent(bounds));
                }
            });

            if selected {
                let cogs = item.cog_assets();
                if cogs.len() > 1 {
                    ui.separator();
                    ui.weak(egui::RichText::new(format!("COG Assets ({}):", cogs.len())).small());
                    ui.horizontal_wrapped(|ui| {
                        for asset in cogs {
                            let label = if let Some(gsd) = asset.gsd {
                                format!("{} ({gsd:.0}m)", asset.key)
                            } else {
                                asset.key.clone()
                            };
                            let tip = asset.title.as_deref().unwrap_or(&asset.key);
                            if ui.small_button(label).on_hover_text(tip).clicked() {
                                actions.push(StacUiAction::StreamCog {
                                    url: asset.href.clone(),
                                    title: format!("{} ({})", item.id, asset.key),
                                    auth_mode: StacCogAuthMode::FromSearchResult,
                                    scale_offset: asset.primary_scale_offset(),
                                });
                            }
                        }
                    });
                }
            }
        })
        .response;
    if resp.contains_pointer() {
        *hovered_any = Some(idx);
    }
}

/// Render the Cloud / STAC tab contents inside the left workbench panel.
pub(crate) fn show_stac_tab_contents(
    ui: &mut egui::Ui,
    stac: &StacSession,
    settings: &AppSettings,
    vectors: &[VectorLayerSummary],
    draw_aoi_active: bool,
    state: &mut StacUiState,
) -> Vec<StacUiAction> {
    let mut actions = Vec::new();

    egui::ScrollArea::vertical()
        .id_salt("stac_workbench_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            show_sample_scenes_section(ui, &mut actions);
            ui.add_space(4.0);
            show_direct_cog_section(ui, state, &mut actions);
            ui.separator();
            show_aoi_section(ui, stac, vectors, draw_aoi_active, state, &mut actions);
            ui.separator();
            show_search_form(ui, stac, settings, state, &mut actions);

            let items = stac.items();
            if !items.is_empty() {
                ui.separator();
                let header = match stac.matched() {
                    Some(total) => format!("Results ({} of {total})", items.len()),
                    None => format!("Results ({})", items.len()),
                };
                ui.strong(header);
                let selected_idx = stac.selected_idx();
                let mut hovered_any = None;
                for (idx, item) in items.iter().enumerate() {
                    show_item_card(
                        ui,
                        idx,
                        item,
                        selected_idx == Some(idx),
                        &mut hovered_any,
                        &mut actions,
                    );
                }
                if hovered_any != stac.hovered_idx() {
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
            direct_cog_url: "https://user:pw@blob.example/a.tif?sig=top_secret".to_owned(),
            ..StacUiState::default()
        };
        let state_dbg = format!("{state:?}");
        assert!(
            !state_dbg.contains("top_secret"),
            "leaked query: {state_dbg}"
        );
        assert!(!state_dbg.contains("pw"), "leaked userinfo: {state_dbg}");

        let action = StacUiAction::StreamCog {
            url: "https://user:pw@blob.example/a.tif?sig=top_secret".to_owned(),
            title: "Test".to_owned(),
            auth_mode: StacCogAuthMode::DirectCustom,
            scale_offset: None,
        };
        let action_dbg = format!("{action:?}");
        assert!(
            !action_dbg.contains("top_secret"),
            "leaked query: {action_dbg}"
        );
        assert!(!action_dbg.contains("pw"), "leaked userinfo: {action_dbg}");
    }
}
