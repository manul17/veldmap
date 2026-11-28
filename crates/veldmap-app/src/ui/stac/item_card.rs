//! Individual STAC search result card displaying precomputed scene metadata
//! (`processing_level` L1C/L2A & `platform` badges), 1-click Band Math scene
//! recipes (`True Color`, `NDVI + SCL`, `NDWI + SCL`, `NBR + SCL`,
//! `NDSI + SCL`, `False Color NIR`, `SWIR`), primary COG stream trigger,
//! camera zoom button, and asset picker.

use eframe::egui;

use crate::stac::{StacCogAuthMode, StacItemView};
use crate::ui::{BADGE_CLOUD_COLOR, BADGE_RECIPE_COLOR, SELECTION_ACCENT_COLOR};

use super::StacUiAction;

pub(super) fn show_item_card(
    ui: &mut egui::Ui,
    idx: usize,
    item: &StacItemView,
    selected: bool,
    hovered_any: &mut Option<usize>,
    actions: &mut Vec<StacUiAction>,
) {
    let frame = egui::Frame::group(ui.style()).stroke(if selected {
        egui::Stroke::new(1.5, SELECTION_ACCENT_COLOR)
    } else {
        ui.style().visuals.widgets.noninteractive.bg_stroke
    });
    let resp = frame
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui
                    .selectable_label(selected, egui::RichText::new(&item.id).strong().small())
                    .clicked()
                {
                    actions.push(StacUiAction::SelectItem(idx));
                }
                if let Some(level) = &item.processing_level {
                    let badge_color = if item.is_surface_reflectance {
                        BADGE_RECIPE_COLOR
                    } else {
                        BADGE_CLOUD_COLOR
                    };
                    ui.colored_label(badge_color, egui::RichText::new(level).small().strong());
                }
                if let Some(platform) = &item.platform {
                    ui.weak(egui::RichText::new(platform).small());
                }
            });
            if let Some(meta_line) = &item.meta_line {
                ui.weak(egui::RichText::new(meta_line).small());
            }

            ui.horizontal_wrapped(|ui| {
                if let Some(asset) = &item.preferred_cog
                    && ui
                        .small_button(format!("Stream {}", asset.key))
                        .on_hover_text("Stream primary Cloud-Optimized GeoTIFF asset")
                        .clicked()
                {
                    actions.push(StacUiAction::StreamCog(asset.to_stream_spec(
                        format!("{} ({})", item.id, asset.key),
                        StacCogAuthMode::FromSearchResult,
                        item.self_href.clone(),
                    )));
                }
                if let Some(bounds) = item.zoom_bounds
                    && ui
                        .small_button("Zoom")
                        .on_hover_text("Fly camera to scene footprint")
                        .clicked()
                {
                    actions.push(StacUiAction::ZoomToExtent(bounds));
                }
            });

            if !item.recipes.is_empty() {
                ui.add_space(2.0);
                ui.horizontal_wrapped(|ui| {
                    for recipe in &item.recipes {
                        if ui
                            .small_button(&recipe.button_label)
                            .on_hover_text(&recipe.tooltip)
                            .clicked()
                        {
                            actions.push(StacUiAction::OpenRecipe {
                                title: format!("{} · {}", item.id, recipe.preset.short_label()),
                                preset: recipe.preset,
                                auth_mode: StacCogAuthMode::FromSearchResult,
                                bands: recipe.bands.clone(),
                                scl: recipe.scl.clone(),
                            });
                        }
                    }
                });
            }

            if selected && item.cog_assets.len() > 1 {
                ui.separator();
                ui.weak(
                    egui::RichText::new(format!("COG Assets ({}):", item.cog_assets.len())).small(),
                );
                ui.horizontal_wrapped(|ui| {
                    for asset in &item.cog_assets {
                        let label = match asset.gsd {
                            Some(gsd) if gsd.is_finite() && gsd >= 1.0 => {
                                format!("{} ({gsd:.0}m)", asset.key)
                            }
                            Some(gsd) if gsd.is_finite() && gsd > 0.0 => {
                                format!("{} ({:.0}cm)", asset.key, gsd * 100.0)
                            }
                            _ => asset.key.clone(),
                        };
                        let tip = asset.title.as_deref().unwrap_or(&asset.key);
                        if ui.small_button(label).on_hover_text(tip).clicked() {
                            actions.push(StacUiAction::StreamCog(asset.to_stream_spec(
                                format!("{} ({})", item.id, asset.key),
                                StacCogAuthMode::FromSearchResult,
                                item.self_href.clone(),
                            )));
                        }
                    }
                });
            }
        })
        .response;
    if resp.contains_pointer() {
        *hovered_any = Some(idx);
    }
}
