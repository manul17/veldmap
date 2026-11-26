//! Collapsible left workbench panel hosting the `Layers` and `Cloud / STAC`
//! tabs with shared tab labeling and action-only close/switch intents.

use eframe::egui;

use crate::layers::LayerSummary;
use crate::stac::StacSession;
use crate::storage::settings::AppSettings;
use crate::vectors::VectorLayerSummary;

use super::layers::{
    LayerUiAction, LayersPanelState, UnavailableLayerNotice, show_layers_tab_contents,
};
use super::stac::{StacUiAction, StacUiState, show_stac_tab_contents};

/// Active tab inside the collapsible left workbench panel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum WorkbenchTab {
    #[default]
    Layers,
    CloudStac,
}

/// Format the shared `"Layers"` / `"Layers (N)"` label used in both the top
/// toolbar and the workbench tab strip.
pub(crate) fn format_layers_tab_label(total_layers: usize) -> String {
    if total_layers > 0 {
        format!("Layers ({total_layers})")
    } else {
        "Layers".to_owned()
    }
}

/// Output emitted by [`show_workbench_panel`].
#[derive(Default)]
pub(crate) struct WorkbenchFrameOutput {
    pub(crate) selected_tab: Option<WorkbenchTab>,
    pub(crate) close_requested: bool,
    pub(crate) layer_actions: Vec<LayerUiAction>,
    pub(crate) stac_actions: Vec<StacUiAction>,
}

/// Borrowed inputs for rendering [`show_workbench_panel`].
pub(crate) struct WorkbenchInputs<'a> {
    pub(crate) active_tab: WorkbenchTab,
    pub(crate) total_layers: usize,
    pub(crate) layer_summaries: &'a [LayerSummary],
    pub(crate) vector_summaries: &'a [VectorLayerSummary],
    pub(crate) unavailable_notices: &'a [UnavailableLayerNotice],
    pub(crate) stac: &'a StacSession,
    pub(crate) settings: &'a AppSettings,
    pub(crate) draw_aoi_active: bool,
}

/// Render the collapsible left workbench panel when open, returning action-only
/// tab/close/layer/STAC intents to `VeldmapApp`.
pub(crate) fn show_workbench_panel(
    ui: &mut egui::Ui,
    inputs: WorkbenchInputs<'_>,
    layers_state: &mut LayersPanelState,
    stac_state: &mut StacUiState,
) -> WorkbenchFrameOutput {
    let mut out = WorkbenchFrameOutput::default();
    let mut tab = inputs.active_tab;

    egui::Panel::left("workbench_left_panel")
        .resizable(true)
        .default_size(310.0)
        .min_size(240.0)
        .max_size(480.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut tab,
                    WorkbenchTab::Layers,
                    format_layers_tab_label(inputs.total_layers),
                );
                ui.selectable_value(&mut tab, WorkbenchTab::CloudStac, "Cloud / STAC");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("×")
                        .on_hover_text("Hide workbench panel")
                        .clicked()
                    {
                        out.close_requested = true;
                    }
                });
            });
            ui.separator();

            match inputs.active_tab {
                WorkbenchTab::Layers => {
                    out.layer_actions = show_layers_tab_contents(
                        ui,
                        inputs.layer_summaries,
                        inputs.vector_summaries,
                        inputs.unavailable_notices,
                        layers_state,
                    );
                }
                WorkbenchTab::CloudStac => {
                    out.stac_actions = show_stac_tab_contents(
                        ui,
                        inputs.stac,
                        inputs.settings,
                        inputs.vector_summaries,
                        inputs.draw_aoi_active,
                        stac_state,
                    );
                }
            }
        });

    if tab != inputs.active_tab {
        out.selected_tab = Some(tab);
    }
    out
}
