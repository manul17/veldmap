//! Presentation-only floating Export Cartographic PNG options window.

use eframe::egui;

use crate::export::CartographicExportOptions;

/// User action emitted by [`show_export_png_window`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExportPngUiAction {
    SaveAs,
}

/// Read-only status inputs for [`show_export_png_window`].
#[derive(Clone, Copy)]
pub(crate) struct ExportPngWindowInputs<'a> {
    pub(crate) has_colorbar: bool,
    pub(crate) busy: bool,
    pub(crate) last_notice: Option<&'a str>,
}

/// Render the floating Export Cartographic PNG options modal when `*open` is
/// `true`.
pub(crate) fn show_export_png_window(
    ctx: &egui::Context,
    open: &mut bool,
    options: &mut CartographicExportOptions,
    inputs: ExportPngWindowInputs<'_>,
) -> Option<ExportPngUiAction> {
    if !*open {
        return None;
    }
    let mut action = None;
    let mut window_open = *open;
    egui::Window::new("Export Cartographic PNG")
        .open(&mut window_open)
        .resizable(false)
        .default_width(320.0)
        .show(ctx, |ui| {
            ui.label("Cartographic overlays composited off the UI thread:");
            ui.add_space(4.0);
            ui.checkbox(&mut options.include_scale_bar, "Metric scale bar");
            ui.checkbox(
                &mut options.include_north_arrow,
                "North arrow (heading-rotated)",
            );
            ui.add_enabled(
                inputs.has_colorbar,
                egui::Checkbox::new(
                    &mut options.include_colorbar,
                    "Scientific / Band Math colorbar",
                ),
            );
            ui.checkbox(
                &mut options.include_caption,
                "Acquisition date / title badge",
            );
            if options.include_caption {
                ui.add(
                    egui::TextEdit::singleline(&mut options.caption_text)
                        .desired_width(280.0)
                        .hint_text("e.g. Sentinel-2 L2A · 2026-11-27 UTC"),
                );
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!inputs.busy, egui::Button::new("Save PNG…"))
                    .clicked()
                {
                    action = Some(ExportPngUiAction::SaveAs);
                }
                if inputs.busy {
                    ui.spinner();
                    ui.weak("Exporting…");
                }
            });
            if let Some(note) = inputs.last_notice {
                ui.separator();
                ui.weak(note);
            }
        });
    *open = window_open;
    action
}
