//! Bottom GIS status bar displaying cursor geodetic coordinates & compact
//! MGRS, zoom-invariant native overlay pixel readouts, camera center, altitude,
//! heading, viewport GSD, native source pixel size, and transient copy/export
//! notices in a strictly single-line fixed-height layout.

use eframe::egui;
use veldmap_core::geo::Camera;

use crate::layers::CursorReadout;

use super::format;

/// Read-only inputs rendered by the bottom GIS status bar.
#[derive(Clone, Copy)]
pub(crate) struct StatusBarInputs<'a> {
    pub(crate) cursor_geo: Option<(f64, f64)>,
    pub(crate) cursor_readout: Option<&'a CursorReadout>,
    pub(crate) topmost_source_px: Option<(f64, f64)>,
    pub(crate) camera: &'a Camera,
    pub(crate) viewport_height_px: f64,
    pub(crate) status_notice: Option<&'a str>,
}

fn render_readout_strip(ui: &mut egui::Ui, inputs: StatusBarInputs<'_>) {
    let StatusBarInputs {
        cursor_geo,
        cursor_readout,
        topmost_source_px,
        camera,
        viewport_height_px,
        ..
    } = inputs;
    ui.horizontal(|ui| {
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        let cursor_text = match cursor_geo {
            Some((lat, lon)) => format!(
                "Cursor: {} · {}",
                format::format_lat_lon(lat, lon),
                format::format_mgrs_compact(lat, lon)
            ),
            None => "Cursor: Off globe".to_owned(),
        };
        ui.monospace(cursor_text).on_hover_text(
            "Right-click globe or press Ctrl/Cmd+Shift+C to copy coordinates (DD / DMS / MGRS)",
        );

        if let Some(readout) = cursor_readout {
            ui.separator();
            let (val_text, tip) = format::format_cursor_readout(readout);
            ui.monospace(val_text).on_hover_text(tip);
        }

        ui.separator();
        let (clat, clon) = camera.center();
        ui.monospace(format!("Center: {}", format::format_lat_lon(clat, clon)));
        ui.separator();

        ui.monospace(format!(
            "Alt: {}",
            format::format_distance_m(camera.height_m())
        ));

        let heading = camera.heading_deg();
        if heading.abs() > 0.05 {
            ui.separator();
            ui.monospace(format!("Hdg: {heading:+.1}°"));
        }

        ui.separator();
        let mpp = camera.metres_per_pixel(viewport_height_px);
        ui.monospace(format!("GSD: {}", format::format_resolution(mpp)));

        let source_px = cursor_readout
            .and_then(|r| r.source_pixel_size_m)
            .or(topmost_source_px);
        if let Some((mx, my)) = source_px {
            ui.separator();
            ui.monospace(format!("Src: {}", format::format_source_pixel_size(mx, my)));
        }
    });
}

/// Render the bottom GIS status bar in a strictly single-line layout (never
/// wrapping or resizing the globe viewport), right-pinning transient notices
/// so copy/export confirmations remain visible on standard window widths.
pub(crate) fn show_status_bar(ui: &mut egui::Ui, inputs: StatusBarInputs<'_>) {
    let frame = egui::Frame::side_top_panel(ui.style()).inner_margin(egui::Margin::symmetric(8, 2));
    egui::Panel::bottom("gis_status_bar")
        .frame(frame)
        .exact_size(24.0)
        .show(ui, |ui| {
            // Strictly single-line clipped layout so hovering multi-band layers
            // or showing MGRS never wraps, resizes the globe viewport, or
            // reallocates GPU offscreen/scratch targets.
            ui.set_clip_rect(ui.max_rect());
            if let Some(notice) = inputs.status_notice {
                egui::Sides::new()
                    .shrink_left()
                    .height(ui.available_height())
                    .show(
                        ui,
                        |ui| {
                            ui.shrink_clip_rect(ui.max_rect());
                            render_readout_strip(ui, inputs);
                        },
                        |ui| {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(notice)
                                        .color(super::BADGE_RECIPE_COLOR)
                                        .monospace()
                                        .small(),
                                )
                                .truncate(),
                            )
                            .on_hover_text(notice);
                            ui.separator();
                        },
                    );
            } else {
                render_readout_strip(ui, inputs);
            }
        });
}
