//! Bottom GIS status bar displaying cursor geodetic coordinates,
//! zoom-invariant native overlay pixel readouts, camera center, altitude,
//! heading, viewport GSD, and native source pixel size.

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
}

/// Render the bottom GIS status bar.
pub(crate) fn show_status_bar(ui: &mut egui::Ui, inputs: StatusBarInputs<'_>) {
    let StatusBarInputs {
        cursor_geo,
        cursor_readout,
        topmost_source_px,
        camera,
        viewport_height_px,
    } = inputs;
    egui::Panel::bottom("gis_status_bar").show(ui, |ui| {
        // Strictly single-line layout so hovering multi-band layers never wraps
        // the status bar, resizes the globe viewport, or reallocates GPU targets.
        ui.horizontal(|ui| {
            let cursor_text = match cursor_geo {
                Some((lat, lon)) => format!("Cursor: {}", format::format_lat_lon(lat, lon)),
                None => "Cursor: Off globe".to_owned(),
            };
            ui.monospace(cursor_text);

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
            ui.monospace(format!("Viewport GSD: {}", format::format_resolution(mpp)));

            let source_px = cursor_readout
                .and_then(|r| r.source_pixel_size_m)
                .or(topmost_source_px);
            if let Some((mx, my)) = source_px {
                ui.separator();
                ui.monospace(format!(
                    "Source px: {}",
                    format::format_source_pixel_size(mx, my)
                ));
            }
        });
    });
}
