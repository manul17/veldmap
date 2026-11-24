//! Bottom GIS status bar displaying cursor geodetic coordinates,
//! zoom-invariant native overlay pixel readouts, camera center, altitude,
//! heading, viewport GSD, and native source pixel size.

use eframe::egui;
use veldmap_core::geo::Camera;

use crate::layers::{CursorReadout, CursorReadoutValue};

use super::format;

/// Render the bottom GIS status bar.
pub(crate) fn show_status_bar(
    ui: &mut egui::Ui,
    cursor_geo: Option<(f64, f64)>,
    cursor_readout: Option<&CursorReadout>,
    topmost_source_px: Option<(f64, f64)>,
    camera: &Camera,
    viewport_height_px: f64,
) {
    egui::Panel::bottom("gis_status_bar").show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            let cursor_text = match cursor_geo {
                Some((lat, lon)) => format!("Cursor: {}", format::format_lat_lon(lat, lon)),
                None => "Cursor: Off globe".to_owned(),
            };
            ui.monospace(cursor_text);

            if let Some(readout) = cursor_readout {
                ui.separator();
                let val_text = match readout.value {
                    CursorReadoutValue::Rgba([r, g, b, a]) => {
                        if a == 255 {
                            format!(
                                "RGB({r},{g},{b}) @ ({},{})",
                                readout.pixel.0, readout.pixel.1
                            )
                        } else {
                            format!(
                                "RGBA({r},{g},{b},{a}) @ ({},{})",
                                readout.pixel.0, readout.pixel.1
                            )
                        }
                    }
                    CursorReadoutValue::Pending => {
                        format!("RGB(…) @ ({},{})", readout.pixel.0, readout.pixel.1)
                    }
                    CursorReadoutValue::Nodata => {
                        format!("nodata @ ({},{})", readout.pixel.0, readout.pixel.1)
                    }
                    CursorReadoutValue::Unavailable => {
                        format!("RGB(n/a) @ ({},{})", readout.pixel.0, readout.pixel.1)
                    }
                };
                ui.monospace(val_text).on_hover_text(format!(
                    "Native base-resolution readout: {}",
                    readout.layer_name
                ));
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
