//! Bottom GIS status bar displaying cursor geodetic coordinates,
//! zoom-invariant native overlay pixel readouts, camera center, altitude,
//! heading, viewport GSD, and native source pixel size.

use eframe::egui;
use veldmap_core::geo::Camera;

use crate::colormap::sentinel2_scl_class_name;
use crate::layers::{CursorReadout, CursorReadoutValue};

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
        ui.horizontal_wrapped(|ui| {
            let cursor_text = match cursor_geo {
                Some((lat, lon)) => format!("Cursor: {}", format::format_lat_lon(lat, lon)),
                None => "Cursor: Off globe".to_owned(),
            };
            ui.monospace(cursor_text);

            if let Some(readout) = cursor_readout {
                ui.separator();
                let val_text = match &readout.value {
                    CursorReadoutValue::Rgba([r, g, b, a]) => {
                        if *a == 255 {
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
                    CursorReadoutValue::Scalar { raw, physical } => {
                        let unit = readout
                            .units
                            .as_deref()
                            .map(|u| format!(" {u}"))
                            .unwrap_or_default();
                        let db_part = if readout.log10_db && *physical > 0.0 {
                            format!(" ({:.2} dB)", 10.0 * physical.log10())
                        } else {
                            String::new()
                        };
                        let raw_part = if readout.has_affine_scaling {
                            format!(" [raw {}]", format::format_scalar(*raw))
                        } else {
                            String::new()
                        };
                        format!(
                            "Val: {}{unit}{db_part}{raw_part} @ ({},{})",
                            format::format_scalar(*physical),
                            readout.pixel.0,
                            readout.pixel.1
                        )
                    }
                    CursorReadoutValue::Categorical {
                        class_id,
                        class_name,
                    } => match class_name {
                        Some(name) => format!(
                            "SCL {class_id}: {name} @ ({},{})",
                            readout.pixel.0, readout.pixel.1
                        ),
                        None => format!(
                            "Class {class_id} @ ({},{})",
                            readout.pixel.0, readout.pixel.1
                        ),
                    },
                    CursorReadoutValue::BandMath(bm) => {
                        let ch_str = bm
                            .channels
                            .iter()
                            .map(|(lbl, val)| match val {
                                Some(v) => format!("{lbl}: {}", format::format_scalar(*v)),
                                None => format!("{lbl}: nodata"),
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        let scl_str = match bm.scl_class {
                            Some(cls) => match sentinel2_scl_class_name(cls) {
                                Some(name) => format!(" · SCL {cls}: {name}"),
                                None => format!(" · SCL {cls}"),
                            },
                            None => String::new(),
                        };
                        let main_str = if bm.scl_masked {
                            format!("{}: masked ({ch_str})", bm.formula_label)
                        } else if bm.is_scalar {
                            match bm.computed {
                                Some(v) => {
                                    let unit = readout
                                        .units
                                        .as_deref()
                                        .map(|u| format!(" {u}"))
                                        .unwrap_or_default();
                                    format!(
                                        "{}: {}{unit} ({ch_str})",
                                        bm.formula_label,
                                        format::format_scalar(v)
                                    )
                                }
                                None => format!("{}: nodata ({ch_str})", bm.formula_label),
                            }
                        } else {
                            format!("{}: ({ch_str})", bm.formula_label)
                        };
                        format!(
                            "{main_str}{scl_str} @ ({},{})",
                            readout.pixel.0, readout.pixel.1
                        )
                    }
                    CursorReadoutValue::Pending => {
                        format!("… @ ({},{})", readout.pixel.0, readout.pixel.1)
                    }
                    CursorReadoutValue::Nodata => {
                        format!("nodata @ ({},{})", readout.pixel.0, readout.pixel.1)
                    }
                    CursorReadoutValue::Unavailable => {
                        format!("n/a @ ({},{})", readout.pixel.0, readout.pixel.1)
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
