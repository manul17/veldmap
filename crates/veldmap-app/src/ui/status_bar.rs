//! Bottom GIS status bar displaying cursor geodetic coordinates,
//! zoom-invariant native overlay pixel readouts, camera center, altitude,
//! heading, viewport GSD, and native source pixel size.

use eframe::egui;
use veldmap_core::geo::Camera;

use crate::colormap::sentinel2_scl_class_name;
use crate::layers::{BandChannelSample, CursorReadout, CursorReadoutValue};

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
                let mut hover_extra: Option<String> = None;
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
                        let fmt_sample = |sample: BandChannelSample| match sample {
                            BandChannelSample::Value(v) => format::format_scalar(v),
                            BandChannelSample::Nodata => "nodata".to_owned(),
                            BandChannelSample::Unavailable => "n/a".to_owned(),
                        };
                        let detailed_channels = bm
                            .channels
                            .iter()
                            .map(|(lbl, sample)| format!("{lbl}: {}", fmt_sample(*sample)))
                            .collect::<Vec<_>>()
                            .join(", ");
                        let detailed_scl = match bm.scl_class {
                            Some(cls) => match sentinel2_scl_class_name(cls) {
                                Some(name) => format!("\nSCL {cls}: {name}"),
                                None => format!("\nSCL {cls}"),
                            },
                            None => String::new(),
                        };
                        hover_extra = Some(format!(
                            "{detailed_channels}{detailed_scl} @ ({},{})",
                            readout.pixel.0, readout.pixel.1
                        ));

                        let masked_tag = if bm.scl_masked { " [masked]" } else { "" };
                        let short_scl = bm
                            .scl_class
                            .map(|cls| format!(" · SCL {cls}"))
                            .unwrap_or_default();
                        let main_str = if bm.is_scalar {
                            match bm.computed {
                                Some(v) => {
                                    let unit = readout
                                        .units
                                        .as_deref()
                                        .map(|u| format!(" {u}"))
                                        .unwrap_or_default();
                                    format!(
                                        "{}: {}{unit}{masked_tag}",
                                        bm.formula_label,
                                        format::format_scalar(v)
                                    )
                                }
                                None => {
                                    let missing =
                                        if bm.channels.iter().take(2).any(|(_, s)| {
                                            matches!(s, BandChannelSample::Unavailable)
                                        }) {
                                            "n/a"
                                        } else {
                                            "nodata"
                                        };
                                    format!("{}: {missing}{masked_tag}", bm.formula_label)
                                }
                            }
                        } else {
                            let compact_vals = bm
                                .channels
                                .iter()
                                .map(|(_, sample)| fmt_sample(*sample))
                                .collect::<Vec<_>>()
                                .join(", ");
                            format!("{}: ({compact_vals}){masked_tag}", bm.formula_label)
                        };
                        format!("{main_str}{short_scl}")
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
                let tip = match hover_extra {
                    Some(details) => format!(
                        "Native base-resolution readout: {}\n{details}",
                        readout.layer_name
                    ),
                    None => format!("Native base-resolution readout: {}", readout.layer_name),
                };
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
