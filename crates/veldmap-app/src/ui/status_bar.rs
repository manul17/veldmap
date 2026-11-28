//! Bottom GIS status bar displaying cursor geodetic coordinates & compact
//! MGRS, zoom-invariant native overlay pixel readouts, camera center, altitude,
//! heading, viewport GSD, native source pixel size, and transient copy/export
//! notices in a strictly single-line fixed-height layout.

use eframe::egui;
use veldmap_core::geo::Camera;

use crate::format;
use crate::layers::{BandChannelSample, BandMathReadout, CursorReadout, CursorReadoutValue};

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

fn format_band_channel_sample(sample: BandChannelSample) -> String {
    match sample {
        BandChannelSample::Value(v) => format::format_scalar(v),
        BandChannelSample::Nodata => "nodata".to_owned(),
        BandChannelSample::Unavailable => "n/a".to_owned(),
    }
}

fn format_band_math_readout(readout: &CursorReadout, bm: &BandMathReadout) -> (String, String) {
    let detailed_channels = bm
        .channels
        .iter()
        .map(|(lbl, sample)| format!("{lbl}: {}", format_band_channel_sample(*sample)))
        .collect::<Vec<_>>()
        .join(", ");
    let detailed_scl = match bm.scl_class {
        Some(cls) => match bm.scl_class_name {
            Some(name) => format!("\nSCL {cls}: {name}"),
            None => format!("\nSCL {cls}"),
        },
        None => String::new(),
    };
    let hover_extra = format!(
        "{detailed_channels}{detailed_scl} @ ({},{})",
        readout.pixel.0, readout.pixel.1
    );

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
                let missing = if bm
                    .channels
                    .iter()
                    .take(2)
                    .any(|(_, s)| matches!(s, BandChannelSample::Unavailable))
                {
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
            .map(|(_, sample)| format_band_channel_sample(*sample))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}: ({compact_vals}){masked_tag}", bm.formula_label)
    };
    (format!("{main_str}{short_scl}"), hover_extra)
}

/// Format a zoom-invariant [`CursorReadout`] into `(status_text, tooltip_text)`
/// for the single-line GIS status bar.
fn format_cursor_readout(readout: &CursorReadout) -> (String, String) {
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
            let (text, extra) = format_band_math_readout(readout, bm);
            hover_extra = Some(extra);
            text
        }
        CursorReadoutValue::Pending => format!("… @ ({},{})", readout.pixel.0, readout.pixel.1),
        CursorReadoutValue::Nodata => format!("nodata @ ({},{})", readout.pixel.0, readout.pixel.1),
        CursorReadoutValue::Unavailable => {
            format!("n/a @ ({},{})", readout.pixel.0, readout.pixel.1)
        }
    };
    let remote_cache_hint = readout.is_remote
        && match &readout.value {
            CursorReadoutValue::Unavailable => true,
            CursorReadoutValue::BandMath(bm) => bm.has_unavailable,
            _ => false,
        };
    let mut tip = match hover_extra {
        Some(details) => format!(
            "Native base-resolution readout: {}\n{details}",
            readout.layer_name
        ),
        None => format!("Native base-resolution readout: {}", readout.layer_name),
    };
    if remote_cache_hint {
        tip.push_str("\nRemote COG hover is cache-only: zoom in to load native-resolution tiles.");
    }
    (val_text, tip)
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
            let (val_text, tip) = format_cursor_readout(readout);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_unavailable_cursor_readout_tooltip_explains_cache_only_zoom_behavior() {
        let base = CursorReadout {
            layer_name: "S2 Red".to_owned(),
            is_remote: true,
            pixel: (10, 20),
            source_pixel_size_m: Some((10.0, 10.0)),
            units: None,
            has_affine_scaling: false,
            log10_db: false,
            value: CursorReadoutValue::Unavailable,
        };
        let (text, tip) = format_cursor_readout(&base);
        assert_eq!(text, "n/a @ (10,20)");
        assert!(tip.contains("cache-only"), "expected remote hint: {tip}");

        let local = CursorReadout {
            is_remote: false,
            ..base.clone()
        };
        let (_, local_tip) = format_cursor_readout(&local);
        assert!(!local_tip.contains("cache-only"));

        // BandMath where float bands are cached but active remote SCL mask is uncached.
        let bm_scl_missing = CursorReadout {
            value: CursorReadoutValue::BandMath(BandMathReadout {
                formula_label: "NDVI".to_owned(),
                is_scalar: true,
                computed: Some(0.5),
                channels: vec![
                    ("NIR".to_owned(), BandChannelSample::Value(0.6)),
                    ("Red".to_owned(), BandChannelSample::Value(0.2)),
                ],
                scl_class: None,
                scl_class_name: None,
                scl_masked: false,
                has_unavailable: true,
            }),
            ..base
        };
        let (_, bm_tip) = format_cursor_readout(&bm_scl_missing);
        assert!(
            bm_tip.contains("cache-only"),
            "expected remote cache-only hint when BandMath has_unavailable is true: {bm_tip}"
        );
    }
}
