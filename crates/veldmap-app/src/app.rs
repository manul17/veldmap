use eframe::egui;
use thiserror::Error;

use crate::render::{ViewportCallback, ViewportPipeline};

#[derive(Debug, Error)]
pub enum AppError {
    #[error("eframe did not provide a wgpu render state")]
    MissingWgpu,
}

pub struct VeldmapApp {
    adapter_label: String,
}

impl VeldmapApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Result<Self, AppError> {
        let render_state = cc.wgpu_render_state.as_ref().ok_or(AppError::MissingWgpu)?;
        let info = render_state.adapter.get_info();
        let adapter_label = format!("{} ({:?})", info.name, info.backend);

        let pipeline = ViewportPipeline::new(&render_state.device, render_state.target_format);
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(pipeline);

        Ok(Self { adapter_label })
    }
}

impl eframe::App for VeldmapApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("status_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong(veldmap_core::APP_NAME);
                ui.separator();
                ui.label(&self.adapter_label);
            });
        });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                let rect = ui.available_rect_before_wrap();
                let (_response, painter) = ui.allocate_painter(rect.size(), egui::Sense::drag());
                painter.add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    ViewportCallback,
                ));
            });
    }
}
