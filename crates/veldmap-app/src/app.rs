use std::sync::Arc;

use eframe::{egui, egui_wgpu};
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
        tracing::info!(
            adapter = %info.name,
            backend = ?info.backend,
            device_type = ?info.device_type,
            target_format = ?render_state.target_format,
            "Initialized wgpu render state"
        );

        render_state.device.on_uncaptured_error(Arc::new(|err| {
            tracing::error!(error = %err, "Uncaptured wgpu device error");
        }));

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
                let available = ui.available_size();
                let (response, painter) = ui.allocate_painter(available, egui::Sense::drag());
                if response.rect.is_positive() {
                    painter.add(egui_wgpu::Callback::new_paint_callback(
                        response.rect,
                        ViewportCallback,
                    ));
                }
            });
    }
}
