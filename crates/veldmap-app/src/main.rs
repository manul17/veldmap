mod app;
mod render;

use eframe::egui;

fn main() -> eframe::Result<()> {
    let mut options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([640.0, 480.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        setup.instance_descriptor.backends = eframe::egui_wgpu::wgpu::Backends::PRIMARY;
        setup.power_preference = eframe::egui_wgpu::wgpu::PowerPreference::HighPerformance;
    }

    eframe::run_native(
        veldmap_core::APP_NAME,
        options,
        Box::new(|cc| Ok(Box::new(app::VeldmapApp::new(cc)?))),
    )
}
