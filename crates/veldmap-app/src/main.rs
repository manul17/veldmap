use eframe::egui;
use eframe::egui_wgpu;
use tracing_subscriber::EnvFilter;
use veldmap_app::app::VeldmapApp;

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(
            "info,veldmap=debug,veldmap_app=debug,veldmap_core=debug,wgpu_core=warn,wgpu_hal=warn",
        )
    });
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

/// Restrict `egui_wgpu`'s environment-resolved backends to `Backends::PRIMARY`
/// (Vulkan, DX12, Metal), explicitly excluding OpenGL/GLES/WebGPU.
fn clamp_to_primary_backends(backends: wgpu::Backends) -> wgpu::Backends {
    let unsupported = backends - wgpu::Backends::PRIMARY;
    if !unsupported.is_empty() {
        tracing::warn!(
            ?unsupported,
            "Ignoring unsupported WGPU_BACKEND entries (only Vulkan, DX12, and Metal are supported)"
        );
    }
    let primary = backends & wgpu::Backends::PRIMARY;
    if primary.is_empty() {
        tracing::warn!(
            "WGPU_BACKEND selected no supported backends; falling back to Backends::PRIMARY"
        );
        wgpu::Backends::PRIMARY
    } else {
        primary
    }
}

fn main() -> eframe::Result<()> {
    init_tracing();

    let mut options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([640.0, 480.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    if let egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        setup.instance_descriptor.backends =
            clamp_to_primary_backends(setup.instance_descriptor.backends);
        tracing::info!(
            backends = ?setup.instance_descriptor.backends,
            power_preference = ?setup.power_preference,
            "Starting VeldMap"
        );
    }

    eframe::run_native(
        veldmap_core::APP_NAME,
        options,
        Box::new(|cc| Ok(Box::new(VeldmapApp::new(cc)?))),
    )
}
