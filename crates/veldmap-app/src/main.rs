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

/// Supported native desktop backends (Vulkan, DX12, Metal), excluding
/// OpenGL/GLES and `BROWSER_WEBGPU` (which `wgpu::Backends::PRIMARY` includes).
const NATIVE_PRIMARY_BACKENDS: wgpu::Backends =
    wgpu::Backends::PRIMARY.difference(wgpu::Backends::BROWSER_WEBGPU);

/// Restrict `egui_wgpu`'s environment-resolved backends to native desktop
/// primary backends (Vulkan, DX12, Metal), explicitly excluding OpenGL/GLES
/// and WebGPU.
fn clamp_to_primary_backends(backends: wgpu::Backends) -> wgpu::Backends {
    let unsupported = backends - NATIVE_PRIMARY_BACKENDS;
    if !unsupported.is_empty() {
        tracing::warn!(
            ?unsupported,
            "Ignoring unsupported WGPU_BACKEND entries (only Vulkan, DX12, and Metal are supported)"
        );
    }
    let primary = backends & NATIVE_PRIMARY_BACKENDS;
    if primary.is_empty() {
        tracing::warn!(
            "WGPU_BACKEND selected no supported backends; falling back to native primary backends"
        );
        NATIVE_PRIMARY_BACKENDS
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

#[cfg(test)]
mod tests {
    use super::{NATIVE_PRIMARY_BACKENDS, clamp_to_primary_backends};

    #[test]
    fn clamps_backends_to_primary_and_falls_back_when_empty() {
        assert_eq!(
            clamp_to_primary_backends(wgpu::Backends::VULKAN),
            wgpu::Backends::VULKAN
        );
        assert_eq!(
            clamp_to_primary_backends(wgpu::Backends::VULKAN | wgpu::Backends::GL),
            wgpu::Backends::VULKAN
        );
        assert_eq!(
            clamp_to_primary_backends(wgpu::Backends::GL | wgpu::Backends::BROWSER_WEBGPU),
            NATIVE_PRIMARY_BACKENDS
        );
        assert_eq!(
            clamp_to_primary_backends(wgpu::Backends::empty()),
            NATIVE_PRIMARY_BACKENDS
        );
    }
}
