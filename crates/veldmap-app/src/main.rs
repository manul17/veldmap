mod app;
mod render;

use eframe::egui;
use eframe::egui_wgpu;
use tracing_subscriber::EnvFilter;

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,veldmap=debug,wgpu_core=warn,wgpu_hal=warn"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

/// Parse `WGPU_BACKEND` while explicitly excluding OpenGL/GLES/WebGPU so the
/// application stays on `Backends::PRIMARY` (Vulkan, DX12, Metal).
fn configured_backends() -> wgpu::Backends {
    let Ok(raw) = std::env::var("WGPU_BACKEND") else {
        return wgpu::Backends::PRIMARY;
    };

    let mut backends = wgpu::Backends::empty();
    for token in raw.split([',', '|', ' ']).filter(|s| !s.is_empty()) {
        match token.to_ascii_lowercase().as_str() {
            "primary" => backends |= wgpu::Backends::PRIMARY,
            "vulkan" | "vk" => backends |= wgpu::Backends::VULKAN,
            "dx12" | "d3d12" => backends |= wgpu::Backends::DX12,
            "metal" | "mtl" => backends |= wgpu::Backends::METAL,
            "gl" | "opengl" | "gles" | "webgpu" => {
                tracing::warn!(
                    backend = token,
                    "Ignoring unsupported WGPU_BACKEND value (only Vulkan, DX12, and Metal are supported)"
                );
            }
            other => {
                tracing::warn!(backend = other, "Ignoring unknown WGPU_BACKEND value");
            }
        }
    }

    if backends.is_empty() {
        tracing::warn!(
            value = raw,
            "WGPU_BACKEND selected no supported backends; falling back to Backends::PRIMARY"
        );
        wgpu::Backends::PRIMARY
    } else {
        backends
    }
}

/// Parse `WGPU_POWER_PREF` (`high`, `low`, `none`).
fn configured_power_preference() -> wgpu::PowerPreference {
    let Ok(raw) = std::env::var("WGPU_POWER_PREF") else {
        return wgpu::PowerPreference::HighPerformance;
    };

    match raw.trim().to_ascii_lowercase().as_str() {
        "high" | "high_performance" | "high-performance" => wgpu::PowerPreference::HighPerformance,
        "low" | "low_power" | "low-power" => wgpu::PowerPreference::LowPower,
        "none" | "default" => wgpu::PowerPreference::None,
        other => {
            tracing::warn!(
                value = other,
                "Ignoring unknown WGPU_POWER_PREF value; using HighPerformance"
            );
            wgpu::PowerPreference::HighPerformance
        }
    }
}

fn main() -> eframe::Result<()> {
    init_tracing();

    let backends = configured_backends();
    let power_preference = configured_power_preference();
    tracing::info!(?backends, ?power_preference, "Starting VeldMap");

    let mut options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([640.0, 480.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    if let egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        setup.instance_descriptor.backends = backends;
        setup.power_preference = power_preference;
    }

    eframe::run_native(
        veldmap_core::APP_NAME,
        options,
        Box::new(|cc| Ok(Box::new(app::VeldmapApp::new(cc)?))),
    )
}
