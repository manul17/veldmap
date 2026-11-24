//! Native desktop binary entrypoint for VeldMap.

use eframe::egui;
use eframe::egui_wgpu;
use tracing_subscriber::EnvFilter;
use veldmap_app::VeldmapApp;
use veldmap_app::render::clamp_to_primary_backends;

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(
            "info,veldmap=debug,veldmap_app=debug,veldmap_core=debug,veldmap_raster=debug,veldmap_net=debug,veldmap_storage=debug,wgpu_core=warn,wgpu_hal=warn",
        )
    });
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
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

    let mut startup_files = Vec::new();
    let mut positional_only = false;
    for arg in std::env::args_os().skip(1) {
        if !positional_only && let Some(flag) = arg.to_str() {
            if flag == "--" {
                positional_only = true;
                continue;
            }
            if flag == "-h" || flag == "--help" {
                println!("Usage: veldmap [OPTIONS] [GEOTIFF_OR_COG_PATH...]");
                return Ok(());
            }
            // Skip unknown flags and macOS Finder launch arguments (`-psn_*`).
            if flag.starts_with('-') {
                tracing::warn!(flag, "Ignoring unrecognized CLI flag");
                continue;
            }
        }
        startup_files.push(std::path::PathBuf::from(arg));
    }

    eframe::run_native(
        veldmap_core::APP_NAME,
        options,
        Box::new(move |cc| {
            Ok(Box::new(VeldmapApp::new_with_startup_files(
                cc,
                startup_files,
            )?))
        }),
    )
}
