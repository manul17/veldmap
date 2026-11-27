//! Cache & GPU tab inside the unified Settings modal (SQLite disk cache
//! statistics, disk/VRAM budget sliders, and off-UI-thread clear trigger).

use eframe::egui;

use crate::storage::settings::{
    DEFAULT_CUSTOM_GPU_CACHE_MAX_MB, DISK_CACHE_MAX_MB_RANGE, GPU_CACHE_MAX_MB_RANGE,
};
use crate::storage::{DiskCacheNotice, DiskCacheStatus};

use super::SettingsAction;

fn format_disk_cache_notice(notice: &DiskCacheNotice) -> String {
    match notice {
        DiskCacheNotice::Cleared => "Disk cache cleared.".to_owned(),
        DiskCacheNotice::ClearFailed(err) => format!("Clear failed: {err}"),
        DiskCacheNotice::WorkerUnavailable => "Disk cache worker unavailable.".to_owned(),
        DiskCacheNotice::WorkerDisconnected => {
            "Clear failed: maintenance worker disconnected.".to_owned()
        }
    }
}

pub(super) fn show_cache_tab(
    ui: &mut egui::Ui,
    disk_status: &DiskCacheStatus,
    disk_cache_max_mb: &mut u64,
    gpu_tile_cache_max_mb: &mut Option<u64>,
    actions: &mut Vec<SettingsAction>,
) {
    ui.heading("Tile Disk & GPU Cache");
    if disk_status.available {
        if let Some(stats) = disk_status.stats {
            let used_mb = (stats.payload_bytes as f64) / (1024.0 * 1024.0);
            let cap_mb = (stats.max_bytes as f64) / (1024.0 * 1024.0);
            ui.monospace(format!(
                "Cached: {} tiles · {:.1} MiB / {:.0} MiB",
                stats.tile_count, used_mb, cap_mb
            ));
        } else {
            ui.label(egui::RichText::new("Loading cache statistics…").weak());
        }
    } else {
        ui.label(egui::RichText::new("Disk cache unavailable in this session.").weak());
    }

    ui.horizontal(|ui| {
        ui.label("Disk cap (MiB):");
        ui.add(egui::Slider::new(disk_cache_max_mb, DISK_CACHE_MAX_MB_RANGE).logarithmic(true));
    });

    ui.horizontal(|ui| {
        let mut custom_vram = gpu_tile_cache_max_mb.is_some();
        if ui
            .checkbox(&mut custom_vram, "Custom GPU VRAM cap (MiB):")
            .changed()
        {
            *gpu_tile_cache_max_mb = if custom_vram {
                Some(DEFAULT_CUSTOM_GPU_CACHE_MAX_MB)
            } else {
                None
            };
        }
        if let Some(mb) = gpu_tile_cache_max_mb {
            ui.add(egui::Slider::new(mb, GPU_CACHE_MAX_MB_RANGE).logarithmic(true));
        } else {
            ui.weak("Auto (host RAM scaled)");
        }
    });

    ui.horizontal(|ui| {
        let clear_label = if disk_status.busy {
            "Clearing…"
        } else {
            "Clear Disk Cache"
        };
        if ui
            .add_enabled(
                disk_status.available && !disk_status.busy,
                egui::Button::new(clear_label),
            )
            .clicked()
        {
            actions.push(SettingsAction::ClearDiskCache);
        }
        if let Some(notice) = &disk_status.notice {
            ui.label(
                egui::RichText::new(format_disk_cache_notice(notice))
                    .weak()
                    .small(),
            );
        }
    });
}
