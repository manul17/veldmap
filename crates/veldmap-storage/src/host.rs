//! Platform physical host RAM inspection for deriving startup [`MemoryBudget`]
//! caps outside pure `veldmap-core`.

use veldmap_core::budget::{DEFAULT_FALLBACK_HOST_RAM_BYTES, MemoryBudget};

/// Probe physical host RAM in bytes using OS-specific facilities, returning
/// `None` if unavailable on this platform.
fn detect_host_ram_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
        parse_proc_meminfo_total_bytes(&contents)
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let text = std::str::from_utf8(&out.stdout).ok()?.trim();
        text.parse::<u64>().ok()
    }
    #[cfg(target_os = "windows")]
    {
        detect_windows_ram_bytes()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

#[cfg(target_os = "windows")]
#[repr(C)]
struct MemoryStatusEx {
    dw_length: u32,
    dw_memory_load: u32,
    ull_total_phys: u64,
    ull_avail_phys: u64,
    ull_total_page_file: u64,
    ull_avail_page_file: u64,
    ull_total_virtual: u64,
    ull_avail_virtual: u64,
    ull_avail_extended_virtual: u64,
}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
}

#[cfg(target_os = "windows")]
fn detect_windows_ram_bytes() -> Option<u64> {
    let mut status = MemoryStatusEx {
        dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };
    // SAFETY: `status` is a properly initialized `MEMORYSTATUSEX` with
    // `dw_length` set to its exact byte size, and `GlobalMemoryStatusEx` only
    // writes into the provided buffer for the duration of the call.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    (ok != 0 && status.ull_total_phys > 0).then_some(status.ull_total_phys)
}

/// Detect physical host RAM and derive startup [`MemoryBudget`] caps, falling
/// back to [`DEFAULT_FALLBACK_HOST_RAM_BYTES`] (`8 GiB`) if unavailable.
pub fn detect_memory_budget() -> MemoryBudget {
    MemoryBudget::for_host_ram(detect_host_ram_bytes().unwrap_or(DEFAULT_FALLBACK_HOST_RAM_BYTES))
}

#[cfg(any(target_os = "linux", test))]
fn parse_proc_meminfo_total_bytes(meminfo: &str) -> Option<u64> {
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kib_str = rest.split_whitespace().next()?;
            let kib = kib_str.parse::<u64>().ok()?;
            return kib.checked_mul(1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_proc_meminfo_and_detects_valid_budget() {
        let sample = "MemTotal:       32845120 kB\nMemFree:        123456 kB\n";
        assert_eq!(
            parse_proc_meminfo_total_bytes(sample),
            Some(32_845_120 * 1024)
        );
        let budget = detect_memory_budget();
        assert!(budget.host_ram_bytes > 0);
        assert!(budget.gpu_tile_cache_bytes >= 256 * 1024 * 1024);
    }
}
