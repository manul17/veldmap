//! Platform physical host RAM inspection for deriving startup [`MemoryBudget`]
//! caps outside pure `veldmap-core`.

use veldmap_core::budget::{DEFAULT_FALLBACK_HOST_RAM_BYTES, MemoryBudget};

/// Probe physical host RAM in bytes using OS-specific facilities, returning
/// `None` if unavailable on this platform.
pub fn detect_host_ram_bytes() -> Option<u64> {
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
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
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
        assert!(budget.host_ram_bytes >= 1024 * 1024 * 1024);
    }
}
