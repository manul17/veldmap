use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use veldmap_core::tile::TileScheme;
use veldmap_storage::DiskTileKey;

use super::tests_support::TempTestRoot;
use super::*;
use crate::settings::BasemapSelection;

#[test]
fn update_settings_skips_noop_writes_and_persists_changes() {
    let dir = TempTestRoot::new("storage-noop");
    let paths = StoragePaths::with_root(dir.path());
    let settings_path = paths.settings_path();
    let wakeup: Wakeup = Arc::new(|| {});
    let mut storage = AppStorage::open(wakeup, Some(paths));

    // No-op update must return false and avoid creating settings.json unnecessarily.
    let same = storage.settings().clone();
    assert!(!storage.update_settings(same));
    assert!(
        !settings_path.exists(),
        "no-op update_settings should not fsync unchanged settings"
    );

    // Real mutation persists settings.json and updates cap.
    assert!(storage.mutate_settings(|s| {
        s.active_basemap = BasemapSelection::EoxSentinel2Cloudless;
        s.disk_cache_max_mb = 128;
    }));
    assert!(settings_path.exists());
    assert_eq!(storage.settings().disk_cache_max_mb, 128);

    storage.begin_shutdown();
    storage.finish_shutdown();
}

#[test]
fn disk_cache_worker_handles_refresh_cap_clear_and_disconnected_recovery() {
    let dir = TempTestRoot::new("storage-worker");
    let paths = StoragePaths::with_root(dir.path());
    let wakes = Arc::new(AtomicUsize::new(0));
    let wakes_c = Arc::clone(&wakes);
    let wakeup: Wakeup = Arc::new(move || {
        wakes_c.fetch_add(1, Ordering::AcqRel);
    });

    let mut storage = AppStorage::open(wakeup, Some(paths));
    let cache = storage.disk_cache().expect("disk cache opened");
    let tile_key = DiskTileKey::new("test-src", TileScheme::WebMercator, 1, 0, 0);
    cache.put(&tile_key, &[7u8; 256]).expect("put tile");

    // Opening Settings triggers an async stats refresh.
    storage.maybe_refresh_disk_stats(true);
    let deadline = Instant::now() + Duration::from_secs(2);
    while storage.disk_cache_ui_state().stats.is_none() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    let stats = storage
        .disk_cache_ui_state()
        .stats
        .expect("stats refreshed");
    assert_eq!(stats.tile_count, 1);
    assert_eq!(stats.payload_bytes, 256);

    // Async clear empties the DB, clears busy flag, sets notice, and bumps clear_gen.
    assert!(!storage.poll_clear_completed());
    storage.request_clear_disk_cache();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !storage.poll_clear_completed() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    let post_clear = storage.disk_cache_ui_state();
    assert!(!post_clear.busy);
    assert_eq!(post_clear.notice.as_deref(), Some("Disk cache cleared."));
    assert_eq!(post_clear.stats.expect("post-clear stats").tile_count, 0);
    assert!(!storage.poll_clear_completed(), "clear_gen consumed once");

    storage.clear_disk_cache_notice();
    assert!(storage.disk_cache_ui_state().notice.is_none());

    // After closing the worker channel, `request_clear_disk_cache()` must NOT
    // permanently wedge `busy = true`.
    storage.begin_shutdown();
    storage.request_clear_disk_cache();
    let after_closed = storage.disk_cache_ui_state();
    assert!(
        !after_closed.busy,
        "request_clear after worker disconnect must never wedge busy=true"
    );
    assert!(after_closed.notice.is_some());

    storage.finish_shutdown();
}
