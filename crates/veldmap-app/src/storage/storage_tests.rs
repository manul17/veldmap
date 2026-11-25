use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use veldmap_core::tile::TileScheme;
use veldmap_storage::DiskTileKey;
use veldmap_storage::test_support::TestTempDir;

use super::settings::BasemapSelection;
use super::*;

#[test]
fn update_settings_skips_noop_writes_and_flushes_async_saves_on_shutdown() {
    let dir = TestTempDir::new("storage-noop");
    let paths = StoragePaths::with_root(dir.path());
    let settings_path = paths.settings_path();
    let wakeup: Wakeup = Arc::new(|| {});
    let mut storage = AppStorage::open(wakeup, Some(paths.clone()));

    // No-op update must return false and avoid creating settings.json unnecessarily.
    let same = storage.settings().clone();
    assert!(!storage.update_settings(same));
    assert!(
        !settings_path.exists(),
        "no-op update_settings should not fsync unchanged settings"
    );

    // Real mutation enqueues async durable save and updates cap.
    assert!(storage.mutate_settings(|s| {
        s.active_basemap = BasemapSelection::EoxSentinel2Cloudless;
        s.disk_cache_max_mb = 128;
    }));
    assert_eq!(storage.settings().disk_cache_max_mb, 128);

    // Deterministic shutdown drains the worker queue and flushes settings.json durably.
    storage.begin_shutdown();
    storage.finish_shutdown();

    assert!(settings_path.exists());
    let reloaded = AppSettings::load_or_default(&paths);
    assert_eq!(
        reloaded.active_basemap,
        BasemapSelection::EoxSentinel2Cloudless
    );
    assert_eq!(reloaded.disk_cache_max_mb, 128);
}

#[test]
fn disk_cache_worker_handles_refresh_cap_clear_and_disconnected_recovery() {
    let dir = TestTempDir::new("storage-worker");
    let paths = StoragePaths::with_root(dir.path());
    let wakes = Arc::new(AtomicUsize::new(0));
    let wakes_c = Arc::clone(&wakes);
    let wakeup: Wakeup = Arc::new(move || {
        wakes_c.fetch_add(1, Ordering::AcqRel);
    });

    let mut storage = AppStorage::open(wakeup, Some(paths));
    let cache = storage.disk_cache().expect("disk cache opened");
    let tile_key = DiskTileKey::new("test-src", TileScheme::WebMercator, 1, 0, 0);
    cache
        .put_encoded_if_uncancelled(
            &tile_key,
            veldmap_storage::TileBlobEncoding::EncodedImage,
            &[7u8; 256],
            &veldmap_core::sync::CancelToken::new(),
        )
        .expect("put tile");

    // Opening Settings triggers an async stats refresh.
    storage.maybe_refresh_disk_stats(true);
    let deadline = Instant::now() + Duration::from_secs(2);
    while storage.disk_cache_status().stats.is_none() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    let stats = storage.disk_cache_status().stats.expect("stats refreshed");
    assert_eq!(stats.tile_count, 1);
    assert_eq!(stats.payload_bytes, 256);

    // Async clear empties the DB, clears busy flag, sets notice, and bumps atomic clear_gen.
    assert!(!storage.poll_clear_completed());
    storage.request_clear_disk_cache();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !storage.poll_clear_completed() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    let post_clear = storage.disk_cache_status();
    assert!(!post_clear.busy);
    assert_eq!(post_clear.notice, Some(DiskCacheNotice::Cleared));
    assert_eq!(post_clear.stats.expect("post-clear stats").tile_count, 0);
    assert!(!storage.poll_clear_completed(), "clear_gen consumed once");

    storage.clear_disk_cache_notice();
    assert!(storage.disk_cache_status().notice.is_none());

    // After closing the worker channel, `request_clear_disk_cache()` must NOT
    // permanently wedge `busy = true`.
    storage.begin_shutdown();
    storage.request_clear_disk_cache();
    let after_closed = storage.disk_cache_status();
    assert!(
        !after_closed.busy,
        "request_clear after worker disconnect must never wedge busy=true"
    );
    assert_eq!(
        after_closed.notice,
        Some(DiskCacheNotice::WorkerUnavailable)
    );

    storage.finish_shutdown();
}

#[test]
fn persistence_fault_surfaces_save_errors_dismisses_and_auto_clears_on_recovery() {
    let dir = TestTempDir::new("storage-fault");
    let paths = StoragePaths::with_root(dir.path());
    let settings_path = paths.settings_path();
    std::fs::create_dir_all(&settings_path).expect("create blocking dir at settings.json path");
    std::fs::write(settings_path.join("blocker"), b"x").expect("populate blocking dir");

    let wakes = Arc::new(AtomicUsize::new(0));
    let wakes_c = Arc::clone(&wakes);
    let wakeup: Wakeup = Arc::new(move || {
        wakes_c.fetch_add(1, Ordering::AcqRel);
    });

    let mut storage = AppStorage::open(wakeup, Some(paths));
    assert!(storage.persistence_fault_summary().is_none());

    // Saving onto a non-empty directory fails atomically on every OS and wakes UI.
    assert!(storage.mutate_settings(|s| s.show_graticule = !s.show_graticule));
    let deadline = Instant::now() + Duration::from_secs(2);
    while storage.persistence_fault_summary().is_none() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    let summary = storage
        .persistence_fault_summary()
        .expect("settings save error surfaced");
    assert!(summary.contains("settings"), "summary={summary}");
    assert!(wakes.load(Ordering::Acquire) > 0);
    // Must stay independent of DiskCacheStatus.notice.
    assert!(storage.disk_cache_status().notice.is_none());

    // Dismissing clears the banner; removing blocker and saving again stays clean.
    storage.dismiss_persistence_fault();
    assert!(storage.persistence_fault_summary().is_none());

    // Trigger failure once more, then remove blocker and verify auto-clear on successful save.
    assert!(storage.mutate_settings(|s| s.show_graticule = !s.show_graticule));
    let deadline = Instant::now() + Duration::from_secs(2);
    while storage.persistence_fault_summary().is_none() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(storage.persistence_fault_summary().is_some());

    std::fs::remove_dir_all(&settings_path).expect("remove blocking dir");
    assert!(storage.mutate_settings(|s| s.show_graticule = !s.show_graticule));
    let deadline = Instant::now() + Duration::from_secs(2);
    while storage.persistence_fault_summary().is_some() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        storage.persistence_fault_summary().is_none(),
        "subsequent successful save must auto-clear persistence fault"
    );

    storage.begin_shutdown();
    storage.finish_shutdown();
}
