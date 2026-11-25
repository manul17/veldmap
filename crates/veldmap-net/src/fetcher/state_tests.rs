use std::time::Duration;

use veldmap_core::tile::{PlanetaryScheme, QuadTileId};

use super::super::failures::MAX_FAILURE_ENTRIES;
use super::*;

fn wm_key(spec: &LiveBasemapSpec, level: u8, x: u32, y: u32) -> TileKey {
    spec.tile_key(QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level,
        x,
        y,
    })
}

fn active_state(spec: &Arc<LiveBasemapSpec>) -> PipelineState {
    PipelineState {
        active_spec: Some(Arc::clone(spec)),
        ..PipelineState::default()
    }
}

#[test]
fn expired_backoffs_route_directly_to_http_or_disk_queues() {
    let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
    let mut st = active_state(&spec);
    let now = Instant::now();
    let http_key = wm_key(&spec, 4, 1, 2);
    let disk_key = wm_key(&spec, 4, 2, 2);

    st.last_wanted.insert(http_key);
    st.last_wanted.insert(disk_key);

    st.failures.insert_backoff(
        http_key,
        BackoffEntry {
            until: now - Duration::from_secs(1),
            retry_disk: false,
        },
    );
    st.failures.insert_backoff(
        disk_key,
        BackoffEntry {
            until: now - Duration::from_secs(1),
            retry_disk: true,
        },
    );

    st.prune_expired_backoffs(now);
    assert!(st.pending.http.contains(&http_key));
    assert!(!st.pending.disk.contains(&http_key));
    assert!(st.pending.disk.contains(&disk_key));
    assert!(!st.pending.http.contains(&disk_key));
}

#[test]
fn retry_failed_preserves_204_empty_and_ignores_stale_generations() {
    let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
    let mut st = active_state(&spec);
    let empty_key = wm_key(&spec, 4, 3, 2);
    let terminal_key = wm_key(&spec, 4, 4, 2);
    st.last_wanted.insert(terminal_key);

    let shutdown = CancelToken::new();
    let (empty_gen, _) = st.register_http_in_flight(empty_key, &shutdown);
    assert_eq!(
        st.mark_empty_204(empty_key, empty_gen.wrapping_add(99)),
        Transition::Stale
    );
    assert_eq!(
        st.mark_empty_204(empty_key, empty_gen),
        Transition::APPLIED_WAKEUP
    );
    st.last_wanted.insert(empty_key);

    let (term_gen, _) = st.register_http_in_flight(terminal_key, &shutdown);
    assert_eq!(
        st.mark_terminal_failure(
            terminal_key,
            term_gen.wrapping_add(99),
            Some("stale".to_owned())
        ),
        Transition::Stale
    );
    assert_eq!(
        st.mark_terminal_failure(terminal_key, term_gen, Some("bad".to_owned())),
        Transition::APPLIED_WAKEUP
    );

    st.retry_failed();
    assert_eq!(
        st.failures.failure_marker(&empty_key),
        Some(FailureMarker::Empty)
    );
    assert!(!st.failures.has_failure(&terminal_key));
    assert!(st.pending.disk.contains(&terminal_key));
}

#[test]
fn active_root_stays_pinned_across_reconcile_and_backoff_expiry() {
    let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
    let mut st = active_state(&spec);
    let now = Instant::now();
    let root_key = wm_key(&spec, 0, 0, 0);

    let mut initial_wanted = HashSet::new();
    initial_wanted.insert(root_key);
    st.reconcile_wanted(&initial_wanted, &|_| false, now);
    assert!(st.pending.disk.contains(&root_key));

    let empty_wanted = HashSet::new();
    st.reconcile_wanted(&empty_wanted, &|_| false, now);
    assert!(
        st.pending.disk.contains(&root_key),
        "active root must remain pinned in pending.disk until ready"
    );

    st.failures.insert_backoff(
        root_key,
        BackoffEntry {
            until: now + Duration::from_secs(5),
            retry_disk: true,
        },
    );
    assert_eq!(
        st.earliest_wanted_backoff(now),
        Some(now + Duration::from_secs(5))
    );
    st.pending.disk.clear();
    st.reconcile_wanted(&empty_wanted, &|_| false, now + Duration::from_secs(6));
    assert!(
        st.pending.disk.contains(&root_key),
        "expired retry_disk backoff for pinned active root must re-queue"
    );
}

#[test]
fn failure_cap_eviction_never_drops_active_backoffs() {
    let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
    let mut st = active_state(&spec);
    let now = Instant::now();
    let shutdown = CancelToken::new();

    let active_backoff = wm_key(&spec, 5, 0, 0);
    st.last_wanted.insert(active_backoff);
    let (bo_gen, _) = st.register_http_in_flight(active_backoff, &shutdown);
    assert_eq!(
        st.mark_transient_backoff(
            active_backoff,
            bo_gen,
            now + Duration::from_secs(10),
            false,
            None,
        ),
        Transition::APPLIED_WAKEUP
    );

    for i in 0..(MAX_FAILURE_ENTRIES as u32 + 16) {
        let k = wm_key(&spec, 12, i, 0);
        st.insert_failure(k, FailureMarker::Terminal);
    }
    assert!(st.failures.failure_len() <= MAX_FAILURE_ENTRIES);
    assert!(st.failures.has_backoff(&active_backoff));
}

#[test]
fn mark_empty_204_wakes_ui_and_clears_offline_status() {
    let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
    let mut st = active_state(&spec);
    let key = wm_key(&spec, 3, 1, 2);
    let shutdown = CancelToken::new();

    st.counters.offline = true;
    st.last_wanted.insert(key);
    let (gen_id, _) = st.register_http_in_flight(key, &shutdown);

    let stale = st.mark_empty_204(key, gen_id.wrapping_add(1));
    assert_eq!(stale, Transition::Stale);
    assert!(!stale.is_applied());
    assert!(!stale.should_wakeup());
    assert!(st.counters.offline);
    assert_eq!(st.status().in_flight, 1);

    let applied = st.mark_empty_204(key, gen_id);
    assert_eq!(applied, Transition::Applied { wakeup: true });
    assert!(applied.is_applied());
    assert!(applied.should_wakeup());
    assert!(!st.counters.offline);
    assert_eq!(st.status().in_flight, 0);
    assert_eq!(st.failures.failure_marker(&key), Some(FailureMarker::Empty));
}

#[test]
fn reconcile_wanted_routes_expired_backoffs_and_skips_ready_roots() {
    let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
    let mut st = active_state(&spec);
    let now = Instant::now();

    let root_key = wm_key(&spec, 0, 0, 0);
    let http_retry_key = wm_key(&spec, 4, 1, 1);
    let disk_retry_key = wm_key(&spec, 4, 2, 1);

    st.failures.insert_backoff(
        root_key,
        BackoffEntry {
            until: now - Duration::from_secs(1),
            retry_disk: true,
        },
    );
    st.failures.insert_backoff(
        http_retry_key,
        BackoffEntry {
            until: now - Duration::from_secs(1),
            retry_disk: false,
        },
    );
    st.failures.insert_backoff(
        disk_retry_key,
        BackoffEntry {
            until: now - Duration::from_secs(1),
            retry_disk: true,
        },
    );

    let wanted = HashSet::from([http_retry_key, disk_retry_key]);
    st.reconcile_wanted(&wanted, &|k| *k == root_key, now);

    assert!(!st.pending.disk.contains(&root_key));
    assert!(!st.pending.http.contains(&root_key));
    assert!(st.pending.http.contains(&http_retry_key));
    assert!(!st.pending.disk.contains(&http_retry_key));
    assert!(st.pending.disk.contains(&disk_retry_key));
    assert!(!st.pending.http.contains(&disk_retry_key));
}

#[test]
fn drain_disk_to_http_streams_only_eligible_keys() {
    let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
    let other_spec = LiveBasemapSpec::nasa_gibs_blue_marble();
    let mut st = active_state(&spec);

    let eligible_key = wm_key(&spec, 2, 0, 0);
    let suppressed_key = wm_key(&spec, 2, 1, 0);
    let wrong_source_key = wm_key(&other_spec, 2, 0, 0);

    st.insert_failure(suppressed_key, FailureMarker::Terminal);
    st.pending
        .disk
        .push_unique([eligible_key, suppressed_key, wrong_source_key]);

    st.drain_disk_to_http(spec.source_id());
    assert!(st.pending.disk.is_empty());
    assert!(st.pending.http.contains(&eligible_key));
    assert!(!st.pending.http.contains(&suppressed_key));
    assert!(!st.pending.http.contains(&wrong_source_key));
}
