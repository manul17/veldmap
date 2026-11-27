use std::time::Duration;

use veldmap_core::tile::{PlanetaryScheme, QuadTileId};

use crate::limiter::PermitDenied;

use super::failures::MAX_FAILURE_ENTRIES;
use super::*;

fn active_state(spec: &Arc<LiveBasemapSpec>, has_disk_cache: bool) -> PipelineState {
    let mut st = PipelineState::new(has_disk_cache);
    st.active_spec = Some(Arc::clone(spec));
    st
}

#[test]
fn expired_backoffs_route_directly_to_http_or_disk_queues() {
    let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
    let mut st = active_state(&spec, true);
    let now = Instant::now();
    let http_key = spec.wm_key(4, 1, 2);
    let disk_key = spec.wm_key(4, 2, 2);

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
    assert!(st.pending.contains_http(&http_key));
    assert!(!st.pending.contains_disk(&http_key));
    assert!(st.pending.contains_disk(&disk_key));
    assert!(!st.pending.contains_http(&disk_key));
}

#[test]
fn retry_failed_preserves_204_empty_and_ignores_stale_generations() {
    let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
    let mut st = active_state(&spec, true);
    let empty_key = spec.wm_key(4, 3, 2);
    let terminal_key = spec.wm_key(4, 4, 2);
    st.last_wanted.insert(terminal_key);

    let shutdown = CancelToken::new();
    let (empty_gen, _) = st.register_in_flight_for_test(empty_key, &shutdown);
    assert_eq!(
        st.mark_empty_204(empty_key, empty_gen.wrapping_add(99)),
        Transition::Stale
    );
    assert_eq!(
        st.mark_empty_204(empty_key, empty_gen),
        Transition::APPLIED_WAKEUP
    );
    st.last_wanted.insert(empty_key);

    let (term_gen, _) = st.register_in_flight_for_test(terminal_key, &shutdown);
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
    assert!(st.pending.contains_disk(&terminal_key));
}

#[test]
fn active_root_stays_pinned_across_reconcile_and_backoff_expiry() {
    let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
    let mut st = active_state(&spec, true);
    let now = Instant::now();
    let root_key = spec.wm_key(0, 0, 0);

    let mut initial_wanted = HashSet::new();
    initial_wanted.insert(root_key);
    st.reconcile_wanted(&initial_wanted, &|_| false, now);
    assert!(st.pending.contains_disk(&root_key));

    let empty_wanted = HashSet::new();
    st.reconcile_wanted(&empty_wanted, &|_| false, now);
    assert!(
        st.pending.contains_disk(&root_key),
        "active root must remain pinned in pending.disk until ready"
    );

    let backoff_until = now + Duration::from_secs(5);
    st.failures.insert_backoff(
        root_key,
        BackoffEntry {
            until: backoff_until,
            retry_disk: true,
        },
    );
    let planned = st.plan_dispatch(4, &CancelToken::new(), now, |_| Ok(()));
    assert!(planned.disk_tasks.is_empty());
    assert!(planned.http_tasks.is_empty());
    assert_eq!(planned.earliest_wakeup, Some(backoff_until));

    st.pending.clear();
    st.reconcile_wanted(&empty_wanted, &|_| false, now + Duration::from_secs(6));
    assert!(
        st.pending.contains_disk(&root_key),
        "expired retry_disk backoff for pinned active root must re-queue"
    );
}

#[test]
fn failure_cap_eviction_never_drops_active_backoffs() {
    let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
    let mut st = active_state(&spec, true);
    let now = Instant::now();
    let shutdown = CancelToken::new();

    let active_backoff = spec.wm_key(5, 0, 0);
    st.last_wanted.insert(active_backoff);
    let (bo_gen, _) = st.register_in_flight_for_test(active_backoff, &shutdown);
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
        let k = spec.wm_key(12, i, 0);
        st.insert_failure(k, FailureMarker::Terminal);
    }
    assert!(st.failures.failure_len() <= MAX_FAILURE_ENTRIES);
    assert!(st.failures.has_backoff(&active_backoff));
}

#[test]
fn mark_empty_204_wakes_ui_and_clears_offline_status() {
    let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
    let mut st = active_state(&spec, true);
    let key = spec.wm_key(3, 1, 2);
    let shutdown = CancelToken::new();

    st.counters.offline = true;
    st.last_wanted.insert(key);
    let (gen_id, _) = st.register_in_flight_for_test(key, &shutdown);

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
    let mut st = active_state(&spec, true);
    let now = Instant::now();

    let root_key = spec.wm_key(0, 0, 0);
    let http_retry_key = spec.wm_key(4, 1, 1);
    let disk_retry_key = spec.wm_key(4, 2, 1);

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

    assert!(!st.pending.contains_disk(&root_key));
    assert!(!st.pending.contains_http(&root_key));
    assert!(st.pending.contains_http(&http_retry_key));
    assert!(!st.pending.contains_disk(&http_retry_key));
    assert!(st.pending.contains_disk(&disk_retry_key));
    assert!(!st.pending.contains_http(&disk_retry_key));
}

#[test]
fn reconcile_without_disk_cache_routes_eligible_keys_directly_to_http() {
    let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
    let other_spec = LiveBasemapSpec::nasa_gibs_blue_marble();
    let mut st = active_state(&spec, false);

    let eligible_key = spec.wm_key(2, 0, 0);
    let suppressed_key = spec.wm_key(2, 1, 0);
    let wrong_source_key = other_spec.wm_key(2, 0, 0);

    st.insert_failure(suppressed_key, FailureMarker::Terminal);
    let wanted = HashSet::from([eligible_key, suppressed_key, wrong_source_key]);
    let now = Instant::now();
    st.reconcile_wanted(&wanted, &|_| false, now);

    let planned = st.plan_dispatch(4, &CancelToken::new(), now, |_| {
        Err::<(), _>(PermitDenied::Saturated)
    });
    assert!(planned.disk_tasks.is_empty());
    assert!(planned.http_tasks.is_empty());
    assert!(st.pending.is_disk_empty());
    assert!(st.pending.contains_http(&eligible_key));
    assert!(!st.pending.contains_http(&suppressed_key));
    assert!(!st.pending.contains_http(&wrong_source_key));
}

#[test]
fn plan_dispatch_carries_quad_id_and_defers_cooled_down_hosts() {
    let spec = LiveBasemapSpec::eox_sentinel2_cloudless();
    let now = Instant::now();
    let shutdown = CancelToken::new();

    let id0 = QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 1,
        x: 0,
        y: 0,
    };
    let id1 = QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 1,
        x: 1,
        y: 0,
    };
    let k0 = spec.tile_key(id0);
    let k1 = spec.tile_key(id1);

    // With disk cache enabled, disk tasks are planned with unpacked QuadTileId
    // and transition atomically to in_flight.
    let mut disk_st = PipelineState::new(true);
    disk_st.set_active_spec(Some(spec.clone()));
    disk_st.reconcile_wanted(&HashSet::from([k0, k1]), &|_| false, now);
    let disk_batch = disk_st.plan_dispatch(1, &shutdown, now, |_| Ok(()));
    assert_eq!(disk_batch.disk_tasks.len(), 1);
    assert_eq!(disk_batch.disk_tasks[0].target.key, k0);
    assert_eq!(disk_batch.disk_tasks[0].target.id, id0);
    assert!(disk_batch.http_tasks.is_empty());
    assert_eq!(disk_st.status().in_flight, 1);

    // Without disk cache, tiles route straight to HTTP and respect host cooldown.
    let mut http_st = PipelineState::new(false);
    http_st.set_active_spec(Some(spec.clone()));
    http_st.reconcile_wanted(&HashSet::from([k1]), &|_| false, now);
    let cooldown_until = now + Duration::from_secs(10);
    let blocked_host = spec.host_for(id1).to_owned();
    let deferred = http_st.plan_dispatch(4, &shutdown, now, |host| {
        if host == blocked_host {
            Err(PermitDenied::Cooldown(cooldown_until))
        } else {
            Ok(())
        }
    });
    assert!(deferred.disk_tasks.is_empty());
    assert!(deferred.http_tasks.is_empty());
    assert_eq!(deferred.earliest_wakeup, Some(cooldown_until));
    assert!(http_st.pending.contains_http(&k1));

    // After cooldown expires, HTTP task is planned with exact QuadTileId and in_flight entry.
    let http_batch = http_st.plan_dispatch(4, &shutdown, cooldown_until, |_| Ok(42u8));
    assert_eq!(http_batch.http_tasks.len(), 1);
    assert_eq!(http_batch.http_tasks[0].target.key, k1);
    assert_eq!(http_batch.http_tasks[0].target.id, id1);
    assert_eq!(http_batch.http_tasks[0].permit, 42u8);
    assert_eq!(http_st.status().in_flight, 1);
}
