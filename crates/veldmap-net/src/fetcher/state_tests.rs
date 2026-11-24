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

    assert!(st.prune_expired_backoffs(now));
    assert!(st.pending_http.contains(&http_key));
    assert!(!st.pending_disk.contains(&http_key));
    assert!(st.pending_disk.contains(&disk_key));
    assert!(!st.pending_http.contains(&disk_key));
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
        Transition::NONE
    );
    assert_eq!(st.mark_empty_204(empty_key, empty_gen), Transition::NOTIFY);
    st.last_wanted.insert(empty_key);

    let (term_gen, _) = st.register_http_in_flight(terminal_key, &shutdown);
    assert_eq!(
        st.mark_terminal_failure(
            terminal_key,
            term_gen.wrapping_add(99),
            Some("stale".to_owned())
        ),
        Transition::NONE
    );
    assert_eq!(
        st.mark_terminal_failure(terminal_key, term_gen, Some("bad".to_owned())),
        Transition::NOTIFY_AND_WAKEUP
    );

    st.retry_failed();
    assert_eq!(
        st.failures.failure_marker(&empty_key),
        Some(FailureMarker::Empty)
    );
    assert!(!st.failures.has_failure(&terminal_key));
    assert!(st.pending_disk.contains(&terminal_key));
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
    assert!(st.pending_disk.contains(&root_key));

    let empty_wanted = HashSet::new();
    st.reconcile_wanted(&empty_wanted, &|_| false, now);
    assert!(
        st.pending_disk.contains(&root_key),
        "active root must remain pinned in pending_disk until ready"
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
    st.pending_disk.clear();
    st.reconcile_wanted(&empty_wanted, &|_| false, now + Duration::from_secs(6));
    assert!(
        st.pending_disk.contains(&root_key),
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
        Transition::NOTIFY_AND_WAKEUP
    );

    for i in 0..(MAX_FAILURE_ENTRIES as u32 + 16) {
        let k = wm_key(&spec, 12, i, 0);
        st.insert_failure(k, FailureMarker::Terminal);
    }
    assert!(st.failures.failure_len() <= MAX_FAILURE_ENTRIES);
    assert!(st.failures.has_backoff(&active_backoff));
}
