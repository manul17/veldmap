use std::collections::HashSet;

use veldmap_core::tile::{BandKey, PixelFormat, TileStreamSpec};

use crate::pipeline::plan::LevelPlan;

use super::*;

fn sample_meta(source: SourceId, pointwise_prefix: usize, total_levels: usize) -> PyramidMeta {
    sample_band_meta(
        source,
        BandKey::Display,
        PixelFormat::Rgba8UnormSrgb,
        0,
        pointwise_prefix,
        total_levels,
    )
}

fn sample_band_meta(
    source: SourceId,
    band: BandKey,
    format: PixelFormat,
    gutter: u8,
    pointwise_prefix: usize,
    total_levels: usize,
) -> PyramidMeta {
    let levels = (0..total_levels)
        .map(|lvl| {
            let serve = if lvl < pointwise_prefix {
                ServeMode::Pointwise
            } else {
                ServeMode::Pass { from: lvl as u8 }
            };
            LevelPlan {
                serve,
                estimated_peak_bytes: 4 * 1024 * 1024,
                fits: true,
            }
        })
        .collect();
    PyramidMeta {
        source,
        spec: TileStreamSpec::new(band, format, gutter, 1),
        levels,
    }
}

fn keys(admitted: &[AdmittedTile]) -> Vec<TileKey> {
    admitted.iter().map(|a| a.key).collect()
}

#[test]
fn pointwise_runs_concurrently_while_pass_is_single_per_pass_key() {
    let mut sched = TileScheduler::new();
    let root_cancel = CancelToken::new();
    let src_a = SourceId::new(10);
    let src_b = SourceId::new(11);
    let meta_a = sample_meta(src_a, 2, 4);
    let meta_b = sample_meta(src_b, 2, 4);
    let pass_a = meta_a.pass_key();

    let k0 = meta_a.tile_key(0, 0, 0);
    let k1 = meta_a.tile_key(0, 1, 0);
    let k2 = meta_a.tile_key(1, 0, 0);
    let adm = sched.admit_pointwise(&meta_a, 0, &[k0, k1], |_| false, &root_cancel);
    assert_eq!(keys(&adm), vec![k0, k1]);
    assert_eq!(sched.in_flight_count_for_source(src_a), 2);

    assert!(
        sched
            .admit_pointwise(&meta_a, 0, &[k0, k2], |_| false, &root_cancel)
            .is_empty()
    );
    let k0_dup = meta_a.tile_key(0, 2, 0);
    assert_eq!(
        keys(&sched.admit_pointwise(&meta_a, 0, &[k0, k0_dup, k0_dup], |_| false, &root_cancel)),
        vec![k0_dup]
    );
    assert_eq!(
        keys(&sched.admit_pointwise(&meta_a, 1, &[k2], |_| false, &root_cancel)),
        vec![k2]
    );
    let pass_tile = meta_a.tile_key(3, 0, 0);
    assert!(
        sched
            .admit_pointwise(&meta_a, 3, &[pass_tile], |_| false, &root_cancel)
            .is_empty()
    );
    assert!(!sched.is_in_flight(&pass_tile));

    // Only 1 active ServeMode::Pass per PassKey.
    let p1 = sched
        .admit_pass(&meta_a, 3, &[pass_tile], |_| false, &root_cancel)
        .expect("first pass succeeds");
    assert!(
        sched
            .admit_pass(
                &meta_a,
                2,
                &[meta_a.tile_key(2, 0, 0)],
                |_| false,
                &root_cancel
            )
            .is_none()
    );
    let p_b = sched
        .admit_pass(
            &meta_b,
            2,
            &[meta_b.tile_key(2, 0, 0)],
            |_| false,
            &root_cancel,
        )
        .expect("independent source pass succeeds");
    assert_ne!(p1.pass_id, p_b.pass_id);

    // Reconciling pointwise tiles must cancel unwanted pointwise jobs for
    // `pass_a` without touching the active pass's flight entries or sibling
    // band streams sharing the same `SourceId`.
    let mut sibling_band_meta = meta_a.clone();
    sibling_band_meta.spec.band = veldmap_core::tile::BandKey::from_name("B08");
    let sibling_key = sibling_band_meta.tile_key(0, 0, 0);
    let sibling_admitted = sched.admit_pointwise(
        &sibling_band_meta,
        0,
        &[sibling_key],
        |_| false,
        &root_cancel,
    );
    assert_eq!(keys(&sibling_admitted), vec![sibling_key]);

    let wanted: HashSet<TileKey> = [k0].into_iter().collect();
    let cancelled = sched.reconcile_pointwise(pass_a, &wanted, |_| false);
    for t in &cancelled {
        t.cancel();
    }
    assert_eq!(cancelled.len(), 3);
    assert!(sched.is_in_flight(&k0));
    assert!(!sched.is_in_flight(&k1));
    assert!(sched.is_in_flight(&pass_tile));
    assert!(sched.is_in_flight(&sibling_key));
    assert!(sched.has_active_pass(pass_a));
    assert!(sched.cancel_in_flight(&sibling_key, sibling_admitted[0].job_id));

    // Zooming IN (target=1 < active_pass_level=3) does NOT cancel coarser pass.
    assert!(sched.update_target(pass_a, 1, None).is_none());
    assert!(!p1.cancel.is_cancelled());
    assert!(sched.has_active_pass(pass_a));
    assert!(sched.has_any_active_pass(src_a));

    // Zooming OUT cancels finer pass.
    assert!(sched.finish_pass(pass_a, p1.pass_id, true));
    let p2 = sched
        .admit_pass(
            &meta_a,
            2,
            &[meta_a.tile_key(2, 0, 0)],
            |_| false,
            &root_cancel,
        )
        .expect("new pass after finish");
    if let Some(tok) = sched.update_target(pass_a, 3, Some(3)) {
        tok.cancel();
    }
    assert!(p2.cancel.is_cancelled());
    assert!(!sched.has_active_pass(pass_a));
    assert!(!sched.has_any_active_pass(src_a));

    // Shared `Pass { from: 0 }` cascade is NOT cancelled when zooming out within same pass.
    let mut shared_meta = meta_a.clone();
    shared_meta.levels[2].serve = ServeMode::Pass { from: 0 };
    shared_meta.levels[3].serve = ServeMode::Pass { from: 0 };
    let p3 = sched
        .admit_pass(
            &shared_meta,
            2,
            &[shared_meta.tile_key(2, 0, 0)],
            |_| false,
            &root_cancel,
        )
        .expect("shared cascade pass");
    assert!(sched.update_target(pass_a, 3, Some(0)).is_none());
    assert!(!p3.cancel.is_cancelled());
    assert!(sched.finish_pass(pass_a, p3.pass_id, true));
}

#[test]
fn multi_band_streams_under_same_source_schedule_and_forgive_independently() {
    let mut sched = TileScheduler::new();
    let root_cancel = CancelToken::new();
    let src = SourceId::new(42);
    let b04 = sample_band_meta(
        src,
        BandKey::from_name("B04"),
        PixelFormat::R32Float,
        0,
        1,
        3,
    );
    let scl = sample_band_meta(src, BandKey::from_name("SCL"), PixelFormat::R8Uint, 0, 1, 3);

    let k_b04 = b04.tile_key(0, 0, 0);
    let k_scl = scl.tile_key(0, 0, 0);
    let adm_b04 = sched.admit_pointwise(&b04, 0, &[k_b04, k_scl], |_| false, &root_cancel);
    assert_eq!(keys(&adm_b04), vec![k_b04]);
    let adm_scl = sched.admit_pointwise(&scl, 0, &[k_b04, k_scl], |_| false, &root_cancel);
    assert_eq!(keys(&adm_scl), vec![k_scl]);

    let p_b04 = b04.tile_key(1, 0, 0);
    let p_scl = scl.tile_key(1, 0, 0);
    let pass_b04 = sched
        .admit_pass(&b04, 1, &[p_b04], |_| false, &root_cancel)
        .expect("B04 pass starts");
    let pass_scl = sched
        .admit_pass(&scl, 1, &[p_scl], |_| false, &root_cancel)
        .expect("SCL pass starts concurrently");

    sched.finish_pass(b04.pass_key(), pass_b04.pass_id, false);
    sched.finish_pass(scl.pass_key(), pass_scl.pass_id, true);
    assert!(sched.is_terminal_failed(&p_b04));

    let pass_b04_2 = sched
        .admit_pass(&b04, 2, &[b04.tile_key(2, 0, 0)], |_| false, &root_cancel)
        .expect("restart B04");
    let pass_scl_2 = sched
        .admit_pass(&scl, 2, &[scl.tile_key(2, 0, 0)], |_| false, &root_cancel)
        .expect("restart SCL");
    let tokens = sched.reset_source(src);
    for t in &tokens {
        t.cancel();
    }
    assert!(pass_b04_2.cancel.is_cancelled());
    assert!(pass_scl_2.cancel.is_cancelled());
    assert!(adm_b04[0].cancel.is_cancelled());
    assert!(adm_scl[0].cancel.is_cancelled());
    assert!(!sched.has_any_active_pass(src));
    assert_eq!(sched.in_flight_count_for_source(src), 0);
    assert!(!sched.is_terminal_failed(&p_b04));
}

#[test]
fn terminal_failures_versioned_job_ids_and_non_penalizing_cancellations() {
    let mut sched = TileScheduler::new();
    let root_cancel = CancelToken::new();
    let meta = sample_meta(SourceId::new(22), 1, 3);
    let pass_key = meta.pass_key();
    let k_term = meta.tile_key(0, 0, 0);
    let k_retry = meta.tile_key(0, 1, 0);

    let adm = sched.admit_pointwise(&meta, 0, &[k_term, k_retry], |_| false, &root_cancel);
    assert_eq!(keys(&adm), vec![k_term, k_retry]);

    // Stale job_id cannot fail or arrive on an active tile.
    assert_eq!(sched.on_pointwise_failed(&k_term, 999_999, true), None);
    assert!(!sched.on_pointwise_arrived(&k_term, 999_999));

    // Terminal failure suppresses immediately.
    assert_eq!(
        sched.on_pointwise_failed(&k_term, adm[0].job_id, true),
        Some(TileFailureOutcome::TerminalFailed)
    );
    assert!(sched.is_terminal_failed(&k_term));

    // First transient failure allows retry; second consecutive suppresses permanently.
    assert_eq!(
        sched.on_pointwise_failed(&k_retry, adm[1].job_id, false),
        Some(TileFailureOutcome::RetryEligible)
    );
    assert!(!sched.is_terminal_failed(&k_retry));
    let retry_adm = sched.admit_pointwise(&meta, 0, &[k_term, k_retry], |_| false, &root_cancel);
    assert_eq!(keys(&retry_adm), vec![k_retry]);
    assert_eq!(
        sched.on_pointwise_failed(&k_retry, retry_adm[0].job_id, false),
        Some(TileFailureOutcome::TerminalFailed)
    );
    assert!(sched.is_terminal_failed(&k_retry));
    assert!(
        sched
            .admit_pointwise(&meta, 0, &[k_term, k_retry], |_| false, &root_cancel)
            .is_empty()
    );

    // Healthy pass forgives single pass failure once; second pass failure escalates.
    let k_pass = meta.tile_key(1, 0, 0);
    let p_fail = sched
        .admit_pass(&meta, 1, &[k_pass], |_| false, &root_cancel)
        .expect("admit failing pass");
    assert!(sched.finish_pass(pass_key, p_fail.pass_id, false));
    assert!(sched.is_terminal_failed(&k_pass));
    let p_heal = sched
        .admit_pass(&meta, 2, &[meta.tile_key(2, 0, 0)], |_| false, &root_cancel)
        .expect("admit healing pass");
    assert!(sched.finish_pass(pass_key, p_heal.pass_id, true));
    assert!(!sched.is_terminal_failed(&k_pass));

    // Forgiven tile is demoted to `RetriedOnce`, so a second failure becomes permanent.
    let p_fail2 = sched
        .admit_pass(&meta, 1, &[k_pass], |_| false, &root_cancel)
        .expect("admit second pass on forgiven tile");
    assert!(sched.finish_pass(pass_key, p_fail2.pass_id, false));
    assert!(sched.is_terminal_failed(&k_pass));
    assert!(
        sched
            .admit_pass(&meta, 1, &[k_pass], |_| false, &root_cancel)
            .is_none()
    );

    // Repeated cancellations/evictions clear in-flight without promoting to Permanent.
    let k_cancel = meta.tile_key(0, 2, 0);
    for _ in 0..5 {
        let a = sched.admit_pointwise(&meta, 0, &[k_cancel], |_| false, &root_cancel);
        assert_eq!(keys(&a), vec![k_cancel]);
        assert!(sched.is_in_flight(&k_cancel));
        assert!(!sched.cancel_in_flight(&k_cancel, a[0].job_id + 1));
        assert!(sched.cancel_in_flight(&k_cancel, a[0].job_id));
        assert!(!sched.is_in_flight(&k_cancel));
        assert!(!sched.is_terminal_failed(&k_cancel));
    }

    // `abort_pass` returns the pass CancelToken, rejects stale pass_id, and
    // does not penalize tiles.
    let k_abort = meta.tile_key(2, 0, 0);
    for _ in 0..5 {
        let p = sched
            .admit_pass(&meta, 2, &[k_abort], |_| false, &root_cancel)
            .expect("admit aborted pass");
        assert!(sched.abort_pass(pass_key, p.pass_id + 1).is_none());
        let tok = sched.abort_pass(pass_key, p.pass_id).expect("matched pass");
        tok.cancel();
        assert!(p.cancel.is_cancelled());
        assert!(!sched.has_active_pass(pass_key));
        assert!(!sched.is_in_flight(&k_abort));
        assert!(!sched.is_terminal_failed(&k_abort));
    }
}

#[test]
fn ladder_unfitting_levels_and_pass_only_recovery_respect_budget_fits() {
    let mut meta = sample_meta(SourceId::new(33), 2, 5);
    let root_cancel = CancelToken::new();
    for lvl in 2..5 {
        meta.levels[lvl].serve = ServeMode::Pass { from: 0 };
    }
    assert_eq!(meta.ladder(0), vec![4, 1, 0]);
    assert_eq!(meta.ladder(3), vec![3]);

    let mut unfitting = meta.clone();
    for row in &mut unfitting.levels {
        row.fits = false;
    }
    assert_eq!(unfitting.finest_fitting_level(), None);
    assert!(unfitting.ladder(0).is_empty());
    let mut sched = TileScheduler::new();
    assert!(
        sched
            .admit_pointwise(
                &unfitting,
                0,
                &[unfitting.tile_key(0, 0, 0)],
                |_| false,
                &root_cancel
            )
            .is_empty()
    );
    assert!(
        sched
            .admit_pass(
                &unfitting,
                4,
                &[unfitting.tile_key(4, 0, 0)],
                |_| false,
                &root_cancel
            )
            .is_none()
    );

    // Pass-only source: first pass failure stays retryable by a recovery pass.
    let pass_only = sample_meta(SourceId::new(34), 0, 3);
    let k_apex = pass_only.tile_key(2, 0, 0);
    let p1 = sched
        .admit_pass(&pass_only, 2, &[k_apex], |_| false, &root_cancel)
        .expect("admit first pass");
    sched.finish_pass(pass_only.pass_key(), p1.pass_id, false);
    assert!(sched.is_terminal_failed(&k_apex));
    let p2 = sched
        .admit_pass(&pass_only, 2, &[k_apex], |_| false, &root_cancel)
        .expect("admit recovery pass");
    assert_eq!(p2.tiles, vec![k_apex]);
    sched.finish_pass(pass_only.pass_key(), p2.pass_id, false);
    assert!(
        sched
            .admit_pass(&pass_only, 2, &[k_apex], |_| false, &root_cancel)
            .is_none()
    );
}
