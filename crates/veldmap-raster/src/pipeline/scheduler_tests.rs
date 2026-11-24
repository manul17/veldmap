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
                source_pixels: 512 * 512,
                estimated_peak_bytes: 4 * 1024 * 1024,
                fits: true,
            }
        })
        .collect();
    PyramidMeta {
        source,
        spec: TileStreamSpec::new(band, format, gutter, 1),
        width: 4096,
        height: 4096,
        tile_size: 512,
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
    let pass_a = meta_a.pass_key();
    let pass_b = PassKey::new(src_b, BandKey::Display, PixelFormat::Rgba8UnormSrgb, 0);

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

    // Only 1 active Serve::Pass per PassKey.
    let cancel1 = CancelToken::new();
    let pid1 = sched
        .begin_pass(pass_a, 3, Some(3), cancel1.clone(), vec![pass_tile])
        .expect("first pass succeeds");
    assert!(
        sched
            .begin_pass(pass_a, 2, Some(2), CancelToken::new(), vec![])
            .is_err()
    );
    let pid_b = sched
        .begin_pass(pass_b, 2, Some(2), CancelToken::new(), vec![])
        .expect("independent source pass succeeds");
    assert_ne!(pid1, pid_b);

    // Reconciling pointwise tiles must cancel unwanted pointwise jobs without
    // touching the active pass's flight entries.
    let wanted: HashSet<TileKey> = [k0].into_iter().collect();
    let cancelled = sched.reconcile_pointwise(src_a, &wanted, |_| false);
    for t in &cancelled {
        t.cancel();
    }
    assert_eq!(cancelled.len(), 3);
    assert!(sched.is_in_flight(&k0));
    assert!(!sched.is_in_flight(&k1));
    assert!(sched.is_in_flight(&pass_tile));
    assert!(sched.has_active_pass(pass_a));

    // Zooming IN (target=1 < active_pass_level=3) does NOT cancel coarser pass.
    assert!(sched.update_target(pass_a, 1, None).is_none());
    assert!(!cancel1.is_cancelled());
    assert!(sched.has_active_pass(pass_a));
    assert!(sched.has_any_active_pass(src_a));

    // Zooming OUT cancels finer pass.
    assert!(sched.finish_pass(pass_a, pid1, true));
    let cancel2 = CancelToken::new();
    let _pid2 = sched
        .begin_pass(
            pass_a,
            2,
            Some(2),
            cancel2.clone(),
            vec![meta_a.tile_key(2, 0, 0)],
        )
        .expect("new pass after finish");
    if let Some(tok) = sched.update_target(pass_a, 3, Some(3)) {
        tok.cancel();
    }
    assert!(cancel2.is_cancelled());
    assert!(!sched.has_active_pass(pass_a));
    assert!(!sched.has_any_active_pass(src_a));

    // Shared `Pass { from: 0 }` cascade is NOT cancelled when zooming out within same pass.
    let cancel3 = CancelToken::new();
    let pid3 = sched
        .begin_pass(
            pass_a,
            2,
            Some(0),
            cancel3.clone(),
            vec![meta_a.tile_key(2, 0, 0)],
        )
        .expect("shared cascade pass");
    assert!(sched.update_target(pass_a, 3, Some(0)).is_none());
    assert!(!cancel3.is_cancelled());
    assert!(sched.finish_pass(pass_a, pid3, true));
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

    let c_b04 = CancelToken::new();
    let c_scl = CancelToken::new();
    let p_b04 = b04.tile_key(1, 0, 0);
    let p_scl = scl.tile_key(1, 0, 0);
    let pid_b04 = sched
        .begin_pass(b04.pass_key(), 1, Some(1), c_b04.clone(), vec![p_b04])
        .expect("B04 pass starts");
    let pid_scl = sched
        .begin_pass(scl.pass_key(), 1, Some(1), c_scl.clone(), vec![p_scl])
        .expect("SCL pass starts concurrently");

    sched.finish_pass(b04.pass_key(), pid_b04, false);
    sched.finish_pass(scl.pass_key(), pid_scl, true);
    assert!(sched.is_terminal_failed(&p_b04));

    let c_b04_2 = CancelToken::new();
    let c_scl_2 = CancelToken::new();
    sched
        .begin_pass(
            b04.pass_key(),
            2,
            Some(2),
            c_b04_2.clone(),
            vec![b04.tile_key(2, 0, 0)],
        )
        .expect("restart B04");
    sched
        .begin_pass(
            scl.pass_key(),
            2,
            Some(2),
            c_scl_2.clone(),
            vec![scl.tile_key(2, 0, 0)],
        )
        .expect("restart SCL");
    let tokens = sched.reset_source(src);
    for t in &tokens {
        t.cancel();
    }
    assert!(c_b04_2.is_cancelled());
    assert!(c_scl_2.is_cancelled());
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
    let meta = sample_meta(SourceId::new(22), 2, 3);
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
    let pid_fail = sched
        .begin_pass(pass_key, 1, None, CancelToken::new(), vec![k_pass])
        .expect("begin failing pass");
    assert!(sched.finish_pass(pass_key, pid_fail, false));
    assert!(sched.is_terminal_failed(&k_pass));
    let pid_heal = sched
        .begin_pass(
            pass_key,
            2,
            Some(2),
            CancelToken::new(),
            vec![meta.tile_key(2, 0, 0)],
        )
        .expect("begin healing pass");
    assert!(sched.finish_pass(pass_key, pid_heal, true));
    assert!(!sched.is_terminal_failed(&k_pass));

    let k_pass_adm = sched.admit_pointwise(&meta, 1, &[k_pass], |_| false, &root_cancel);
    assert_eq!(keys(&k_pass_adm), vec![k_pass]);
    assert_eq!(
        sched.on_pointwise_failed(&k_pass, k_pass_adm[0].job_id, false),
        Some(TileFailureOutcome::TerminalFailed)
    );
    assert!(sched.is_terminal_failed(&k_pass));

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
        let p_cancel = CancelToken::new();
        let pid = sched
            .begin_pass(pass_key, 2, Some(2), p_cancel.clone(), vec![k_abort])
            .expect("begin aborted pass");
        assert!(sched.abort_pass(pass_key, pid + 1).is_none());
        let tok = sched.abort_pass(pass_key, pid).expect("matched pass");
        tok.cancel();
        assert!(p_cancel.is_cancelled());
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
            .check_pass_candidates(&unfitting, 4, &[unfitting.tile_key(4, 0, 0)], |_| false)
            .is_empty()
    );

    // Pass-only source: first pass failure stays retryable by a recovery pass.
    let pass_only = sample_meta(SourceId::new(34), 0, 3);
    let k_apex = pass_only.tile_key(2, 0, 0);
    let pid1 = sched
        .begin_pass(
            pass_only.pass_key(),
            2,
            Some(2),
            CancelToken::new(),
            vec![k_apex],
        )
        .expect("begin first pass");
    sched.finish_pass(pass_only.pass_key(), pid1, false);
    assert!(sched.is_terminal_failed(&k_apex));
    assert_eq!(
        sched.check_pass_candidates(&pass_only, 2, &[k_apex], |_| false),
        vec![k_apex]
    );
    let pid2 = sched
        .begin_pass(
            pass_only.pass_key(),
            2,
            Some(2),
            CancelToken::new(),
            vec![k_apex],
        )
        .expect("begin second pass");
    sched.finish_pass(pass_only.pass_key(), pid2, false);
    assert!(
        sched
            .check_pass_candidates(&pass_only, 2, &[k_apex], |_| false)
            .is_empty()
    );
}
