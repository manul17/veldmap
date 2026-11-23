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

#[test]
fn pointwise_runs_concurrently_while_pass_is_single_per_pass_key() {
    let mut sched = TileScheduler::new();
    let src_a = SourceId::new(10);
    let src_b = SourceId::new(11);
    let meta_a = sample_meta(src_a, 2, 4);
    let pass_a = meta_a.pass_key();
    let pass_b = PassKey::new(src_b, BandKey::Display, PixelFormat::Rgba8UnormSrgb, 0);

    // Concurrent pointwise admissions on level 0 and level 1 succeed together.
    let k0 = meta_a.tile_key(0, 0, 0);
    let k1 = meta_a.tile_key(0, 1, 0);
    let k2 = meta_a.tile_key(1, 0, 0);
    assert_eq!(
        sched.admit_pointwise(&meta_a, 0, &[k0, k1], |_| false),
        vec![k0, k1]
    );
    // Duplicate in-flight or mismatched-level tiles are ignored, while matching-level tiles admit.
    assert!(
        sched
            .admit_pointwise(&meta_a, 0, &[k0, k2], |_| false)
            .is_empty()
    );
    let k0_dup = meta_a.tile_key(0, 2, 0);
    assert_eq!(
        sched.admit_pointwise(&meta_a, 0, &[k0, k0_dup, k0_dup], |_| false),
        vec![k0_dup]
    );
    assert_eq!(
        sched.admit_pointwise(&meta_a, 1, &[k2], |_| false),
        vec![k2]
    );
    // Calling admit_pointwise on a Pass level returns empty without mutating state.
    let pass_tile = meta_a.tile_key(3, 0, 0);
    assert!(
        sched
            .admit_pointwise(&meta_a, 3, &[pass_tile], |_| false)
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
    // Another source can run its own pass concurrently.
    let pid_b = sched
        .begin_pass(pass_b, 2, Some(2), CancelToken::new(), vec![])
        .expect("independent source pass succeeds");
    assert_ne!(pid1, pid_b);

    // Zooming IN (target=1 < active_pass_level=3) does NOT cancel coarser pass.
    assert!(!sched.update_target(pass_a, 1, None));
    assert!(!cancel1.is_cancelled());
    assert!(sched.has_active_pass(pass_a));
    assert!(sched.has_any_active_pass(src_a));

    // Zooming OUT cancels finer pass.
    sched.finish_pass(pass_a, pid1, true);
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
    assert!(sched.update_target(pass_a, 3, Some(3)));
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
    assert!(!sched.update_target(pass_a, 3, Some(0)));
    assert!(!cancel3.is_cancelled());
    sched.finish_pass(pass_a, pid3, true);
}

#[test]
fn multi_band_streams_under_same_source_schedule_and_forgive_independently() {
    let mut sched = TileScheduler::new();
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

    // Pointwise admission filters out keys from other bands/formats sharing `SourceId`.
    let k_b04 = b04.tile_key(0, 0, 0);
    let k_scl = scl.tile_key(0, 0, 0);
    assert_eq!(
        sched.admit_pointwise(&b04, 0, &[k_b04, k_scl], |_| false),
        vec![k_b04]
    );
    assert_eq!(
        sched.admit_pointwise(&scl, 0, &[k_b04, k_scl], |_| false),
        vec![k_scl]
    );

    // Concurrent passes on `B04` and `SCL` succeed under the same `SourceId`.
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

    // Failing B04 while SCL succeeds must NOT forgive B04's transient pass failure.
    sched.finish_pass(b04.pass_key(), pid_b04, false);
    sched.finish_pass(scl.pass_key(), pid_scl, true);
    assert!(sched.is_terminal_failed(&p_b04));

    // `reset_source` cancels active passes across all bands of `src`.
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
    sched.reset_source(src);
    assert!(c_b04_2.is_cancelled());
    assert!(c_scl_2.is_cancelled());
    assert!(!sched.has_any_active_pass(src));
    assert!(!sched.is_terminal_failed(&p_b04));
}

#[test]
fn terminal_and_repeated_failures_are_suppressed() {
    let mut sched = TileScheduler::new();
    let meta = sample_meta(SourceId::new(22), 2, 3);
    let pass_key = meta.pass_key();
    let k_term = meta.tile_key(0, 0, 0);
    let k_retry = meta.tile_key(0, 1, 0);

    assert_eq!(
        sched.admit_pointwise(&meta, 0, &[k_term, k_retry], |_| false),
        vec![k_term, k_retry]
    );

    // Terminal failure suppresses immediately.
    assert_eq!(
        sched.on_tile_failed(&k_term, true),
        TileFailureOutcome::TerminalFailed
    );
    assert!(sched.is_terminal_failed(&k_term));

    // First transient failure allows retry; second consecutive suppresses permanently.
    assert_eq!(
        sched.on_tile_failed(&k_retry, false),
        TileFailureOutcome::RetryEligible
    );
    assert!(!sched.is_terminal_failed(&k_retry));
    assert_eq!(
        sched.admit_pointwise(&meta, 0, &[k_term, k_retry], |_| false),
        vec![k_retry]
    );
    assert_eq!(
        sched.on_tile_failed(&k_retry, false),
        TileFailureOutcome::TerminalFailed
    );
    assert!(sched.is_terminal_failed(&k_retry));
    assert!(
        sched
            .admit_pointwise(&meta, 0, &[k_term, k_retry], |_| false)
            .is_empty()
    );

    // A healthy pass never forgives permanently rejected or twice-failed tiles.
    let pid_ok = sched
        .begin_pass(
            pass_key,
            2,
            Some(2),
            CancelToken::new(),
            vec![meta.tile_key(2, 0, 0)],
        )
        .expect("begin healthy pass");
    sched.finish_pass(pass_key, pid_ok, true);
    assert!(sched.is_terminal_failed(&k_term));
    assert!(sched.is_terminal_failed(&k_retry));

    // First-time pass failure is suppressed until a healthy pass forgives it once.
    let k_pass = meta.tile_key(1, 0, 0);
    let pid_fail = sched
        .begin_pass(pass_key, 1, None, CancelToken::new(), vec![k_pass])
        .expect("begin failing pass");
    sched.finish_pass(pass_key, pid_fail, false);
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
    sched.finish_pass(pass_key, pid_heal, true);
    assert!(!sched.is_terminal_failed(&k_pass));
    // Second failure after forgiveness escalates permanently.
    assert_eq!(
        sched.on_tile_failed(&k_pass, false),
        TileFailureOutcome::TerminalFailed
    );
    assert!(sched.is_terminal_failed(&k_pass));
}

#[test]
fn ladder_unfitting_levels_and_pass_only_recovery_respect_budget_fits() {
    let mut meta = sample_meta(SourceId::new(33), 2, 5);
    for lvl in 2..5 {
        meta.levels[lvl].serve = ServeMode::Pass { from: 0 };
    }
    // Pointwise target keeps apex pass once + pointwise rungs.
    assert_eq!(meta.ladder(0), vec![4, 1, 0]);
    // Pass target skips redundant coarser rungs sharing `Pass { from: 0 }`.
    assert_eq!(meta.ladder(3), vec![3]);

    // When no level fits, ladder must return empty and admissions must reject.
    let mut unfitting = meta.clone();
    for row in &mut unfitting.levels {
        row.fits = false;
    }
    assert_eq!(unfitting.finest_fitting_level(), None);
    assert!(unfitting.ladder(0).is_empty());
    let mut sched = TileScheduler::new();
    let k_unfit = unfitting.tile_key(0, 0, 0);
    assert!(
        sched
            .admit_pointwise(&unfitting, 0, &[k_unfit], |_| false)
            .is_empty()
    );
    let k_unfit_pass = unfitting.tile_key(3, 0, 0);
    assert!(
        sched
            .check_pass_candidates(&unfitting, 3, &[k_unfit_pass], |_| false)
            .is_empty()
    );

    // Pass-only source admits a recovery pass after a single pass failure, then stops on 2nd failure.
    let pass_only = PyramidMeta {
        levels: (0..2)
            .map(|_| LevelPlan {
                serve: ServeMode::Pass { from: 0 },
                source_pixels: 512 * 512,
                estimated_peak_bytes: 1024,
                fits: true,
            })
            .collect(),
        ..meta
    };
    let pass_key = pass_only.pass_key();
    let k = pass_only.tile_key(1, 0, 0);
    let pid1 = sched
        .begin_pass(pass_key, 1, Some(0), CancelToken::new(), vec![k])
        .expect("pass 1");
    sched.finish_pass(pass_key, pid1, false);
    assert_eq!(
        sched.check_pass_candidates(&pass_only, 1, &[k], |_| false),
        vec![k]
    );
    let pid2 = sched
        .begin_pass(pass_key, 1, Some(0), CancelToken::new(), vec![k])
        .expect("pass 2");
    sched.finish_pass(pass_key, pid2, false);
    assert!(
        sched
            .check_pass_candidates(&pass_only, 1, &[k], |_| false)
            .is_empty()
    );
}
