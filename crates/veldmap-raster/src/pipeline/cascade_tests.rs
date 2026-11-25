use super::*;

#[test]
fn cascade_handles_non_tile_row_batches_npot_dims_and_short_slices() {
    let (w, h) = (700u32, 600u32);
    let mut cascade = Cascade::new(0, w, h);
    assert!(
        cascade
            .push_rows(&[0u8; 8], 1, &mut |_, _, _, _, _, _| Ok(()))
            .is_err()
    );

    let mut emitted = Vec::new();
    let row_bytes = (w as usize) * 4;
    // Feed in 130-row batches (non-divisor of 512) to exercise multi-band flushing.
    let mut fed = 0u32;
    while fed < h {
        let n = (h - fed).min(130);
        let buf = vec![42u8; (n as usize) * row_bytes];
        cascade
            .push_rows(&buf, n, &mut |lvl, tx, ty, tw, th, rgba| {
                emitted.push((lvl, tx, ty, tw, th, rgba.len()));
                Ok(())
            })
            .expect("push_rows");
        fed += n;
    }
    cascade
        .finish(&mut |lvl, tx, ty, tw, th, rgba| {
            emitted.push((lvl, tx, ty, tw, th, rgba.len()));
            Ok(())
        })
        .expect("finish");

    // Level 0: 2x2 = 4 tiles; Level 1 (350x300 <= 512): 1 apex tile.
    assert_eq!(emitted.len(), 5);
    assert!(emitted.contains(&(1, 0, 0, 350, 300, 350 * 300 * 4)));
    assert!(bytes(w, h) > bands_bytes(w, h, 4));
    // Apex-only raster has zero halving buffer cost.
    assert_eq!(
        flush_bytes(256, 256, 4),
        256 * 256 * 4,
        "apex-only flush_bytes must not count halving"
    );
}

#[test]
fn cascade_supports_r32float_and_r8uint_with_format_semantics() {
    // R8Uint 600x4 -> 300x2 nearest-neighbor (never averaging class 1 and class 9 into 5).
    let mut cat = Cascade::new_for_format(0, 600, 4, PixelFormat::R8Uint);
    let mut rows = vec![9u8; 600 * 4];
    rows[0] = 1;
    let mut apex_u8 = None;
    cat.push_rows(&rows, 4, &mut |lvl, _tx, _ty, _tw, _th, bytes| {
        if lvl == 1 {
            apex_u8 = Some(bytes);
        }
        Ok(())
    })
    .expect("push");
    cat.finish(&mut |lvl, _tx, _ty, _tw, _th, bytes| {
        if lvl == 1 {
            apex_u8 = Some(bytes);
        }
        Ok(())
    })
    .expect("finish");
    let apex = apex_u8.expect("apex emitted");
    assert_eq!(apex[0], 1, "categorical cascade must pick exact class id");
    assert_eq!(apex[1], 9);

    // R16Uint 600x4 -> 300x2 nearest-neighbor preserves 16-bit class IDs verbatim.
    let mut cat16 = Cascade::new_for_format(0, 600, 4, PixelFormat::R16Uint);
    let mut rows16 = Vec::with_capacity(600 * 4 * 2);
    for i in 0..(600 * 4) {
        let v: u16 = if i == 0 { 1024 } else { 4096 };
        rows16.extend_from_slice(&v.to_ne_bytes());
    }
    let mut apex_u16 = None;
    cat16
        .push_rows(&rows16, 4, &mut |lvl, _tx, _ty, _tw, _th, bytes| {
            if lvl == 1 {
                apex_u16 = Some(bytes);
            }
            Ok(())
        })
        .expect("push16");
    cat16
        .finish(&mut |lvl, _tx, _ty, _tw, _th, bytes| {
            if lvl == 1 {
                apex_u16 = Some(bytes);
            }
            Ok(())
        })
        .expect("finish16");
    let apex16 = apex_u16.expect("apex16 emitted");
    assert_eq!(u16::from_ne_bytes([apex16[0], apex16[1]]), 1024);
    assert_eq!(u16::from_ne_bytes([apex16[2], apex16[3]]), 4096);

    // R32Float 600x4 -> 300x2 NaN-skipping average (cell 0: [10, NaN, 20, NaN] -> 15; cell 1: all NaN -> NaN).
    let mut sci = Cascade::new_for_format(0, 600, 4, PixelFormat::R32Float);
    let mut rows_f32 = Vec::with_capacity(600 * 4 * 4);
    for y in 0..4u32 {
        for x in 0..600u32 {
            let v = match (x, y) {
                (0, 0) => 10.0f32,
                (0, 1) => 20.0f32,
                (1, 0) | (1, 1) | (2, 0) | (3, 0) | (2, 1) | (3, 1) => f32::NAN,
                _ => 5.0f32,
            };
            rows_f32.extend_from_slice(&v.to_ne_bytes());
        }
    }
    let mut apex_f32 = None;
    sci.push_rows(&rows_f32, 4, &mut |lvl, _tx, _ty, _tw, _th, bytes| {
        if lvl == 1 {
            apex_f32 = Some(bytes);
        }
        Ok(())
    })
    .expect("push f32");
    sci.finish(&mut |lvl, _tx, _ty, _tw, _th, bytes| {
        if lvl == 1 {
            apex_f32 = Some(bytes);
        }
        Ok(())
    })
    .expect("finish f32");
    let af = apex_f32.expect("apex f32 emitted");
    let c0 = f32::from_ne_bytes([af[0], af[1], af[2], af[3]]);
    let c1 = f32::from_ne_bytes([af[4], af[5], af[6], af[7]]);
    assert!(
        (c0 - 15.0).abs() < 1e-5,
        "expected NaN-skipping avg 15.0, got {c0}"
    );
    assert!(c1.is_nan(), "all-NaN 2x2 cell must remain NaN, got {c1}");
}
