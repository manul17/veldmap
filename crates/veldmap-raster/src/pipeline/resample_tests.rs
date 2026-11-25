use super::*;

fn resample_rgba8(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let whole = Window {
        x0: 0.0,
        y0: 0.0,
        x1: f64::from(sw),
        y1: f64::from(sh),
    };
    resample_rgba8_window(src, sw, sh, whole, dw, dh)
}

fn solid(w: u32, h: u32, value: u8) -> Vec<u8> {
    vec![value; (w * h * 4) as usize]
}

#[test]
fn uniform_stays_uniform_and_fractional_window_seams_align() {
    for (sw, sh, dw, dh) in [(7, 5, 3, 2), (10, 10, 5, 5), (3, 3, 4, 4), (513, 2, 257, 1)] {
        let out = resample_rgba8(&solid(sw, sh, 137), sw, sh, dw, dh);
        assert!(out.iter().all(|&v| v == 137));
    }

    let values: Vec<u8> = (0..9).map(|v| v * 25).collect();
    let mut src = Vec::new();
    for v in &values {
        src.extend_from_slice(&[*v, *v, *v, 255]);
    }
    let whole = resample_rgba8(&src, 9, 1, 6, 1);

    let mut joined = Vec::new();
    for tile in 0..2u32 {
        let (from, to) = (f64::from(tile) * 4.5, f64::from(tile + 1) * 4.5);
        let (sx0, sx1) = (from.floor() as u32, (to.ceil() as u32).min(9));
        let region = &src[(sx0 as usize) * 4..(sx1 as usize) * 4];
        let window = Window {
            x0: from - f64::from(sx0),
            y0: 0.0,
            x1: to - f64::from(sx0),
            y1: 1.0,
        };
        joined.extend(resample_rgba8_window(region, sx1 - sx0, 1, window, 3, 1));
    }
    assert_eq!(joined, whole);
}

#[test]
fn halve_and_bleed_handle_transparency_and_edges() {
    // Checkerboard [0, 255] averages to 0.5 in linear light (~188 sRGB, not 128).
    let checker = [
        0, 0, 0, 255, 255, 255, 255, 255, //
        255, 255, 255, 255, 0, 0, 0, 255,
    ];
    let halved = halve_rgba8(&checker, 2, 2);
    assert!(halved[0].abs_diff(188) <= 2 && halved[3] == 255);
    let resampled = resample_rgba8(&checker, 2, 2, 1, 1);
    assert!(resampled[0].abs_diff(188) <= 2 && resampled[3] == 255);

    // 3x3 odd boundary keeps 2x2, 1x2, 2x1, and 1x1 edge cells exact in linear light.
    let mut src = Vec::new();
    for v in [10u8, 20, 30, 40, 50, 60, 70, 80, 90] {
        src.extend_from_slice(&[v, v, v, 255]);
    }
    let out = halve_rgba8(&src, 3, 3);
    assert_eq!(out.len(), 16);
    assert_eq!(&out[12..16], &[90, 90, 90, 255]);
    assert!(
        out[0] > 30 && out[0] < 40,
        "linear avg > gamma 30: {}",
        out[0]
    );

    let mixed = [
        255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0,
    ];
    assert_eq!(halve_rgba8(&mixed, 2, 2), vec![255, 255, 255, 128]);
    assert_eq!(resample_rgba8(&mixed, 2, 2, 1, 1), vec![255, 255, 255, 128]);

    let mut bleed_buf = vec![200, 100, 50, 255, 0, 0, 0, 0, 0, 0, 0, 0];
    bleed_alpha(&mut bleed_buf, 3, 1);
    assert_eq!(&bleed_buf[4..8], &[200, 100, 50, 0]);
    assert_eq!(&bleed_buf[8..12], &[200, 100, 50, 0]);
}

#[test]
fn scientific_and_categorical_resamplers_match_halve_and_skip_nan_slivers() {
    // R32Float NaN-skipping halve & sliver-guarded window resample.
    let f32_src: Vec<u8> = [10.0f32, f32::NAN, 30.0, f32::NAN]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect();
    let h = halve_f32(&f32_src, 2, 2);
    assert!((read_f32_ne(&h, 0) - 20.0).abs() < 1e-6);

    // Tiny floating-point boundary sliver (1e-13) into finite neighbor must NOT bleed into all-NaN cell.
    let sliver_src: Vec<u8> = [100.0f32, f32::NAN]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect();
    let sliver_out = resample_f32_window(
        &sliver_src,
        2,
        1,
        Window {
            x0: 1.0 - 1e-13,
            y0: 0.0,
            x1: 2.0,
            y1: 1.0,
        },
        1,
        1,
    );
    assert!(read_f32_ne(&sliver_out, 0).is_nan());

    // Categorical 2x exact decimation: pointwise window and cascade halve must agree verbatim.
    let u8_grid: Vec<u8> = (0u8..16).collect();
    let halved_u8 = halve_uint_nearest(&u8_grid, 4, 4, 1);
    let win_u8 = resample_uint_nearest_window(
        &u8_grid,
        4,
        4,
        Window {
            x0: 0.0,
            y0: 0.0,
            x1: 4.0,
            y1: 4.0,
        },
        2,
        2,
        1,
    );
    assert_eq!(win_u8, halved_u8);

    let u16_grid: Vec<u8> = (100u16..116).flat_map(u16::to_ne_bytes).collect();
    let halved_u16 = halve_uint_nearest(&u16_grid, 4, 4, 2);
    let win_u16 = resample_uint_nearest_window(
        &u16_grid,
        4,
        4,
        Window {
            x0: 0.0,
            y0: 0.0,
            x1: 4.0,
            y1: 4.0,
        },
        2,
        2,
        2,
    );
    assert_eq!(win_u16, halved_u16);
}
