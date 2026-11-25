use super::*;

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_ne_bytes());
    }
    out
}

#[test]
fn nodata_and_percentile_stretch_behave_consistently() {
    let mapping = DisplayMapping::stretched(0.0, 100.0, Some(-9999.0));
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -9999.0] {
        assert!(!is_data_f64(f64::from(bad), Some(-9999.0)));
        let out = raw_bytes_to_rgba(
            &mapping,
            &f32_bytes(&[bad]),
            SampleType::F32,
            PixelLayout::named(1),
            1,
        );
        assert_eq!(out, vec![0, 0, 0, 0]);
    }

    // RGB nodata only clears pixels where every color channel equals nodata.
    let rgb_map = DisplayMapping::stretched(0.0, 255.0, Some(0.0));
    let out = raw_bytes_to_rgba(
        &rgb_map,
        &[0, 0, 0, 0, 0, 1],
        SampleType::U8,
        PixelLayout::named(3),
        2,
    );
    assert_eq!(&out[..4], &[0, 0, 0, 0]);
    assert_eq!(&out[4..], &[0, 0, 1, 255]);

    // Flat field lands mid-scale rather than black (including f32::MAX).
    for val in [0.0f32, 42.0, -273.15, 1.0e7, f32::MAX] {
        let mut vals = vec![val; 10];
        let (lo, hi) = percentile_stretch(&mut vals).expect("non-empty");
        assert!(hi > lo);
        let px = raw_bytes_to_rgba(
            &DisplayMapping::stretched(lo, hi, None),
            &f32_bytes(&[val]),
            SampleType::F32,
            PixelLayout::named(1),
            1,
        );
        assert!((100..=160).contains(&px[0]));
        assert_eq!(px[3], 255);
    }
    assert_eq!(percentile_stretch(&mut []), None);

    // Large finite f32 ranges must not overflow `hi - lo` to infinity, and non-finite alpha clears pixel.
    let wide = DisplayMapping::stretched(-2.0e38, 2.0e38, None);
    let out_wide = raw_bytes_to_rgba(
        &wide,
        &f32_bytes(&[0.0]),
        SampleType::F32,
        PixelLayout::named(1),
        1,
    );
    assert!((120..=135).contains(&out_wide[0]));
    let inf_alpha = raw_bytes_to_rgba(
        &DisplayMapping::stretched(0.0, 1.0, None),
        &f32_bytes(&[0.5, f32::INFINITY]),
        SampleType::F32,
        PixelLayout::named(2),
        1,
    );
    assert_eq!(inf_alpha, vec![0, 0, 0, 0]);
    let nan_alpha = raw_bytes_to_rgba(
        &DisplayMapping::stretched(0.0, 1.0, None),
        &f32_bytes(&[0.5, f32::NAN]),
        SampleType::F32,
        PixelLayout::named(2),
        1,
    );
    assert_eq!(nan_alpha, vec![0, 0, 0, 0]);
}

#[test]
fn f16_f64_nodata_white_is_zero_non_last_alpha_and_band_stats() {
    // Binary16 edge cases: ±0, normal, subnormal, ±Inf, NaN.
    assert_eq!(f16_bits_to_f32(0x0000), 0.0);
    assert!(f16_bits_to_f32(0x8000).is_sign_negative());
    assert!((f16_bits_to_f32(0x3C00) - 1.0).abs() < 1e-6);
    assert!((f16_bits_to_f32(0x0001) - 5.9604645e-8).abs() < 1e-12);
    assert!(f16_bits_to_f32(0x7C00).is_infinite() && f16_bits_to_f32(0x7C00).is_sign_positive());
    assert!(f16_bits_to_f32(0xFC00).is_infinite() && f16_bits_to_f32(0xFC00).is_sign_negative());
    assert!(f16_bits_to_f32(0x7E00).is_nan());

    // Exact i32 nodata beyond f32 24-bit precision must mask cleanly on Display.
    let exact_nd = i32::MAX - 1; // 2_147_483_646 (not exact in f32)
    let i32_raw: Vec<u8> = [exact_nd, 100i32]
        .into_iter()
        .flat_map(i32::to_ne_bytes)
        .collect();
    let i32_map = DisplayMapping::stretched(0.0, 200.0, Some(f64::from(exact_nd)));
    let rgba = raw_bytes_to_rgba(
        &i32_map,
        &i32_raw,
        SampleType::I32,
        PixelLayout::named(1),
        2,
    );
    assert_eq!(&rgba[0..4], &[0, 0, 0, 0]);
    assert_eq!(rgba[7], 255);

    // WhiteIsZero display inversion happens AFTER raw nodata check.
    let wiz = DisplayMapping::identity(Some(0.0)).with_invert(true);
    let wiz_rgba = raw_bytes_to_rgba(
        &wiz,
        &[0, 10, 255],
        SampleType::U8,
        PixelLayout::named(1),
        3,
    );
    assert_eq!(&wiz_rgba[0..4], &[0, 0, 0, 0], "nodata=0 stays transparent");
    assert_eq!(&wiz_rgba[4..8], &[245, 245, 245, 255]);
    assert_eq!(&wiz_rgba[8..12], &[0, 0, 0, 255]);

    // Non-last alpha channel (5-band R,G,B,Alpha,NIR) + I8 max alpha == 255.
    let rgb_a_nir = PixelLayout::rgb(5, Some(3));
    assert_eq!(rgb_a_nir.alpha_channel(), Some(3));
    let px5 = to_rgba8(&[10, 20, 30, 128, 250], rgb_a_nir, 1);
    assert_eq!(px5, vec![10, 20, 30, 128]);
    let i8_alpha = raw_bytes_to_rgba(
        &DisplayMapping::stretched(-128.0, 127.0, None),
        &[0u8, 127u8],
        SampleType::I8,
        PixelLayout::named(2),
        1,
    );
    assert_eq!(i8_alpha[3], 255);

    // BandStats filters NaN/Inf, computes dB without re-sorting, and handles non-positive sets.
    let mut vals = vec![100.0, f64::NAN, 1.0, 10.0, f64::INFINITY, -5.0];
    let stats = compute_band_stats(&mut vals).expect("stats");
    assert_eq!(stats.sample_count, 4);
    assert_eq!(stats.min, -5.0);
    assert_eq!(stats.max, 100.0);
    assert_eq!(stats.db_min, Some(0.0));
    assert!((stats.db_max.expect("db_max") - 20.0).abs() < 1e-9);
    assert_eq!(stats.histogram.iter().sum::<u32>(), 4);
    assert_eq!(stats.db_histogram.iter().sum::<u32>(), 3);

    let mut neg_only = vec![-10.0, 0.0, -1.0];
    let neg_stats = compute_band_stats(&mut neg_only).expect("neg stats");
    assert!(neg_stats.db_min.is_none());
    assert!(neg_stats.db_max.is_none());
}
