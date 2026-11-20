// Extracted from master:veldmodules/image-tiler/src/adapters/radiometry.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_bytes_match_plain_expansion() {
        let mapping = Mapping::identity(None);
        let bytes = [7u8, 9, 200];
        assert_eq!(
            mapping.rgba(&Samples::U8(&bytes), Pixel::named(1), 3),
            super::super::to_rgba(&bytes, Pixel::named(1), 3)
        );
    }

    #[test]
    fn nodata_pixel_turns_transparent_black() {
        // Серый u8 с nodata 7: первый пиксель — поле, второй — данные.
        let mapping = Mapping::identity(Some(7.0));
        let out = mapping.rgba(&Samples::U8(&[7, 9]), Pixel::named(1), 2);
        assert_eq!(out, vec![0, 0, 0, 0, 9, 9, 9, 255]);
    }

    #[test]
    fn nodata_keys_only_when_every_colour_matches() {
        // RGB: (0,0,0) — поле, (0,0,1) — законный почти-чёрный.
        let mapping = Mapping::stretched(0.0, 255.0, Some(0.0));
        let out = mapping.rgba(&Samples::U8(&[0, 0, 0, 0, 0, 1]), Pixel::named(3), 2);
        assert_eq!(&out[..4], &[0, 0, 0, 0]);
        assert_eq!(&out[4..], &[0, 0, 1, 255]);
    }

    #[test]
    fn stretch_maps_range_and_clamps_edges() {
        let mapping = Mapping::stretched(100.0, 355.0, None);
        let out = mapping.rgba(&Samples::U16(&[100, 355, 50, 1000, 22750]), Pixel::named(1), 4);
        assert_eq!(&out[0..4], &[0, 0, 0, 255]); // lo → 0
        assert_eq!(&out[4..8], &[255, 255, 255, 255]); // hi → 255
        assert_eq!(&out[8..12], &[0, 0, 0, 255]); // ниже lo — кламп
        assert_eq!(&out[12..16], &[255, 255, 255, 255]); // выше hi — кламп
    }

    #[test]
    fn f32_nodata_is_compared_raw_before_stretch() {
        // DEM: nodata −32767 не участвует ни в цвете, ни в растяге.
        let mapping = Mapping::stretched(0.0, 100.0, Some(-32767.0));
        let out = mapping.rgba(&Samples::F32(&[-32767.0, 50.0]), Pixel::named(1), 2);
        assert_eq!(out, vec![0, 0, 0, 0, 128, 128, 128, 255]);
    }

    #[test]
    fn nan_sample_is_transparent_even_without_declared_nodata() {
        let mapping = Mapping::stretched(0.0, 100.0, None);
        let out = mapping.rgba(&Samples::F32(&[f32::NAN, 50.0]), Pixel::named(1), 2);
        assert_eq!(out, vec![0, 0, 0, 0, 128, 128, 128, 255]);
    }

    #[test]
    fn alpha_channel_is_scaled_not_stretched() {
        // Серый с альфой u16: цвет растягивается, альфа — старший байт.
        let mapping = Mapping::stretched(0.0, 1000.0, None);
        let out = mapping.rgba(&Samples::U16(&[500, 0x8000]), Pixel::named(2), 1);
        assert_eq!(out, vec![128, 128, 128, 0x80]);
    }

    #[test]
    fn percentiles_ignore_hot_outliers() {
        // 0..=100 с одиночным выбросом: растяг остаётся у тела распределения.
        let mut values: Vec<f32> = (0..=100).map(|v| v as f32).collect();
        values.push(1_000_000.0);
        let (lo, hi) = percentile_stretch(&mut values).unwrap();
        assert_eq!(lo, 2.0);
        assert!((98.0..=100.0).contains(&hi), "hi = {}", hi);
    }

    /// Ровное поле встаёт посередине шкалы, а не в её нижний край: чёрный
    /// кадр неотличим от пустоты, а поле однотонно, а не пусто.
    #[test]
    fn flat_field_lands_in_the_middle_of_the_scale() {
        for value in [0.0f32, 42.0, -273.15, 1.0e7, 9.969_21e36] {
            let mut values = vec![value; 10];
            let (lo, hi) = percentile_stretch(&mut values).expect("выборка не пуста");
            assert!(hi > lo, "растяг вырожден при {}: {}..{}", value, lo, hi);
            let grey = Mapping::stretched(lo, hi, None)
                .rgba(&Samples::F32(&[value]), Pixel::named(1), 1);
            assert!((100..=160).contains(&grey[0]), "{} дало яркость {}", value, grey[0]);
            assert_eq!(grey[3], 255, "ровное поле не пусто");
        }
        assert_eq!(percentile_stretch(&mut []), None);
    }

    /// Один и тот же вопрос — годен ли отсчёт — у выборки растяга и у показа.
    /// Разойдясь, они дают белый непрозрачный пиксель там, где выборка
    /// посчитала значение негодным.
    #[test]
    fn what_the_sample_throws_out_the_frame_does_not_show() {
        let nodata = Some(-9999.0);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -9999.0] {
            assert!(!is_data(bad, nodata), "{} сочтено данными", bad);
            let out = Mapping::stretched(0.0, 100.0, nodata)
                .rgba(&Samples::F32(&[bad]), Pixel::named(1), 1);
            assert_eq!(&out[..], &[0, 0, 0, 0], "{} вышло непрозрачным", bad);
        }
        assert!(is_data(0.0, nodata) && is_data(-9999.0, None));
    }
}
