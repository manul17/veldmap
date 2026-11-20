// Extracted from master:veldmodules/image-tiler/src/resample.rs

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, value: u8) -> Vec<u8> {
        vec![value; (w * h * 4) as usize]
    }

    #[test]
    fn uniform_stays_uniform() {
        // Любое отношение сторон: среднее константы — та же константа.
        for (sw, sh, dw, dh) in [(7, 5, 3, 2), (10, 10, 5, 5), (3, 3, 4, 4), (513, 2, 257, 1)] {
            let out = resample(&solid(sw, sh, 137), sw, sh, dw, dh);
            assert!(out.iter().all(|&v| v == 137), "{}x{} → {}x{}", sw, sh, dw, dh);
        }
    }

    /// Ужимается окно, а не весь прочитанный кусок. Читать приходится целыми
    /// пикселями, и прочитанное шире того, что тайлу принадлежит; растянутое
    /// целиком, оно уехало бы на долю пикселя — у соседнего тайла в другую
    /// сторону, и на стыке остался бы шов.
    #[test]
    fn only_the_window_is_squeezed() {
        let mut src = Vec::new();
        for v in [10u8, 20, 30, 40] {
            src.extend_from_slice(&[v, v, v, 255]);
        }
        let window = Window { x0: 1.0, y0: 0.0, x1: 3.0, y1: 1.0 };
        let out = resample_window(&src, 4, 1, window, 2, 1);
        assert_eq!(&out[0..4], &[20, 20, 20, 255], "первая ячейка — второй пиксель");
        assert_eq!(&out[4..8], &[30, 30, 30, 255], "вторая — третий");
        // Без окна тот же буфер даёт средние по парам — то самое смещение.
        assert_eq!(&resample(&src, 4, 1, 2, 1)[0..4], &[15, 15, 15, 255]);
    }

    /// Два соседних тайла, взятые каждый из своего расширенного куска,
    /// складываются в то же самое, что и одно ужатие всей строки: шва нет.
    #[test]
    fn neighbouring_tiles_join_without_a_seam() {
        // Строка из девяти пикселей ужимается в шесть — масштаб 3/2, то есть
        // границы тайлов приходятся на середины исходных пикселей.
        let values: Vec<u8> = (0..9).map(|v| v * 25).collect();
        let mut src = Vec::new();
        for v in &values {
            src.extend_from_slice(&[*v, *v, *v, 255]);
        }
        let whole = resample(&src, 9, 1, 6, 1);

        // Тайл шириной три: первый берёт долю [0, 4.5), второй — [4.5, 9).
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
            joined.extend(resample_window(region, sx1 - sx0, 1, window, 3, 1));
        }
        assert_eq!(joined, whole);
    }

    #[test]
    fn exact_two_to_one_is_block_average() {
        // 4×2 → 2×1: каждая ячейка — среднее своего блока 2×2.
        let mut src = Vec::new();
        for v in [10u8, 20, 30, 40, 50, 60, 70, 80] {
            src.extend_from_slice(&[v, v, v, 255]);
        }
        let out = resample(&src, 4, 2, 2, 1);
        // Блок левый: 10,20,50,60 → 35; правый: 30,40,70,80 → 55.
        assert_eq!(&out[0..4], &[35, 35, 35, 255]);
        assert_eq!(&out[4..8], &[55, 55, 55, 255]);
    }

    #[test]
    fn fractional_ratio_preserves_mean() {
        // 3 → 2 по ширине: средний пиксель делится между ячейками пополам,
        // и среднее по картинке сохраняется.
        let mut src = Vec::new();
        for v in [0u8, 90, 180] {
            src.extend_from_slice(&[v, 0, 0, 255]);
        }
        let out = resample(&src, 3, 1, 2, 1);
        // Ячейка 0: (0·1 + 90·0.5)/1.5 = 30; ячейка 1: (90·0.5 + 180·1)/1.5 = 150.
        assert_eq!(out[0], 30);
        assert_eq!(out[4], 150);
    }

    #[test]
    fn identity_is_copy() {
        let src: Vec<u8> = (0..4 * 6 * 4).map(|i| (i % 251) as u8).collect();
        assert_eq!(resample(&src, 4, 6, 4, 6), src);
    }

    #[test]
    fn halve_averages_integer_blocks() {
        // 3×3 → 2×2: полный блок, обрезанные по краю и угловой из одного.
        let mut src = Vec::new();
        for v in [10u8, 20, 30, 40, 50, 60, 70, 80, 90] {
            src.extend_from_slice(&[v, v, v, 255]);
        }
        let out = halve(&src, 3, 3);
        // Блоки: (10,20,40,50)=30; (30,60)=45; (70,80)=75; (90)=90.
        assert_eq!(&out[0..4], &[30, 30, 30, 255]);
        assert_eq!(&out[4..8], &[45, 45, 45, 255]);
        assert_eq!(&out[8..12], &[75, 75, 75, 255]);
        assert_eq!(&out[12..16], &[90, 90, 90, 255]);
    }

    #[test]
    fn halve_rounds_to_nearest() {
        // (0 + 1) / 2 = 0.5 → 1: округление, а не отбрасывание.
        let src = [0, 0, 0, 255, 1, 1, 1, 255];
        assert_eq!(&halve(&src, 2, 1)[..3], &[1, 1, 1]);
    }

    #[test]
    fn transparent_does_not_bleed_into_colour() {
        // Блок 2×2: два белых и два прозрачных чёрных (поле гранулы). Цвет
        // остаётся белым, полупрозрачность несёт одна альфа.
        let src = [255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0];
        assert_eq!(halve(&src, 2, 2), vec![255, 255, 255, 128]);
        assert_eq!(resample(&src, 2, 2, 1, 1), vec![255, 255, 255, 128]);
    }

    #[test]
    fn fully_transparent_block_stays_empty() {
        let src = [9, 9, 9, 0, 9, 9, 9, 0];
        assert_eq!(halve(&src, 2, 1), vec![0, 0, 0, 0]);
        assert_eq!(resample(&src, 2, 1, 1, 1), vec![0, 0, 0, 0]);
    }
}
