// Extracted from master:veldmodules/data-provider/src/mgrs.rs

#[cfg(test)]
mod tests {
    use super::*;

    /// Окно выбора ряда обязано накрывать полосу с запасом с обеих сторон:
    /// буква ряда называет нортинг с точностью до периода в 2000 км, и
    /// разрешает её середина полосы. Полоса X двенадцатиградусная, и посчитай
    /// мы её середину общим восьмиградусным правилом — окно съехало бы на два
    /// градуса к югу, оставив северному краю 112 км запаса вместо 354.
    #[test]
    fn every_band_fits_the_row_window_from_both_sides() {
        for band in 0..BANDS.len() {
            let height = if band == BANDS.len() - 1 { LAST_BAND_DEG } else { BAND_DEG };
            let south = -80.0 + band as f64 * BAND_DEG;
            let target = band_middle_deg(band).abs() * 110_946.0;

            for (edge, side) in [(south, "юг"), (south + height, "север")] {
                let margin = ROW_CYCLE_M / 2.0 - (edge.abs() * 110_946.0 - target).abs();
                assert!(
                    margin > 250_000.0,
                    "полоса {} с {}а: запас {:.0} м",
                    BANDS[band] as char, side, margin
                );
            }
        }
    }

    /// Полоса X шире прочих, и это записано в самом правиле, а не подогнано
    /// числом: её середина 78°, а не 76°.
    #[test]
    fn the_x_band_is_twelve_degrees_wide() {
        assert_eq!(band_middle_deg(BANDS.len() - 1), 78.0);
        assert_eq!(band_middle_deg(0), -76.0, "полоса C — обычная, от −80° до −72°");
    }

    #[test]
    fn tile_is_read_from_product_name() {
        assert_eq!(
            tile_of("S2C_MSIL2A_20260812T081601_N0512_R121_T40WFC_20260812T121315.SAFE"),
            Some("40WFC")
        );
        assert_eq!(tile_of("S1C_IW_GRDH_1SDV_20260810T031955_011C6E_EDC8.SAFE"), None);
    }

    /// Имя приходит из хранилища, и на `T` там начинается не только гранула.
    /// Короткий сегмент — отказ, а не срез мимо длины: тир Landsat (`_T1`)
    /// стоит в имени каждого его продукта.
    #[test]
    fn short_and_odd_segments_are_refused_not_sliced() {
        assert_eq!(tile_of("LC09_L2SP_02_T1"), None);
        assert_eq!(tile_of("A_T_B"), None);
        assert_eq!(tile_of("T"), None);
        // Пять знаков, но не пять байт: срез по байтам границы символа не
        // спрашивает, а код гранулы латиницей.
        assert_eq!(tile_of("T40WФC"), None);
        assert_eq!(tile_of("T40wfc"), None);
    }

    /// Опорная гранула T40WFC: числа сверены с её MTD_TL.xml
    /// (EPSG:32640, ULX 600000, ULY 7800000, 10980 пикселей по 10 м).
    #[test]
    fn reference_granule_frame_matches_its_xml() {
        let frame = frame("40WFC").unwrap();
        assert_eq!((frame.zone, frame.south), (40, false));
        assert_eq!((frame.x0, frame.y1), (600_000.0, 7_800_000.0));
        assert_eq!((frame.x1 - frame.x0, frame.y1 - frame.y0), (109_800.0, 109_800.0));
    }

    /// Нечётная зона без сдвига рядов: у экваториального тайла зоны 31 ряд
    /// с буквы A. T31NAA — юго-западный угол зоны на экваторе.
    #[test]
    fn odd_zone_rows_start_at_a() {
        let frame = frame("31NAA").unwrap();
        assert_eq!((frame.x0, frame.y1), (100_000.0, 100_000.0));
        assert!(!frame.south);
    }

    /// Южное полушарие: нортинг отсчитан вниз от ложного сдвига 10 000 000 и
    /// остаётся в его пределах.
    #[test]
    fn southern_band_stays_under_false_northing() {
        let frame = frame("33HVB").unwrap();
        assert!(frame.south);
        assert!(frame.y1 < 10_000_000.0 && frame.y0 > 5_000_000.0, "{}..{}", frame.y0, frame.y1);
    }

    #[test]
    fn garbage_codes_are_refused() {
        assert!(frame("40WIC").is_err()); // I в столбцах не живёт
        assert!(frame("00WFC").is_err());
        assert!(frame("4WFC").is_err());
        assert!(frame("40ZFC").is_err()); // полосы Z нет
    }
}
