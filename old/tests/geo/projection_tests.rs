// Extracted from master:veldmodules/globe/src/projection.rs

#[cfg(test)]
mod tests {
    use super::*;

    /// Веб-Меркатор кладёт мир в квадрат со стороной в два раза по 20 037 508
    /// метров — на этом стои́т вся тайловая сетка степеней двойки, и числа эти
    /// не наши, а её.
    #[test]
    fn мир_веб_меркатора_квадратный() {
        let map = Projection::WebMercator;
        let side = 20_037_508.34;

        let (x, y) = map.from_geodetic(0.0, 0.0);
        assert!(x.abs() < 1e-6 && y.abs() < 1e-6, "нуль градусов — нуль метров: {x} {y}");

        let (x, _) = map.from_geodetic(0.0, 180.0);
        assert!((x - side).abs() < 0.5, "край по долготе: {x}");

        let (_, y) = map.from_geodetic(WEB_MERCATOR_LIMIT_DEG, 0.0);
        assert!((y - side).abs() < 0.5, "край по широте: {y}");
    }

    /// Числа настоящих мест — против таблиц, а не против нашей же формулы,
    /// переписанной в тест.
    #[test]
    fn настоящие_места_ложатся_куда_положено() {
        let map = Projection::WebMercator;
        for (lat, lon, ex, ey) in [
            (47.2357, 39.7015, 4_419_550.76, 5_980_631.53),
            (-33.8688, 151.2093, 16_832_542.28, -4_011_198.65),
        ] {
            let (x, y) = map.from_geodetic(lat, lon);
            assert!((x - ex).abs() < 0.5 && (y - ey).abs() < 0.5, "{lat} {lon} → {x} {y}");
        }
    }

    /// Прямая и обратная сходятся. Это единственная проверка проекции, не
    /// сводящаяся к переписыванию её же формул: ошибись в одной — пара
    /// разойдётся.
    #[test]
    fn веб_меркатор_обратим() {
        let map = Projection::WebMercator;
        for lat in [-84.0, -45.0, -0.5, 0.0, 12.25, 60.0, 84.9] {
            for lon in [-179.5, -90.0, 0.0, 33.3, 179.5] {
                let (x, y) = map.from_geodetic(lat, lon);
                let (back_lat, back_lon) = map.to_geodetic(x, y);
                assert!(
                    (back_lat - lat).abs() < 1e-9 && (back_lon - lon).abs() < 1e-9,
                    "{lat} {lon} → {x} {y} → {back_lat} {back_lon}"
                );
            }
        }
    }

    /// За пределом сетки широта зажимается — чтобы прямая оставалась
    /// определённой и обратимой. Варп-сетка сюда не ходит (она зовёт только
    /// обратную), так что защищает этот зажим не её, а саму пару.
    #[test]
    fn за_пределом_широта_зажимается() {
        let map = Projection::WebMercator;
        let (_, edge) = map.from_geodetic(WEB_MERCATOR_LIMIT_DEG, 0.0);
        let (_, beyond) = map.from_geodetic(89.9, 0.0);
        assert_eq!(edge, beyond, "за пределом координата обязана встать");
        assert!(beyond.is_finite(), "и остаться конечной");
    }

    /// Код 3857 узнаётся, а неизвестный по-прежнему остаётся неизвестным:
    /// растр с чужой системой ложится по контуру каталога, а не мимо себя.
    #[test]
    fn веб_меркатор_узнаётся_по_коду() {
        assert_eq!(Projection::from_epsg(3857), Some(Projection::WebMercator));
        assert!(matches!(Projection::from_epsg(32637), Some(Projection::Transverse(_))));
        assert_eq!(Projection::from_epsg(3035), None, "равновеликую азимутальную мы не умеем");
    }

    /// Насколько разошлись две точки по земле, метры. Грубой метрики здесь
    /// довольно: проверяется сходство в сантиметрах, а не форма Земли.
    fn ground_m(first: (f64, f64), second: (f64, f64)) -> f64 {
        let lat = ((first.0 + second.0) / 2.0).to_radians();
        let per_lat = 111_132.92 - 559.82 * (2.0 * lat).cos();
        let per_lon = 111_412.84 * lat.cos() - 93.5 * (3.0 * lat).cos();
        ((first.0 - second.0) * per_lat).hypot((first.1 - second.1) * per_lon)
    }

    /// **Опорная проверка всей цепочки** — числами стороннего пересчётчика
    /// (geobridge, коды 63361xx).
    ///
    /// Сходятся здесь сразу четыре вещи, и порознь ни одна не проверяема:
    /// эллипсоид Красовского, ряды Крюгера с единичным масштабом, отступы зоны
    /// и семь элементов Гельмерта. Ошибись любая — разъедется всё.
    ///
    /// Все точки лежат **внутри** своих зон, и это условие проверки, а не
    /// удобство: вне зоны сверять не с чем. Классические формулы
    /// Гаусса-Крюгера раскладываются по степеням разности долгот и объявлены
    /// годными в пределах трёх градусов от осевого меридиана — в тринадцати
    /// градусах от него пересчётчики расходятся между собой на метры, и
    /// совпадение там означало бы лишь общую ошибку.
    #[test]
    fn the_local_grid_matches_an_outside_converter() {
        // Зона, север, восток → широта и долгота WGS84.
        let checks = [
            (1, 500_000.0, 1_330_000.0, 47.932_102_711, 38.383_287_711),
            (2, 500_000.0, 2_330_000.0, 47.932_149_061, 41.383_334_064),
            (3, 500_000.0, 3_330_000.0, 47.932_194_230, 44.383_385_063),
            (2, 500_000.0, 2_350_000.0, 47.930_905_649, 41.650_973_905),
        ];
        for (zone, north, east, lat, lon) in checks {
            let system = System::msk61(zone).expect("зона МСК-61");
            let got = system.to_geodetic(east, north);
            let off = ground_m(got, (lat, lon));
            // Два миллиметра, а не сантиметр: поправка масштаба действует почти
            // целиком по высоте, и по горизонтали её переворот виден лишь на
            // пять миллиметров. Собственное расхождение здесь — четыре десятых.
            assert!(off < 0.002, "зона {}: {:?} против {:?} — {} м", zone, got, (lat, lon), off);
        }
    }

    /// Перевод между датумами — не подмена эллипсоида: в Ростовской области он
    /// сдвигает точку на сто с лишним метров, и почти всё это сдвиг центра.
    /// Выброшенный, он оставил бы поле на соседнем поле.
    #[test]
    fn the_datum_shift_is_worth_a_hundred_metres() {
        let rostov = (47.23, 39.72);
        let moved = PULKOVO_1942.to_wgs84(rostov.0, rostov.1);
        let shift = ground_m(rostov, moved);
        assert!((100.0..130.0).contains(&shift), "{} м", shift);

        // Сам WGS84 никуда не едет: переходить ему некуда.
        assert_eq!(WGS84_DATUM.to_wgs84(rostov.0, rostov.1), rostov);
    }

    /// Гельмерт и обратный к нему — пара, и в геоцентрике она точная: сходится
    /// до нанометра на радиусе Земли, то есть до последних разрядов f64.
    ///
    /// Проверяется здесь, а не на широте с долготой, потому что только здесь
    /// пара полна: наружу уезжают два числа из трёх, и выброшенная высота даёт
    /// свой остаток, к матрице отношения не имеющий (см. соседний тест).
    /// Развёрнутый знаками вместо настоящей обратной, он разошёлся бы тут на
    /// полмиллиметра — и спрятался бы за этим остатком, проверяй мы их вместе.
    #[test]
    fn the_helmert_and_its_inverse_are_exact() {
        let points = [
            [3_000_000.0, 2_000_000.0, 5_200_000.0],
            [-1_500_000.0, 5_800_000.0, 2_000_000.0],
            [4_000_000.0, -3_000_000.0, -3_500_000.0],
        ];
        // Второй датум — с тремя ненулевыми разворотами: у СК-42 разворот вокруг
        // первой оси ровно нулевой, и на ней первая строка матрицы не
        // проверяется вовсе. Числа его выдуманы, но проверяется здесь не они, а
        // тождество матрицы с обратной.
        let turned = Datum { rotate_sec: [-1.5, -0.35, -0.79], ..PULKOVO_1942 };
        for datum in [PULKOVO_1942, turned] {
            for point in points {
                let back = datum.helmert_back(datum.helmert(point));
                for axis in 0..3 {
                    let off = (back[axis] - point[axis]).abs();
                    assert!(off < 1e-6, "{:?}, ось {}: {} м", datum.rotate_sec, axis, off);
                }
            }
        }
    }

    /// Переход между датумами и обратный сходятся и в градусах — но не точно, и
    /// причина этому одна: наружу отсюда уезжают два числа из трёх.
    ///
    /// Гельмерт разворачивает и вертикаль, а высоту двумерный договор не несёт
    /// — привязка растра её не знает. Оттого остаток пропорционален тому, на
    /// сколько эллипсоиды разошлись по высоте: около двадцати микрон на каждый
    /// метр этого расхождения. В Ростовской области, ради которой МСК-61 и
    /// построена, это десяток микрон; на дальнем краю страны — шесть сотых
    /// миллиметра.
    #[test]
    fn the_datum_shift_and_its_reverse_converge_where_it_is_fitted() {
        // Область, ради которой СК-42 и построена.
        for (lat, lon) in [(47.23, 39.72), (55.75, 37.62), (66.53, 66.60), (43.1, 131.9)] {
            let (there_lat, there_lon) = PULKOVO_1942.to_wgs84(lat, lon);
            let back = PULKOVO_1942.from_wgs84(there_lat, there_lon);
            assert!(ground_m(back, (lat, lon)) < 1e-3, "{:?} → {:?}", (lat, lon), back);
        }
    }

    /// Код EPSG называет систему только внутри своих полос, и границы у них
    /// точные: соседние коды — не «зона ноль» и не «зона шестьдесят один», а
    /// совсем другие системы, и принятые за UTM они увезли бы снимок за тысячи
    /// километров.
    #[test]
    fn epsg_names_a_system_only_inside_its_bands() {
        assert_eq!(System::from_epsg(32601), Some(System::utm(1, false)));
        assert_eq!(System::from_epsg(32660), Some(System::utm(60, false)));
        assert_eq!(System::from_epsg(32701), Some(System::utm(1, true)));
        assert_eq!(System::from_epsg(32760), Some(System::utm(60, true)));
        assert_eq!(System::from_epsg(28401), Some(System::gauss_kruger(1)));
        assert_eq!(System::from_epsg(28408), Some(System::gauss_kruger(8)));
        assert_eq!(System::from_epsg(28432), Some(System::gauss_kruger(32)));
        assert_eq!(System::from_epsg(6_336_101), System::msk61(1));
        assert_eq!(System::from_epsg(6_336_102), System::msk61(2));
        assert_eq!(System::from_epsg(6_336_103), System::msk61(3));

        assert_eq!(System::from_epsg(32600), None, "полоса начинается с первой зоны");
        assert_eq!(System::from_epsg(32700), None, "полоса начинается с первой зоны");
        assert_eq!(System::from_epsg(32661), None, "это полярная стереографическая");
        assert_eq!(System::from_epsg(32761), None, "это полярная стереографическая");
        assert_eq!(System::from_epsg(28400), None, "зоны с нулевым номером нет");
        assert_eq!(System::from_epsg(28433), None, "зон Гаусса-Крюгера тридцать две");
        assert_eq!(System::from_epsg(6_336_100), None, "зоны с нулевым номером нет");
        assert_eq!(System::from_epsg(6_336_104), None, "у МСК-61 три зоны");
        assert_eq!(System::from_epsg(4326), None, "градусы вообще не проекция");
        assert_eq!(System::from_epsg(3031), None, "антарктическая полярная — не она");
    }

    /// Три семейства отличаются не устройством, а числами — и числа эти обязаны
    /// быть именно теми. Осевой меридиан UTM отсчитывается от антимеридиана,
    /// Гаусса-Крюгера — от нуля, и перепутанные, они разъедутся на полмира.
    #[test]
    fn each_family_keeps_its_own_numbers() {
        let utm = System::from_epsg(32638).expect("зона 38 северная");
        assert!((utm.central_meridian_deg - 45.0).abs() < 1e-12);
        assert_eq!(utm.scale, 0.9996);
        assert_eq!(utm.datum, WGS84_DATUM);

        let gk = System::gauss_kruger(8);
        assert!((gk.central_meridian_deg - 45.0).abs() < 1e-12, "тот же меридиан, другой счёт");
        assert_eq!(gk.scale, 1.0);
        assert_eq!(gk.false_easting_m, 8_500_000.0, "номер зоны вписан в координату");
        assert_eq!(gk.datum.ellipsoid, KRASOVSKY);

        let msk = System::msk61(2).expect("зона 2");
        // Со счётом из определения, а не с копией той же литералы: 40°59′.
        assert!((msk.central_meridian_deg - (40.0 + 59.0 / 60.0)).abs() < 1e-10);
        assert_eq!(msk.false_easting_m, 2_300_000.0);
        assert_eq!(msk.false_northing_m, -4_811_057.628);
    }

    /// Прямая и обратная сходятся по всей зоне — на этом стоит варп-сетка:
    /// её узлы считаются обратной, а привязка растра задана прямой. Гоняется
    /// это на всех трёх семействах: у каждого свой эллипсоид и свой датум, и
    /// пара обязана сойтись у каждого.
    #[test]
    fn roundtrip_converges_across_zone() {
        // Широты у каждого семейства свои, и не для удобства: UTM объявлен на
        // всю Землю, а СК-42 подгонялась под страну, и гонять её у полюса
        // значило бы мерить не проекцию, а остаток выброшенной высоты (см.
        // `the_datum_shift_and_its_reverse_converge_where_it_is_fitted`).
        // Допуск у семейств разный, и разница у него причина, а не снисхождение:
        // у WGS84 перехода между датумами нет вовсе, и остаётся одна проекция —
        // это доли микрона. У систем на Красовском к ней добавляется остаток
        // выброшенной высоты, и меньше его не сделать, не расширив договор до
        // трёх чисел.
        let cases = [
            (System::utm(40, false), [-79.0, -8.0, 0.0, 45.0, 83.5], 1e-4),
            (System::utm(40, true), [-79.0, -8.0, 0.0, 45.0, 83.5], 1e-4),
            (System::gauss_kruger(8), [42.0, 50.0, 55.0, 65.0, 75.0], 1e-3),
            (System::msk61(2).expect("зона 2"), [46.0, 47.0, 48.0, 49.0, 50.0], 1e-3),
        ];
        for (system, latitudes, tolerance_m) in cases {
            let middle = system.central_meridian_deg;
            for lat in latitudes {
                for offset in [-2.5, 0.0, 1.7, 2.9] {
                    let lon = middle + offset;
                    let (e, n) = system.from_geodetic(lat, lon);
                    let back = system.to_geodetic(e, n);
                    let off = ground_m(back, (lat, lon));
                    assert!(off < tolerance_m, "{:?}: {:?} — {} м", system, back, off);
                }
            }
        }
    }

    /// Точка за швом ложится в свою зону, а не в бессмыслицу. Зона 1 стои́т на
    /// осевом меридиане −177°, и 179° в.д. лежит от него в четырёх градусах к
    /// западу, хотя разность долгот выходит в триста пятьдесят шесть градусов к
    /// востоку. Ряд эту разность берёт синусом и косинусом, и ветвь ему
    /// безразлична — но безразлична она только ему, а не всякому счёту, и это
    /// стои́т держать проверенным.
    #[test]
    fn a_longitude_across_the_seam_lands_in_its_zone() {
        let zone = System::utm(1, false);
        assert!((zone.central_meridian_deg + 177.0).abs() < 1e-12);

        let (east, north) = zone.from_geodetic(10.0, 179.0);
        // Четыре градуса к западу — меньше пятисот километров, то есть отступ
        // заведомо не выходит за ложный сдвиг.
        assert!((0.0..500_000.0).contains(&east), "{}", east);
        assert!((1_000_000.0..1_200_000.0).contains(&north), "{}", north);

        // И обратно — той же ветвью, что у осевого меридиана: 179° в.д. здесь
        // называется −181°, потому что зона лежит по западную сторону шва.
        let (lat, lon) = zone.to_geodetic(east, north);
        assert!((lat - 10.0).abs() < 1e-9, "{}", lat);
        assert!((lon + 181.0).abs() < 1e-9, "{}", lon);
    }

    /// Ветвь долготы одинакова при любом датуме. Зона 32 Гаусса-Крюгера стои́т
    /// на осевом меридиане 189° — за швом, — а переход между датумами
    /// кончается `atan2` и свернул бы её к −171°. У UTM перехода нет, и без
    /// разворота эти двое вели бы себя по-разному: рвалась бы рамка только у
    /// той системы, у которой датум свой.
    #[test]
    fn a_system_beyond_the_seam_keeps_its_branch() {
        let far = System::gauss_kruger(32);
        assert!((far.central_meridian_deg - 189.0).abs() < 1e-12);

        let (lat, lon) = far.to_geodetic(far.false_easting_m, 5_000_000.0);
        assert!(lon > 180.0, "долгота осталась за швом: {}", lon);
        assert!((lon - 189.0).abs() < 0.01, "{}", lon);
        assert!((44.0..46.0).contains(&lat), "{}", lat);
    }

    /// Короткое замыкание на WGS84 смотрит на весь датум, а не на его
    /// эллипсоид: датум бывает на том же эллипсоиде и всё же сдвинутым, и
    /// принятый за WGS84 он не сдвинул бы точку никуда.
    #[test]
    fn a_shifted_datum_on_the_same_ellipsoid_still_moves() {
        let nudged = Datum { shift_m: [100.0, 0.0, 0.0], ..WGS84_DATUM };
        // Точка выбрана там, где ось X лежит горизонтально: на экваторе под
        // 90° в.д. сдвиг вдоль неё виден целиком, а на нулевом меридиане ушёл
        // бы в одну высоту.
        let here = (0.0, 90.0);
        let moved = nudged.to_wgs84(here.0, here.1);
        assert!((ground_m(moved, here) - 100.0).abs() < 0.5, "{:?}", moved);

        let back = nudged.from_wgs84(moved.0, moved.1);
        assert!(ground_m(back, here) < 1e-3, "{:?}", back);
    }

    /// Точка на самой полярной оси даёт полюс, а не не-число: счёт широты там
    /// делит на косинус прямого угла. С нынешних мест вызова такая точка не
    /// приходит, но молчаливое `NaN` уехало бы дальше рамкой, у которой
    /// протяжённость формально есть.
    #[test]
    fn the_axis_gives_a_pole_and_not_a_nan() {
        let polar = KRASOVSKY.semi_major_m * (1.0 - KRASOVSKY.flattening);
        let (north, lon) = KRASOVSKY.from_earth_centred([0.0, 0.0, polar]);
        assert!((north - 90.0).abs() < 1e-9, "{}", north);
        assert_eq!(lon, 0.0, "на полюсе долгота произвольна");

        let (south, _) = KRASOVSKY.from_earth_centred([0.0, 0.0, -polar]);
        assert!((south + 90.0).abs() < 1e-9, "{}", south);
    }

    /// Якоря, известные без счёта: точка осевого меридиана на экваторе — это
    /// ложный сдвиг в чистом виде, в обоих полушариях.
    #[test]
    fn axis_anchors_are_exact() {
        let north = System::utm(31, false);
        let (e, n) = north.from_geodetic(0.0, 3.0);
        assert!((e - 500_000.0).abs() < 1e-6 && n.abs() < 1e-6, "{} {}", e, n);

        let south = System::utm(31, true);
        let (_, n) = south.from_geodetic(0.0, 3.0);
        assert!((n - 10_000_000.0).abs() < 1e-6, "{}", n);
    }

    /// Полушария зеркальны: та же дуга к югу — тот же метраж вниз от экватора.
    #[test]
    fn hemispheres_mirror() {
        let zone = System::utm(40, false);
        let (e1, n1) = zone.from_geodetic(30.0, 58.0);
        let (e2, n2) = zone.from_geodetic(-30.0, 58.0);
        assert!((e1 - e2).abs() < 1e-6);
        assert!((n1 + n2).abs() < 1e-6, "{} {}", n1, n2);
    }

    /// Оборотов у обратного перехода столько, что дальше улучшать нечего.
    ///
    /// Проверяется полосой широт, а не точкой: множитель сходимости зависит от
    /// широты в сотню раз, и удобная точка сказала бы о числе оборотов не то.
    /// Утверждается пол округления, а не последний разряд: разряд у нативной
    /// libm и у той, с которой модуль отгружается, свой. Отсюда и то, чего
    /// этот тест не сторожит: запас поверх сошедшегося можно срезать, и он
    /// смолчит. Потребовать запаса он и не может — запас взят как раз на ту
    /// libm, которой здесь нет.
    #[test]
    fn the_conformal_turn_has_nothing_left_to_improve() {
        let ellipsoid = WGS84;
        let e = ellipsoid.eccentricity_sq().sqrt();
        let radius = ellipsoid.semi_major_m;

        let (mut settled, mut short) = (0.0f64, 0.0f64);
        for step in 0..=1680 {
            let lat = f64::from(step) * 0.05;
            // Конформная широта этой точки — вход цикла.
            let tau_c = {
                let phi = lat.to_radians();
                (phi.tan().asinh() - e * (e * phi.sin()).atanh()).sinh()
            };
            let whole = geodetic_from_conformal(e, tau_c, CONFORMAL_ROUNDS + 4);
            settled = settled
                .max((geodetic_from_conformal(e, tau_c, CONFORMAL_ROUNDS) - whole).abs() * radius);
            short = short.max((geodetic_from_conformal(e, tau_c, 4) - whole).abs() * radius);
        }
        assert!(settled < 1e-8, "оборотов не хватает: {} м", settled);
        assert!(short > 1e-6, "проверять было бы нечего: четырёх хватило на {} м", short);
    }

    /// А у перехода от геоцентрики широта ищется зеркальным циклом, и худший
    /// случай у него не экватор, а полюс.
    ///
    /// Проверяется на точке, снятой с эллипсоида: лежащую на нём цикл не
    /// трогает вовсе — начальное приближение там точное, и тест, гоняющий его
    /// через датум, прошёл бы и при нуле оборотов.
    #[test]
    fn the_centred_turn_works_only_above_the_ellipsoid() {
        let ellipsoid = WGS84;
        // Своим подъёмом, а не `to_earth_centred`: та кладёт точку ровно на
        // эллипсоид, а цикл на такой точке холост.
        let lifted = |lat_deg: f64, height_m: f64| {
            let lat = lat_deg.to_radians();
            let e2 = ellipsoid.eccentricity_sq();
            let curvature = ellipsoid.semi_major_m / (1.0 - e2 * lat.sin() * lat.sin()).sqrt();
            (
                (curvature + height_m) * lat.cos(),
                (curvature * (1.0 - e2) + height_m) * lat.sin(),
            )
        };

        // На самом эллипсоиде цикл холост: сколько ни крути, ответ тот же.
        for step in 0..=90 {
            let (flat, z) = lifted(f64::from(step), 0.0);
            assert_eq!(
                ellipsoid.latitude_at(flat, z, 0),
                ellipsoid.latitude_at(flat, z, CENTRED_ROUNDS),
                "на широте {} цикл что-то делает над самим эллипсоидом",
                step
            );
        }

        let (mut settled, mut short) = (0.0f64, 0.0f64);
        for step in 0..=1800 {
            let (flat, z) = lifted(f64::from(step) * 0.05, 271.0);
            let whole = ellipsoid.latitude_at(flat, z, CENTRED_ROUNDS + 6);
            let radius = ellipsoid.semi_major_m;
            settled = settled
                .max((ellipsoid.latitude_at(flat, z, CENTRED_ROUNDS) - whole).abs() * radius);
            short = short.max((ellipsoid.latitude_at(flat, z, 2) - whole).abs() * radius);
        }
        assert!(settled < 1e-8, "оборотов не хватает: {} м", settled);
        assert!(short > 1e-7, "проверять было бы нечего: двух хватило на {} м", short);
    }

    /// Опорная точка гранулы Sentinel-2 T40WFC: её верхний левый угол
    /// (ULX 600000, ULY 7800000 из MTD_TL.xml, EPSG:32640) обязан вернуться
    /// теми градусами, где тайл и лежит — ~70.2° с.ш. у восточного края зоны.
    #[test]
    fn sentinel_tile_corner_lands_where_the_tile_is() {
        let zone = System::utm(40, false);
        let (lat, lon) = zone.to_geodetic(600_000.0, 7_800_000.0);
        assert!((69.9..70.5).contains(&lat), "{}", lat);
        assert!((59.0..60.3).contains(&lon), "{}", lon);
        // И обратно в те же метры — не только здесь.
        //
        // Одной точкой это утверждение не держится: множитель сходимости у
        // обратного перехода идёт от `e²` на экваторе до сотой его доли к
        // восьмидесятым широтам, и здешние 70,3° — почти вдевятеро легче
        // худшего случая. Проверка, стоящая на одном углу гранулы, зелена и
        // при цикле, урезанном вдвое; полоса отступов её держит.
        // До восьми с половиной миллионов, а не выше: дальше отступ уводит за
        // предел UTM в восемьдесят четыре градуса, и точка вышла бы из своей
        // зоны — условие, которое соседний тест держит намеренно.
        for step in 0..9 {
            let northing = 500_000.0 + f64::from(step) * 1_000_000.0;
            let (lat, lon) = zone.to_geodetic(600_000.0, northing);
            let (e, n) = zone.from_geodetic(lat, lon);
            assert!((e - 600_000.0).abs() < 1e-6, "восток на {}: {}", northing, e);
            assert!((n - northing).abs() < 1e-6, "север на {}: {}", northing, n);
        }
    }
}
