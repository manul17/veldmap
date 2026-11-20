// Extracted from master:veldmodules/image-tiler/src/adapters/tiff.rs

#[cfg(test)]
mod tests {
    use super::*;

    /// Байт на пиксель RGB8 — тестам геометрии сетки глубина сэмпла безразлична.
    const RGB8: u32 = 3;
    use super::super::super::pyramid::{self, TILE};
    use super::super::grid::{region_of, spanned, CHUNK_CACHE_BYTES, REGION_CAP};

    /// Внутренний тайл, каким его пишет GDAL, — тестам, которым важен размер
    /// копии, а не сетка чанков.
    const TILES: (u32, u32) = (TILE, TILE);

    /// Растр без копий — под ним `pick_source` смотрит только на размеры.
    fn bare(width: u32, height: u32) -> Info {
        Info::plain(width, height, Kind::Tiff(Layout::of(true, TILES, Vec::new(), RGB8)))
    }

    /// Копии настоящего снимка — гранула Sentinel-1 GRDH
    /// `S1C_IW_GRDH_1SDV_…_COG.SAFE`, растр 26553×16668, шесть IFD.
    ///
    /// Числа сняты с файла, а не выведены: правило выбора стои́т на том, как
    /// пишет стороны копий GDAL, и проверять эту модель ею же бессмысленно.
    /// Три нижних IFD (1659×1041, 829×520, 414×260) прочитаны из строки
    /// описания, три верхних продолжают ту же цепочку.
    fn sentinel1_grdh() -> (Info, Layout) {
        let overviews = [
            (13276, 8334),
            (6638, 4167),
            (3319, 2083),
            (1659, 1041),
            (829, 520),
            (414, 260),
        ];
        let overviews = overviews
            .iter()
            .enumerate()
            .map(|(step, &(width, height))| Overview {
                image: step + 1,
                width,
                height,
                chunk: TILES,
            })
            .collect();
        (bare(26553, 16668), Layout::of(true, TILES, overviews, RGB8))
    }

    /// Копии, записанные делением стороны пополам вниз, — так их пишет GDAL.
    fn halved_down(width: u32, height: u32, count: usize) -> Layout {
        let (mut w, mut h) = (width, height);
        let overviews = (1..=count)
            .map(|image| {
                w /= 2;
                h /= 2;
                Overview { image, width: w, height: h, chunk: TILES }
            })
            .collect();
        Layout::of(true, TILES, overviews, RGB8)
    }

    /// Копия годится своему уровню, хотя на пиксель у́же его. Проверять надо
    /// на нечётной стороне: у чётной оба счёта совпадают, и правило на ней не
    /// проверяется вовсе.
    ///
    /// Это и есть та пара, которая обязана сойтись механически: уровень
    /// считает `pyramid::level_size` округлением вверх, копии в файле — GDAL
    /// делением вниз. Разойдясь, они стоят чтения вчетверо большей копии на
    /// каждом уровне, кроме нулевого.
    #[test]
    fn копия_годится_своему_уровню_у_нечётной_стороны() {
        // Sentinel-1 GRDH: 25437×16729, шесть IFD.
        let (width, height) = (25437u32, 16729u32);
        let layout = halved_down(width, height, 5);
        let info = bare(width, height);

        for level in 1..=5usize {
            let (lw, lh) = (
                pyramid::level_size(width, level as u32),
                pyramid::level_size(height, level as u32),
            );
            let (image, chosen, chosen_h) = layout.grid.source_for((info.width, info.height), lw, lh);
            assert_eq!(
                (image, chosen, chosen_h),
                (level, lw - 1, lh - 1),
                "уровню {} ({}×{}) досталась чужая копия",
                level, lw, lh
            );
        }
    }

    /// То же на снятых с настоящего файла числах, а не на выведенных нашей
    /// же моделью деления.
    ///
    /// Ширина у гранулы нечётная, поэтому расходится она на каждом уровне —
    /// это и есть промах, ради которого допуск заведён. Высота делится на
    /// четыре, и на первых двух уровнях расхождения нет вовсе: допуск нужен
    /// не всегда и не всем сторонам сразу.
    #[test]
    fn у_настоящей_гранулы_каждый_уровень_берёт_свою_копию() {
        let (info, layout) = sentinel1_grdh();

        for level in 1..=6usize {
            let (lw, lh) = (
                pyramid::level_size(info.width, level as u32),
                pyramid::level_size(info.height, level as u32),
            );
            let copy = &layout.grid.overviews[level - 1];
            assert_eq!(lw - copy.width, 1, "ширина уровня {} и его копии", level);
            assert!(lh - copy.height <= 1, "высота уровня {} и его копии", level);
            assert_eq!(layout.grid.source_for((info.width, info.height), lw, lh).0, level, "уровню {} — своя копия", level);
        }
        assert_eq!(
            pyramid::level_size(info.height, 1) - layout.grid.overviews[0].height,
            0,
            "у чётной высоты расхождения нет — иначе допуск проверялся бы вхолостую"
        );
    }

    /// Нулевому уровню копий не бывает: он и есть родное разрешение, и
    /// прощать там нечего — округления не было.
    #[test]
    fn нулевой_уровень_читается_из_базового_ifd() {
        let (width, height) = (25437u32, 16729u32);
        let layout = halved_down(width, height, 5);
        let found = layout.grid.source_for((width, height), width, height);
        assert_eq!(found, (0, width, height));

        // Вырожденный случай той же ловушки: у растра в два пикселя копия
        // ровно вдвое мельче, и абсолютный допуск пустил бы её под уровень 0.
        let tiny = Layout::of(true, TILES, vec![Overview { image: 1, width: 1, height: 1, chunk: TILES }], RGB8);
        assert_eq!(tiny.grid.source_for((2, 2), 2, 2), (0, 2, 2), "родному разрешению копий нет");
    }

    /// Допуск ровно в пиксель, а не «примерно». Копия у́же на два уже не
    /// годится: прощается округление, а не близость.
    #[test]
    fn допуск_ровно_в_один_пиксель() {
        let short = |w: u32| Layout::of(true, TILES, vec![Overview { image: 1, width: w, height: w, chunk: TILES }], RGB8);
        let info = bare(2000, 2000);
        assert_eq!(short(499).grid.source_for((info.width, info.height), 500, 500).0, 1, "у́же на пиксель — своя");
        assert_eq!(short(498).grid.source_for((info.width, info.height), 500, 500).0, 0, "у́же на два — чужая");
    }

    /// Стороны проверяются порознь: копия, годная по ширине, может не годиться
    /// по высоте. У вытянутого растра пиксель короткой стороны — это разы, и
    /// уровень собрался бы растягиванием вдвое.
    #[test]
    fn узкая_копия_не_годится_по_высоте() {
        let info = bare(513, 3);
        let layout = Layout::of(true, TILES, vec![Overview { image: 1, width: 256, height: 1, chunk: TILES }], RGB8);
        let (lw, lh) = (pyramid::level_size(513, 1), pyramid::level_size(3, 1));
        assert_eq!((lw, lh), (257, 2));
        assert_eq!(
            layout.grid.source_for((info.width, info.height), lw, lh),
            (0, 513, 3),
            "копия высотой 1 под уровень высотой 2 не годится"
        );
    }

    /// Годных копий у уровня бывает несколько, и берётся самая мелкая — она
    /// дешевле всех по чтению. Порядок в файле при этом ничего не решает:
    /// копии перечислены в порядке обнаружения, а не по размеру.
    #[test]
    fn из_годных_копий_берётся_самая_мелкая() {
        let layout = Layout::of(true, TILES, vec![
                Overview { image: 3, width: 800, height: 800, chunk: TILES },
                Overview { image: 1, width: 3200, height: 3200, chunk: TILES },
                Overview { image: 2, width: 1600, height: 1600, chunk: TILES },
            ], RGB8);
        let found = layout.grid.source_for((6400, 6400), 800, 800);
        assert_eq!(found, (3, 800, 800), "годная мельче — та, что ровно под уровень");
    }

    /// Копия грубее уровня больше, чем на округление, не годится: тайл
    /// собрался бы растягиванием, а не ужатием.
    #[test]
    fn копия_грубее_уровня_не_годится() {
        let layout = Layout::of(true, TILES, vec![Overview { image: 1, width: 400, height: 400, chunk: TILES }], RGB8);
        let found = layout.grid.source_for((1000, 1000), 500, 500);
        assert_eq!(found, (0, 1000, 1000), "уровню в 500 копия в 400 не годится");
    }

    /// Заголовок каталога геоключей: версия, ревизия и число ключей.
    fn geokeys(model: u16) -> Vec<u16> {
        vec![1, 1, 0, 1, 1024, 0, 1, model]
    }

    /// Тот же каталог, но с GTRasterTypeGeoKey: 1 — координата названа для
    /// угла пикселя, 2 — для его середины.
    fn geokeys_raster(model: u16, raster: u16) -> Vec<u16> {
        vec![1, 1, 0, 2, 1024, 0, 1, model, 1025, 0, 1, raster]
    }

    /// Негодное число из тега отменяет решётку целиком.
    ///
    /// Тег читается из файла как есть, а файл приезжает из сети. Место, которого
    /// на Земле нет, решётку отменяет; бесконечность — тем более: доехав до
    /// глобуса, она увела бы разворот долгот в арифметику, у которой ответа нет.
    #[test]
    fn a_tie_that_is_not_a_place_drops_the_whole_lattice() {
        let node = |px: f64, lon: f64, lat: f64| vec![px, 0.0, 0.0, lon, lat, 0.0];
        let lattice = |a: Vec<f64>, b: Vec<f64>| {
            let mut points = a;
            points.extend(b);
            geo_ties(&geokeys(2), &points, &[], &[], 10, 20)
        };

        let whole = lattice(node(0.0, 10.0, 50.0), node(9.0, 11.0, 50.0));
        assert_eq!(whole.len(), 2, "решётка из двух узлов проходит целиком");
        assert_eq!(whole[0].lon, 10.0);

        assert!(lattice(node(0.0, 10.0, 50.0), node(9.0, 11.0, 500.0)).is_empty(), "широты нет");
        assert!(lattice(node(0.0, 10.0, 50.0), node(9.0, 400.0, 50.0)).is_empty(), "долготы нет");
        // Круг у долготы шире, чем у широты, и подставлять их друг за друга
        // нельзя: файл, пишущий долготу как 0…360, потерял бы решётку целиком.
        assert_eq!(lattice(node(0.0, 10.0, 50.0), node(9.0, 359.0, 50.0)).len(), 2);
        assert!(
            lattice(node(0.0, 10.0, 50.0), node(9.0, f64::INFINITY, 50.0)).is_empty(),
            "бесконечная долгота"
        );
        assert!(
            lattice(node(0.0, 10.0, 50.0), node(f64::NAN, 11.0, 50.0)).is_empty(),
            "пиксель, которого нет"
        );
        let mut broken = node(0.0, 10.0, 50.0);
        broken.extend(vec![9.0, f64::INFINITY, 0.0, 11.0, 50.0, 0.0]);
        assert!(geo_ties(&geokeys(2), &broken, &[], &[], 10, 20).is_empty(), "строка, которой нет");
    }

    /// Углы, выведенные из опоры и шага, проверяются конечностью, а не кругом:
    /// у растра, чья опора стои́т далеко от угла, угол законно уходит за полюс.
    #[test]
    fn a_derived_corner_may_leave_the_sphere_but_not_the_numbers() {
        let far = vec![100.0, 200.0, 0.0, 13.0, 50.0, 0.0];
        let corners = geo_ties(&geokeys(2), &far, &[0.5, 0.25, 0.0], &[], 10, 20);
        assert_eq!(corners.len(), 4);
        assert_eq!(corners[0].lat, 100.0, "угол ушёл за полюс, и это не отказ");

        let endless = vec![100.0, 200.0, 0.0, f64::INFINITY, 50.0, 0.0];
        assert!(geo_ties(&geokeys(2), &endless, &[0.5, 0.25, 0.0], &[], 10, 20).is_empty());
        let northless = vec![100.0, 200.0, 0.0, 13.0, f64::INFINITY, 0.0];
        assert!(geo_ties(&geokeys(2), &northless, &[0.5, 0.25, 0.0], &[], 10, 20).is_empty());

        // Матрица бывает конечной, а углы из неё — нет: члены её велики, и
        // произведение на сторону растра переполняется. Вырожденность такую не
        // ловит — определитель выходит числом.
        let huge = vec![
            1.0e308, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        assert!(geo_ties(&geokeys(2), &[], &[], &huge, 10, 20).is_empty());

        // Та же матрица в проекционной ветви: члены конечны, определитель тоже,
        // а дальний угол уходит в бесконечность.
        assert!(geo_placement(&projected_keys(32638), &[], &[], &huge, 10, 20).is_none());
    }

    /// Решётка узлов проходит как есть: пиксель берётся из первых двух чисел
    /// шестёрки, место — из четвёртого и пятого. Порядок узлов — файла: им и
    /// сказано, каким пикселем куда лёг растр.
    #[test]
    fn tiepoint_lattice_keeps_pixel_and_place() {
        // Два узла верхнего ребра гранулы Sentinel-1: слева восток, справа
        // запад — снимок нисходящего витка лежит поперёк меридианов.
        let points = vec![
            0.0, 0.0, 0.0, 2.707, 73.395, 0.0, //
            529.0, 0.0, 0.0, 2.096, 73.470, 0.0,
        ];
        let ties = geo_ties(&geokeys(2), &points, &[], &[], 10572, 9993);
        assert_eq!(ties.len(), 2);
        assert_eq!((ties[0].px, ties[0].py), (0.0, 0.0));
        assert_eq!((ties[0].lat, ties[0].lon), (73.395, 2.707));
        assert_eq!((ties[1].px, ties[1].py), (529.0, 0.0));
    }

    /// Одна точка с шагом пикселя — четырьмя углами, и строки идут на юг:
    /// нижний край южнее верхнего, а не наоборот.
    #[test]
    fn pixel_scale_becomes_four_corners_facing_south() {
        let points = vec![0.0, 0.0, 0.0, 10.0, 50.0, 0.0];
        let ties = geo_ties(&geokeys(2), &points, &[0.5, 0.25, 0.0], &[], 100, 200);
        assert_eq!(ties.len(), 4);
        assert_eq!((ties[0].lon, ties[0].lat), (10.0, 50.0));
        assert_eq!((ties[1].lon, ties[1].lat), (60.0, 50.0), "правый край восточнее");
        assert_eq!((ties[2].lon, ties[2].lat), (10.0, 0.0), "нижний край южнее");
    }

    /// Координата, названная для середины пикселя, сдвигает узел на его
    /// половину: наружу узел уезжает долей растра, где ноль — край. Так
    /// привязан Copernicus DSM, и половина его секундного шага — пятнадцать
    /// метров на местности.
    ///
    /// Сдвигается при этом весь растр, а не растягивается: размах между
    /// краями остаётся тем же.
    #[test]
    fn a_point_raster_shifts_the_node_by_half_a_pixel() {
        let points = vec![0.0, 0.0, 0.0, 13.0, 1.0, 0.0];
        let step = [1.0 / 3600.0, 1.0 / 3600.0, 0.0];
        let corner = geo_ties(&geokeys_raster(2, 1), &points, &step, &[], 3600, 3600);
        let middle = geo_ties(&geokeys_raster(2, 2), &points, &step, &[], 3600, 3600);

        assert_eq!((corner[0].lon, corner[0].lat), (13.0, 1.0), "угол назван как есть");
        let half = 0.5 / 3600.0;
        assert!((middle[0].lon - (13.0 - half)).abs() < 1e-12, "{}", middle[0].lon);
        assert!((middle[0].lat - (1.0 + half)).abs() < 1e-12, "{}", middle[0].lat);
        assert!(
            ((middle[1].lon - middle[0].lon) - (corner[1].lon - corner[0].lon)).abs() < 1e-12,
            "растр растянулся вместо сдвига по долготе"
        );
        assert!(
            ((middle[2].lat - middle[0].lat) - (corner[2].lat - corner[0].lat)).abs() < 1e-12,
            "растр растянулся вместо сдвига по широте"
        );
    }

    /// Один и тот же растр, названный парой «точка + шаг» и равной ей матрицей,
    /// ложится одинаково — и с серединой пикселя тоже. Ветки разные, и сдвиг в
    /// них берётся с разных концов: у пары — с начала отсчёта, у матрицы — с
    /// довода. Разойдись они, и один и тот же файл лежал бы по-разному в
    /// зависимости от того, каким тегом его привязали.
    #[test]
    fn the_matrix_and_the_step_place_a_point_raster_alike() {
        let (dx, dy) = (1.0 / 3600.0, 1.0 / 3600.0);
        let points = vec![0.0, 0.0, 0.0, 13.0, 1.0, 0.0];
        let matrix = vec![dx, 0.0, 0.0, 13.0, 0.0, -dy, 0.0, 1.0];
        let keys = geokeys_raster(2, 2);

        let by_step = geo_ties(&keys, &points, &[dx, dy, 0.0], &[], 3600, 3600);
        let by_matrix = geo_ties(&keys, &[], &[], &matrix, 3600, 3600);

        assert_eq!(by_step.len(), by_matrix.len());
        for (step, matrix) in by_step.iter().zip(&by_matrix) {
            assert!((step.lon - matrix.lon).abs() < 1e-12, "{} против {}", step.lon, matrix.lon);
            assert!((step.lat - matrix.lat).abs() < 1e-12, "{} против {}", step.lat, matrix.lat);
        }
        // И это не совпадение двух нулей: сдвиг в обеих ветках произошёл.
        let corner = geo_ties(&geokeys_raster(2, 1), &[], &[], &matrix, 3600, 3600);
        assert!((corner[0].lon - by_matrix[0].lon).abs() > dy / 4.0);
    }

    /// Решётка узлов сдвигается той же половиной пикселя: у неё пиксель назван
    /// теми же координатами, что и у остальных, и оставленная без сдвига она
    /// разошлась бы с ними на растре, привязанном обоими способами разом.
    #[test]
    fn a_tiepoint_lattice_shifts_by_half_a_pixel_too() {
        let points = vec![
            0.0, 0.0, 0.0, 2.707, 73.395, 0.0, //
            529.0, 0.0, 0.0, 2.096, 73.470, 0.0,
        ];
        let ties = geo_ties(&geokeys_raster(2, 2), &points, &[], &[], 10572, 9993);
        assert_eq!((ties[0].px, ties[0].py), (0.5, 0.5));
        assert_eq!((ties[1].px, ties[1].py), (529.5, 0.5));
        // Место узла при этом не трогают: сдвинулся пиксель, а не координата.
        assert_eq!((ties[0].lat, ties[0].lon), (73.395, 2.707));
    }

    /// Умолчание спецификации — угол: файл без GTRasterTypeGeoKey читается так
    /// же, как файл, объявивший угол прямо.
    #[test]
    fn a_raster_without_the_key_is_placed_by_its_corner() {
        let points = vec![0.0, 0.0, 0.0, 13.0, 1.0, 0.0];
        let step = [1.0 / 3600.0, 1.0 / 3600.0, 0.0];
        let silent = geo_ties(&geokeys(2), &points, &step, &[], 3600, 3600);
        let said = geo_ties(&geokeys_raster(2, 1), &points, &step, &[], 3600, 3600);
        assert_eq!((silent[0].lon, silent[0].lat), (said[0].lon, said[0].lat));
    }

    /// Копией считается уменьшенная картинка, но не уменьшенная маска: GDAL
    /// пишет копии маски как 5 = 1|4, и взятая за копию однобитная маска
    /// показалась бы вместо снимка.
    #[test]
    fn a_mask_is_not_an_overview() {
        assert!(is_overview(1), "уменьшенная копия");
        assert!(!is_overview(5), "уменьшенная копия маски");
        assert!(!is_overview(4), "маска в полный размер");
        assert!(!is_overview(0), "сам снимок");
        assert!(is_overview(3), "копия страницы многостраничного — всё ещё копия");
    }

    /// Повёрнутый растр привязан матрицей, и углы её слушаются: у повёрнутого
    /// верхний правый угол не на той же широте, что верхний левый, — а пара
    /// «точка + шаг» такого сказать не умеет вовсе.
    #[test]
    fn a_transformation_matrix_places_the_rotated_corners() {
        // Поворот на 90°: столбцы растра идут на восток, строки — на север.
        let matrix = vec![
            0.0, 0.5, 0.0, 10.0, //
            0.25, 0.0, 0.0, 50.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let ties = geo_ties(&geokeys(2), &[], &[], &matrix, 100, 200);
        assert_eq!(ties.len(), 4);
        assert_eq!((ties[0].lon, ties[0].lat), (10.0, 50.0));
        assert_eq!((ties[1].lon, ties[1].lat), (10.0, 75.0), "вправо по растру — на север");
        assert_eq!((ties[2].lon, ties[2].lat), (110.0, 50.0), "вниз по растру — на восток");
    }

    /// Матрица читается только там, где пары «точка + шаг» нет: спецификация
    /// разрешает ровно одну из двух привязок, и файл с обеими врёт хотя бы
    /// одной. Верить в таком случае надо той, которую понимают все.
    #[test]
    fn the_pair_outranks_the_matrix() {
        let points = vec![0.0, 0.0, 0.0, 10.0, 50.0, 0.0];
        let matrix = vec![
            0.0, 0.5, 0.0, 999.0, //
            0.25, 0.0, 0.0, 999.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let ties = geo_ties(&geokeys(2), &points, &[0.5, 0.25, 0.0], &matrix, 100, 200);
        assert_eq!((ties[0].lon, ties[0].lat), (10.0, 50.0));
    }

    /// Вырожденная матрица — не привязка: масштаба у неё нет, и все четыре
    /// угла сошлись бы в одну точку.
    #[test]
    fn a_degenerate_matrix_is_no_binding() {
        let flat = vec![0.0; 16];
        assert!(geo_ties(&geokeys(2), &[], &[], &flat, 100, 200).is_empty());
        // Как и обрезанная: восьми чисел на две строки не набралось.
        assert!(geo_ties(&geokeys(2), &[], &[], &[1.0, 0.0, 0.0], 100, 200).is_empty());
    }

    /// Привязка к проекции узлами в градусах не притворяется: перевести её мог
    /// бы только знающий саму проекцию, а тайлер про Землю не знает ничего.
    /// Уезжает она отдельным полем (см. [`geo_placement`]).
    #[test]
    fn projected_files_yield_nothing() {
        let points = vec![0.0, 0.0, 0.0, 600_000.0, 7_800_000.0, 0.0];
        assert!(geo_ties(&geokeys(1), &points, &[10.0, 10.0, 0.0], &[], 10, 10).is_empty());
        // Как и файл вовсе без геотегов.
        assert!(geo_ties(&[], &points, &[10.0, 10.0, 0.0], &[], 10, 10).is_empty());
    }

    /// Каталог ключей проекционного растра: модель — метры, плюс код системы.
    fn projected_keys(epsg: u16) -> Vec<u16> {
        vec![1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, epsg]
    }

    /// Он же с GTRasterTypeGeoKey: 1 — координата названа для угла пикселя,
    /// 2 — для его середины.
    fn projected_keys_raster(epsg: u16, raster: u16) -> Vec<u16> {
        vec![1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, raster, 3072, 0, 1, epsg]
    }

    /// Растр в проекции отдаётся кодом системы и преобразованием: пиксель (0,0)
    /// ложится ровно в опорную точку, шаг идёт по осям, а узлов в градусах у
    /// такого файла нет вовсе.
    ///
    /// Числа — с настоящего файла Landsat-5 (`LT51780121988065ESA00_B1.TIF`,
    /// зона 38 северная, шаг 30 м): ровно тот случай, ради которого поле и
    /// заведено.
    #[test]
    fn a_projected_raster_reports_its_system_and_transform() {
        let points = vec![0.0, 0.0, 0.0, 499_536.218_75, 7_693_329.5, 0.0];
        let found = geo_placement(&projected_keys(32638), &points, &[30.0, 30.0, 0.0], &[], 10, 20)
            .expect("привязка к зоне 38");

        assert_eq!(found.epsg, 32638);
        assert_eq!((found.affine[2], found.affine[5]), (499_536.218_75, 7_693_329.5));
        assert_eq!(found.affine[0], 30.0, "шаг на восток");
        // Шаг по Y положителен, а строки растра идут на юг.
        assert_eq!(found.affine[4], -30.0, "шаг на юг");
        assert_eq!((found.affine[1], found.affine[3]), (0.0, 0.0), "поворота у пары «точка+шаг» нет");

        assert!(
            geo_ties(&projected_keys(32638), &points, &[30.0, 30.0, 0.0], &[], 7956, 7740)
                .is_empty(),
            "метры зоны не должны уехать узлами в градусах"
        );
    }

    /// Опорная точка не обязана стоять в углу растра, а шаг — быть квадратным.
    /// На настоящем файле этого не видно: у Landsat опора в (0,0) и шаг 30×30, и
    /// на таких числах перепутанные оси неотличимы друг от друга.
    #[test]
    fn a_tiepoint_off_the_corner_places_the_whole_raster() {
        let points = vec![100.0, 200.0, 0.0, 500_000.0, 7_000_000.0, 0.0];
        let found = geo_placement(&projected_keys(32638), &points, &[30.0, 15.0, 0.0], &[], 10, 20)
            .expect("привязка");

        assert_eq!(found.affine[0], 30.0, "шаг на восток");
        assert_eq!(found.affine[4], -15.0, "шаг на юг — свой, а не тот же");
        // Пиксель (0,0) стои́т на сто шагов западнее опоры и на двести севернее.
        assert_eq!(found.affine[2], 500_000.0 - 100.0 * 30.0);
        assert_eq!(found.affine[5], 7_000_000.0 + 200.0 * 15.0);
    }

    /// То же в градусной ветке: обе привязки называют опору одинаково, и
    /// разойтись в этом им нельзя — два растра одного снимка легли бы врозь.
    #[test]
    fn a_degree_tiepoint_off_the_corner_places_the_whole_raster() {
        let points = vec![100.0, 200.0, 0.0, 13.0, 50.0, 0.0];
        let ties = geo_ties(&geokeys(2), &points, &[0.5, 0.25, 0.0], &[], 10, 20);
        assert_eq!(ties[0].lon, 13.0 - 100.0 * 0.5, "сто шагов западнее опоры");
        assert_eq!(ties[0].lat, 50.0 + 200.0 * 0.25, "двести шагов севернее");
    }

    /// Полпикселя у повёрнутой матрицы снимается по обеим осям сразу: у растра
    /// по осям второй член нулевой, и выброшенный, он там не виден вовсе.
    #[test]
    fn the_half_pixel_of_a_rotated_matrix_takes_both_axes() {
        let matrix = vec![
            0.0, 30.0, 0.0, 500_000.0, //
            30.0, 0.0, 0.0, 7_000_000.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let found =
            geo_placement(&projected_keys_raster(32638, 2), &[], &[], &matrix, 10, 20).expect("поворот");
        assert_eq!(found.affine[2], 500_000.0 - 15.0, "полшага по второму члену");
        assert_eq!(found.affine[5], 7_000_000.0 - 15.0, "и по нему же в другой строке");
    }

    /// Матрица со сложенными строками вырождена, хотя ни один её член не ноль:
    /// растр складывается в линию. Держится это знаком в определителе — при
    /// сложении вместо вычитания такая привязка проехала бы насквозь.
    #[test]
    fn a_shear_that_collapses_the_raster_is_no_binding() {
        let shear = vec![
            30.0, 30.0, 0.0, 500_000.0, //
            30.0, 30.0, 0.0, 7_000_000.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        assert!(geo_placement(&projected_keys(32638), &[], &[], &shear, 10, 20).is_none());
        assert!(geo_ties(&geokeys(2), &[], &[], &shear, 10, 10).is_empty());
    }

    /// Не-число привязкой не является ни в одной ветке и ни в одном теге.
    /// Сравнение с нулём его не ловит — `NaN != 0` истинно, — а доехав до рамки,
    /// оно даёт слой, который не выберет уровень пирамиды никогда.
    #[test]
    fn a_transform_of_not_a_number_binds_nothing() {
        let points = vec![0.0, 0.0, 0.0, 500_000.0, 7_000_000.0, 0.0];
        let sick_step = [f64::NAN, 30.0, 0.0];
        assert!(geo_placement(&projected_keys(32638), &points, &sick_step, &[], 10, 20).is_none(), "шаг");
        assert!(geo_ties(&geokeys(2), &points, &sick_step, &[], 10, 10).is_empty(), "он же в градусах");

        // Шаг годен, а место названо не числом: рамка вышла бы целой с виду.
        let sick_tie = vec![0.0, 0.0, 0.0, f64::NAN, 7_000_000.0, 0.0];
        assert!(
            geo_placement(&projected_keys(32638), &sick_tie, &[30.0, 30.0, 0.0], &[], 10, 20).is_none(),
            "опорная точка"
        );

        let sick_matrix = vec![
            30.0, 0.0, 0.0, f64::NAN, //
            0.0, -30.0, 0.0, 7_000_000.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        assert!(geo_placement(&projected_keys(32638), &[], &[], &sick_matrix, 10, 20).is_none(), "матрица");
    }

    /// Градусный файл проекцией не притворяется — и наоборот. Ветки
    /// взаимоисключимы: одна из них обязана промолчать на числах другой, иначе
    /// метры зоны уехали бы широтой.
    #[test]
    fn the_two_bindings_do_not_answer_for_each_other() {
        let points = vec![0.0, 0.0, 0.0, 13.0, 1.0, 0.0];
        let step = [1.0 / 3600.0, 1.0 / 3600.0, 0.0];

        assert!(geo_placement(&geokeys(2), &points, &step, &[], 10, 20).is_none(), "градусы — не проекция");
        assert!(!geo_ties(&geokeys(2), &points, &step, &[], 3600, 3600).is_empty());

        let metres = vec![0.0, 0.0, 0.0, 500_000.0, 7_000_000.0, 0.0];
        let keys = projected_keys(32638);
        assert!(geo_ties(&keys, &metres, &[30.0, 30.0, 0.0], &[], 10, 10).is_empty(),
            "проекция — не градусы");
        assert!(geo_placement(&keys, &metres, &[30.0, 30.0, 0.0], &[], 10, 20).is_some());
    }

    /// Полпикселя середины снимается у проекции так же, как у градусной ветки:
    /// начало преобразования уезжает на полшага в ту же сторону. Разойдись эти
    /// две конвенции — и два растра одного снимка легли бы со сдвигом друг
    /// относительно друга.
    #[test]
    fn the_half_pixel_moves_both_bindings_the_same_way() {
        let points = vec![0.0, 0.0, 0.0, 500_000.0, 7_000_000.0, 0.0];
        let step = [30.0, 30.0, 0.0];

        let corner = geo_placement(&projected_keys_raster(32638, 1), &points, &step, &[], 10, 20)
            .expect("угол пикселя");
        let middle = geo_placement(&projected_keys_raster(32638, 2), &points, &step, &[], 10, 20)
            .expect("середина пикселя");

        assert_eq!((corner.affine[2], corner.affine[5]), (500_000.0, 7_000_000.0));
        assert_eq!(middle.affine[2] - corner.affine[2], -15.0, "на полшага на запад");
        assert_eq!(middle.affine[5] - corner.affine[5], 15.0, "на полшага на север");

        // Та же пара для градусной ветки: знаки обязаны совпасть.
        let degrees = [1.0 / 3600.0, 1.0 / 3600.0, 0.0];
        let node = vec![0.0, 0.0, 0.0, 13.0, 1.0, 0.0];
        let by_corner = geo_ties(&geokeys_raster(2, 1), &node, &degrees, &[], 10, 10);
        let by_middle = geo_ties(&geokeys_raster(2, 2), &node, &degrees, &[], 10, 10);
        assert!(by_middle[0].lon < by_corner[0].lon, "на полшага на запад");
        assert!(by_middle[0].lat > by_corner[0].lat, "на полшага на север");
    }

    /// Матрица и пара «точка + шаг» задают одно преобразование и обязаны дать
    /// одинаковое аффинное — на одних и тех же числах. Порознь эти две ветки
    /// разошлись бы знаком по Y и никто бы этого не заметил.
    #[test]
    fn the_matrix_and_the_step_describe_one_projection() {
        let step = 30.0;
        let points = vec![0.0, 0.0, 0.0, 499_536.0, 7_693_329.0, 0.0];
        let matrix = vec![
            step, 0.0, 0.0, 499_536.0, //
            0.0, -step, 0.0, 7_693_329.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let keys = projected_keys_raster(32638, 2);
        let by_step = geo_placement(&keys, &points, &[step, step, 0.0], &[], 10, 20).expect("шагом");
        let by_matrix = geo_placement(&keys, &[], &[], &matrix, 10, 20).expect("матрицей");
        assert_eq!(by_step.affine, by_matrix.affine);
    }

    /// Повёрнутая матрица доезжает поворотом, а не своей диагональю: у растра,
    /// снятого не по осям зоны, внедиагональные члены и есть вся привязка.
    #[test]
    fn a_rotated_matrix_keeps_its_rotation() {
        // Поворот на 90°: восток растра идёт на север зоны.
        let matrix = vec![
            0.0, 30.0, 0.0, 500_000.0, //
            30.0, 0.0, 0.0, 7_000_000.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let found = geo_placement(&projected_keys(32638), &[], &[], &matrix, 10, 20).expect("поворот");
        assert_eq!((found.affine[0], found.affine[1]), (0.0, 30.0));
        assert_eq!((found.affine[3], found.affine[4]), (30.0, 0.0));
    }

    /// Система, названная непонятно, — это не система. Ноль и user-defined
    /// кодом не являются: параметры такой проекции лежат отдельными ключами, и
    /// собрать её из них значило бы разбирать проекции, а не читать растр.
    #[test]
    fn a_system_without_a_code_is_no_system() {
        let points = vec![0.0, 0.0, 0.0, 500_000.0, 7_000_000.0, 0.0];
        let step = [30.0, 30.0, 0.0];
        let bare = vec![1, 1, 0, 1, 1024, 0, 1, 1];

        assert!(geo_placement(&bare, &points, &step, &[], 10, 20).is_none(), "кода нет вовсе");
        assert!(geo_placement(&projected_keys(32767), &points, &step, &[], 10, 20).is_none(), "user-defined");
        assert!(geo_placement(&projected_keys(0), &points, &step, &[], 10, 20).is_none(), "ноль");
    }

    /// Решётка опорных точек в метрах зоны — это отказ, а не привязка по
    /// первому узлу: решётка описывает нелинейную раскладку, и шестёркой чисел
    /// она не выражается. Взятая по одному узлу, она врала бы тем сильнее, чем
    /// дальше от него.
    #[test]
    fn a_projected_lattice_is_refused_not_flattened() {
        let points = vec![
            0.0, 0.0, 0.0, 500_000.0, 7_000_000.0, 0.0, //
            100.0, 0.0, 0.0, 503_000.0, 7_000_100.0, 0.0,
        ];
        assert!(geo_placement(&projected_keys(32638), &points, &[30.0, 30.0, 0.0], &[], 10, 20).is_none());
    }

    /// Вырожденное преобразование привязкой не является — ни в одной ветке и ни
    /// в одной модели координат: растр сложился бы в линию или в точку, а
    /// снаружи такая рамка неотличима от настоящей.
    #[test]
    fn a_degenerate_transform_binds_nothing() {
        let points = vec![0.0, 0.0, 0.0, 500_000.0, 7_000_000.0, 0.0];
        let flat = vec![
            30.0, 0.0, 0.0, 500_000.0, //
            0.0, 0.0, 0.0, 7_000_000.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        for step in [[0.0, 30.0, 0.0], [30.0, 0.0, 0.0]] {
            assert!(geo_placement(&projected_keys(32638), &points, &step, &[], 10, 20).is_none(), "шаг");
            assert!(
                geo_ties(&geokeys(2), &points, &step, &[], 10, 10).is_empty(),
                "тот же нулевой шаг в градусах"
            );
        }
        assert!(geo_placement(&projected_keys(32638), &[], &[], &flat, 10, 20).is_none(), "матрица");
        assert!(geo_ties(&geokeys(2), &[], &[], &flat, 10, 10).is_empty(), "она же в градусах");
    }

    /// Чужой датум называется вслух: места для него дальше по течению нет — и
    /// узлы, и рамка объявлены в WGS84, — так что файл на Пулково-42 уехал бы
    /// туда молча, а это сотня метров в средней полосе.
    ///
    /// Молчание файла и user-defined чужим датумом не являются: сказать о них
    /// нечего, и строка о них была бы шумом в каждом логе.
    #[test]
    fn a_foreign_datum_is_named_aloud() {
        let with = |code: u16| vec![1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, code];
        assert_eq!(foreign_datum(&with(4284)), Some(4284), "Пулково-42");
        assert_eq!(foreign_datum(&with(4258)), Some(4258), "ETRS89");
        assert_eq!(foreign_datum(&with(4326)), None, "он и есть WGS84");
        assert_eq!(foreign_datum(&with(4979)), None, "он же, трёхмерный");
        assert_eq!(
            foreign_datum(&with(4055)),
            None,
            "сфера веб-Меркатора: координаты те же, что у WGS84"
        );
        assert_eq!(foreign_datum(&with(32767)), None, "user-defined — не ответ");
        assert_eq!(foreign_datum(&geokeys(2)), None, "ключа нет вовсе");
    }

    /// Оговорка о привязке говорит про один род беды — «файл о ней заговорил, а
    /// прочитать нечем», — и молчит обо всём остальном.
    ///
    /// Иначе поле врёт своим именем. Чужой датум сюда попасть не должен: у него
    /// привязка как раз взялась, и снимок лежит на своём месте с точностью до
    /// сотни метров, а подпись объявляла бы место неизвестным. Молчание файла —
    /// тем более не беда: так лежит всякий квиклук, и жалоба у каждого была бы
    /// шумом, за которым не видно настоящей.
    ///
    /// Код системы называется числом: отладочное `Some(1)` в подписи под
    /// снимком не говорит смотрящему ничего.
    #[test]
    fn оговорка_о_привязке_говорит_только_о_непрочитанной() {
        // 1024 = 2 (географическая), 3072 не назван.
        let keys = geokeys(2);
        let said = binding_trouble(true, false, &keys, 1).expect("несёт, а взять нечем");
        assert!(said.contains("модель координат 2"), "{}", said);
        assert!(said.contains("система не названа"), "{}", said);
        assert!(said.contains("опорных точек 1"), "{}", said);
        assert!(!said.contains("Some("), "отладочная печать уехала бы подписью: {}", said);

        assert_eq!(binding_trouble(true, true, &keys, 1), None, "привязка взята — беды нет");
        assert_eq!(binding_trouble(false, false, &keys, 0), None, "файл о привязке не говорил");
        assert_eq!(binding_trouble(false, true, &keys, 0), None);
    }

    /// Цветовая модель разбирается один раз и отвечает обоим спрашивающим — и
    /// о раскладке пикселя, и о разрядности. Многополосный без цветовой
    /// интерпретации — это стопка величин: показывается первая, а вторая не
    /// прозрачность, хотя сэмплов у неё, как у серого с альфой, ровно два.
    #[test]
    fn colour_model_answers_channels_and_depth() {
        assert_eq!(model(tiff::ColorType::Gray(16)).unwrap().1, 16);
        assert_eq!(model(tiff::ColorType::RGBA(8)).unwrap().0.channels, 4);
        assert_eq!(pixel(tiff::ColorType::RGB(8)).unwrap().channels, 3);
        let (stack, bits) = model(tiff::ColorType::Multiband { bit_depth: 8, num_samples: 2 })
            .expect("стопка величин — не отказ");
        assert_eq!((stack.channels, stack.colors(), stack.has_alpha(), bits), (2, 1, false, 8));
        assert!(pixel(tiff::ColorType::CMYK(8)).is_err());
    }

    /// Отказ по формату отсчёта и разворот чанка — одно правило, названное
    /// дважды. Разойдись они, файл либо получил бы отказ над тем, что
    /// читается, либо доехал бы до чтения ради отказа на первом чанке — а это
    /// уже скачанные гигабайты.
    ///
    /// Пустые векторы годятся: `typed` смотрит на разновидность, а не на
    /// содержимое.
    #[test]
    fn отказ_по_формату_сходится_с_разворотом_чанка() {
        use tiff::decoder::DecodingResult as R;
        let cases: [(u16, u8, DecodingResult); 9] = [
            (1, 8, R::U8(Vec::new())),
            (1, 16, R::U16(Vec::new())),
            (2, 16, R::I16(Vec::new())),
            (3, 32, R::F32(Vec::new())),
            (1, 32, R::U32(Vec::new())),
            (1, 64, R::U64(Vec::new())),
            (2, 8, R::I8(Vec::new())),
            (3, 16, R::F16(Vec::new())),
            (3, 64, R::F64(Vec::new())),
        ];
        for (format, bits, data) in &cases {
            assert_eq!(
                sampled(*format, *bits),
                typed(data).is_ok(),
                "формат {format} по {bits} бит: описание и чанк разошлись"
            );
        }
        // Комплексный отсчёт — тот самый Sentinel-1 SLC: свой разновидности у
        // него в выходе декодера нет вовсе, и поймать его можно только здесь.
        assert!(!sampled(5, 32), "комплексный целый");
        assert!(!sampled(6, 64), "комплексный дробный");
    }

    /// Несжатый читается всегда, а сжатия, которого не собрано, в таблице нет.
    /// Согласие таблицы с фичами крейта стережёт тест buildgen — здесь только
    /// то, что она вообще спрашивается.
    #[test]
    fn сжатие_вне_таблицы_не_разбирается() {
        assert!(decodable(1), "несжатый");
        assert!(decodable(0xC350), "ZSTD — им сжат c_gls_LST");
        assert!(!decodable(0xC351), "WebP в наборе не собран");
        assert!(!decodable(34_887), "LERC крейт не знает ни под какой фичей");
    }

    /// Обрезанный чанк кончается строкой, а не выходом за границу среза.
    /// Разница здесь не стилистическая: срез уронил бы инстанс трапом, после
    /// которого заказчик ждёт ответа, которого никто не пришлёт.
    #[test]
    fn a_short_chunk_is_refused_not_blitted() {
        let mapping = Mapping::identity(None);
        let whole = DecodingResult::U8(vec![7u8; 4 * 4]);
        let rgba = chunk_rgba(&mapping, &whole, Pixel::named(1), 4, 4).expect("целый чанк");
        assert_eq!(rgba.len(), 4 * 4 * 4);

        let short = DecodingResult::U8(vec![7u8; 4 * 3]);
        let refused = chunk_rgba(&mapping, &short, Pixel::named(1), 4, 4).expect_err("чанк неполон");
        assert!(refused.contains("неполным"), "{}", refused);
    }

    /// Копия, расходящаяся с уровнем на пиксель округления, читается своей
    /// сеткой: тайл берётся ровно там, где лежит, и занимает ровно чанк файла.
    ///
    /// Цена ошибки здесь не в виде картинки, а в скорости: пиксель, приписанный
    /// области сверх нужного, делает её 513×513 и накрывает 2×2 чанка вместо
    /// одного, а собираются они рядами, отстоящими на весь ряд чанков. Чтение
    /// прыгает по файлу вперёд-назад на каждом тайле, хост читает это как
    /// случайный доступ и сбрасывает разгон упреждающего чтения.
    ///
    /// Числа взяты у настоящего снимка: Sentinel-1 GRD COG 25309×9217, уровень
    /// 1 (12655×4609) из копии 12654×4608.
    #[test]
    fn копия_в_пиксель_от_уровня_читается_своей_сеткой() {
        let (lw, lh) = (12655u32, 4609u32);
        let (sw, sh) = (12654u32, 4608u32);
        let tile = |tx, ty| region_of((tx, ty), (TILE, TILE), (lw, lh), (sw, sh), true);

        // Внутренний тайл: ровно тайл, ровно на границе чанка, окно — всё
        // прочитанное.
        let at = tile(3, 1);
        assert_eq!((at.sx0, at.sx1), (3 * 512, 4 * 512));
        assert_eq!((at.sy0, at.sy1), (512, 1024));
        assert_eq!((at.window.x0, at.window.y0), (0.0, 0.0));
        assert_eq!((at.window.x1, at.window.y1), (512.0, 512.0));

        // Начало тайла кратно стороне чанка у всех столбцов и рядов — это и
        // есть «один чанк на тайл».
        for tx in 0..(lw.div_ceil(TILE) - 1) {
            let at = tile(tx, 0);
            assert_eq!(at.sx0 % u64::from(TILE), 0, "столбец {tx} съехал с границы чанка");
            assert_eq!(at.sx1 - at.sx0, u64::from(TILE), "столбец {tx} шире тайла");
        }

        // И НИ ОДНОМУ тайлу сетки, включая краевые, не достаётся пустой
        // области: пустая — это не пустой тайл, а отказ всего прохода, и с ним
        // приговор всем его ячейкам разом.
        for ty in 0..lh.div_ceil(TILE) {
            for tx in 0..lw.div_ceil(TILE) {
                let at = region_of(
                    (tx, ty),
                    (pyramid::tile_extent(tx, lw), pyramid::tile_extent(ty, lh)),
                    (lw, lh),
                    (sw, sh),
                    true,
                );
                assert!(at.sx1 > at.sx0 && at.sy1 > at.sy0, "тайлу {tx}:{ty} не досталось пикселей");
                assert!(at.sx1 <= u64::from(sw) && at.sy1 <= u64::from(sh), "тайл {tx}:{ty} за копией");
            }
        }

        // А пропорциональный пересчёт — тот самый дефект: область на пиксель
        // левее и на пиксель шире, то есть два чанка вместо одного.
        let askew = region_of((3, 1), (TILE, TILE), (lw, lh), (sw, sh), false);
        assert_eq!(askew.sx0, 3 * 512 - 1, "пересчёт не сдвинул область — проверять нечего");
        assert_eq!(askew.sx1 - askew.sx0, u64::from(TILE) + 1, "область шириной не в тайл");
    }

    /// Краевой тайл копия не дотягивает на тот же пиксель, и растягивать его
    /// нельзя: у него своя, короткая длина, а ресемпл доводит её до стороны
    /// тайла — как и всякую краевую.
    /// Ряд, которому на уровне досталась одна строка, а в копии не досталось
    /// ни одной: сторона 4609 — это десять рядов, последний в один пиксель, а
    /// копия кончается на 4608. Такому ряду отдаётся последняя строка копии.
    ///
    /// Ошибка здесь не косметическая: пустая область — отказ всего прохода, а
    /// отказ приговаривает все его ячейки, и нижний ряд уровня не появился бы
    /// уже никогда.
    #[test]
    fn ряду_за_краем_копии_достаётся_её_последняя_строка() {
        let (lw, lh) = (12655u32, 4609u32);
        let (sw, sh) = (12654u32, 4608u32);
        let last = lh.div_ceil(TILE) - 1;
        let th = pyramid::tile_extent(last, lh);
        assert_eq!((last, th), (9, 1), "проверяется именно ряд в один пиксель");

        let at = region_of((0, last), (TILE, th), (lw, lh), (sw, sh), true);
        assert_eq!((at.sy0, at.sy1), (4607, 4608), "ряду не досталось ни строки");
        assert_eq!(at.window.y1, 1.0);

        // Тот же ряд у прежнего, пропорционального пути — та же строка: клэмп
        // не выдумывает пикселей, а берёт те же, что брались всегда.
        let askew = region_of((0, last), (TILE, th), (lw, lh), (sw, sh), false);
        assert_eq!((askew.sy0, askew.sy1), (at.sy0, at.sy1));
    }

    #[test]
    fn краевой_тайл_доводится_а_не_обрезается() {
        let (lw, lh) = (12655u32, 4609u32);
        let (sw, sh) = (12654u32, 4608u32);
        let last = lw.div_ceil(TILE) - 1;
        let tw = pyramid::tile_extent(last, lw);

        let at = region_of((last, 0), (tw, TILE), (lw, lh), (sw, sh), true);
        assert_eq!(at.sx1, u64::from(sw), "край области ушёл за копию");
        assert_eq!(at.sx1 - at.sx0, u64::from(tw) - 1, "краевому досталось не то, что осталось");
        assert_eq!(at.window.x1, (at.sx1 - at.sx0) as f64, "окно разошлось с прочитанным");
    }

    /// Дробная копия своей сеткой не читается: там пересчёт и есть правда, а
    /// прочитанное шире тайла нарочно — усреднению нужен каждый задетый
    /// пиксель.
    #[test]
    fn дробная_копия_читается_пересчётом() {
        // Копия вдвое мельче уровня: тайл в 512 пикселей уровня — это 256
        // пикселей копии, и ресемпл разворачивает их обратно.
        let at = region_of((2, 0), (TILE, TILE), (2048, 2048), (1024, 1024), false);
        assert_eq!((at.sx0, at.sx1), (512, 768), "половинная копия считается ровно вдвое");
        assert_eq!(at.window.x1 - at.window.x0, 256.0, "тайлу принадлежит всё прочитанное");

        // А трёхкратная копия даёт дробные границы, и прочитанное выходит шире
        // окна — ровно та доля пикселя, ради которой границы и разводят.
        let odd = region_of((1, 0), (TILE, TILE), (1536, 1536), (512, 512), false);
        assert!(odd.sx1 - odd.sx0 >= 170, "прочитанное у́же задетого");
        assert!(odd.window.x1 - odd.window.x0 <= (odd.sx1 - odd.sx0) as f64);
    }

    /// Окно не смеет обещать больше, чем отдаст `produce_direct`.
    ///
    /// Это главное его свойство, и цена ошибки здесь не «медленно», а «никогда»:
    /// отказ по [`REGION_CAP`] валит ВЕСЬ проход, проход приговаривает все свои
    /// ячейки разом (`Fetch::produced`), а лестницы у такого источника нет —
    /// отступить на ступень грубее и добрать оттуда не выйдет. Уровень просто
    /// не появляется.
    ///
    /// Проверяется обходом настоящей сетки, а не идеальным тайлом: границы
    /// разводятся наружу, и худший тайл бывает на пиксель шире.
    #[test]
    fn окно_не_обещает_того_чего_проход_не_отдаст() {
        let sizes = [
            (25309u32, 17408u32), // Sentinel-1 GRD
            (25437, 16709),       // он же другой нарезки
            (8271, 8391),         // Landsat 7
            (7681, 7801),
            (10980, 10980), // Sentinel-2
            (1025, 1025),   // сторона ровно на границе тайла плюс пиксель
            (513, 3),
        ];
        for (w, h) in sizes {
            let layout = Layout::of(false, (w, 1), Vec::new(), RGB8);
            for level in 0..pyramid::level_count(w, h) {
                if !layout.grid.pointwise((w, h), level) {
                    continue;
                }
                let (lw, lh) =
                    (pyramid::level_size(w, level).max(1), pyramid::level_size(h, level).max(1));
                for ty in 0..lh.div_ceil(TILE) {
                    for tx in 0..lw.div_ceil(TILE) {
                        let at = region_of(
                            (tx, ty),
                            (pyramid::tile_extent(tx, lw), pyramid::tile_extent(ty, lh)),
                            (lw, lh),
                            (w, h),
                            lw.abs_diff(w) <= 1 && lh.abs_diff(h) <= 1,
                        );
                        let (rw, rh) = (at.sx1 - at.sx0, at.sy1 - at.sy0);
                        assert!(rw > 0 && rh > 0, "{w}x{h} уровень {level} тайл {tx}:{ty} пуст");
                        assert!(
                            rw * rh <= REGION_CAP,
                            "{w}x{h} уровень {level}: окно обещало, а тайл {tx}:{ty} даёт {rw}x{rh}"
                        );
                    }
                }
            }
        }
    }

    /// Порог окна стои́т там, где полосы под тайл перестают влезать в кэш
    /// чанков. Дальше соседний тайл того же ряда перечитывает их заново, и ряд
    /// из семи тайлов обходится в семь чтений файла вместо одного — то есть
    /// окно становится дороже прохода, который оно заменяет.
    ///
    /// Числа настоящие: Sentinel-1 GRD 25309×17408 без копий и без внутренних
    /// тайлов. Полосы одного тайла — это его строки во всю ширину растра.
    #[test]
    fn порог_окна_стои́т_на_кэше_чанков() {
        let (w, h) = (25309u32, 17408u32);
        let layout = Layout::of(false, (w, 1), Vec::new(), RGB8);

        let held = |level: u32| -> u64 {
            let lh = pyramid::level_size(h, level).max(1);
            let rh = (u64::from(TILE) * u64::from(h)).div_ceil(u64::from(lh)) + 1;
            rh * u64::from(w) * 4
        };

        for level in 0..pyramid::level_count(w, h) {
            let fits = held(level) <= CHUNK_CACHE_BYTES as u64;
            assert_eq!(
                layout.grid.pointwise((w, h), level),
                fits,
                "уровень {level}: полос на {} МБ при кэше {} МБ",
                held(level) / (1024 * 1024),
                CHUNK_CACHE_BYTES / (1024 * 1024)
            );
        }

        // Вблизи окно есть, издали его нет — иначе правило не о чем.
        assert!(layout.grid.pointwise((w, h), 0), "подробный уровень не взялся окном");
        assert!(!layout.grid.pointwise((w, h), 4), "грубый уровень объявлен окном");
    }

    /// Тайловому файлу с ПОЛНОЙ цепочкой копий окно не кончается: у всякого
    /// уровня своя копия, и область тайла в ней всегда ровно тайл.
    #[test]
    fn у_полной_цепочки_копий_окно_не_кончается() {
        let overviews = (1..7)
            .map(|level| Overview {
                image: level as usize,
                width: pyramid::level_size(25309, level),
                height: pyramid::level_size(17408, level),
                chunk: TILES,
            })
            .collect();
        let layout =
            Layout::of(true, TILES, overviews, RGB8);

        for level in 0..7 {
            assert!(layout.grid.pointwise((25309, 17408), level), "уровень {level} не взялся окном");
        }
    }

    /// А оборванная цепочка окно кончает — и это главное, ради чего окно
    /// спрашивают у файла с произвольным доступом вообще.
    ///
    /// `gdaladdo 2 4` кладёт две копии и на этом останавливается. Уровню,
    /// которому досталась самая мелкая из них, область под тайл считается
    /// пропорцией — и растёт вдвое с каждой недостающей ступенью, пока не
    /// перевалит [`REGION_CAP`]. Дальше `produce_direct` отказал бы, отказ
    /// свалил бы весь проход, а проход приговаривает свои ячейки разом: без
    /// этой проверки уровень не появился бы уже никогда.
    #[test]
    fn оборванная_цепочка_копий_кончает_окно() {
        let (w, h) = (65536u32, 65536u32);
        let layout = Layout::of(true, TILES, vec![
                Overview { image: 1, width: w / 2, height: h / 2, chunk: TILES },
                Overview { image: 2, width: w / 4, height: h / 4, chunk: TILES },
            ], RGB8);

        assert!(layout.grid.pointwise((w, h), 0), "нулевому уровню копия не нужна");
        assert!(layout.grid.pointwise((w, h), 2), "уровню 2 досталась своя копия");

        let top = pyramid::level_count(w, h) - 1;
        assert!(
            !layout.grid.pointwise((w, h), top),
            "вершине копий не досталось — окном её брать нельзя"
        );

        // И то же самое глазами потребителя: `Info::windowed` считает уровни
        // снизу и обязана остановиться там же.
        let counted = Info::plain(w, h, Kind::Tiff(layout)).windowed();
        assert!(counted > 0 && counted <= top, "точечных уровней {counted} при вершине {top}");
    }

    /// Выборка растяга — самая мелкая копия, а нет копий — базовый растр.
    /// Правило это одно на оба рукава производства, и держится показ на том,
    /// что ответ у файла один: посчитанный по разным копиям, растяг развёл бы
    /// соседние тайлы одного уровня разной яркостью.
    #[test]
    fn выборка_растяга_берётся_из_самой_мелкой_копии() {
        let (_, layout) = sentinel1_grdh();
        let smallest = layout
            .grid.overviews
            .iter()
            .min_by_key(|overview| overview.width)
            .expect("копии у гранулы есть");
        assert_eq!(layout.stats(), smallest.image, "выборка — из самой мелкой копии");
        assert_eq!(smallest.width, 414, "самая мелкая копия гранулы");

        let bare = Layout::of(false, TILES, Vec::new(), RGB8);
        assert_eq!(bare.stats(), 0, "копий нет — выборка из базового растра");
    }

    /// Сколько чанков задевает область — считается формулой, а проверяется
    /// перебором смещений. Пара эта обязана сойтись механически: занизив,
    /// [`windowed`] обещает окно там, где память кончится посреди прохода;
    /// завысив — гонит в проход уровень, который читался бы точечно.
    ///
    /// Своя сетка и чужая проверяются порознь: у своей начало кратно тайлу, и
    /// перебирать надо только такие начала.
    #[test]
    fn задетые_чанки_сходятся_с_перебором_смещений() {
        let tile = u64::from(TILE);
        // Правда: сколько чанков накрывает область в `side` пикселей,
        // начавшаяся на `start`.
        let touched =
            |start: u64, side: u64, chunk: u64| (start + side - 1) / chunk - start / chunk + 1;

        for chunk in [1u64, 3, 5, 256, 512, 1024, 4096] {
            for side in [1u64, 2, 3, 511, 512, 513, 1024, 1025, 4096] {
                let source = 1u32 << 20;

                // Чужая сетка: начало где угодно.
                let worst = (0..chunk).map(|start| touched(start, side, chunk)).max().unwrap();
                assert_eq!(
                    spanned(side, chunk as u32, source, false),
                    worst * chunk,
                    "чужая сетка: чанк {chunk}, сторона {side}"
                );

                // Своя сетка — только там, где сетки соразмерны; иначе
                // формула честно берёт худшее из всех смещений.
                if chunk % tile != 0 && tile % chunk != 0 {
                    assert_eq!(
                        spanned(side, chunk as u32, source, true),
                        worst * chunk,
                        "несоразмерные сетки: чанк {chunk}, сторона {side}"
                    );
                    continue;
                }
                let flush = (0..chunk)
                    .filter(|start| start % tile == 0)
                    .map(|start| touched(start, side, chunk))
                    .max()
                    .expect("хотя бы начало нуль подходит всегда");
                assert_eq!(
                    spanned(side, chunk as u32, source, true),
                    flush * chunk,
                    "своя сетка: чанк {chunk}, сторона {side}"
                );
            }
        }

        // Чанков у копии конечное число — больше, чем есть, не задеть.
        assert_eq!(spanned(4096, 512, 1000, false), 1024, "область шире копии");
    }

    /// Внутренний тайл вчетверо крупнее экранного — обычный вывод GDAL, и он
    /// обязан оставаться оконным: экранный тайл своей сетки лежит целиком
    /// внутри одного чанка, сколько бы тот ни был.
    #[test]
    fn крупный_внутренний_тайл_окна_не_отнимает() {
        let (w, h) = (40000u32, 40000u32);
        let chunk = (4096u32, 4096u32);
        let overviews = (1..7)
            .map(|level| Overview {
                image: level as usize,
                width: pyramid::level_size(w, level),
                height: pyramid::level_size(h, level),
                chunk,
            })
            .collect();
        let layout = Layout::of(true, chunk, overviews, RGB8);

        // Один чанк 4096² в RGBA — 64 МиБ, и это ровно половина кэша чанков:
        // мерка обязана насчитать один, а не четыре.
        assert!(layout.grid.pointwise((w, h), 0), "нулевой уровень своей сетки");
        assert!(layout.grid.pointwise((w, h), 3), "уровень со своей копией");
    }

    /// Полосный файл, у которого полоса — весь растр (TIFF без `RowsPerStrip`
    /// пишется одной полосой по спецификации), окном не берётся ни на одном
    /// уровне: чанк распаковывается целиком, то есть «окно» в один тайл
    /// разворачивает весь снимок.
    #[test]
    fn полоса_во_весь_растр_окна_не_даёт() {
        let (w, h) = (8000u32, 8000u32);
        let whole =
            Layout::of(false, (w, h), Vec::new(), RGB8);
        let rows =
            Layout::of(false, (w, 512), Vec::new(), RGB8);

        for level in 0..pyramid::level_count(w, h) {
            assert!(!whole.grid.pointwise((w, h), level), "уровень {level} обещал окно на целом растре");
        }
        assert!(rows.grid.pointwise((w, h), 0), "полоса в 512 строк окно даёт");
    }
}
