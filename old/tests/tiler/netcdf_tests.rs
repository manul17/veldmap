// Extracted from master:veldmodules/image-tiler/src/adapters/netcdf.rs

#[cfg(test)]
mod tests {
    use super::super::super::budget;
    use super::*;

    /// Узлы стоят по краям отрезка и равномерно между ними, без повторов:
    /// повтор сложил бы решётку не в прямоугольник, и привязка отвалилась бы
    /// целиком. Отрезок в один отсчёт узлов не даёт — делить не на что.
    #[test]
    fn nodes_span_the_stretch_without_repeats() {
        assert_eq!(nodes(0, 1), vec![0, 1]);
        assert_eq!(nodes(0, 20).len(), 21);
        let wide = nodes(0, 449);
        assert_eq!(wide.len(), 21);
        assert_eq!((wide[0], *wide.last().unwrap()), (0, 449));
        assert!(wide.windows(2).all(|pair| pair[0] < pair[1]), "{:?}", wide);

        // Отступившая решётка стои́т на отрезке, а не на всей стороне, и концы
        // у неё те же самые — иначе крайние узлы уехали бы обратно на дыру.
        let inset = nodes(11, 320);
        assert_eq!(inset.len(), 21);
        assert_eq!((inset[0], *inset.last().unwrap()), (11, 320));
        assert!(inset.windows(2).all(|pair| pair[0] < pair[1]), "{:?}", inset);

        assert!(nodes(5, 5).is_empty());
        assert!(nodes(6, 5).is_empty());
    }

    /// Годность узла: незаполненный отсчёт помечен и `NaN`, и числом, и
    /// проверяются обе координаты порознь.
    #[test]
    fn a_node_is_sound_when_written_and_inside_the_circle() {
        assert!(placed(55.0, 37.0));
        assert!(placed(0.0, 359.0), "долгота бывает записана и как 0…360");
        assert!(!placed(f64::NAN, 37.0), "круг отвергает NaN сам");
        assert!(!placed(55.0, f64::NAN));
        assert!(!placed(f64::INFINITY, 37.0), "и бесконечность тоже");
        assert!(!placed(120.0, 37.0), "широта у шара своя, и она у́же долготной");
        assert!(!placed(-999.0, 37.0), "заполнитель SYNERGY");
        assert!(!placed(9.969_21e36, 37.0), "заполнитель CF");
        assert!(!placed(55.0, -999.0));
        assert!(!placed(55.0, 9.969_21e36));
    }

    /// Чистая сетка стои́т там, где стои́т: отступ на ней не двигает ничего.
    /// Отрезок, на котором решётки не построить, отступ не принимает.
    #[test]
    fn a_clean_grid_stands_where_it_is() {
        let stand = footing((0, 265), (0, 1499), |_, _| true);
        assert_eq!(stand, Some(((0, 265), (0, 1499))));
        assert_eq!(footing((0, 0), (0, 5), |_, _| true), None, "сторона в один отсчёт — не сетка");
    }

    /// Рваный край отступ обходит, а не отменяет им привязку.
    ///
    /// Размах и границы негодного сняты с гранулы
    /// `S3A_SY_2_AOD____20260812T011939` (растр 4022×324: столбцы 0…4 и 323
    /// негодны целиком, строка 0 целиком, столбцы 5…10 и 321…322 рвано), а сама
    /// рванина смоделирована полосой: настоящая гуще с одного конца витка.
    /// Ответ у модели тот же, что правило даёт на снятой с файла маске, —
    /// строки 1…4021 и столбцы 11…320, то есть 95,7 % растра вместо пропавшей
    /// привязки.
    #[test]
    fn a_ragged_edge_is_retreated_from() {
        let sound = |row: u32, column: u32| {
            !(column <= 4
                || column == 323
                || row == 0
                || (((5..=10).contains(&column) || (321..=322).contains(&column)) && row < 3000))
        };
        assert_eq!(footing((0, 4021), (0, 323), sound), Some(((1, 4021), (11, 320))));
    }

    /// Узлов на стороне — по её длине, а не одним числом на обе.
    ///
    /// Числа настоящие: у сетки OLCI 15076 строк и 77 столбцов, и решётка
    /// выходит 236×21 вместо 21×21; у AOD 4022×324 — 63×21. Пол держит короткую
    /// сторону, потолок — длинную, а отсчёты стороны — совсем короткую, где
    /// узел пришёлся бы на каждый.
    #[test]
    fn the_nodes_are_counted_by_the_side() {
        assert_eq!((count(15076), count(77)), (236, 21));
        assert_eq!((count(4022), count(324)), (63, 21));
        assert_eq!(count(266), 21, "короткой стороне довольно пола");
        assert_eq!((count(1344), count(1345)), (21, 22), "пол кончается там, где шаг догнал его");
        assert_eq!((count(16320), count(16321)), (255, 256), "и упирается в потолок");
        assert_eq!(count(1_000_000), 256);
        assert_eq!((count(2), count(3)), (2, 3), "узел не встаёт дважды на один отсчёт");
        assert_eq!(nodes(0, 1), vec![0, 1]);
        assert_eq!(nodes(0, 15075).len(), 236);
    }

    /// Отступает та сторона, которая переложит нужную ось. Сплошной негодный
    /// столбец, попавший ровно на узловой, даёт по одному негодному узлу на
    /// первой и последней строке решётки — и сторона, где негодных больше
    /// сейчас, увела бы отступ в строки, где он не помогает вовсе.
    #[test]
    fn a_bad_line_on_a_node_moves_the_axis_that_helps() {
        let bad = nodes(0, 319)[10];
        assert_eq!(bad, 160);
        let stand = footing((0, 3999), (0, 319), |_, column| column != bad);
        assert_eq!(stand, Some(((0, 3999), (0, 318))), "решётка сдвинула столбцы, а не строки");
    }

    /// Дыра не у края отступом не чинится: сдвигать нечего, и привязки нет.
    #[test]
    fn a_hole_in_the_middle_is_not_retreated_from() {
        let sound =
            |row: u32, column: u32| !((1900..2100).contains(&row) && (130..190).contains(&column));
        assert_eq!(footing((0, 3999), (0, 319), sound), None);
    }

    /// Бюджет — мельчайший шаг решётки, и мельчайший из двух, а не свой у
    /// каждой оси.
    ///
    /// Числа настоящие: у растра AOD 4022×324 решётка выходит 63×21, шаг по
    /// строкам 64 отсчёта, по столбцам 16. Отступ на шестнадцать столбцов
    /// проходит, на семнадцать — нет.
    ///
    /// Мельчайшая ось бывает и строчной: у отрезка в 21 строку и 4022 столбца
    /// шаг по строкам — один отсчёт, и отступать по столбцам он разрешает ровно
    /// столько же, хотя своя их мерка вшестидесятеро свободнее.
    #[test]
    fn the_budget_is_the_finest_step_of_the_two() {
        let bad_to = |edge: u32| move |_row: u32, column: u32| column >= edge;
        assert_eq!(footing((0, 4021), (0, 323), bad_to(16)), Some(((0, 4021), (16, 323))));
        assert_eq!(footing((0, 4021), (0, 323), bad_to(17)), None);

        assert_eq!(footing((0, 20), (0, 4021), bad_to(1)), Some(((0, 20), (1, 4021))));
        assert_eq!(footing((0, 20), (0, 4021), bad_to(2)), None);
    }

    /// Шаг оси меряется промежутками между узлами, а не отсчётами стороны, и
    /// сторона в два отсчёта — это шаг в один, а не в ноль.
    #[test]
    fn the_step_of_an_axis_is_measured_between_its_nodes() {
        let bad_to = |edge: u32| move |_row: u32, column: u32| column >= edge;
        // Промежутков двадцать, отсчётов сорок: у отрезка (0, 39) шаг равен
        // одному, а не двум, и на два столбца отступать уже нечем.
        assert_eq!(footing((0, 39), (0, 39), bad_to(1)), Some(((0, 39), (1, 39))));
        assert_eq!(footing((0, 39), (0, 39), bad_to(2)), None);
        // Сторона в два отсчёта — решётка в два узла, и шаг у неё есть.
        assert_eq!(footing((0, 1), (0, 40), bad_to(1)), Some(((0, 1), (1, 40))));
    }

    /// Бюджет — предел каждой стороне порознь, и предел этот точный.
    ///
    /// У отрезка в 40 промежутков мельчайший шаг решётки — два отсчёта, и
    /// сплошная негодная полоса ровно в два столбца проходится, а в три —
    /// уже нет.
    #[test]
    fn the_budget_is_spent_side_by_side() {
        assert_eq!(footing((0, 40), (0, 40), |_, column| column > 1), Some(((0, 40), (2, 40))));
        assert_eq!(footing((0, 40), (0, 40), |_, column| column > 2), None);
    }

    /// Грязнейшая сторона узнаётся своей: марш ведёт та сторона, на которой
    /// негодное и лежит, а не её противоположная.
    #[test]
    fn the_march_goes_by_the_side_that_is_dirty() {
        assert_eq!(footing((0, 40), (0, 40), |row, _| row < 39), Some(((0, 38), (0, 40))));
        assert_eq!(footing((0, 40), (0, 40), |_, column| column < 39), Some(((0, 40), (0, 38))));
    }

    /// Ничью решает грязь: когда обе стороны уводят узлы с дефекта, идёт та, на
    /// которой он и лежит. Негодная точка стои́т здесь на первом столбце и на
    /// внутренней строке — отступить можно и строками, и столбцами, а верно
    /// столбцами.
    #[test]
    fn a_tie_is_broken_by_the_dirty_side() {
        let sound = |row: u32, column: u32| (row, column) != (11, 0);
        assert_eq!(footing((0, 44), (0, 44), sound), Some(((0, 44), (1, 44))));
    }

    /// Марш идёт по грязнейшей стороне, а при равной грязи — туда, где дыр
    /// останется меньше.
    ///
    /// Расстановки найдены перебором: правило разбирается тут не рассуждением,
    /// и закрепить его можно только свидетелем.
    #[test]
    fn the_march_weighs_what_is_left_behind() {
        let holes = [(18u32, 53u32), (33, 35), (46, 35), (53, 58)];
        let sound = |row: u32, column: u32| !holes.contains(&(row, column));
        assert_eq!(footing((0, 59), (0, 59), sound), Some(((0, 58), (1, 59))));

        let holes = [(7u32, 13u32), (24, 35), (29, 20), (41, 13)];
        let sound = |row: u32, column: u32| !holes.contains(&(row, column));
        assert_eq!(footing((0, 44), (0, 44), sound), Some(((2, 44), (0, 43))));
    }

    /// Убавляющий шаг берётся и с чистой стороны — иначе решётка топчется по
    /// грязи, которая ничего не решает, и сдаётся при живом ответе.
    ///
    /// Негодны здесь угол первой строки и полоса, до которой решётка дотянется
    /// только после сдвига строк: по грязи отступать пришлось бы столбцам, и
    /// бюджет утёк бы в них весь.
    #[test]
    fn a_step_that_helps_is_taken_from_any_side() {
        let holes = [(0, 4), (0, 5), (1, 36), (1, 37), (1, 38), (1, 39), (31, 38)];
        let sound = |row: u32, column: u32| !holes.contains(&(row, column));
        assert_eq!(footing((0, 40), (0, 40), sound), Some(((2, 39), (0, 40))));
    }

    /// Бюджет меряется мельчайшим шагом решётки, а не своим у каждой оси.
    ///
    /// Клин тонок в узлах и толст в отсчётах: по мерке шага собственной оси
    /// (201 отсчёт у растра в 4000 строк) отступ прошагал бы внутрь 191 строку
    /// и бросил бы их годные данные линейной экстраполяции — сотни километров
    /// трека молча.
    #[test]
    fn a_retreat_deeper_than_the_finest_step_is_refused() {
        let sound = |row: u32, column: u32| !(column == 0 || (row <= 190 && column <= 300));
        assert_eq!(footing((0, 3999), (0, 319), sound), None);
    }

    /// Решётке, которой некуда отступать, одна негодная точка отменяет всё:
    /// бюджет у неё нулевой, а решётка с дырой уложила бы снимок куда попало.
    #[test]
    fn a_single_broken_node_drops_the_whole_lattice() {
        let rows = (0u32, 1u32);
        let columns = (0u32, 1u32);
        let whole = lattice(rows, columns, Seating::SAME, |row, column| {
            Some((f64::from(row), f64::from(column)))
        });
        assert_eq!(whole.len(), 4);
        assert_eq!(whole[0].px, 0.5, "узел стои́т в середине отсчёта");

        let holed = lattice(rows, columns, Seating::SAME, |row, column| match (row, column) {
            (1, 1) => Some((f64::NAN, 0.0)),
            _ => Some((f64::from(row), f64::from(column))),
        });
        assert!(holed.is_empty());

        // Широта за пределами шара — тоже дыра, только записанная числом:
        // так лежит незаполненный край полосы съёмки. И долгота тоже: у
        // решётки, проверенной лишь по широте, такая дыра проходила бы.
        let filled = lattice(rows, columns, Seating::SAME, |_, _| Some((9.969_21e36, 0.0)));
        assert!(filled.is_empty());
        let eastless = lattice(rows, columns, Seating::SAME, |_, _| Some((0.0, 9.969_21e36)));
        assert!(eastless.is_empty());

        // Опорная сетка разрежена: отсчёт координат стои́т не на каждом пикселе,
        // и узел уезжает туда, куда указывает шаг.
        let step = Seating { step: (64.0, 1.0), origin: (0.0, 0.0) };
        let sparse = lattice(rows, columns, step, |row, column| {
            Some((f64::from(row), f64::from(column)))
        });
        assert_eq!(sparse[1].px, 64.5, "второй столбец опорной сетки — 64-й пиксель");
        assert_eq!(sparse[1].py, 0.5);

        // У посадки есть и начало: решётка `tx` SLSTR свисает за левый край
        // растра, и её нулевой узел приходится на 26 пикселей левее нуля.
        let shifted = lattice(
            rows,
            columns,
            Seating { step: (16.0, 1.0), origin: (-26.0, 0.0) },
            |row, column| Some((f64::from(row), f64::from(column))),
        );
        assert_eq!(shifted[0].px, -25.5, "нулевой узел стои́т левее растра");
        assert_eq!(shifted[1].px, -9.5);
        assert_eq!(shifted[1].py, 0.5, "по строкам начала нет — они совпадают отсчёт в отсчёт");
    }

    /// Отступившая решётка и строится на отступе: узлы стоят на тех отсчётах,
    /// которые нашёл [`footing`], а не на тех, о которых спросили.
    #[test]
    fn a_retreated_lattice_stands_on_what_it_found() {
        let edge = lattice((0, 40), (0, 40), Seating::SAME, |_, column| {
            Some((if column == 0 { -999.0 } else { 1.0 }, 37.0))
        });
        assert_eq!(edge.len(), 441);
        assert_eq!(edge[0].px, 1.5, "нулевой узел ушёл с негодного столбца");
        assert_eq!(edge[0].py, 0.5, "по строкам отступать было не от чего");

        // Отступ не помог — привязки нет, и решётка не строится на негодном.
        let blocked = lattice((0, 40), (0, 40), Seating::SAME, |row, column| {
            let inside = (15..=25).contains(&row) && (15..=25).contains(&column);
            Some((if inside { -999.0 } else { 1.0 }, 37.0))
        });
        assert!(blocked.is_empty());
    }

    /// Отсчёт прибора у гранулы SLSTR: растр `in` и решётка `tx` стоят в
    /// разных отсчётах, и сходятся они только через смещения.
    ///
    /// Числа настоящие — `S3A_SL_2_LST____20260824T174507`: растр 266×1500 при
    /// `track_offset` 998 и разрешении 1000 м, решётка 266×130 при
    /// `track_offset` 64 и разрешении 16000 м поперёк трека.
    #[test]
    fn the_slstr_tie_grid_sits_by_the_instrument_frame() {
        let raster = Frame { across_at: 998.0, along_at: 3598.0, across: 1000.0, along: 1000.0 };
        let grid = Frame { across_at: 64.0, along_at: 3598.0, across: 16000.0, along: 1000.0 };
        let seat = seating(Some(raster), Some(grid), None, (1500, 266), (130, 266))
            .expect("отсчёты названы обоими файлами");
        assert_eq!(seat.step, (16.0, 1.0), "шаг — отношение разрешений");
        assert_eq!(seat.origin, (-26.0, 0.0), "нулевой узел свисает за левый край растра");
        // Крайний узел приходится далеко за правый край — решётка шире снимка.
        assert_eq!(129.0 * seat.step.0 + seat.origin.0, 2038.0);
    }

    /// Опорная сетка OLCI названа подвыборкой, и та обязана сойтись с
    /// размерами: узлы стоят на краях растра.
    ///
    /// Числа настоящие — `S3A_OL_2_LRR____20260824T161836`: растр 15076×1217,
    /// сетка 15076×77, `ac_subsampling_factor` 16, `al_subsampling_factor` 1.
    #[test]
    fn the_olci_tie_grid_sits_by_its_subsampling() {
        let seat = seating(None, None, Some((16.0, 1.0)), (1217, 15076), (77, 15076))
            .expect("подвыборка сходится с размерами");
        assert_eq!(seat.step, (16.0, 1.0));
        assert_eq!(seat.origin, (0.0, 0.0), "узел ноль стои́т на пикселе ноль");
        assert_eq!(76.0 * seat.step.0 + seat.origin.0, 1216.0, "крайний узел — крайний столбец");
    }

    /// Сетка размером с растр садится отсчёт в отсчёт, и подвыборку у неё не
    /// спрашивают. Правило это не про дешевизну: `ac_subsampling_factor`
    /// описывает опорную сетку продукта, и поотсчётный `geo_coordinates.nc`
    /// OLCI несёт его тоже. Спрошенный раньше размеров, он растянул бы
    /// поотсчётную сетку в шестнадцать раз.
    #[test]
    fn a_grid_the_size_of_the_raster_ignores_the_subsampling_attribute() {
        let seat = seating(None, None, Some((16.0, 1.0)), (1217, 15076), (1217, 15076))
            .expect("сетка размером с растр садится всегда");
        assert_eq!(seat, Seating::SAME);
    }

    /// Подвыборка, не сошедшаяся с размерами, — это подвыборка не про эту
    /// пару. Посадка по ней вышла бы такой же молчаливой, как выведенная из
    /// размеров, поэтому здесь отказ.
    #[test]
    fn a_subsampling_that_misses_the_sizes_is_refused() {
        let off = seating(None, None, Some((16.0, 1.0)), (1217, 15076), (130, 15076));
        assert!(off.is_err(), "77 узлов ожидалось, а сетка о 130");
        // Допуск — пиксель, а не «примерно»: 77 узлов шагом 16 покрывают 1217
        // столбцов, и растру шириной 1220 та же подвыборка уже не отвечает.
        let near = seating(None, None, Some((16.0, 1.0)), (1220, 15076), (77, 15076));
        assert!(near.is_err(), "разошлись на три столбца — это уже другая пара");
    }

    /// Разрежённая сетка без единого слова о том, чем она связана с растром,
    /// не садится вовсе: снимок ляжет по контуру каталога, а не мимо себя.
    #[test]
    fn a_sparse_grid_that_says_nothing_about_itself_is_refused() {
        let mute = seating(None, None, None, (1500, 266), (130, 266));
        assert!(mute.is_err());
    }

    /// Полукилометровая сетка `an` и косой обзор `fo` — те случаи, где обе оси
    /// и оба начала разные, и симметричная описка в них была бы видна.
    ///
    /// Числа настоящие — `S3A_SL_1_RBT____20260824T174507`: у `an` растр
    /// 533×3000 при `track_offset` 1996, `start_offset` 7195 и разрешении 500 м
    /// по обеим осям; у `fo` растр 266×900 при `track_offset` 450. Решётка `tx`
    /// им обоим одна и та же.
    #[test]
    fn every_slstr_grid_seats_on_the_same_tie_lattice() {
        let tie = Frame { across_at: 64.0, along_at: 3598.0, across: 16000.0, along: 1000.0 };

        let half_kilometre = Frame { across_at: 1996.0, along_at: 7195.0, across: 500.0, along: 500.0 };
        let seat = seating(Some(half_kilometre), Some(tie), None, (3000, 533), (130, 266))
            .expect("отсчёты названы обоими файлами");
        assert_eq!(seat.step, (32.0, 2.0), "полукилометровому пикселю узел вдвое дороже");
        // Начало уезжает по обеим осям, и в разные стороны: смещения считаются
        // от разного (см. `Frame`). Проверено сличением поотсчётных сеток `an`
        // и `in` одной гранулы, без участия опорной: строка `in` отвечает
        // строке `an` с номером `2·i + 1`, а не `2·i − 1`.
        assert_eq!(seat.origin, (-52.0, 1.0), "поперёк влево, вдоль вправо");

        let oblique = Frame { across_at: 450.0, along_at: 3598.0, across: 1000.0, along: 1000.0 };
        let seat = seating(Some(oblique), Some(tie), None, (900, 266), (130, 266))
            .expect("отсчёты названы обоими файлами");
        assert_eq!(seat.step, (16.0, 1.0));
        assert_eq!(seat.origin, (-574.0, 0.0), "косой обзор смещён поперёк трека сильнее надирного");
    }

    /// Прочитанное сверяется с видимым: посадка обязана накрыть растр.
    ///
    /// Атрибут бывает не про эту пару файлов. `resolution` у OLCI одинаков и у
    /// поотсчётного, и у опорного файла — он свойство продукта, — и отношение
    /// разрешений дало бы шестнадцатикратно разрежённой сетке шаг единица.
    #[test]
    fn a_seating_that_leaves_the_raster_uncovered_is_refused() {
        let same = Frame { across_at: 0.0, along_at: 0.0, across: 1080.0, along: 1176.0 };
        let short = seating(Some(same), Some(same), None, (1217, 15076), (77, 15076));
        assert!(short.is_err(), "77 узлов шагом единица накрывают шестнадцатую долю снимка");

        // Настоящая посадка растр накрывает — с запасом слева и справа.
        let raster = Frame { across_at: 998.0, along_at: 3598.0, across: 1000.0, along: 1000.0 };
        let tie = Frame { across_at: 64.0, along_at: 3598.0, across: 16000.0, along: 1000.0 };
        assert!(seating(Some(raster), Some(tie), None, (1500, 266), (130, 266)).is_ok());
    }

    /// Отсчёт прибора отвечает раньше размеров и раньше подвыборки: он
    /// прочитан, а те выведены.
    #[test]
    fn the_instrument_frame_outranks_both_guesses() {
        let raster = Frame { across_at: 998.0, along_at: 3598.0, across: 1000.0, along: 1000.0 };
        let tie = Frame { across_at: 64.0, along_at: 3598.0, across: 16000.0, along: 1000.0 };

        // Подвыборка названа тоже, и она о другом — отсчёт главнее.
        let seat = seating(Some(raster), Some(tie), Some((4.0, 4.0)), (1500, 266), (130, 266))
            .expect("отсчёты названы обоими файлами");
        assert_eq!(seat.step, (16.0, 1.0), "шаг взят у отсчёта, а не у подвыборки");
        assert_eq!(seat.origin, (-26.0, 0.0));

        // Формы совпали, а отсчёты разошлись — это не сетка этого растра, и
        // равенство форм её таковой не делает: отказ, а не посадка один в один.
        let elsewhere = Frame { across_at: 450.0, along_at: 3598.0, across: 1000.0, along: 1000.0 };
        let apart = seating(Some(raster), Some(elsewhere), None, (1500, 266), (1500, 266));
        assert!(apart.is_err(), "сетка, съехавшая на 548 столбцов, растр не накрывает");
    }

    /// Отсчёт прибора и подвыборка читаются из глобальных атрибутов, и каждая
    /// ось берётся из своего: перепутанные, они разъехались бы молча.
    #[test]
    fn the_global_attributes_are_read_axis_by_axis() {
        let attrs: HashMap<String, AttrValue> = [
            ("resolution".to_string(), AttrValue::String("[ 16000 1000 ]".to_string())),
            ("track_offset".to_string(), AttrValue::I32(64)),
            ("start_offset".to_string(), AttrValue::I32(3598)),
            ("ac_subsampling_factor".to_string(), AttrValue::I32(16)),
            ("al_subsampling_factor".to_string(), AttrValue::I32(1)),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            frame(&attrs),
            Some(Frame { across_at: 64.0, along_at: 3598.0, across: 16000.0, along: 1000.0 })
        );
        assert_eq!(subsampling(&attrs), Some((16.0, 1.0)));

        // Без единого из трёх отсчёта не выходит.
        for missing in ["resolution", "track_offset", "start_offset"] {
            let mut lean = attrs.clone();
            lean.remove(missing);
            assert_eq!(frame(&lean), None, "без '{}' отсчёт не собрать", missing);
        }
        let mut lean = attrs.clone();
        lean.remove("al_subsampling_factor");
        assert_eq!(subsampling(&lean), None, "подвыборка нужна по обеим осям");
    }

    /// Отсчёт прибора без пары бесполезен: одно смещение не говорит ни о шаге,
    /// ни о начале.
    #[test]
    fn one_instrument_frame_without_the_other_is_not_enough() {
        let raster = Frame { across_at: 998.0, along_at: 3598.0, across: 1000.0, along: 1000.0 };
        assert!(seating(Some(raster), None, None, (1500, 266), (130, 266)).is_err());
        assert!(seating(None, Some(raster), None, (1500, 266), (130, 266)).is_err());
    }

    /// `resolution` записан строкой в скобках, и читаются оба числа.
    #[test]
    fn the_resolution_attribute_is_read_out_of_its_brackets() {
        assert_eq!(resolution("[ 16000 1000 ]"), Some((16000.0, 1000.0)));
        assert_eq!(resolution("[ 1000 1000 ]"), Some((1000.0, 1000.0)));
        assert_eq!(resolution("[ 1080 1176 ]"), Some((1080.0, 1176.0)));
        assert_eq!(resolution("1000"), None, "одного числа мало");
        assert_eq!(resolution(""), None);
        assert_eq!(resolution("[ 0 1000 ]"), None, "нулевая ячейка — не разрешение");
        // Число берётся целиком: разрезанное по цифрам, `1.5` дало бы единицу и
        // пятёрку, и обе прошли бы дальше как правдоподобные разрешения.
        assert_eq!(resolution("[ 1.5 2.5 ]"), Some((1.5, 2.5)));
        assert_eq!(resolution("[ 1e3 2e3 ]"), Some((1000.0, 2000.0)));
        assert_eq!(resolution("[ -1000 -1000 ]"), None, "знак не теряется");
        assert_eq!(resolution("[ 1.2.3 1000 ]"), None, "не число — не разрешение");
    }

    /// Окно строк режется по оси строк плоскости: чанк во всю высоту — окно
    /// ростом с плоскость; иначе окно — связка чанков файла не длиннее тайла,
    /// а чанк выше тайла — сам. Сколько единичных осей стоит перед строками,
    /// окно не спрашивает.
    ///
    /// Числа настоящие: у величины SYNERGY AOD 324 строки одним чанком, у
    /// OLCI 15076 строк чанками по 5026.
    #[test]
    fn окно_строк_режется_по_оси_строк_плоскости() {
        assert_eq!(rows_of(324, Some(324)), 324, "один чанк — плоскость");
        assert_eq!(rows_of(15076, Some(100)), 500, "связка чанков не длиннее тайла");
        assert_eq!(rows_of(15076, Some(TILE as u64)), TILE);
        assert_eq!(rows_of(15076, Some(300)), 300, "второй чанк уже не влезает в тайл");
        assert_eq!(rows_of(15076, Some(600)), 600, "чанк выше тайла берётся сам");
        assert_eq!(rows_of(15076, Some(5026)), 5026);
        assert_eq!(rows_of(15076, None), TILE, "непрерывной раскладке окно — тайл");
        assert_eq!(rows_of(300, None), 300, "и не длиннее самой величины");
        assert_eq!(rows_of(1217, Some(1)), TILE, "чанк в строку — связка в тайл");
    }

    /// Регион окна: по оси строк — окно, обрезанное высотой величины, по оси
    /// столбцов — вся ширина, по единичным осям — нулевой отсчёт, где бы они
    /// ни стояли.
    #[test]
    fn регион_окна_ложится_по_осям_плоскости() {
        let mut layout = Layout::of(450, 4172, 256, DType::F32);
        assert_eq!(layout.region(512, 256), (vec![512, 0], vec![256, 450]));
        assert_eq!(layout.region(4096, 256), (vec![4096, 0], vec![76, 450]), "нижнее окно короче");
        layout.shape = vec![1, 4172, 450];
        layout.axes = (1, 2);
        assert_eq!(layout.region(512, 256), (vec![0, 512, 0], vec![1, 256, 450]));
        layout.shape = vec![4172, 1, 450, 1];
        layout.axes = (0, 2);
        assert_eq!(layout.region(0, 256), (vec![0, 0, 0, 0], vec![256, 1, 450, 1]));
    }

    /// Глубина свежего чанка — большая из двух фаз чтения крейта: сборка
    /// окна (окно, сжатое, распакованный чанк — три отсчёта на отсчёт) и
    /// развёртка (сырое и типизированное либо f32). Занижена — трап вместо
    /// отказа.
    #[test]
    fn глубина_чанка_считает_копии_крейта() {
        assert_eq!(depth_of(&DType::U8), 3, "байтам развёртка не нужна, решает сборка");
        assert_eq!(depth_of(&DType::U16), 6);
        assert_eq!(depth_of(&DType::I16), 6);
        assert_eq!(depth_of(&DType::F32), 12);
        assert_eq!(depth_of(&DType::F64), 24, "восьмибайтовый отсчёт: сборка дороже развёртки в f32");
        assert_eq!(depth_of(&DType::I32), 12);
        assert_eq!(depth_of(&DType::I8), 5, "у байта со знаком развёртка в f32 дороже сборки");
        assert_eq!(depth_of(&DType::String), 24, "неузнанный тип считается восьмибайтовым");
        let layout = Layout::of(450, 4172, 4172, DType::F32);
        assert_eq!(layout.grid.depth, 12);
        assert_eq!(layout.grid.chunk, (450, 4172));
    }

    /// Шаг выборки разводится со строкой: общий делитель у шага и ширины
    /// оставил бы выборку в одних и тех же столбцах каждой строки, а у полосы
    /// съёмки крайние столбцы не измерены вовсе.
    #[test]
    fn the_sampling_step_never_walks_one_column() {
        // Гранула OLCI: 4865×1687 отсчётов на выборку в миллион даёт шаг 7, а
        // 4865 = 5·7·139 — то есть исходный шаг делит ширину, и выборка ходила
        // бы по одной седьмой столбцов. Разведённый шаг — 8.
        let (width, height) = (4865usize, 1687usize);
        let step = sampling_step(width * height, width);
        assert_eq!(step, 8, "шаг разводится с шириной");
        assert_eq!(gcd_for_test(step, width), 1);

        // Плоскость мельче выборки: шаг единичный, разводить нечего.
        assert_eq!(sampling_step(450 * 372, 450), 1);

        // И общее правило — на ширинах, где шаг заведомо больше единицы.
        for width in 2..64usize {
            let step = sampling_step(width * STRETCH_SAMPLES * 3, width);
            assert!(step > 1, "шаг должен быть больше единицы: {}", step);
            assert_eq!(
                gcd_for_test(step, width), 1,
                "шаг {} и ширина {} не взаимно просты", step, width
            );
        }
    }

    fn gcd_for_test(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }

    /// Растяг считается по годным значениям: метка «нет данных» утянула бы
    /// нижний край, и весь снимок вышел бы белым.
    #[test]
    fn stretch_ignores_the_fill_value() {
        let mut values = vec![-9999.0f32; 100];
        values.extend((0..=100i16).map(|value| f32::from(value) + 300.0));
        let mapping = mapping("/проба", &values, Some(-9999.0));
        // Поле «нет данных» — прозрачное, а живое значение попадает в середину
        // растяга, а не в белое.
        let out = mapping.rgba(&Samples::F32(&[-9999.0, 350.0]), Pixel::named(1), 2);
        assert_eq!(&out[..4], &[0, 0, 0, 0]);
        assert!((100..=160).contains(&out[4]), "середина растяга: {}", out[4]);
    }

    /// Пустая величина — не изображение гранулы, однотонная — изображение
    /// последней очереди. Ловится это только по самим отсчётам: заголовок у
    /// пустой и у заполненной один и тот же.
    #[test]
    fn an_unfilled_variable_is_not_a_picture() {
        let fill = Some(65535.0);
        assert_eq!(spread(&[65535.0; 8], fill), Spread::Empty);
        assert_eq!(spread(&[], fill), Spread::Empty);
        assert_eq!(spread(&[f32::NAN, f32::INFINITY], None), Spread::Empty);
        assert_eq!(spread(&[0.0, 65535.0, 0.0], fill), Spread::Flat);
        assert_eq!(spread(&[0.0, 1.0, 65535.0], fill), Spread::Varying);
        // Без метки «нет данных» пустоту делают только нечисла.
        assert_eq!(spread(&[7.0, 7.0], None), Spread::Flat);
    }

    /// Имена координат разрешаются в пути: относительное — от группы величины,
    /// абсолютное — как есть.
    #[test]
    fn coordinate_names_become_paths() {
        assert_eq!(resolve("/PRODUCT", "latitude"), "/PRODUCT/latitude");
        assert_eq!(resolve("/PRODUCT", "/OTHER/latitude"), "/OTHER/latitude");
        assert_eq!(resolve("", "lat"), "/lat");
        let names: Vec<String> = words("Q_FLAGS, ERRORBAR_LST, TIME_DELTA".to_string()).collect();
        assert_eq!(names, ["Q_FLAGS", "ERRORBAR_LST", "TIME_DELTA"]);
    }

    /// «Нет данных» записано типом самой величины: у температуры поверхности
    /// это целое −9999, у индекса аэрозоля — дробное 9.96921e36. Прочитать
    /// одной лестницей нельзя, а потерянное «нет данных» утягивает растяг и
    /// красит весь снимок в один цвет.
    #[test]
    fn the_fill_value_is_read_whatever_its_type() {
        assert_eq!(number(&AttrValue::F64Array(vec![9.969_209_968_386_869e36])), Some(9.969_21e36));
        assert_eq!(number(&AttrValue::I64Array(vec![-9999])), Some(-9999.0));
        assert_eq!(number(&AttrValue::I32(-32767)), Some(-32767.0));
        assert_eq!(number(&AttrValue::F64(0.5)), Some(0.5));
        assert_eq!(number(&AttrValue::AsciiString("K".to_string())), None);
    }

    /// Величина, которой негде лечь, не изображение этой гранулы, чем бы она ни
    /// была измерена, — и спрашивается место раньше всех прочих доводов.
    ///
    /// Живой случай, ради которого правило и заведено: у гранулы Sentinel-5P
    /// уровня 1B в одной группе с широтой и долготой лежат углы наблюдения, а в
    /// соседней — таблица длин волн прибора. Таблица дробная, не угловая и той
    /// же глубины, поэтому по прежним доводам она побеждала все четыре угла — и
    /// показ уходил в то, что вообще не про Землю, а слой кончался словами
    /// «привязки нет», хотя привязка была не нужна ей одной.
    ///
    /// Отбором, а не отсевом: файл, о месте не сказавший ничего, показывается
    /// по-прежнему — он ляжет по контуру каталога.
    #[test]
    fn место_спрашивается_раньше_всех_прочих_доводов() {
        let plane = |path: &str, group: &str, shape: (u32, u32)| Item {
            path: path.to_string(),
            group: group.to_string(),
            depth: 3,
            plane: Some(shape),
            line: None,
            real: true,
            angular: false,
            candidate: true,
            fill: None,
            packing: (1.0, 0.0),
            said: String::new(),
            units: String::new(),
            coordinates: Vec::new(),
            ancillary: Vec::new(),
            shape: vec![10, 10],
            axes: (0, 1),
            chunk_rows: None,
        };
        let coordinate = |path: &str, group: &str, units: &str| Item {
            units: units.to_string(),
            candidate: false,
            ..plane(path, group, (497, 77))
        };
        const GEO: &str = "/BAND1/STANDARD_MODE/GEODATA";
        const INSTR: &str = "/BAND1/STANDARD_MODE/INSTRUMENT";
        let items = vec![
            // Таблица прибора: дробная, не угловая, но её форма — канал на
            // пиксель, и координат такой формы в файле нет.
            plane(&format!("{}/nominal_wavelength", INSTR), INSTR, (497, 77)),
            // Угол наблюдения: угловой, то есть по прежним доводам младше, —
            // зато широта с долготой лежат рядом и той же формы.
            Item { angular: true, ..plane(&format!("{}/solar_zenith_angle", GEO), GEO, (497, 77)) },
            coordinate(&format!("{}/latitude", GEO), GEO, "degrees_north"),
            coordinate(&format!("{}/longitude", GEO), GEO, "degrees_east"),
        ];
        let order = preferred(&items);
        assert_eq!(
            order.first().map(|item| item.path.as_str()),
            Some("/BAND1/STANDARD_MODE/GEODATA/solar_zenith_angle"),
            "победила величина, которой негде лечь",
        );

        // Место старше и глубины — довода, который до него был первым. Проверить
        // это на живом файле нечем: у Sentinel-5P обе группы лежат на одной
        // глубине, и там два правила отвечают одинаково. А разойтись они могут,
        // и ответ должен быть тот же: величина, которой негде лечь, не
        // становится изображением оттого, что лежит ближе к корню.
        let deeper = vec![
            Item { depth: 1, ..plane("/shallow/nominal_wavelength", "/shallow", (497, 77)) },
            Item { depth: 5, ..plane(&format!("{}/solar_zenith_angle", GEO), GEO, (497, 77)) },
            coordinate(&format!("{}/latitude", GEO), GEO, "degrees_north"),
            coordinate(&format!("{}/longitude", GEO), GEO, "degrees_east"),
        ];
        assert_eq!(
            preferred(&deeper).first().map(|item| item.path.as_str()),
            Some("/BAND1/STANDARD_MODE/GEODATA/solar_zenith_angle"),
            "мелкая величина без места победила глубокую с местом",
        );

        // Те же две величины, но файл о месте не сказал ничего: прежний порядок
        // цел — место не отменяет доводов, оно только идёт раньше них.
        let mute = vec![
            plane(&format!("{}/nominal_wavelength", INSTR), INSTR, (497, 77)),
            Item { angular: true, ..plane(&format!("{}/solar_zenith_angle", GEO), GEO, (497, 77)) },
        ];
        assert_eq!(
            preferred(&mute).first().map(|item| item.path.as_str()),
            Some("/BAND1/STANDARD_MODE/INSTRUMENT/nominal_wavelength"),
            "без координат выбор обязан остаться прежним",
        );
    }

    /// Из годных величин показывается та, что ближе к корню и дробная:
    /// подробности расчёта лежат в подгруппах, а целое в CF — код или счётчик.
    #[test]
    fn the_shallowest_measurement_wins() {
        let item = |path: &str, depth: usize, real: bool| Item {
            path: path.to_string(),
            group: String::new(),
            depth,
            plane: Some((10, 10)),
            line: None,
            real,
            angular: false,
            candidate: true,
            fill: None,
            packing: (1.0, 0.0),
            said: String::new(),
            units: String::new(),
            coordinates: Vec::new(),
            ancillary: Vec::new(),
            shape: vec![10, 10],
            axes: (0, 1),
            chunk_rows: None,
        };
        let items = vec![
            item("/PRODUCT/SUPPORT_DATA/cloud_fraction", 2, true),
            item("/PRODUCT/qa_value", 1, false),
            item("/PRODUCT/aerosol_index_354_388", 1, true),
            item("/PRODUCT/aerosol_index_335_367", 1, true),
        ];
        // Глубина важнее дробности, дробность — важнее алфавита, алфавит —
        // последний, и он же делает выбор одним и тем же от запуска к запуску.
        let order = preferred(&items);
        // Угловая величина идёт после всех прочих той же глубины и дробности:
        // замкнутый угол в яркость не растягивается (см. `angular`).
        let angles = vec![
            Item { angular: true, ..item("/PRODUCT/aerosol_index_000", 1, true) },
            item("/PRODUCT/zzz_last_by_alphabet", 1, true),
        ];
        assert_eq!(
            preferred(&angles).iter().map(|item| item.path.as_str()).collect::<Vec<&str>>(),
            ["/PRODUCT/zzz_last_by_alphabet", "/PRODUCT/aerosol_index_000"],
        );
        assert!(angular("degrees") && angular(" deg ") && angular("radians"));
        assert!(!angular("m s-1") && !angular("degrees_north") && !angular("k"));

        // Время — «когда», а не «что»: по первому слову единиц, иначе «m s-1»
        // прочиталось бы секундами.
        let with_units = |units: &str| Item {
            units: units.to_string(),
            ..item("/x", 1, true)
        };
        assert!(timing(&with_units("microseconds")));
        assert!(timing(&with_units("seconds since 2000-01-01 00:00:00")));
        assert!(!timing(&with_units("m s-1")));
        assert!(!timing(&with_units("kelvin")));
        assert!(!timing(&with_units("")));
        // Единицы хранятся как записаны — сравнение с CF регистра не видит.
        assert!(timing(&with_units("Seconds since 2000-01-01 00:00:00")));
        assert!(northing(&with_units("Degrees_North")) && easting(&with_units("degrees_East")));
        assert!(!northing(&with_units("K")) && !easting(&with_units("K")));
        assert_eq!(
            order.iter().map(|item| item.path.as_str()).collect::<Vec<&str>>(),
            [
                "/PRODUCT/aerosol_index_335_367",
                "/PRODUCT/aerosol_index_354_388",
                "/PRODUCT/qa_value",
                "/PRODUCT/SUPPORT_DATA/cloud_fraction",
            ],
        );

        // Не осталось годных — сказано, чего именно не нашлось.
        let none: Vec<Item> = items
            .into_iter()
            .map(|item| Item { candidate: false, ..item })
            .collect();
        assert!(preferred(&none).is_empty());
        assert!(explain(&none).contains("координаты"), "{}", explain(&none));
    }

    /// Кэш заголовков помещается в запас, отложенный на всё несчитаемое: он и
    /// есть самая крупная его часть, названная числом.
    #[test]
    fn кэш_заголовков_помещается_в_запас() {
        assert!(
            u64::try_from(METADATA_CACHE).is_ok_and(|cache| cache < budget::RESERVE),
            "кэш метаданных не помещается в запас"
        );
    }
}
