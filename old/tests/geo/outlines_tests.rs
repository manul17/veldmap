// Extracted from master:veldmodules/globe/src/outlines.rs

#[cfg(test)]
mod tests {

    /// Ребро, размах которого не с Земли, ведётся числом шагов, а не миллиардом.
    ///
    /// Числа приходят по шине, и приведение к `u32` насыщается вместо
    /// переполнения: незажатое, оно дало бы четыре миллиарда вершин. А ребро,
    /// размаха у которого нет вовсе, ведётся одним шагом — не полным кругом.
    #[test]
    fn an_edge_wider_than_the_earth_is_not_drawn_forever() {
        let mut endless = Vec::new();
        edge(&mut endless, (0.0, 0.0), (1.0e9, 0.0));
        let most = (geodesy::TURN_DEG / max_edge_deg()).ceil() as usize;
        assert_eq!(endless.len(), most, "вершин {}", endless.len());

        let mut nowhere = Vec::new();
        edge(&mut nowhere, (0.0, 0.0), (f64::NAN, 0.0));
        assert_eq!(nowhere.len(), 1, "вести нечего — один шаг");
    }

    use super::*;
    use crate::proto::globe::GeoPoint;

    fn outline(points: &[(f64, f64)]) -> Outline {
        Outline {
            points: points.iter().map(|&(lat, lon)| GeoPoint { lat, lon }).collect(),
            style: OutlineStyle::OutlinePlain as i32,
        }
    }

    /// Лента везёт соседей смещениями, и смещения эти — настоящие: `next`
    /// смотрит к следующей вершине по обходу, `prev` к предыдущей, и одно
    /// обратно другому. Перепутай их местами — и на экране не изменится ничего
    /// (обе стороны ленты симметричны, излом отражается целиком), а направление
    /// звена станет обратным. Тестов на саму ленту не было вовсе.
    #[test]
    fn a_ribbon_carries_true_offsets_to_its_neighbours() {
        let built =
            Outlines::build(&[styled(&[(0.0, 0.0), (0.0, 1.0), (1.0, 0.5)], OutlineStyle::OutlineSelected)]);
        // На каждый узел кольца — пара вершин, левый край и правый.
        let count = built.ribbon.len() / 2;
        assert!(count >= 3, "кольцо из {} узлов", count);

        let apart = |a: geodesy::World, b: geodesy::World| {
            f64::from((glam::Vec3::from(a) - glam::Vec3::from(b)).length()) * geodesy::SEMI_MAJOR_M
        };
        let at = |node: usize| {
            let vertex = built.ribbon[node * 2];
            (vertex.position, vertex.position_low)
        };
        for index in 0..count {
            let ahead = (index + 1) % count;
            let step = geodesy::offset(at(ahead), at(index));
            let span = f64::from(glam::Vec3::from(step).length()) * geodesy::SEMI_MAJOR_M;
            assert!(span > 1.0, "узел {}: звено в {} м — проверять нечего", index, span);

            assert!(apart(built.ribbon[index * 2].next, step) < 1e-3, "узел {}: next", index);
            // Обратный ход у соседа — то же звено, взятое назад.
            let back = geodesy::offset(at(index), at(ahead));
            assert!(apart(built.ribbon[ahead * 2].prev, back) < 1e-3, "узел {}: prev", ahead);
            // И обе копии узла везут одно и то же: в стороны их разводит шейдер.
            assert_eq!(built.ribbon[index * 2].next, built.ribbon[index * 2 + 1].next);
        }
    }

    fn styled(points: &[(f64, f64)], style: OutlineStyle) -> Outline {
        Outline { style: style as i32, ..outline(points) }
    }

    /// Три вида разъезжаются по своим буферам: рисуются они разными
    /// пайплайнами, и слипшиеся буферы означали бы ленту вокруг обычного
    /// контура либо штриховку без края.
    ///
    /// Едущий на шар при этом рисуется дважды — линией и штриховкой: линия
    /// говорит, где он ляжет, штриховка — сколько места займёт.
    #[test]
    fn each_style_goes_to_its_own_draw() {
        let square = [(10.0, 10.0), (10.0, 12.0), (12.0, 12.0), (12.0, 10.0)];

        let plain = Outlines::build(&[outline(&square)]);
        assert!(!plain.indices.is_empty(), "очерченный рисуется линиями");
        assert!(plain.ribbon_indices.is_empty() && plain.hatch_indices.is_empty());

        let selected = Outlines::build(&[styled(&square, OutlineStyle::OutlineSelected)]);
        assert!(!selected.ribbon_indices.is_empty(), "выделенный рисуется лентой");
        assert!(selected.indices.is_empty() && selected.hatch_indices.is_empty());

        let pending = Outlines::build(&[styled(&square, OutlineStyle::OutlinePending)]);
        assert_eq!(pending.indices, plain.indices, "едущий очерчен той же линией");
        assert!(!pending.hatch_indices.is_empty(), "область не заштрихована");
        assert!(pending.ribbon_indices.is_empty(), "едущий не выделяют лентой");
    }

    /// Заливка лежит на поверхности, а не режет её хордой: соседние узлы веера
    /// расходятся не дальше, чем вершины самого контура.
    ///
    /// Ради этого веер и разбит по радиусу. Треугольник от середины кольца до
    /// его края — это хорда через десятки градусов дуги, и середина такой
    /// хорды уходит под поверхность на сотню километров, где её съедает тест
    /// глубины: от области осталась бы одна кромка.
    #[test]
    fn the_filled_area_follows_the_surface() {
        // Двадцать градусов поперёк — обычная гранула, и хорда через неё
        // проседает почти на сотню километров.
        let wide = [(0.0, 0.0), (0.0, 20.0), (20.0, 20.0), (20.0, 0.0)];
        let built = Outlines::build(&[styled(&wide, OutlineStyle::OutlinePending)]);

        let between = |left: u32, right: u32| {
            let ends = [left, right].map(|index| {
                let (lat, lon) = angles_of(&built.hatch[index as usize]);
                geodesy::unit(lat, lon)
            });
            ends[0].dot(ends[1]).clamp(-1.0, 1.0).acos().to_degrees()
        };
        let mut widest = 0.0f64;
        for triangle in built.hatch_indices.chunks(3) {
            widest = widest
                .max(between(triangle[0], triangle[1]))
                .max(between(triangle[1], triangle[2]))
                .max(between(triangle[2], triangle[0]));
        }

        // Меряется не сам шаг, а то, ради чего он выбран: середина хорды в
        // столько-то градусов проседает под поверхность на `1 − cos(θ/2)`
        // (радиус Земли здесь — единица), и просесть глубже, чем контур поднят,
        // ей нельзя — там её съест тест глубины.
        let sag = 1.0 - (widest.to_radians() / 2.0).cos();
        let lift = f64::from(geodesy::metres(height_m()));
        assert!(
            sag < lift,
            "ячейка в {} градусов проседает на {} при выносе {}",
            widest,
            sag,
            lift
        );
    }

    /// Контур садится на общий вынос — тот самый, на котором лежит снимок под
    /// ним. Разойдись они, и обвод поехал бы относительно обведённого тем
    /// сильнее, чем ближе к краю кадра.
    #[test]
    fn an_outline_lands_on_the_common_lift() {
        assert!(
            (height_m() - mesh::SURFACE_LIFT_M).abs() < 1e-9,
            "контур поднялся над общим выносом: {} м", height_m()
        );
    }

    /// Кольцу, разошедшемуся на пол-Земли, середины нет вовсе: веер из любой
    /// точки пошёл бы поверх обратной стороны. Такое кольцо остаётся одним
    /// контуром — он и есть главное, а заливка при нём необязательна.
    #[test]
    fn a_ring_around_the_earth_gets_no_fill() {
        let band = outline(&[
            (-70.0125, -180.0125),
            (70.0125, -180.0125),
            (70.0125, 179.9875),
            (-70.0125, 179.9875),
        ]);
        let built = Outlines::build(&[Outline {
            style: OutlineStyle::OutlinePending as i32,
            ..band
        }]);
        assert!(!built.indices.is_empty(), "контур пропал вместе с заливкой");
        assert!(built.hatch_indices.is_empty(), "веер построен поверх обратной стороны");
    }

    /// А шапка вокруг полюса Землю по долготе тоже обходит — и заливается:
    /// середина у неё есть, и это сам полюс.
    #[test]
    fn a_polar_cap_is_filled_all_the_same() {
        let cap: Vec<(f64, f64)> =
            (0..12).map(|step| (85.0, f64::from(step) * 30.0 - 180.0)).collect();
        let built = Outlines::build(&[styled(&cap, OutlineStyle::OutlinePending)]);
        assert!(!built.hatch_indices.is_empty(), "шапка осталась незалитой");
    }

    /// Контур, которого нет: двумя точками замкнутой ломаной не выйдет, и ни
    /// линии, ни заливки с ним не заводится.
    #[test]
    fn a_degenerate_outline_draws_nothing() {
        let built =
            Outlines::build(&[styled(&[(10.0, 10.0), (10.0, 12.0)], OutlineStyle::OutlinePending)]);
        assert!(built.indices.is_empty());
        assert!(built.hatch_indices.is_empty());
    }

    /// Пара углов вершины — по её нормали: она единичная, и выводит из неё
    /// широту с долготой та же функция, что и всюду. Наружу нормаль уезжает
    /// в f32, поэтому здесь она расширяется обратно.
    fn angles_of(vertex: &Vertex) -> (f64, f64) {
        geodesy::angles(glam::Vec3::from(vertex.normal).as_dvec3())
    }

    /// Прямоугольник в градусной сетке обводится по своим краям, а не по дугам
    /// между углами. У глобального продукта края — это две полные параллели:
    /// сведи их дуга к хорде, и от контура остался бы один меридиан, потому что
    /// −180 и +180 — одна и та же точка шара.
    #[test]
    fn a_whole_earth_band_keeps_its_parallels() {
        let band = outline(&[
            (-70.0125, -180.0125),
            (70.0125, -180.0125),
            (70.0125, 179.9875),
            (-70.0125, 179.9875),
        ]);
        let loops = ring(&band);
        assert_eq!(loops.len(), 2, "пояс — это две параллели, и петель у него две");

        let ring = loops.concat();
        let lons: Vec<f64> = ring.iter().map(|v| angles_of(v).1).collect();
        let east = lons.iter().filter(|lon| **lon > 60.0).count();
        let west = lons.iter().filter(|lon| **lon < -60.0).count();
        assert!(east > 30 && west > 30, "контур собрался в меридиан: {} восточных, {} западных", east, west);

        // А вершин на шве не осталось ни одной: боковых краёв у пояса нет, они
        // записаны только потому, что прямоугольник записан четырьмя углами.
        // Проверяется широтами, а не долготами: шов — это меридиан, и на нём
        // лежат все широты между параллелями.
        let stray = ring
            .iter()
            .map(|v| angles_of(v).0)
            .filter(|lat| (-69.0..=69.0).contains(lat))
            .count();
        assert_eq!(stray, 0, "на шве осталось {} вершин", stray);

        // И каждая петля — целая параллель: своя широта и весь круг долгот.
        for run in &loops {
            let lats: Vec<f64> = run.iter().map(|v| angles_of(v).0).collect();
            let spread = lats.iter().fold(f64::MIN, |a, b| a.max(*b))
                - lats.iter().fold(f64::MAX, |a, b| a.min(*b));
            assert!(spread < 0.1, "петля гуляет по широте на {}°", spread);
            let angles: Vec<(f64, f64)> = run.iter().map(angles_of).collect();
            assert!(geodesy::encircles(&angles), "петля не обошла Землю");
        }
    }

    /// Пояс режется одинаково, сколькими бы вершинами его ни записали.
    ///
    /// Тот же пояс, у которого параллели разбиты на четверти: полного круга нет
    /// теперь ни в одном ребре, и признак «ребро в круг» на нём бы отказал. А
    /// пробег между разрезами круг обходит по-прежнему — сумма от дробления не
    /// меняется. Ровно эту независимость от записи держит у щелчка
    /// `footprint::covers_a_band_written_with_extra_vertices`.
    #[test]
    fn a_band_is_cut_the_same_however_densely_it_is_written() {
        let dense = outline(&[
            (-70.0125, -180.0125),
            (70.0125, -180.0125),
            (70.0125, -90.0),
            (70.0125, 0.0),
            (70.0125, 90.0),
            (70.0125, 179.9875),
            (-70.0125, 179.9875),
            (-70.0125, 90.0),
            (-70.0125, 0.0),
            (-70.0125, -90.0),
        ]);
        let loops = ring(&dense);
        assert_eq!(loops.len(), 2, "гуще записанный пояс не разрезан");
        let stray = loops
            .concat()
            .iter()
            .map(|v| angles_of(v).0)
            .filter(|lat| (-69.0..=69.0).contains(lat))
            .count();
        assert_eq!(stray, 0, "на шве осталось {} вершин", stray);
    }

    /// А поясу с прорезью боковые рёбра — настоящий край, и резать их нельзя.
    ///
    /// Сетка, не сомкнувшаяся на десять градусов, Землю по долготе всё ещё
    /// обходит — меридиана, свободного от неё, нет. Но пробег круга не
    /// набирает, и выброшенные края замкнули бы каждую петлю хордой прямо
    /// через прорезь, то есть через то, чего в снимке нет.
    #[test]
    fn a_band_with_a_gap_keeps_its_edges() {
        let mut points = vec![(-70.0, 10.0), (70.0, 10.0)];
        for lon in [90.0, 170.0, -110.0, -30.0, 0.0] {
            points.push((70.0, lon));
        }
        points.push((-70.0, 0.0));
        for lon in [-30.0, -110.0, 170.0, 90.0] {
            points.push((-70.0, lon));
        }
        let gapped = outline(&points);

        let ring = ring(&gapped);
        assert_eq!(ring.len(), 1, "настоящий край выброшен как шов");
        let angles: Vec<(f64, f64)> = ring.concat().iter().map(angles_of).collect();
        assert!(geodesy::encircles(&angles), "проверять было бы нечего: круга нет");
    }

    /// Пояс режется одинаково, с какой бы вершины поставщик его ни начал.
    ///
    /// Обход начинается сразу после разреза, а не с головы списка: начатый с
    /// середины пробег распался бы надвое — его начало попало бы в одну петлю,
    /// а продолжение в другую, и параллель нарисовалась бы двумя дугами с
    /// хордой между ними.
    #[test]
    fn a_band_is_cut_the_same_wherever_its_list_begins() {
        let corners = [
            (-70.0125, -180.0125),
            (70.0125, -180.0125),
            (70.0125, -90.0),
            (70.0125, 0.0),
            (70.0125, 90.0),
            (70.0125, 179.9875),
            (-70.0125, 179.9875),
            (-70.0125, 90.0),
            (-70.0125, 0.0),
            (-70.0125, -90.0),
        ];
        for turn in 0..corners.len() {
            let mut points = corners.to_vec();
            points.rotate_left(turn);
            let loops = ring(&outline(&points));
            assert_eq!(loops.len(), 2, "начав с {}-й вершины, получили {} петель", turn, loops.len());
        }
    }

    /// Пробег, не вернувшийся в ту же точку шара, разрезом не считается.
    ///
    /// Шов — это переход с края на край, то есть то, что он соединяет, и без
    /// него одно место. Кольцо, обошедшее Землю по параллели и спустившееся на
    /// десять градусов, тоже имеет ребро без долготы — но это его настоящий
    /// край, и выброшенный, он стянул бы петлю хордой через нетронутое.
    #[test]
    fn a_run_that_does_not_close_is_no_seam() {
        let dented = outline(&[(70.0, 0.0), (70.0, 120.0), (70.0, -120.0), (60.0, -120.0)]);
        let loops = ring(&dented);
        assert_eq!(loops.len(), 1, "настоящий край выброшен как шов");

        // Ребро в десять градусов стои́т по меридиану −120°, и вершины на нём
        // лежат только у него: соседнее ребро уходит от той же точки на восток.
        let standing = loops
            .concat()
            .iter()
            .map(angles_of)
            .filter(|(lat, lon)| (60.5..69.5).contains(lat) && (lon + 120.0).abs() < 1e-6)
            .count();
        assert!(standing > 20, "ребро в десять градусов пропало: вершин {}", standing);
    }

    /// Повторённая угловая вершина ничего не меняет: между двумя разрезами
    /// подряд петли нет, и пустой пробег не мешает соседям.
    #[test]
    fn a_repeated_corner_does_not_split_the_band() {
        let doubled = outline(&[
            (-70.0125, -180.0125),
            (70.0125, -180.0125),
            (70.0125, -180.0125),
            (70.0125, 179.9875),
            (-70.0125, 179.9875),
        ]);
        let loops = ring(&doubled);
        assert_eq!(loops.len(), 2, "повторённый угол сбил разрез");
        let stray = loops
            .concat()
            .iter()
            .map(|v| angles_of(v).0)
            .filter(|lat| (-69.0..=69.0).contains(lat))
            .count();
        assert_eq!(stray, 0, "на шве осталось {} вершин", stray);
    }

    /// Заливки нет и у неразрезанного кольца, разошедшегося на пол-Земли.
    ///
    /// Пояс с прорезью петлёй остаётся одной — резать у него нечего, — а
    /// середины у него всё равно нет: веер из любой точки пошёл бы поверх
    /// обратной стороны. Держит это [`FILL_REACH`], и держать больше некому:
    /// у сомкнувшегося пояса до него дело не доходит, там раньше отвечают
    /// петли.
    #[test]
    fn a_gapped_band_is_whole_and_still_gets_no_fill() {
        let mut points = vec![(-70.0, 10.0), (70.0, 10.0)];
        for lon in [90.0, 170.0, -110.0, -30.0, 0.0] {
            points.push((70.0, lon));
        }
        points.push((-70.0, 0.0));
        for lon in [-30.0, -110.0, 170.0, 90.0] {
            points.push((-70.0, lon));
        }
        let gapped = outline(&points);
        assert_eq!(ring(&gapped).len(), 1, "проверять было бы нечего: кольцо разрезано");

        let built =
            Outlines::build(&[Outline { style: OutlineStyle::OutlinePending as i32, ..gapped }]);
        assert!(!built.indices.is_empty(), "контур пропал");
        assert!(built.hatch_indices.is_empty(), "веер построен поверх обратной стороны");
    }

    /// А честная шапка заливается, даже когда записана прямоугольником до
    /// полюса: вторая петля у неё вырождена в точку, рисуется одна, и середина
    /// у этой одной есть.
    #[test]
    fn a_cap_written_to_the_pole_keeps_its_fill() {
        let polar = outline(&[
            (60.0, -180.0125),
            (90.0, -180.0125),
            (90.0, 179.9875),
            (60.0, 179.9875),
        ]);
        let built =
            Outlines::build(&[Outline { style: OutlineStyle::OutlinePending as i32, ..polar }]);
        assert!(!built.hatch_indices.is_empty(), "шапка осталась незалитой");
    }

    /// Петля, выродившаяся в точку, не рисуется.
    ///
    /// У прямоугольника, упёршегося в полюс, верхняя параллель — сама точка:
    /// круг долгот там сходится в ноль дуги. Разрез такую петлю рождает
    /// законно, а рисовать её нечем — двумя вершинами замкнутой линии не
    /// выйдет.
    #[test]
    fn a_loop_shrunk_to_a_point_draws_nothing() {
        let polar = outline(&[
            (60.0, -180.0125),
            (90.0, -180.0125),
            (90.0, 179.9875),
            (60.0, 179.9875),
        ]);
        let loops = ring(&polar);
        assert_eq!(loops.len(), 2, "полюс не отрезан от параллели");
        assert!(loops.iter().any(|run| run.len() < 3), "вырожденной петли нет");

        let built = Outlines::build(&[Outline { style: OutlineStyle::OutlinePlain as i32, ..polar }]);
        let lats: Vec<f64> = built.vertices.iter().map(angles_of).map(|(lat, _)| lat).collect();
        assert!(!lats.is_empty(), "контур пропал целиком");
        assert!(lats.iter().all(|lat| *lat < 89.0), "вырожденная петля нарисована");
    }

    /// А коробка поперёк шва — не круг, и обводится она короткой стороной:
    /// записанная как 170 → −170, прочитанная буквально она шла бы вокруг всей
    /// Земли мимо снятого.
    #[test]
    fn a_box_across_the_seam_takes_the_short_way() {
        let box_ = outline(&[
            (65.0, 170.0),
            (65.0, -170.0),
            (60.0, -170.0),
            (60.0, 170.0),
        ]);
        let loops = ring(&box_);
        assert_eq!(loops.len(), 1, "обычная коробка разрезана");
        let ring = loops.concat();
        // Ни одна вершина не уходит за пределы снятого: 170°…190° (то же, что
        // 170°…−170°), а середина Земли остаётся нетронутой.
        let inside = |lon: f64| !(-160.0..=160.0).contains(&lon);
        assert!(
            ring.iter().all(|v| inside(angles_of(v).1)),
            "контур ушёл через полмира: {:?}",
            ring.iter().map(|v| angles_of(v).1).take(8).collect::<Vec<f64>>()
        );
        // И уплотнён он по своей длине, а не по чужой: короткая сторона — двадцать
        // градусов долготы, длинная — триста сорок, и уплотнение по длинной
        // стоило бы кольца на порядок гуще.
        let long_way = (340.0 / max_edge_deg()).ceil() as usize;
        assert!(
            ring.len() < long_way / 3,
            "уплотнено как трёхсотградусное ребро: {} вершин против {} у длинного пути",
            ring.len(), long_way
        );
    }

    /// А у полосы съёмки углы лежат где придётся, и её рёбра по-прежнему ведутся
    /// дугами: середина ребра севернее прямой в градусах — на этом и держится
    /// совпадение контура со снимком на высоких широтах.
    #[test]
    fn a_swath_edge_still_follows_the_arc() {
        let swath = outline(&[(73.0, 10.0), (73.4, 30.0), (72.0, 31.0), (71.6, 11.0)]);
        let loops = ring(&swath);
        assert_eq!(loops.len(), 1, "полоса съёмки разрезана");
        let ring = loops.concat();
        let north = ring
            .iter()
            .map(angles_of)
            .filter(|(_, lon)| (19.0..21.0).contains(lon))
            .map(|(lat, _)| lat)
            .fold(f64::MIN, f64::max);
        assert!(north > 73.3, "ребро пошло локсодромой: {}", north);
    }
}
