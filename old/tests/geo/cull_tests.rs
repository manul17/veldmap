// Extracted from master:veldmodules/globe/src/cull.rs

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::camera::Camera;
    use super::super::geodesy::Geodetic;

    fn at(lat: f64, lon: f64, height_m: f64) -> World {
        geodesy::position(Geodetic { lat_deg: lat, lon_deg: lon, height_m })
    }

    /// Камера, наведённая на это место с этой высоты. Наводка едет, поэтому шаг
    /// длиннее перелёта ставит её ровно в цель (см. `Camera::advance`).
    fn looking(lat: f64, lon: f64, radius_deg: f64) -> (Mat4, World) {
        let mut camera = Camera::default();
        camera.focus(lat, lon, radius_deg);
        camera.advance(10.0);
        (camera.view_projection(1.5), camera.eye())
    }

    /// Точка, попавшая в кадр по проекции, обязана пройти все четыре плоскости,
    /// и не попавшая — обязана хоть одну провалить.
    ///
    /// Этим и держится раскладка матрицы: плоскости выводятся из строк, а
    /// проекция умножает столбцы, и перепутанные они разошлись бы молча —
    /// картинка осталась бы прежней, а отсекаться начало бы не то.
    #[test]
    fn a_point_inside_the_frame_clears_every_plane() {
        let (view_proj, eye) = looking(30.0, 20.0, 20.0);
        let frame = Frame::new(&view_proj, eye);
        let (mut inside, mut outside) = (0, 0);
        for lat in (-80..=80).step_by(5) {
            for lon in (-180..180).step_by(5) {
                let point = at(f64::from(lat), f64::from(lon), 0.0);
                let clip = super::super::camera::project(&view_proj, point, eye);
                if clip[3] <= 0.0 {
                    continue;
                }
                let (x, y) = (clip[0] / clip[3], clip[1] / clip[3]);
                let framed = (-1.0..=1.0).contains(&x) && (-1.0..=1.0).contains(&y);
                let admitted = frame
                    .inward
                    .iter()
                    .all(|inward| inward.dot(Vec3::from(point) - frame.eye) >= 0.0);
                assert_eq!(framed, admitted, "точка {},{}: кадр {} против плоскостей {}",
                    lat, lon, framed, admitted);
                match framed {
                    true => inside += 1,
                    false => outside += 1,
                }
            }
        }
        assert!(inside > 20 && outside > 200, "выборка вырождена: {} внутри, {} снаружи", inside, outside);
    }

    /// Горизонт отсекает обратную сторону Земли и не трогает того, что под
    /// глазом.
    #[test]
    fn the_far_side_of_the_earth_is_refused_and_the_near_side_is_not() {
        let (view_proj, eye) = looking(0.0, 0.0, 25.0);
        let frame = Frame::new(&view_proj, eye);
        let under = Ball { centre: at(0.0, 0.0, 0.0), radius: 0.0 };
        let behind = Ball { centre: at(0.0, 180.0, 0.0), radius: 0.0 };
        assert!(frame.admits(&under), "точка под глазом отвергнута");
        assert!(!frame.admits(&behind), "обратная сторона Земли не отвергнута");
        // Тот же дальний край, но накрытый шаром во всю Землю: отвергать его
        // нельзя — шар достаёт до ближней стороны.
        assert!(
            frame.admits(&Ball { centre: [0.0; 3], radius: 1.01 }),
            "шар со всей Землёй отвергнут"
        );
    }

    /// Горизонт меряется полярным радиусом, а не экваториальным.
    ///
    /// Разводит их сжатие — двадцать один километр, — и взятый экваториальным
    /// горизонт поднимается выше настоящего: точка у полюса, ещё видимая с
    /// краю, объявляется зашедшей за него. Проверяется это точкой, лежащей
    /// ровно на касательной к полярному шару: она видна впритык, и всякая
    /// мерка крупнее полярной её отвергнет.
    #[test]
    fn the_horizon_is_measured_by_the_smallest_radius_the_earth_has() {
        // Камера видит шар почти целиком, так что лимб попадает в кадр.
        let (view_proj, eye) = looking(0.0, 0.0, 70.0);
        let frame = Frame::new(&view_proj, eye);
        let polar = f64::from(at(90.0, 0.0, 0.0)[2]);
        let (eye, away) = (glam::DVec3::from(Vec3::from(eye).as_dvec3()), glam::DVec3::Z);
        let (towards, distance) = (eye.normalize(), eye.length());
        // Точка касания луча из глаза к шару полярного радиуса.
        let along = polar * polar / distance;
        let across = polar * (1.0 - (polar / distance).powi(2)).sqrt();
        let side = towards.cross(away).cross(towards).normalize();
        let grazing = towards * along + side * across;
        let touching = Ball { centre: grazing.as_vec3().to_array(), radius: 0.0 };
        assert!(frame.admits(&touching), "точка на касательной к полюсу отвергнута");
        // Та же точка, отодвинутая за горизонт на пятую долю сжатия, обязана
        // быть отвергнута — иначе тест прошёл бы и без всякого горизонта.
        let squashed = polar * polar - f64::from(at(0.0, 0.0, 0.0)[0]).powi(2);
        let beyond = towards * (along + squashed * 0.2 / distance) + side * across;
        assert!(
            !frame.admits(&Ball { centre: beyond.as_vec3().to_array(), radius: 0.0 }),
            "точка за горизонтом принята"
        );
    }

    /// Глаз внутри шара — «видно», без особой ветки: у верхних ступеней шар
    /// накрывает Землю целиком вместе с камерой.
    #[test]
    fn a_ball_that_swallows_the_eye_is_admitted() {
        let (view_proj, eye) = looking(45.0, 10.0, 0.01);
        let frame = Frame::new(&view_proj, eye);
        let swallowing = Ball { centre: [0.0; 3], radius: 2.0 };
        assert!(frame.admits(&swallowing), "шар с камерой внутри отвергнут");
    }

    /// Шар, чей центр за краем кадра, а край дотягивается внутрь, — виден.
    ///
    /// В этом весь смысл запаса: ячейку описывает не точка, а шар, и ячейка,
    /// середина которой ушла за край, обыкновенно всё ещё видна половиной.
    /// Проверяющий одну середину отрезал бы у кадра кайму шириной в ячейку.
    #[test]
    fn a_ball_reaching_into_the_frame_is_admitted_though_its_centre_is_out() {
        let (view_proj, eye) = looking(30.0, 20.0, 20.0);
        let frame = Frame::new(&view_proj, eye);
        // Точка выше того, куда смотрят: за верхним краем кадра, но недалеко.
        // Вверх, а не вбок: угол обзора задан по вертикали, а вширь кадр шире
        // на своё отношение сторон, и вбок за край уходят много позже.
        let outside = at(60.0, 20.0, 0.0);
        let deficit = frame
            .inward
            .iter()
            .map(|inward| inward.dot(Vec3::from(outside) - frame.eye))
            .fold(f32::MAX, f32::min);
        assert!(deficit < 0.0, "точка оказалась в кадре: запас проверять не на чем");
        assert!(!frame.admits(&Ball { centre: outside, radius: 0.0 }), "середина не за краем");
        assert!(
            frame.admits(&Ball { centre: outside, radius: -deficit * 1.01 }),
            "шар, дотянувшийся до кадра, отвергнут"
        );
        assert!(
            !frame.admits(&Ball { centre: outside, radius: -deficit * 0.99 }),
            "шар, не дотянувшийся до кадра, принят"
        );
    }

    /// Шар накрывает свои точки — иначе ячейка теряется молча.
    #[test]
    fn a_ball_covers_every_point_it_was_built_on() {
        for (lat, lon, span) in [(0.0, 0.0, 60.0), (45.0, 10.0, 2.0), (80.0, 100.0, 0.05)] {
            // Середина первой, углы следом: набор, у которого дальняя точка
            // стои́т не первой. Иначе шар, построенный по одной лишь первой,
            // случайно накрыл бы всё, и проверка перестала бы что-либо держать.
            let points: Vec<World> = [4usize, 0, 1, 2, 3, 5, 6, 7, 8]
                .iter()
                .map(|at_index| {
                    let (row, col) = (at_index / 3, at_index % 3);
                    at(lat + span * f64::from(row as u32) / 2.0,
                       lon + span * f64::from(col as u32) / 2.0, 80.0)
                })
                .collect();
            let ball = Ball::over(&points);
            for point in &points {
                assert!(
                    distance(ball.centre, *point) <= ball.radius + 1e-6,
                    "точка вне шара при размахе {}°", span
                );
            }
        }
    }

}
