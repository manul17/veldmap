// Extracted from master:veldmodules/globe/src/mesh.rs

#[cfg(test)]
mod tests {
    use super::*;

    /// Узел везёт обе половины, и младшая — не пустое место: сантиметр по земле
    /// старшей неразличим, а парой берётся точно.
    ///
    /// Обнули младшую — и приём с половинами отменится разом для тела, сетки,
    /// контура и ленты: рисоваться всё продолжит, только шагом в метр, и
    /// заметить это будет нечем.
    #[test]
    fn a_node_carries_a_low_half_that_matters() {
        let step_m = 0.01;
        let step_deg = step_m / 111_320.0;
        let here = vertex(Geodetic::surface(45.0, 60.0));
        let there = vertex(Geodetic::surface(45.0 + step_deg, 60.0));

        assert_ne!(here.position_low, there.position_low, "сантиметр не виден и младшей");

        let moved = f64::from(
            glam::Vec3::from(geodesy::offset(
                (there.position, there.position_low),
                (here.position, here.position_low),
            ))
            .length(),
        ) * geodesy::SEMI_MAJOR_M;
        assert!((moved - step_m).abs() < 1e-4, "{} м вместо {}", moved, step_m);
    }

    /// Предельная хорда и провал обратны друг другу: провал такой хорды — это
    /// ровно то, что вынос способен принять, оставив просвет.
    #[test]
    fn the_limit_chord_sags_exactly_into_the_lift() {
        let sag = sag_m(max_chord_deg());
        assert!(
            (sag - (SURFACE_LIFT_M - CLEARANCE_M)).abs() < 1e-6,
            "провал предельной хорды {} м", sag
        );
    }

    /// Поделивший себя до предела садится на общий вынос — и ровная решётка,
    /// которая меряется диагональю квада, и ломаная, которая меряется звеном.
    #[test]
    fn what_divides_finely_lands_on_the_common_lift() {
        let diagonal = max_quad_side_deg() * std::f64::consts::SQRT_2;
        assert!(diagonal <= max_chord_deg() + 1e-9, "диагональ квада длиннее предельной хорды");
        for chord in [max_chord_deg(), diagonal, max_quad_side_deg(), 0.0] {
            assert!(
                (lift_m(chord) - SURFACE_LIFT_M).abs() < 1e-6,
                "хорда в {}° подняла носитель до {} м", chord, lift_m(chord)
            );
        }
    }

    /// Сетка и рисуется на общем выносе, а не только считается по нему: легшая
    /// на само тело, она спорила бы с ним за глубину и пропадала клочками.
    ///
    /// Высота читается из самой вершины: длина её вектора против длины такого
    /// же вектора на поверхности. Прямо разностью с единицей её не взять — у
    /// эллипсоида радиус меняется с широтой.
    #[test]
    fn the_graticule_is_drawn_on_the_lift() {
        let mesh = Mesh::build();
        for step in [0usize, 7, 64] {
            let index = mesh.indices[mesh.grid.start as usize + step] as usize;
            let node = mesh.vertices[index];
            let (lat_deg, lon_deg) = geodesy::angles(glam::Vec3::from(node.normal).as_dvec3());
            let ground = geodesy::world(Geodetic { lat_deg, lon_deg, height_m: 0.0 }).length();
            let raised = f64::from(glam::Vec3::from(node.position).length());
            let height = (raised - ground) * geodesy::SEMI_MAJOR_M;
            assert!(
                (height - SURFACE_LIFT_M).abs() < 3.0,
                "узел сетки на {} м вместо общего выноса", height
            );
        }
    }

    /// А не поделившийся поднимается ровно на свой провал и на просвет: иначе
    /// его середина ушла бы под поверхность, где её съедает тест глубины.
    #[test]
    fn what_cannot_divide_rises_by_its_own_sag() {
        let coarse = 8.0;
        assert!(lift_m(coarse) > SURFACE_LIFT_M, "грубое звено не поднялось");
        assert!(
            (lift_m(coarse) - sag_m(coarse) - CLEARANCE_M).abs() < 1e-9,
            "подъём {} м не равен провалу {} м с просветом", lift_m(coarse), sag_m(coarse)
        );
    }

    /// Сетка делится по тому же правилу, что и всё остальное: её звено
    /// умещается под общий вынос, и добирать высотой ей нечего. Ослабь деление,
    /// и линии сетки пошли бы под той поверхностью, на которой лежат снимок и
    /// контур.
    #[test]
    fn the_graticule_fits_under_the_common_lift() {
        let step = 360.0 / f64::from(GRID_SEGMENTS);
        assert!(
            (lift_m(step) - SURFACE_LIFT_M).abs() < 1e-9,
            "звено сетки в {}° не умещается под вынос: {} м", step, lift_m(step)
        );
        // И узлы её при этом лежат на рёбрах тела.
        assert_eq!(f64::from(GRID_STEP_DEG) % LON_STEP_DEG, 0.0, "сетка не на рёбрах тела");
        assert_eq!(
            GRID_SEGMENTS % (360.0 / LON_STEP_DEG) as u32,
            0,
            "узлы параллелей сошли с рёбер тела"
        );
        assert_eq!(
            GRID_SEGMENTS % (180.0 / LAT_STEP_DEG) as u32,
            0,
            "узлы меридианов сошли с рёбер тела"
        );
    }
}
