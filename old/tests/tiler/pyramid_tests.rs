// Extracted from master:veldmodules/image-tiler/src/pyramid.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_size_is_iterated_halving() {
        // Композиция округлений: два деления пополам = одно на четыре.
        for side in [1u32, 2, 3, 511, 512, 513, 1000, 10980] {
            for level in 0..8 {
                assert_eq!(level_size(side, level), side.div_ceil(1 << level));
            }
        }
        assert_eq!(level_size(10980, 0), 10980);
        assert_eq!(level_size(10980, 1), 5490);
        assert_eq!(level_size(10980, 5), 344);
    }

    #[test]
    fn level_count_boundaries() {
        // Помещается в тайл — уровень один; на пиксель больше — уже два.
        assert_eq!(level_count(TILE, TILE), 1);
        assert_eq!(level_count(TILE + 1, 1), 2);
        assert_eq!(level_count(1, 1), 1);
        assert_eq!(level_count(0, 100), 0);
        // TCI Sentinel-2: 10980 → 5490 → 2745 → 1373 → 687 → 344.
        assert_eq!(level_count(10980, 10980), 6);
        // Вершина действительно помещается в один тайл.
        for (w, h) in [(10980, 10980), (7000, 3), (513, 513)] {
            let top = level_count(w, h) - 1;
            assert!(level_size(w, top) <= TILE && level_size(h, top) <= TILE);
        }
    }

    #[test]
    fn grid_and_extent_cover_side_exactly() {
        for side in [1u32, TILE - 1, TILE, TILE + 1, 3 * TILE, 10980] {
            let tiles = grid(side);
            let covered: u32 = (0..tiles).map(|i| tile_extent(i, side)).sum();
            assert_eq!(covered, side, "сторона {}", side);
            // Все тайлы, кроме последнего, полные.
            for i in 0..tiles.saturating_sub(1) {
                assert_eq!(tile_extent(i, side), TILE);
            }
            assert_eq!(tile_extent(tiles, side), 0);
        }
    }

    #[test]
    fn exact_tile_uses_whole_texture() {
        let cell = cell_image_rect(1, 0, 0, 1024, 1024);
        assert_eq!(cell, [512.0, 0.0, 1024.0, 512.0]);
        assert_eq!(cell_uv(cell, (0, 1, 0), 512, 512), [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn parent_serves_its_quadrant() {
        // Ячейка (0; 1,1) — правый нижний квадрант тайла-предка (1; 0,0).
        let cell = cell_image_rect(1, 1, 0, 2048, 2048);
        assert_eq!(cell_uv(cell, (1, 0, 0), 512, 512), [0.5, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn edge_cell_is_clipped_by_image() {
        // 700 px: уровень 0 — ячейки 0..2, правая короче; краевой тайл
        // 188×40 используется целиком.
        let cell = cell_image_rect(1, 0, 0, 700, 40);
        assert_eq!(cell, [512.0, 0.0, 700.0, 40.0]);
        assert_eq!(cell_uv(cell, (0, 1, 0), 188, 40), [0.0, 0.0, 1.0, 1.0]);
    }
}
