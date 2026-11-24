//! Thin re-export of canonical raster pyramid arithmetic from
//! [`veldmap_core::tile::pyramid`].

pub use veldmap_core::tile::pyramid::{
    RASTER_TILE_SIZE, cell_image_rect, cell_uv, grid_dim, level_count, level_size, tile_extent,
};
