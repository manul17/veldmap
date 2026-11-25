//! Cross-module integration tests exercising `veldmap-core::tile` together
//! with `veldmap-core::geo::Camera` (`quadtree::traverse` + `TileCache` carrier
//! resolution + `compose_uv_rect` + `DecodedTile`).

use veldmap_core::geo::Camera;
use veldmap_core::tile::{
    DecodedTile, GlobeScheme, PixelFormat, PlanetaryScheme, QuadTileId, QuadtreeConfig, SourceId,
    TileCache, TileKey, compose_uv_rect, traverse,
};

#[test]
fn quadtree_traverse_composes_with_tile_cache_carrier_and_decoded_tile() {
    let cam = Camera::at(35.0, 20.0, 200_000.0);
    let config = QuadtreeConfig {
        scheme: GlobeScheme::WebMercator,
        max_tex_level: 8,
        polar_max_tex_level: 0,
        ..QuadtreeConfig::default()
    };
    let visible = traverse(&cam, 1280, 720, &config);
    assert!(!visible.is_empty());

    let tile_bytes = 256 * 256 * PixelFormat::Rgba8UnormSrgb.bytes_per_pixel();
    let mut cache = TileCache::new(tile_bytes * 16);

    let root = QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 0,
        x: 0,
        y: 0,
    };
    let root_key = TileKey::display_quad(SourceId::BUNDLED_BASEMAP, root);
    let decoded_root =
        DecodedTile::try_new(256, 256, PixelFormat::Rgba8UnormSrgb, vec![255; tile_bytes])
            .expect("valid RGBA8 root tile");
    cache
        .insert(root_key, decoded_root, tile_bytes)
        .expect("insert root");

    for tile in visible
        .iter()
        .filter(|t| t.tex_id.scheme == PlanetaryScheme::WebMercator)
    {
        let tex_key = TileKey::display_quad(SourceId::BUNDLED_BASEMAP, tile.tex_id);
        let (carrier, carrier_uv, decoded) =
            cache.carrier(&tex_key).expect("root carrier covers leaf");
        assert_eq!(carrier, root_key);
        assert_eq!(decoded.width(), 256);
        let combined = compose_uv_rect(carrier_uv, tile.uv_rect);
        assert!(combined[0] >= 0.0 && combined[1] >= 0.0);
        assert!(combined[0] + combined[2] <= 1.0001);
        assert!(combined[1] + combined[3] <= 1.0001);
    }
}
