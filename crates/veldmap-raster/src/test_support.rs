//! Synthetic raster fixture helpers shared across workspace tests.

use std::fs::File;
use std::io;
use std::path::Path;

use ::tiff::encoder::TiffEncoder;
use ::tiff::tags::Tag;
use veldmap_core::tile::{RASTER_TILE_SIZE, level_count};

fn tiff_io_err(err: ::tiff::TiffError) -> io::Error {
    io::Error::other(err)
}

fn write_cog_level(
    enc: &mut TiffEncoder<&mut File>,
    w: u32,
    h: u32,
    tile: u32,
    rgb: [u8; 3],
    is_base: bool,
) -> io::Result<()> {
    let mut dir = enc.image_directory().map_err(tiff_io_err)?;
    dir.write_tag(Tag::ImageWidth, w).map_err(tiff_io_err)?;
    dir.write_tag(Tag::ImageLength, h).map_err(tiff_io_err)?;
    dir.write_tag(Tag::BitsPerSample, &[8u16, 8, 8][..])
        .map_err(tiff_io_err)?;
    dir.write_tag(Tag::Compression, 1u16).map_err(tiff_io_err)?;
    dir.write_tag(Tag::PhotometricInterpretation, 2u16)
        .map_err(tiff_io_err)?;
    dir.write_tag(Tag::SamplesPerPixel, 3u16)
        .map_err(tiff_io_err)?;
    dir.write_tag(Tag::PlanarConfiguration, 1u16)
        .map_err(tiff_io_err)?;
    dir.write_tag(Tag::TileWidth, tile).map_err(tiff_io_err)?;
    dir.write_tag(Tag::TileLength, tile).map_err(tiff_io_err)?;
    if !is_base {
        dir.write_tag(Tag::NewSubfileType, 1u32)
            .map_err(tiff_io_err)?;
    }
    let cols = w.div_ceil(tile);
    let rows = h.div_ceil(tile);
    let tile_pixels = (tile as usize).saturating_mul(tile as usize);
    let mut tile_bytes = Vec::with_capacity(tile_pixels.saturating_mul(3));
    for _ in 0..tile_pixels {
        tile_bytes.extend_from_slice(&rgb);
    }
    let tile_len = u32::try_from(tile_bytes.len())
        .map_err(|_| io::Error::other("tile byte count overflow"))?;
    let total_tiles = (cols as usize).saturating_mul(rows as usize);
    let mut offsets = Vec::with_capacity(total_tiles);
    let mut counts = Vec::with_capacity(total_tiles);
    for _ in 0..total_tiles {
        let at = dir.write_data(&tile_bytes[..]).map_err(tiff_io_err)?;
        let off = u32::try_from(at).map_err(|_| io::Error::other("tile offset overflow"))?;
        offsets.push(off);
        counts.push(tile_len);
    }
    dir.write_tag(Tag::TileOffsets, &offsets[..])
        .map_err(tiff_io_err)?;
    dir.write_tag(Tag::TileByteCounts, &counts[..])
        .map_err(tiff_io_err)?;
    if is_base {
        let geokeys: [u16; 16] = [
            1, 1, 0, 3, // header
            1024, 0, 1, 1, // Projected
            1025, 0, 1, 1, // PixelIsArea
            3072, 0, 1, 32637, // EPSG:32637
        ];
        dir.write_tag(Tag::GeoKeyDirectoryTag, &geokeys[..])
            .map_err(tiff_io_err)?;
        dir.write_tag(Tag::ModelPixelScaleTag, &[10.0_f64, 10.0, 0.0][..])
            .map_err(tiff_io_err)?;
        dir.write_tag(
            Tag::ModelTiepointTag,
            &[0.0_f64, 0.0, 0.0, 500_000.0, 5_250_000.0, 0.0][..],
        )
        .map_err(tiff_io_err)?;
    }
    dir.finish().map_err(tiff_io_err)
}

/// Write a multi-level tiled GeoTIFF COG (`width x height`, `512 x 512` RGB8
/// tiles, `EPSG:32637`, `10 m/px` anchored at `(500_000, 5_250_000)`) to
/// `path` filled with solid `rgb`.
pub fn write_tiled_geotiff_cog(
    path: &Path,
    width: u32,
    height: u32,
    rgb: [u8; 3],
) -> io::Result<()> {
    let mut file = File::create(path)?;
    let mut enc = TiffEncoder::new(&mut file).map_err(tiff_io_err)?;
    let levels = level_count(width, height);
    let mut cur_w = width;
    let mut cur_h = height;
    for idx in 0..levels {
        write_cog_level(&mut enc, cur_w, cur_h, RASTER_TILE_SIZE, rgb, idx == 0)?;
        cur_w = cur_w.div_ceil(2);
        cur_h = cur_h.div_ceil(2);
    }
    Ok(())
}
