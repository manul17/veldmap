//! Shared synthetic TIFF/COG fixture builders for `veldmap-raster` integration tests.

use std::io::Cursor;

use ::tiff::encoder::{TiffEncoder, colortype};
use ::tiff::tags::Tag;
use veldmap_core::tile::{DecodedTile, PixelFormat, SourceFingerprint};
use veldmap_raster::{RASTER_TILE_SIZE, generate_linear_mipmaps, level_size};

pub fn fake_fingerprint(bytes: &[u8]) -> SourceFingerprint {
    let mut digest = [0u8; 32];
    for (i, b) in bytes.iter().take(32).enumerate() {
        digest[i] = *b;
    }
    digest[31] ^= (bytes.len() & 0xFF) as u8;
    SourceFingerprint::from_digest(digest)
}

pub fn rgb_pattern(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            out.push((x * 7 + y * 3) as u8);
            out.push((x ^ y) as u8);
            out.push(((x * y) / 97) as u8);
        }
    }
    out
}

fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .flat_map(|px| [px[0], px[1], px[2], 255])
        .collect()
}

fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect()
}

fn halve_rgb_linear(rgb: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut tile =
        DecodedTile::try_new(width, height, PixelFormat::Rgba8UnormSrgb, rgb_to_rgba(rgb))
            .expect("build decoded tile for halving");
    generate_linear_mipmaps(&mut tile);
    let mip1 = tile.mip_data(1).expect("mip level 1");
    rgba_to_rgb(mip1)
}

fn tile_bytes(rgb: &[u8], width: u32, height: u32, tx: u32, ty: u32) -> Vec<u8> {
    let mut out = vec![0u8; (RASTER_TILE_SIZE * RASTER_TILE_SIZE * 3) as usize];
    for row in 0..RASTER_TILE_SIZE {
        let y = ty * RASTER_TILE_SIZE + row;
        if y >= height {
            break;
        }
        let x0 = tx * RASTER_TILE_SIZE;
        let x1 = (x0 + RASTER_TILE_SIZE).min(width);
        let from = ((y * width + x0) * 3) as usize;
        let len = ((x1 - x0) * 3) as usize;
        let to = (row * RASTER_TILE_SIZE * 3) as usize;
        out[to..to + len].copy_from_slice(&rgb[from..from + len]);
    }
    out
}

/// Build an uncompressed tiled TIFF with `levels` 2x2-halved overview IFDs,
/// returning `(bytes, base_tile_byte_ranges, ifd_offsets)`.
pub fn build_tiled_cog(
    width: u32,
    height: u32,
    levels: u32,
) -> (Vec<u8>, Vec<(u64, u64)>, Vec<u64>) {
    let mut file = Cursor::new(Vec::new());
    let mut base_tiles = Vec::new();
    let mut ifds = Vec::new();
    {
        let mut encoder = TiffEncoder::new(&mut file).expect("tiff encoder");
        let mut rgb = rgb_pattern(width, height);
        let (mut w, mut h) = (width, height);

        for level in 0..=levels {
            let mut dir = encoder.image_directory().expect("image_directory");
            if level > 0 {
                dir.write_tag(Tag::NewSubfileType, 1u32).expect("subfile");
            }
            dir.write_tag(Tag::ImageWidth, w).expect("width");
            dir.write_tag(Tag::ImageLength, h).expect("height");
            dir.write_tag(Tag::BitsPerSample, &[8u16, 8, 8][..])
                .expect("bps");
            dir.write_tag(Tag::Compression, 1u16).expect("compression");
            dir.write_tag(Tag::PhotometricInterpretation, 2u16)
                .expect("photometric");
            dir.write_tag(Tag::SamplesPerPixel, 3u16).expect("spp");
            dir.write_tag(Tag::PlanarConfiguration, 1u16)
                .expect("planar");
            dir.write_tag(Tag::TileWidth, RASTER_TILE_SIZE).expect("tw");
            dir.write_tag(Tag::TileLength, RASTER_TILE_SIZE)
                .expect("th");

            let (across, down) = (w.div_ceil(RASTER_TILE_SIZE), h.div_ceil(RASTER_TILE_SIZE));
            let mut offsets = Vec::new();
            let mut counts = Vec::new();
            for ty in 0..down {
                for tx in 0..across {
                    let tile = tile_bytes(&rgb, w, h, tx, ty);
                    let at = dir.write_data(&tile[..]).expect("write tile");
                    offsets.push(at as u32);
                    counts.push(tile.len() as u32);
                    if level == 0 {
                        base_tiles.push((at, tile.len() as u64));
                    }
                }
            }
            dir.write_tag(Tag::TileOffsets, &offsets[..]).expect("offs");
            dir.write_tag(Tag::TileByteCounts, &counts[..])
                .expect("cnts");
            let finish = dir.finish_with_offsets().expect("finish ifd");
            ifds.push(u64::from(finish.offset));

            if level < levels {
                rgb = halve_rgb_linear(&rgb, w, h);
                w = level_size(w, 1);
                h = level_size(h, 1);
            }
        }
    }
    (file.into_inner(), base_tiles, ifds)
}

pub fn build_stripped_tiff(width: u32, height: u32, rows_per_strip: u32) -> Vec<u8> {
    let mut file = Cursor::new(Vec::new());
    {
        let mut encoder = TiffEncoder::new(&mut file).expect("tiff encoder");
        let mut image = encoder
            .new_image::<colortype::RGB8>(width, height)
            .expect("new_image");
        image.rows_per_strip(rows_per_strip).expect("rows");
        image
            .write_data(&rgb_pattern(width, height))
            .expect("write stripped");
    }
    file.into_inner()
}

pub fn build_multi_tile_tiff(tiles_x: u32, tiles_y: u32, chunk_side: u32) -> Vec<u8> {
    let width = tiles_x * chunk_side;
    let height = tiles_y * chunk_side;
    let mut file = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut file).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        dir.write_tag(Tag::ImageWidth, width).expect("tag");
        dir.write_tag(Tag::ImageLength, height).expect("tag");
        dir.write_tag(Tag::BitsPerSample, &[8u16, 8, 8][..])
            .expect("tag");
        dir.write_tag(Tag::Compression, 1u16).expect("tag");
        dir.write_tag(Tag::PhotometricInterpretation, 2u16)
            .expect("tag");
        dir.write_tag(Tag::SamplesPerPixel, 3u16).expect("tag");
        dir.write_tag(Tag::PlanarConfiguration, 1u16).expect("tag");
        dir.write_tag(Tag::TileWidth, chunk_side).expect("tag");
        dir.write_tag(Tag::TileLength, chunk_side).expect("tag");

        let chunk_bytes = (chunk_side * chunk_side * 3) as usize;
        let mut offsets = Vec::new();
        let mut counts = Vec::new();
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let pattern = vec![((tx * 37 + ty * 17) & 0xFF) as u8; chunk_bytes];
                let at = dir.write_data(&pattern[..]).expect("write");
                offsets.push(at as u32);
                counts.push(pattern.len() as u32);
            }
        }
        dir.write_tag(Tag::TileOffsets, &offsets[..]).expect("tag");
        dir.write_tag(Tag::TileByteCounts, &counts[..])
            .expect("tag");
        dir.finish().expect("finish");
    }
    file.into_inner()
}
