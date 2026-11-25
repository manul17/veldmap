//! Shared synthetic TIFF/COG fixture builders for `veldmap-raster` integration tests.

use std::io::{Cursor, Seek, Write};

use ::tiff::encoder::{
    DirectoryEncoder, TiffEncoder, TiffKind, TiffKindBig, TiffKindStandard, colortype,
};
use ::tiff::tags::Tag;
use veldmap_core::tile::{
    BandKey, DecodedTile, PixelFormat, RASTER_TILE_SIZE, SourceFingerprint, level_size,
};
use veldmap_raster::{BandSample, DecodeContext, RasterError, RasterSource, generate_tile_mipmaps};

/// Test helper wrapping [`RasterSource::sample_band`] for display RGBA8 readouts.
pub fn sample_display_rgba(
    src: &dyn RasterSource,
    px: u32,
    py: u32,
    ctx: &DecodeContext<'_>,
) -> Result<Option<[u8; 4]>, RasterError> {
    match src.sample_band(BandKey::Display, PixelFormat::Rgba8UnormSrgb, px, py, ctx)? {
        BandSample::Rgba(px) => Ok(Some(px)),
        BandSample::Nodata | BandSample::Unavailable => Ok(None),
        other => panic!("unexpected sample for display stream: {other:?}"),
    }
}

/// Core TIFF IFD tags shared across synthetic test fixtures.
pub struct TiffIfdHeader<'a> {
    pub width: u32,
    pub height: u32,
    pub bits_per_sample: &'a [u16],
    pub compression: u16,
    pub photometric: u16,
    pub samples_per_pixel: u16,
    pub planar_configuration: u16,
    pub sample_format: Option<u16>,
    pub is_overview: bool,
}

impl TiffIfdHeader<'static> {
    pub const fn rgb8(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bits_per_sample: &[8, 8, 8],
            compression: 1,
            photometric: 2,
            samples_per_pixel: 3,
            planar_configuration: 1,
            sample_format: None,
            is_overview: false,
        }
    }

    pub const fn gray8(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bits_per_sample: &[8],
            compression: 1,
            photometric: 1,
            samples_per_pixel: 1,
            planar_configuration: 1,
            sample_format: None,
            is_overview: false,
        }
    }
}

pub fn write_ifd_header<W: Write + Seek, K: TiffKind>(
    dir: &mut DirectoryEncoder<'_, W, K>,
    header: &TiffIfdHeader<'_>,
) {
    if header.is_overview {
        dir.write_tag(Tag::NewSubfileType, 1u32).expect("subfile");
    }
    dir.write_tag(Tag::ImageWidth, header.width).expect("width");
    dir.write_tag(Tag::ImageLength, header.height)
        .expect("height");
    if header.bits_per_sample.len() == 1 {
        dir.write_tag(Tag::BitsPerSample, header.bits_per_sample[0])
            .expect("bps");
    } else {
        dir.write_tag(Tag::BitsPerSample, header.bits_per_sample)
            .expect("bps");
    }
    if let Some(fmt) = header.sample_format {
        dir.write_tag(Tag::SampleFormat, fmt)
            .expect("sample format");
    }
    dir.write_tag(Tag::Compression, header.compression)
        .expect("compression");
    dir.write_tag(Tag::PhotometricInterpretation, header.photometric)
        .expect("photometric");
    dir.write_tag(Tag::SamplesPerPixel, header.samples_per_pixel)
        .expect("spp");
    dir.write_tag(Tag::PlanarConfiguration, header.planar_configuration)
        .expect("planar");
}

pub fn write_single_strip_u32<W: Write + Seek>(
    dir: &mut DirectoryEncoder<'_, W, TiffKindStandard>,
    rows_per_strip: u32,
    payload: &[u8],
) {
    dir.write_tag(Tag::RowsPerStrip, rows_per_strip)
        .expect("rows");
    let at = dir.write_data(payload).expect("write strip");
    dir.write_tag(Tag::StripOffsets, &[at as u32][..])
        .expect("strip offsets");
    dir.write_tag(Tag::StripByteCounts, &[payload.len() as u32][..])
        .expect("strip counts");
}

pub fn write_single_strip_u64<W: Write + Seek>(
    dir: &mut DirectoryEncoder<'_, W, TiffKindBig>,
    rows_per_strip: u32,
    payload: &[u8],
) {
    dir.write_tag(Tag::RowsPerStrip, rows_per_strip)
        .expect("rows");
    let at = dir.write_data(payload).expect("write strip");
    dir.write_tag(Tag::StripOffsets, &[at][..])
        .expect("strip offsets");
    dir.write_tag(Tag::StripByteCounts, &[payload.len() as u64][..])
        .expect("strip counts");
}

pub fn write_tile_grid_u32<W: Write + Seek>(
    dir: &mut DirectoryEncoder<'_, W, TiffKindStandard>,
    tile_w: u32,
    tile_h: u32,
    tiles_x: u32,
    tiles_y: u32,
    mut make_tile: impl FnMut(u32, u32) -> Vec<u8>,
) -> Vec<(u64, u64)> {
    dir.write_tag(Tag::TileWidth, tile_w).expect("tile width");
    dir.write_tag(Tag::TileLength, tile_h).expect("tile height");
    let total = (tiles_x * tiles_y) as usize;
    let mut offsets = Vec::with_capacity(total);
    let mut counts = Vec::with_capacity(total);
    let mut ranges = Vec::with_capacity(total);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let tile = make_tile(tx, ty);
            let at = dir.write_data(&tile[..]).expect("write tile");
            offsets.push(at as u32);
            counts.push(tile.len() as u32);
            ranges.push((at, tile.len() as u64));
        }
    }
    dir.write_tag(Tag::TileOffsets, &offsets[..])
        .expect("tile offsets");
    dir.write_tag(Tag::TileByteCounts, &counts[..])
        .expect("tile counts");
    ranges
}

pub fn fake_fingerprint(bytes: &[u8]) -> SourceFingerprint {
    let mut digest = [0u8; 32];
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (bytes.len() as u64);
    for (i, &b) in bytes.iter().enumerate() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        digest[i & 31] ^= (h >> ((i & 7) * 8)) as u8;
    }
    digest[0..8].copy_from_slice(&h.to_le_bytes());
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
    generate_tile_mipmaps(&mut tile);
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
            let mut header = TiffIfdHeader::rgb8(w, h);
            header.is_overview = level > 0;
            write_ifd_header(&mut dir, &header);

            let (across, down) = (w.div_ceil(RASTER_TILE_SIZE), h.div_ceil(RASTER_TILE_SIZE));
            let ranges = write_tile_grid_u32(
                &mut dir,
                RASTER_TILE_SIZE,
                RASTER_TILE_SIZE,
                across,
                down,
                |tx, ty| tile_bytes(&rgb, w, h, tx, ty),
            );
            if level == 0 {
                base_tiles = ranges;
            }
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
        write_ifd_header(&mut dir, &TiffIfdHeader::rgb8(width, height));

        let chunk_bytes = (chunk_side * chunk_side * 3) as usize;
        let _ = write_tile_grid_u32(
            &mut dir,
            chunk_side,
            chunk_side,
            tiles_x,
            tiles_y,
            |tx, ty| vec![((tx * 37 + ty * 17) & 0xFF) as u8; chunk_bytes],
        );
        dir.finish().expect("finish");
    }
    file.into_inner()
}

/// Build a minimal classic little-endian single-strip TIFF with explicit IFD
/// entries (sorted by tag id).
pub fn build_raw_tiff(entries: &mut [(u16, u16, u32, Vec<u8>)], pixel_payload: &[u8]) -> Vec<u8> {
    entries.sort_by_key(|(tag, _, _, _)| *tag);
    // Header: 8 bytes (II, 42, ifd_offset=8)
    let mut out = vec![b'I', b'I', 42, 0, 8, 0, 0, 0];
    let n_entries = entries.len() as u16;
    out.extend_from_slice(&n_entries.to_le_bytes());

    let ifd_size = 2 + entries.len() * 12 + 4;
    let extra_data_offset = (8 + ifd_size) as u32;
    let mut extra_blobs: Vec<u8> = Vec::new();

    // Reserve pixel_payload right after extra blobs.
    let mut inline_or_deferred: Vec<(u16, u16, u32, [u8; 4], bool)> = Vec::new();
    for (tag, ty, count, raw) in entries.iter() {
        if *tag == 273 {
            // StripOffsets placeholder (patched once extra_blobs finishes)
            inline_or_deferred.push((*tag, *ty, *count, [0; 4], true));
            continue;
        }
        if raw.len() <= 4 {
            let mut val = [0u8; 4];
            val[..raw.len()].copy_from_slice(raw);
            inline_or_deferred.push((*tag, *ty, *count, val, false));
        } else {
            let off = extra_data_offset + extra_blobs.len() as u32;
            extra_blobs.extend_from_slice(raw);
            if extra_blobs.len() % 2 == 1 {
                extra_blobs.push(0);
            }
            inline_or_deferred.push((*tag, *ty, *count, off.to_le_bytes(), false));
        }
    }

    let pixel_offset = extra_data_offset + extra_blobs.len() as u32;
    for (tag, ty, count, mut val, is_strip_off) in inline_or_deferred {
        if is_strip_off {
            val = pixel_offset.to_le_bytes();
        }
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&ty.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&val);
    }
    // Next IFD = 0
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&extra_blobs);
    out.extend_from_slice(pixel_payload);
    out
}

pub fn build_rgb_4x4_tiff(extra: &[(u16, u16, u32, Vec<u8>)], fill: u8) -> Vec<u8> {
    let payload = vec![fill; 4 * 4 * 3];
    let mut entries = vec![
        tag_u32(256, 4),
        tag_u32(257, 4),
        tag_u16_vec(258, &[8, 8, 8]),
        tag_u16(259, 1),
        tag_u16(262, 2),
        tag_u32(273, 0),
        tag_u16(277, 3),
        tag_u32(278, 4),
        tag_u32(279, payload.len() as u32),
        tag_u16(284, 1),
    ];
    entries.extend_from_slice(extra);
    build_raw_tiff(&mut entries, &payload)
}

pub fn tag_u16(tag: u16, v: u16) -> (u16, u16, u32, Vec<u8>) {
    (tag, 3, 1, v.to_le_bytes().to_vec())
}

pub fn tag_u16_vec(tag: u16, vals: &[u16]) -> (u16, u16, u32, Vec<u8>) {
    let mut b = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    (tag, 3, vals.len() as u32, b)
}

pub fn tag_u32(tag: u16, v: u32) -> (u16, u16, u32, Vec<u8>) {
    (tag, 4, 1, v.to_le_bytes().to_vec())
}

pub fn tag_f64_vec(tag: u16, vals: &[f64]) -> (u16, u16, u32, Vec<u8>) {
    let mut b = Vec::with_capacity(vals.len() * 8);
    for v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    (tag, 12, vals.len() as u32, b)
}

pub fn tag_rational_vec(tag: u16, vals: &[(u32, u32)]) -> (u16, u16, u32, Vec<u8>) {
    let mut b = Vec::with_capacity(vals.len() * 8);
    for &(n, d) in vals {
        b.extend_from_slice(&n.to_le_bytes());
        b.extend_from_slice(&d.to_le_bytes());
    }
    (tag, 5, vals.len() as u32, b)
}
