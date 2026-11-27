use veldmap_core::io::testing::{CountingByteSource, MemoryByteSource};

use super::super::is_overview;
use super::super::tag_ids::{
    TAG_COMPRESSION, TAG_GDAL_METADATA, TAG_IMAGE_WIDTH, TAG_MODEL_PIXEL_SCALE,
    TAG_NEW_SUBFILE_TYPE, TAG_TILE_OFFSETS,
};
use super::*;

fn put_u16(buf: &mut Vec<u8>, order: TiffByteOrder, v: u16) {
    match order {
        TiffByteOrder::Little => buf.extend_from_slice(&v.to_le_bytes()),
        TiffByteOrder::Big => buf.extend_from_slice(&v.to_be_bytes()),
    }
}

fn put_u32(buf: &mut Vec<u8>, order: TiffByteOrder, v: u32) {
    match order {
        TiffByteOrder::Little => buf.extend_from_slice(&v.to_le_bytes()),
        TiffByteOrder::Big => buf.extend_from_slice(&v.to_be_bytes()),
    }
}

fn put_u64(buf: &mut Vec<u8>, order: TiffByteOrder, v: u64) {
    match order {
        TiffByteOrder::Little => buf.extend_from_slice(&v.to_le_bytes()),
        TiffByteOrder::Big => buf.extend_from_slice(&v.to_be_bytes()),
    }
}

#[test]
fn rejects_cyclic_ifd_chains_and_truncated_or_invalid_bigtiff_headers() {
    let cancel = CancelToken::new();

    // Truncated BigTIFF header (12 bytes).
    let mut trunc = b"II".to_vec();
    put_u16(&mut trunc, TiffByteOrder::Little, 43);
    put_u16(&mut trunc, TiffByteOrder::Little, 8);
    put_u16(&mut trunc, TiffByteOrder::Little, 0);
    trunc.extend_from_slice(&[0u8; 4]);
    let err = SinglePassIfdReader::open(
        Arc::new(MemoryByteSource::new(trunc.into())),
        u64::MAX,
        &cancel,
    )
    .err()
    .expect("truncated BigTIFF must fail");
    assert!(matches!(err, RasterError::Decode(_)));

    // Bad BigTIFF offset_size / reserved.
    let mut bad_big = b"MM".to_vec();
    put_u16(&mut bad_big, TiffByteOrder::Big, 43);
    put_u16(&mut bad_big, TiffByteOrder::Big, 16);
    put_u16(&mut bad_big, TiffByteOrder::Big, 1);
    put_u64(&mut bad_big, TiffByteOrder::Big, 16);
    let err = SinglePassIfdReader::open(
        Arc::new(MemoryByteSource::new(bad_big.into())),
        u64::MAX,
        &cancel,
    )
    .err()
    .expect("bad BigTIFF header must fail");
    assert!(matches!(err, RasterError::Decode(_)));

    // Cyclic Classic TIFF: IFD0 at 8 -> IFD1 at 14 -> IFD0 at 8.
    let mut cyc = b"II".to_vec();
    put_u16(&mut cyc, TiffByteOrder::Little, 42);
    put_u32(&mut cyc, TiffByteOrder::Little, 8);
    // IFD0 at offset 8: 0 entries, next_ifd = 14
    put_u16(&mut cyc, TiffByteOrder::Little, 0);
    put_u32(&mut cyc, TiffByteOrder::Little, 14);
    // IFD1 at offset 14: 0 entries, next_ifd = 8
    put_u16(&mut cyc, TiffByteOrder::Little, 0);
    put_u32(&mut cyc, TiffByteOrder::Little, 8);
    let mut reader = SinglePassIfdReader::open(
        Arc::new(MemoryByteSource::new(cyc.into())),
        u64::MAX,
        &cancel,
    )
    .expect("header opens");
    let d0 = reader
        .read_directory(reader.first_ifd, &cancel)
        .expect("IFD0 reads");
    let d1 = reader
        .read_directory(d0.next_ifd(), &cancel)
        .expect("IFD1 reads");
    let err = reader
        .read_directory(d1.next_ifd(), &cancel)
        .err()
        .expect("cycle must fail");
    assert!(matches!(err, RasterError::Decode(_)));
}

#[test]
fn big_endian_bigtiff_reads_inline_out_of_prefix_and_float_tags_with_coalescing() {
    let cancel = CancelToken::new();
    let order = TiffByteOrder::Big;
    let ifd_off: u64 = (INITIAL_HEADER_PREFIX_BYTES as u64) + 512;
    let payload_a_off = ifd_off + 256;
    let payload_b_off = payload_a_off + 128; // within 16 KiB coalesce gap

    let mut data = vec![0u8; (payload_b_off + 32) as usize];
    // Header: Big-Endian BigTIFF pointing past 256 KiB prefix.
    let mut hdr = b"MM".to_vec();
    put_u16(&mut hdr, order, 43);
    put_u16(&mut hdr, order, 8);
    put_u16(&mut hdr, order, 0);
    put_u64(&mut hdr, order, ifd_off);
    data[..16].copy_from_slice(&hdr);

    // Build IFD with 4 entries:
    // 1. ImageWidth (inline u32 = 512, left-justified in BigTIFF 8-byte slot)
    // 2. ModelPixelScale (out-of-prefix FLOAT x3 at payload_a_off)
    // 3. TileOffsets (out-of-prefix LONG8 x2 at payload_b_off)
    // 4. Compression (unknown field_type = 99 -> preserved so strict read fails!)
    let mut ifd = Vec::new();
    put_u64(&mut ifd, order, 4);
    // Entry 1: ImageWidth
    put_u16(&mut ifd, order, TAG_IMAGE_WIDTH);
    put_u16(&mut ifd, order, 4); // LONG
    put_u64(&mut ifd, order, 1);
    put_u32(&mut ifd, order, 512);
    put_u32(&mut ifd, order, 0);
    // Entry 2: ModelPixelScale (FLOAT = 11, count = 3 -> 12 bytes > 8 inline)
    put_u16(&mut ifd, order, TAG_MODEL_PIXEL_SCALE);
    put_u16(&mut ifd, order, 11);
    put_u64(&mut ifd, order, 3);
    put_u64(&mut ifd, order, payload_a_off);
    // Entry 3: TileOffsets (LONG8 = 16, count = 2 -> 16 bytes)
    put_u16(&mut ifd, order, TAG_TILE_OFFSETS);
    put_u16(&mut ifd, order, 16);
    put_u64(&mut ifd, order, 2);
    put_u64(&mut ifd, order, payload_b_off);
    // Entry 4: Compression with invalid field_type 99 and huge crafted count
    put_u16(&mut ifd, order, TAG_COMPRESSION);
    put_u16(&mut ifd, order, 99);
    put_u64(&mut ifd, order, 10_000_000_000);
    put_u64(&mut ifd, order, 0);
    // next_ifd = 0
    put_u64(&mut ifd, order, 0);
    data[ifd_off as usize..ifd_off as usize + ifd.len()].copy_from_slice(&ifd);

    // Write FLOAT values [10.5, 20.25, 0.0] at payload_a_off
    let mut pa = Vec::new();
    for f in [10.5f32, 20.25, 0.0] {
        pa.extend_from_slice(&f.to_be_bytes());
    }
    data[payload_a_off as usize..payload_a_off as usize + pa.len()].copy_from_slice(&pa);

    // Write LONG8 values [1000, 2000] at payload_b_off
    let mut pb = Vec::new();
    put_u64(&mut pb, order, 1000);
    put_u64(&mut pb, order, 2000);
    data[payload_b_off as usize..payload_b_off as usize + pb.len()].copy_from_slice(&pb);

    let counting = Arc::new(CountingByteSource::new(Arc::new(MemoryByteSource::new(
        data.into(),
    ))));
    let mut reader = SinglePassIfdReader::open(
        Arc::clone(&counting) as Arc<dyn ByteSource>,
        u64::MAX,
        &cancel,
    )
    .expect("open BigTIFF");
    let mut dir = reader
        .read_directory(reader.first_ifd, &cancel)
        .expect("read out-of-prefix IFD");
    let reads_before = counting.snapshot().spans.len();
    dir.prefetch_tags(
        &mut reader,
        &[TAG_MODEL_PIXEL_SCALE, TAG_TILE_OFFSETS, TAG_COMPRESSION],
        1024 * 1024,
        &cancel,
    )
    .expect("coalesced prefetch");
    // Both out-of-prefix tags coalesced into a single positioned read.
    assert_eq!(counting.snapshot().spans.len() - reads_before, 1);

    assert_eq!(
        dir.read_u32_strict(&reader, TAG_IMAGE_WIDTH).unwrap(),
        Some(512)
    );
    assert_eq!(
        dir.read_f64_vec_opt(&reader, TAG_MODEL_PIXEL_SCALE),
        vec![10.5, 20.25, 0.0]
    );
    assert_eq!(
        dir.read_u64_vec_strict(&reader, TAG_TILE_OFFSETS).unwrap(),
        Some(vec![1000, 2000])
    );
    // Unknown field_type on strict tag must error (not default or OOM).
    assert!(matches!(
        dir.read_u16_strict(&reader, TAG_COMPRESSION),
        Err(RasterError::Decode(_))
    ));
}

#[test]
fn enforces_cumulative_fetch_budget_and_skips_mask_ifds_without_tag_reads() {
    let cancel = CancelToken::new();
    let order = TiffByteOrder::Little;
    let out_off: u64 = (INITIAL_HEADER_PREFIX_BYTES as u64) + 64;
    let mut data = vec![0u8; (out_off + 2048) as usize];

    // Classic TIFF: IFD0 at 8 is a mask IFD (NewSubfileType = 4) with a huge out-of-prefix tag.
    let mut hdr = b"II".to_vec();
    put_u16(&mut hdr, order, 42);
    put_u32(&mut hdr, order, 8);
    data[..8].copy_from_slice(&hdr);

    let mut ifd = Vec::new();
    put_u16(&mut ifd, order, 2);
    // NewSubfileType = 4 (mask)
    put_u16(&mut ifd, order, TAG_NEW_SUBFILE_TYPE);
    put_u16(&mut ifd, order, 4);
    put_u32(&mut ifd, order, 1);
    put_u32(&mut ifd, order, 4);
    // TileOffsets out-of-prefix (500 LONGs = 2000 bytes at out_off)
    put_u16(&mut ifd, order, TAG_TILE_OFFSETS);
    put_u16(&mut ifd, order, 4);
    put_u32(&mut ifd, order, 500);
    put_u32(&mut ifd, order, out_off as u32);
    // next_ifd = 0
    put_u32(&mut ifd, order, 0);
    data[8..8 + ifd.len()].copy_from_slice(&ifd);

    let counting = Arc::new(CountingByteSource::new(Arc::new(MemoryByteSource::new(
        data.into(),
    ))));
    let budget = (INITIAL_HEADER_PREFIX_BYTES as u64) + 1000;
    let mut reader = SinglePassIfdReader::open(
        Arc::clone(&counting) as Arc<dyn ByteSource>,
        budget,
        &cancel,
    )
    .expect("open within budget");
    assert_eq!(counting.snapshot().spans.len(), 1);
    let mut dir = reader
        .read_directory(reader.first_ifd, &cancel)
        .expect("IFD inside prefix");
    assert_eq!(counting.snapshot().spans.len(), 1);
    let subfile = dir.read_u32_opt(&reader, TAG_NEW_SUBFILE_TYPE).unwrap_or(0);
    assert!(!is_overview(subfile));
    assert_eq!(
        counting.snapshot().spans.len(),
        1,
        "mask check issued zero extra reads"
    );

    // Attempting to prefetch the 2000 B out-of-prefix tag exceeds the remaining 1000 B budget.
    let err = dir
        .prefetch_tags(&mut reader, &[TAG_TILE_OFFSETS], 4096, &cancel)
        .expect_err("must enforce cumulative reader budget");
    assert!(matches!(err, RasterError::DecoderLimitsExceeded(_)));
    assert_eq!(
        counting.snapshot().spans.len(),
        1,
        "rejected before issuing read_at"
    );
}

#[test]
fn read_ascii_opt_preserves_latin1_tags_lossily_without_blanking() {
    let cancel = CancelToken::new();
    let order = TiffByteOrder::Little;
    let payload: &[u8] = b"<GDALMetadata><Item name=\"NOTE\">20\xb0C</Item></GDALMetadata>\0";
    let payload_off: u32 = 32;
    let mut data = vec![0u8; payload_off as usize + payload.len()];

    let mut hdr = b"II".to_vec();
    put_u16(&mut hdr, order, 42);
    put_u32(&mut hdr, order, 8);
    data[..8].copy_from_slice(&hdr);

    let mut ifd = Vec::new();
    put_u16(&mut ifd, order, 1);
    put_u16(&mut ifd, order, TAG_GDAL_METADATA);
    put_u16(&mut ifd, order, 2); // ASCII
    put_u32(&mut ifd, order, payload.len() as u32);
    put_u32(&mut ifd, order, payload_off);
    put_u32(&mut ifd, order, 0);
    data[8..8 + ifd.len()].copy_from_slice(&ifd);
    data[payload_off as usize..payload_off as usize + payload.len()].copy_from_slice(payload);

    let mut reader = SinglePassIfdReader::open(
        Arc::new(MemoryByteSource::new(data.into())),
        u64::MAX,
        &cancel,
    )
    .expect("open");
    let dir = reader
        .read_directory(reader.first_ifd, &cancel)
        .expect("ifd");
    let decoded = dir.read_ascii_opt(&reader, TAG_GDAL_METADATA);
    assert!(decoded.contains("<GDALMetadata>"));
    assert!(decoded.contains("20\u{FFFD}C"));
    assert_eq!(
        dir.read_ascii_bytes_opt(&reader, TAG_GDAL_METADATA),
        &payload[..payload.len() - 1]
    );
}
