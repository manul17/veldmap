//! Format-dispatched lossless disk-cache helpers (`TileBlobEncoding::Qoi` for
//! `Rgba8UnormSrgb` display tiles; byte-shuffled `TileBlobEncoding::Lz4Raw` for
//! `R32Float` / `R8Uint` / `R16Uint` scientific & categorical tiles).

use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{DecodedTile, PixelFormat};
use veldmap_raster::{
    MAX_LZ4_RAW_DECODED_BYTES, MAX_QOI_DECODED_BYTES, decode_lz4_raw_tile, decode_qoi_rgba8,
    encode_lz4_raw_tile, encode_qoi_rgba8, generate_tile_mipmaps,
};
use veldmap_storage::{DiskTileKey, TileBlobEncoding, TileDiskCache};

/// Maximum tiles buffered before committing a single SQLite batch write during
/// `ServeMode::Pass`.
const PASS_BATCH_MAX_TILES: usize = 16;

/// Maximum encoded bytes buffered before flushing a batch write (`4 MiB`).
const PASS_BATCH_MAX_BYTES: usize = 4 * 1024 * 1024;

fn encoding_for_format(format: PixelFormat) -> TileBlobEncoding {
    match format {
        PixelFormat::Rgba8UnormSrgb => TileBlobEncoding::Qoi,
        PixelFormat::R32Float | PixelFormat::R8Uint | PixelFormat::R16Uint => {
            TileBlobEncoding::Lz4Raw
        }
    }
}

fn encode_tile_blob(tile: &DecodedTile) -> Option<(TileBlobEncoding, Vec<u8>)> {
    let res = match tile.format() {
        PixelFormat::Rgba8UnormSrgb => encode_qoi_rgba8(
            tile.width(),
            tile.height(),
            tile.data(),
            MAX_QOI_DECODED_BYTES,
        )
        .map(|bytes| (TileBlobEncoding::Qoi, bytes)),
        PixelFormat::R32Float | PixelFormat::R8Uint | PixelFormat::R16Uint => {
            encode_lz4_raw_tile(tile, MAX_LZ4_RAW_DECODED_BYTES)
                .map(|bytes| (TileBlobEncoding::Lz4Raw, bytes))
        }
    };
    match res {
        Ok(encoded) => Some(encoded),
        Err(err) => {
            tracing::warn!(
                format = ?tile.format(),
                width = tile.width(),
                height = tile.height(),
                error = %err,
                "Failed to encode overlay tile for disk cache"
            );
            None
        }
    }
}

/// Look up and decode a cached overlay tile (`Qoi` or `Lz4Raw`) from `disk`,
/// recording its debounced LRU touch ONLY after validation succeeds and
/// regenerating CPU mip chains (`Rgba8UnormSrgb` / `R32Float`).
pub(super) fn load_cached_tile(
    disk: &TileDiskCache,
    disk_key: &DiskTileKey,
) -> Option<DecodedTile> {
    let expected_encoding = encoding_for_format(disk_key.format);
    let (entry, _, payload) = disk
        .get_encoded_entry(disk_key, Some(expected_encoding))
        .ok()??;

    let decoded = match disk_key.format {
        PixelFormat::Rgba8UnormSrgb => {
            decode_qoi_rgba8(&payload, MAX_QOI_DECODED_BYTES).and_then(|(w, h, rgba)| {
                DecodedTile::try_new(w, h, PixelFormat::Rgba8UnormSrgb, rgba)
                    .map_err(|e| veldmap_raster::RasterError::Decode(e.to_string()))
            })
        }
        PixelFormat::R32Float | PixelFormat::R8Uint | PixelFormat::R16Uint => {
            decode_lz4_raw_tile(&payload, disk_key.format, MAX_LZ4_RAW_DECODED_BYTES)
        }
    };

    match decoded {
        Ok(mut tile) => {
            disk.touch_entry(entry);
            generate_tile_mipmaps(&mut tile);
            Some(tile)
        }
        Err(err) => {
            tracing::warn!(
                source_key = %disk_key.source_key,
                format = ?disk_key.format,
                level = disk_key.level,
                x = disk_key.x,
                y = disk_key.y,
                error = %err,
                "Purging corrupt cached overlay tile entry"
            );
            let _ = disk.delete_entry(entry);
            None
        }
    }
}

/// Encode and persist one mip-0 overlay tile (`Qoi` or `Lz4Raw`) into `disk`
/// unless caching is disabled or `cancel` has fired.
pub(super) fn store_cached_tile(
    disk: &TileDiskCache,
    disk_key: &DiskTileKey,
    tile: &DecodedTile,
    cancel: &CancelToken,
) {
    if disk.max_bytes() == 0 || cancel.is_cancelled() {
        return;
    }
    if let Some((encoding, encoded)) = encode_tile_blob(tile) {
        let _ = disk.put_encoded_if_uncancelled(disk_key, encoding, &encoded, cancel);
    }
}

/// Bounded write buffer amortizing SQLite WAL commits across sequential-pass
/// tile emissions (`Qoi` or `Lz4Raw`).
pub(super) struct TileWriteBatch {
    entries: Vec<(DiskTileKey, TileBlobEncoding, Vec<u8>)>,
    bytes: usize,
}

impl TileWriteBatch {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::with_capacity(PASS_BATCH_MAX_TILES),
            bytes: 0,
        }
    }

    pub(super) fn push_and_maybe_flush(
        &mut self,
        disk: &TileDiskCache,
        disk_key: DiskTileKey,
        tile: &DecodedTile,
        cancel: &CancelToken,
    ) {
        if disk.max_bytes() == 0 || cancel.is_cancelled() {
            return;
        }
        let Some((encoding, encoded)) = encode_tile_blob(tile) else {
            return;
        };
        self.bytes = self.bytes.saturating_add(encoded.len());
        self.entries.push((disk_key, encoding, encoded));
        if self.entries.len() >= PASS_BATCH_MAX_TILES || self.bytes >= PASS_BATCH_MAX_BYTES {
            self.flush(disk, cancel);
        }
    }

    pub(super) fn flush(&mut self, disk: &TileDiskCache, cancel: &CancelToken) {
        if self.entries.is_empty() {
            return;
        }
        if !cancel.is_cancelled() && disk.max_bytes() > 0 {
            let refs: Vec<(&DiskTileKey, TileBlobEncoding, &[u8])> = self
                .entries
                .iter()
                .map(|(k, enc, bytes)| (k, *enc, bytes.as_slice()))
                .collect();
            let _ = disk.put_encoded_batch_if_uncancelled(&refs, cancel);
        }
        self.clear();
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
}
