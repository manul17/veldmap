//! Lossless QOI RGBA8 display-tile disk-cache helpers (`load_qoi_display_tile`,
//! `store_qoi_display_tile`, and batched sequential-pass writer [`QoiWriteBatch`]).

use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{DecodedTile, PixelFormat};
use veldmap_raster::{
    MAX_QOI_DECODED_BYTES, decode_qoi_rgba8, encode_qoi_rgba8, generate_linear_mipmaps,
};
use veldmap_storage::{DiskTileKey, TileBlobEncoding, TileDiskCache};

/// Maximum QOI tiles buffered before committing a single SQLite batch write
/// during `ServeMode::Pass`.
const PASS_QOI_BATCH_MAX_TILES: usize = 16;

/// Maximum encoded QOI bytes buffered before flushing a batch write (`4 MiB`).
const PASS_QOI_BATCH_MAX_BYTES: usize = 4 * 1024 * 1024;

/// Look up and decode a lossless QOI RGBA8 tile from `disk`, recording its
/// debounced LRU touch ONLY after validation succeeds and purging corrupt rows
/// safely by `(tile_id, row_last_used_ms)`.
pub(super) fn load_qoi_display_tile(
    disk: &TileDiskCache,
    disk_key: &DiskTileKey,
) -> Option<DecodedTile> {
    let (entry, _, payload) = disk
        .get_encoded_entry(disk_key, Some(TileBlobEncoding::Qoi))
        .ok()??;
    match decode_qoi_rgba8(&payload, MAX_QOI_DECODED_BYTES) {
        Ok((w, h, rgba)) => {
            if let Ok(mut tile) = DecodedTile::try_new(w, h, PixelFormat::Rgba8UnormSrgb, rgba) {
                disk.touch_entry(entry);
                generate_linear_mipmaps(&mut tile);
                Some(tile)
            } else {
                let _ = disk.delete_entry(entry);
                None
            }
        }
        Err(err) => {
            tracing::warn!(
                source_key = %disk_key.source_key,
                level = disk_key.level,
                x = disk_key.x,
                y = disk_key.y,
                error = %err,
                "Purging corrupt QOI disk tile entry"
            );
            let _ = disk.delete_entry(entry);
            None
        }
    }
}

/// Encode and persist one RGBA8 display tile into `disk` unless caching is
/// disabled or `cancel` has fired.
pub(super) fn store_qoi_display_tile(
    disk: &TileDiskCache,
    disk_key: &DiskTileKey,
    width: u32,
    height: u32,
    rgba: &[u8],
    cancel: &CancelToken,
) {
    if disk.max_bytes() == 0 || cancel.is_cancelled() {
        return;
    }
    if let Ok(encoded) = encode_qoi_rgba8(width, height, rgba, MAX_QOI_DECODED_BYTES) {
        let _ = disk.put_encoded_if_uncancelled(disk_key, TileBlobEncoding::Qoi, &encoded, cancel);
    }
}

/// Bounded write buffer amortizing SQLite WAL commits across sequential-pass
/// QOI tile emissions.
pub(super) struct QoiWriteBatch {
    entries: Vec<(DiskTileKey, Vec<u8>)>,
    bytes: usize,
}

impl QoiWriteBatch {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::with_capacity(PASS_QOI_BATCH_MAX_TILES),
            bytes: 0,
        }
    }

    pub(super) fn push_and_maybe_flush(
        &mut self,
        disk: &TileDiskCache,
        disk_key: DiskTileKey,
        width: u32,
        height: u32,
        rgba: &[u8],
        cancel: &CancelToken,
    ) {
        if disk.max_bytes() == 0 || cancel.is_cancelled() {
            return;
        }
        let Ok(encoded) = encode_qoi_rgba8(width, height, rgba, MAX_QOI_DECODED_BYTES) else {
            return;
        };
        self.bytes = self.bytes.saturating_add(encoded.len());
        self.entries.push((disk_key, encoded));
        if self.entries.len() >= PASS_QOI_BATCH_MAX_TILES || self.bytes >= PASS_QOI_BATCH_MAX_BYTES
        {
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
                .map(|(k, bytes)| (k, TileBlobEncoding::Qoi, bytes.as_slice()))
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
