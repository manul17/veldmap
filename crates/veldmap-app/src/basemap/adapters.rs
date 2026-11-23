//! Adapters bridging `veldmap-storage` and `veldmap-raster` into `veldmap-net`'s
//! storage- and codec-agnostic traits (`TileBlobStore` and `TileDecodeExecutor`).

use std::sync::Arc;

use veldmap_core::sync::{CancelToken, Wakeup};
use veldmap_core::tile::QuadTileId;
use veldmap_net::{
    BlobStoreError, DecodeFailure, DecodeRequest, TileBlobStore, TileDecodeExecutor,
};
use veldmap_raster::{DecodeJob, DecodeJobKey, DecodePoolHandle, decode_map_tile_with_mipmaps};
use veldmap_storage::{DiskTileKey, StorageError, TileDiskCache};

use crate::gpu_tiles::UploadSender;

/// Adapter bridging `veldmap-storage`'s [`TileDiskCache`] into `veldmap-net`'s
/// storage-agnostic [`TileBlobStore`] trait.
pub(crate) struct SqliteBlobStore(pub(crate) Arc<TileDiskCache>);

impl SqliteBlobStore {
    fn disk_key(source_key: &str, id: QuadTileId) -> DiskTileKey {
        DiskTileKey::new(
            source_key,
            id.scheme.into_tile_scheme(),
            id.level,
            id.x,
            id.y,
        )
    }

    fn map_err(err: StorageError) -> BlobStoreError {
        if err.is_cancelled() {
            BlobStoreError::Cancelled
        } else {
            BlobStoreError::Backend(err.to_string())
        }
    }
}

impl TileBlobStore for SqliteBlobStore {
    fn get(&self, source_key: &str, id: QuadTileId) -> Result<Option<Vec<u8>>, BlobStoreError> {
        let key = Self::disk_key(source_key, id);
        self.0.get(&key).map_err(Self::map_err)
    }

    fn put(&self, source_key: &str, id: QuadTileId, payload: &[u8]) -> Result<(), BlobStoreError> {
        let key = Self::disk_key(source_key, id);
        self.0.put(&key, payload).map(|_| ()).map_err(Self::map_err)
    }

    fn delete(&self, source_key: &str, id: QuadTileId) -> Result<(), BlobStoreError> {
        let key = Self::disk_key(source_key, id);
        self.0.delete(&key).map(|_| ()).map_err(Self::map_err)
    }
}

/// Adapter bridging `veldmap-raster`'s generic [`DecodePoolHandle`] into
/// `veldmap-net`'s codec-agnostic [`TileDecodeExecutor`] trait.
pub(crate) struct RasterDecodeExecutor {
    pub(crate) pool: DecodePoolHandle,
    pub(crate) done_tx: UploadSender,
    pub(crate) wakeup: Wakeup,
    pub(crate) shutdown: CancelToken,
}

impl TileDecodeExecutor for RasterDecodeExecutor {
    fn submit(&self, req: DecodeRequest) -> bool {
        let key = req.key;
        let queued_bytes = req.bytes.len();
        let cancel = req.cancel.clone();
        let done_tx = self.done_tx.clone();
        let wakeup = Arc::clone(&self.wakeup);
        let shutdown = self.shutdown.clone();
        let bytes = req.bytes;
        let on_decoded = req.on_decoded;

        self.pool.submit(DecodeJob {
            key: DecodeJobKey::Tile(key),
            queued_bytes,
            cancel: req.cancel,
            work: Box::new(move || match decode_map_tile_with_mipmaps(&bytes) {
                Ok(tile) => {
                    if cancel.is_cancelled() || shutdown.is_cancelled() {
                        on_decoded.complete(Err(DecodeFailure::Aborted));
                        return;
                    }
                    if done_tx.send((key, tile)).is_err() {
                        on_decoded.complete(Err(DecodeFailure::Aborted));
                        return;
                    }
                    on_decoded.complete(Ok(()));
                    (wakeup)();
                }
                Err(err) => {
                    tracing::debug!(
                        ?key,
                        error = %err,
                        "Failed to decode map tile payload"
                    );
                    on_decoded.complete(Err(DecodeFailure::InvalidPayload(err.to_string())));
                }
            }),
        })
    }
}
