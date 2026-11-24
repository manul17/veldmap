//! Bounded async disk-cache lookup, corrupt-entry deletion, and best-effort
//! write persistence for [`super::LiveTilePipeline`].

use std::sync::Arc;

use tokio::sync::Semaphore;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{QuadTileId, TileKey};

use crate::basemap::LiveBasemapSpec;
use crate::{BlobStoreError, TileBlobStore};

/// Maximum concurrent blocking disk-cache write/delete tasks (`cache.put()` /
/// `cache.delete()`) so burst completions serialize cleanly without piling up
/// contending Tokio blocking threads on a single-writer backend.
const MAX_CONCURRENT_DISK_WRITES: usize = 1;

/// Maximum total pending + active post-decode disk-cache write/delete tasks so
/// slow disk writes never accumulate unbounded queued tasks and payloads.
const MAX_QUEUED_DISK_WRITES: usize = 32;

/// Outcome of a bounded blocking disk-cache read.
pub(super) enum DiskReadOutcome {
    Hit(Vec<u8>),
    Miss,
    Cancelled,
    Error(String),
}

/// Encapsulates the optional [`TileBlobStore`] together with its bounded
/// write/delete backpressure semaphores so non-disk pipelines allocate zero
/// write-gate semaphores and both permits always stay held until the blocking
/// call completes.
#[derive(Clone)]
pub(super) struct DiskCacheGate {
    store: Arc<dyn TileBlobStore>,
    put_queue_sem: Arc<Semaphore>,
    put_sem: Arc<Semaphore>,
}

impl DiskCacheGate {
    pub(super) fn new(store: Arc<dyn TileBlobStore>) -> Self {
        Self {
            store,
            put_queue_sem: Arc::new(Semaphore::new(MAX_QUEUED_DISK_WRITES)),
            put_sem: Arc::new(Semaphore::new(MAX_CONCURRENT_DISK_WRITES)),
        }
    }

    /// Perform a cancellable blocking lookup for `(spec.cache_key(), id)`.
    pub(super) async fn get(
        &self,
        spec: &Arc<LiveBasemapSpec>,
        id: QuadTileId,
        cancel: &CancelToken,
    ) -> DiskReadOutcome {
        let store = Arc::clone(&self.store);
        let spec = Arc::clone(spec);
        let res = tokio::select! {
            _ = cancel.cancelled() => return DiskReadOutcome::Cancelled,
            res = tokio::task::spawn_blocking(move || store.get(spec.cache_key(), id)) => res,
        };
        if cancel.is_cancelled() {
            return DiskReadOutcome::Cancelled;
        }
        match res {
            Ok(Ok(Some(bytes))) => DiskReadOutcome::Hit(bytes),
            Ok(Ok(None)) => DiskReadOutcome::Miss,
            Ok(Err(BlobStoreError::Cancelled)) => DiskReadOutcome::Cancelled,
            Ok(Err(BlobStoreError::Backend(msg))) => DiskReadOutcome::Error(msg),
            Err(join_err) => DiskReadOutcome::Error(format!("blocking task failed: {join_err}")),
        }
    }

    /// Best-effort bounded deletion of a corrupt cached tile entry.
    ///
    /// Both `queue_permit` and `put_permit` are moved into `spawn_blocking` so
    /// cancellation mid-delete never drops `queue_permit` early while the
    /// blocking SQLite call is still executing.
    pub(super) async fn delete_corrupt(
        &self,
        spec: Arc<LiveBasemapSpec>,
        key: TileKey,
        id: QuadTileId,
        cancel: &CancelToken,
        shutdown: &CancelToken,
    ) {
        if cancel.is_cancelled() || shutdown.is_cancelled() {
            return;
        }
        let Ok(queue_permit) = Arc::clone(&self.put_queue_sem).try_acquire_owned() else {
            tracing::debug!(
                ?key,
                "Skipping corrupt disk tile deletion under write-queue backpressure"
            );
            return;
        };
        let put_permit = tokio::select! {
            _ = cancel.cancelled() => return,
            _ = shutdown.cancelled() => return,
            res = Arc::clone(&self.put_sem).acquire_owned() => match res {
                Ok(p) => p,
                Err(_) => return,
            },
        };
        let store = Arc::clone(&self.store);
        let cancel_for_delete = cancel.clone();
        let shutdown_for_delete = shutdown.clone();
        let del_res = tokio::select! {
            _ = cancel.cancelled() => return,
            _ = shutdown.cancelled() => return,
            res = tokio::task::spawn_blocking(move || {
                let _queue_permit = queue_permit;
                let _put_permit = put_permit;
                if cancel_for_delete.is_cancelled() || shutdown_for_delete.is_cancelled() {
                    return Ok(());
                }
                store.delete(spec.cache_key(), id)
            }) => res,
        };
        match del_res {
            Ok(Ok(())) | Ok(Err(BlobStoreError::Cancelled)) => {}
            Ok(Err(del_err)) => {
                tracing::debug!(
                    ?key,
                    error = %del_err,
                    "Failed to delete corrupt cached tile entry"
                );
            }
            Err(join_err) => {
                tracing::debug!(
                    ?key,
                    error = %join_err,
                    "Corrupt cached tile deletion blocking task failed"
                );
            }
        }
    }

    /// Spawn a bounded best-effort background write persisting decode-validated
    /// bytes to disk.
    pub(super) fn persist_async(
        &self,
        spec: Arc<LiveBasemapSpec>,
        key: TileKey,
        id: QuadTileId,
        bytes: Arc<[u8]>,
        shutdown: CancelToken,
    ) {
        if shutdown.is_cancelled() {
            return;
        }
        let Ok(queue_permit) = Arc::clone(&self.put_queue_sem).try_acquire_owned() else {
            tracing::debug!(
                ?key,
                "Skipping best-effort tile disk write under write-queue backpressure"
            );
            return;
        };
        let store = Arc::clone(&self.store);
        let put_sem = Arc::clone(&self.put_sem);
        tokio::spawn(async move {
            let put_permit = tokio::select! {
                _ = shutdown.cancelled() => return,
                res = put_sem.acquire_owned() => match res {
                    Ok(p) => p,
                    Err(_) => return,
                },
            };
            let _ = tokio::task::spawn_blocking(move || {
                let _queue_permit = queue_permit;
                let _put_permit = put_permit;
                if shutdown.is_cancelled() {
                    return;
                }
                match store.put(spec.cache_key(), id, &bytes) {
                    Ok(()) | Err(BlobStoreError::Cancelled) => {}
                    Err(err) => {
                        tracing::debug!(
                            error = %err,
                            "Failed to persist validated live tile to disk cache"
                        );
                    }
                }
            })
            .await;
        });
    }
}
