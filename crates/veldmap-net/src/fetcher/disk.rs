//! Bounded async disk-cache lookup, corrupt-entry deletion, and best-effort
//! write persistence for [`crate::LiveTilePipeline`].

use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{QuadTileId, TileKey};

use crate::basemap::LiveBasemapSpec;
use crate::{BlobLease, BlobStoreError, TileBlobStore};

/// Maximum concurrent blocking disk-cache write/delete tasks (`cache.put()` /
/// `cache.delete()`) so burst completions serialize cleanly without piling up
/// contending Tokio blocking threads on a single-writer backend.
const MAX_CONCURRENT_DISK_WRITES: usize = 1;

/// Maximum total pending + active post-decode disk-cache write/delete tasks so
/// slow disk writes never accumulate unbounded queued tasks and payloads.
const MAX_QUEUED_DISK_WRITES: usize = 32;

/// Outcome of a bounded blocking disk-cache read.
pub(super) enum DiskReadOutcome {
    Hit(Vec<u8>, BlobLease),
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
    ///
    /// Moves `disk_slot` into `spawn_blocking` so cancelling mid-read never
    /// releases the disk-lookup concurrency slot while the blocking SQLite
    /// `store.get()` call is still executing.
    pub(super) async fn get(
        &self,
        spec: &Arc<LiveBasemapSpec>,
        id: QuadTileId,
        cancel: &CancelToken,
        disk_slot: impl Send + 'static,
    ) -> DiskReadOutcome {
        if cancel.is_cancelled() {
            return DiskReadOutcome::Cancelled;
        }
        let store = Arc::clone(&self.store);
        let spec = Arc::clone(spec);
        let cancel_for_get = cancel.clone();
        let res = tokio::select! {
            _ = cancel.cancelled() => return DiskReadOutcome::Cancelled,
            res = tokio::task::spawn_blocking(move || {
                let _disk_slot = disk_slot;
                if cancel_for_get.is_cancelled() {
                    return DiskReadOutcome::Cancelled;
                }
                match store.get(spec.cache_key(), id) {
                    Ok(Some((bytes, lease))) => DiskReadOutcome::Hit(bytes, lease),
                    Ok(None) => DiskReadOutcome::Miss,
                    Err(BlobStoreError::Backend(msg)) => DiskReadOutcome::Error(msg),
                }
            }) => res,
        };
        if cancel.is_cancelled() {
            return DiskReadOutcome::Cancelled;
        }
        match res {
            Ok(outcome) => outcome,
            Err(join_err) => DiskReadOutcome::Error(format!("blocking task failed: {join_err}")),
        }
    }

    async fn run_bounded_write(
        store: Arc<dyn TileBlobStore>,
        put_sem: Arc<Semaphore>,
        queue_permit: OwnedSemaphorePermit,
        cancel: CancelToken,
        key: TileKey,
        op_label: &'static str,
        op: impl FnOnce(&dyn TileBlobStore, &CancelToken) -> Result<(), BlobStoreError> + Send + 'static,
    ) {
        let put_permit = tokio::select! {
            _ = cancel.cancelled() => return,
            res = put_sem.acquire_owned() => match res {
                Ok(p) => p,
                Err(_) => return,
            },
        };
        let cancel_for_write = cancel.clone();
        let res = tokio::select! {
            _ = cancel.cancelled() => return,
            res = tokio::task::spawn_blocking(move || {
                let _queue_permit = queue_permit;
                let _put_permit = put_permit;
                if cancel_for_write.is_cancelled() {
                    return Ok(());
                }
                op(store.as_ref(), &cancel_for_write)
            }) => res,
        };
        match res {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                tracing::debug!(?key, error = %err, op = op_label, "Disk cache write failed");
            }
            Err(join_err) => {
                tracing::debug!(
                    ?key,
                    error = %join_err,
                    op = op_label,
                    "Disk cache write blocking task failed"
                );
            }
        }
    }

    /// Record an LRU recency touch for a decode-validated [`BlobLease`] on the
    /// blocking thread pool so opportunistic touch flushes never stall Tokio
    /// async worker threads.
    pub(super) fn touch_async(&self, lease: BlobLease, shutdown: &CancelToken) {
        if shutdown.is_cancelled() {
            return;
        }
        let store = Arc::clone(&self.store);
        let shutdown = shutdown.clone();
        tokio::task::spawn_blocking(move || {
            if !shutdown.is_cancelled() {
                store.touch(lease);
            }
        });
    }

    /// Best-effort bounded deletion of a corrupt cached tile entry represented
    /// by `lease`.
    ///
    /// Both `queue_permit` and `put_permit` are moved into `spawn_blocking` so
    /// cancellation mid-delete never drops `queue_permit` early while the
    /// blocking SQLite call is still executing.
    pub(super) async fn delete_corrupt(
        &self,
        lease: BlobLease,
        key: TileKey,
        cancel: &CancelToken,
    ) {
        if cancel.is_cancelled() {
            return;
        }
        let Ok(queue_permit) = Arc::clone(&self.put_queue_sem).try_acquire_owned() else {
            tracing::debug!(
                ?key,
                "Skipping corrupt disk tile deletion under write-queue backpressure"
            );
            return;
        };
        Self::run_bounded_write(
            Arc::clone(&self.store),
            Arc::clone(&self.put_sem),
            queue_permit,
            cancel.clone(),
            key,
            "delete_corrupt",
            move |store, _| store.delete(lease),
        )
        .await;
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
        tokio::spawn(Self::run_bounded_write(
            store,
            put_sem,
            queue_permit,
            shutdown,
            key,
            "persist_tile",
            move |store, cancel| store.put(spec.cache_key(), id, &bytes, cancel),
        ));
    }
}
