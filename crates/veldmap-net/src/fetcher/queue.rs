//! Coarsest-first deduplicated `TileKey` priority queue backed by a `BTreeSet`
//! for O(log N) insertion, deduplication, lookup, in-place retain, and
//! coarsest-first pop.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use veldmap_core::tile::TileKey;

/// Total-ordering wrapper over [`TileKey`] ordered strictly coarsest-first
/// via [`TileKey::coarsest_first_key`] (which is injective with `Eq`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OrderedTileKey(TileKey);

impl Ord for OrderedTileKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .coarsest_first_key()
            .cmp(&other.0.coarsest_first_key())
    }
}

impl PartialOrd for OrderedTileKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
struct CoarsestTileQueue {
    items: BTreeSet<OrderedTileKey>,
}

impl CoarsestTileQueue {
    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn clear(&mut self) {
        self.items.clear();
    }

    fn pop(&mut self) -> Option<TileKey> {
        self.items.pop_first().map(|entry| entry.0)
    }

    fn contains(&self, key: &TileKey) -> bool {
        self.items.contains(&OrderedTileKey(*key))
    }

    /// Insert a single `TileKey` into the queue without duplicating entries.
    fn insert_unique(&mut self, key: TileKey) -> bool {
        self.items.insert(OrderedTileKey(key))
    }

    fn remove(&mut self, key: &TileKey) -> bool {
        self.items.remove(&OrderedTileKey(*key))
    }

    /// Retain only entries matching `keep` in place with zero auxiliary
    /// allocations.
    fn retain(&mut self, mut keep: impl FnMut(&TileKey) -> bool) {
        self.items.retain(|entry| keep(&entry.0));
    }

    fn drain(&mut self) -> impl Iterator<Item = TileKey> + use<> {
        std::mem::take(&mut self.items)
            .into_iter()
            .map(|entry| entry.0)
    }
}

/// Disjoint two-stage coarsest-first pending queues structurally maintaining
/// `disk ∩ http = ∅` so disk misses never bounce back to disk lookup.
#[derive(Default)]
pub(super) struct PendingQueues {
    /// Coarsest-first queue awaiting fast local disk-cache lookup.
    disk: CoarsestTileQueue,
    /// Coarsest-first queue of disk misses awaiting an HTTP per-host slot.
    http: CoarsestTileQueue,
}

impl PendingQueues {
    pub(super) fn clear(&mut self) {
        self.disk.clear();
        self.http.clear();
    }

    pub(super) fn len(&self) -> usize {
        self.disk.len().saturating_add(self.http.len())
    }

    pub(super) fn contains(&self, key: &TileKey) -> bool {
        self.disk.contains(key) || self.http.contains(key)
    }

    /// Enqueue `key` for disk lookup unless it is already awaiting HTTP
    /// (preserving `disk ∩ http = ∅` so disk misses never bounce back).
    pub(super) fn insert_disk(&mut self, key: TileKey) -> bool {
        if self.http.contains(&key) {
            return false;
        }
        self.disk.insert_unique(key)
    }

    /// Enqueue `key` for HTTP fetch, removing any stale entry in `disk` so
    /// `disk ∩ http = ∅` always holds.
    pub(super) fn insert_http(&mut self, key: TileKey) -> bool {
        self.disk.remove(&key);
        self.http.insert_unique(key)
    }

    pub(super) fn insert_unique(&mut self, key: TileKey, retry_disk: bool) {
        if retry_disk {
            self.insert_disk(key);
        } else {
            self.insert_http(key);
        }
    }

    pub(super) fn push_http(&mut self, keys: impl IntoIterator<Item = TileKey>) {
        for key in keys {
            self.insert_http(key);
        }
    }

    pub(super) fn pop_disk(&mut self) -> Option<TileKey> {
        self.disk.pop()
    }

    pub(super) fn pop_http(&mut self) -> Option<TileKey> {
        self.http.pop()
    }

    /// Move all eligible entries from `disk` into `http`, leaving `disk` empty.
    pub(super) fn drain_disk_to_http(&mut self, mut keep: impl FnMut(&TileKey) -> bool) {
        if self.disk.is_empty() {
            return;
        }
        for key in self.disk.drain() {
            if keep(&key) {
                self.http.insert_unique(key);
            }
        }
    }

    /// Retain still-wanted / active-root entries in place across both stages.
    pub(super) fn retain(&mut self, mut keep: impl FnMut(&TileKey) -> bool) {
        self.http.retain(&mut keep);
        self.disk.retain(&mut keep);
    }

    #[cfg(test)]
    pub(super) fn contains_disk(&self, key: &TileKey) -> bool {
        self.disk.contains(key)
    }

    #[cfg(test)]
    pub(super) fn contains_http(&self, key: &TileKey) -> bool {
        self.http.contains(key)
    }

    #[cfg(test)]
    pub(super) fn is_disk_empty(&self) -> bool {
        self.disk.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basemap::LiveBasemapSpec;

    #[test]
    fn insert_unique_and_retain_maintain_coarsest_first_order() {
        let spec = LiveBasemapSpec::eox_sentinel2_cloudless();
        let mut queue = CoarsestTileQueue::default();
        let k0 = spec.wm_key(0, 0, 0);
        let k1 = spec.wm_key(2, 1, 1);
        let k2 = spec.wm_key(1, 0, 0);

        assert!(queue.insert_unique(k1));
        assert!(queue.insert_unique(k0));
        assert!(queue.insert_unique(k2));
        assert!(!queue.insert_unique(k1));
        assert!(queue.contains(&k2));

        queue.retain(|k| *k != k2);
        assert!(!queue.contains(&k2));
        assert_eq!(queue.pop(), Some(k0));
        assert_eq!(queue.pop(), Some(k1));
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn pending_queues_enforces_disk_http_disjointness() {
        let spec = LiveBasemapSpec::eox_sentinel2_cloudless();
        let mut pending = PendingQueues::default();
        let k0 = spec.wm_key(1, 0, 0);
        let k1 = spec.wm_key(1, 1, 0);

        assert!(pending.insert_disk(k0));
        assert!(pending.contains_disk(&k0));
        assert!(!pending.contains_http(&k0));

        // Promoting to HTTP removes from disk and prevents bouncing back to disk.
        assert!(pending.insert_http(k0));
        assert!(!pending.contains_disk(&k0));
        assert!(pending.contains_http(&k0));
        assert!(!pending.insert_disk(k0));
        assert!(!pending.contains_disk(&k0));

        // Batch HTTP push also evicts any matching disk entry.
        assert!(pending.insert_disk(k1));
        pending.push_http([k1]);
        assert!(!pending.contains_disk(&k1));
        assert!(pending.contains_http(&k1));
        assert_eq!(pending.len(), 2);
    }
}
