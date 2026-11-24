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
pub(super) struct CoarsestTileQueue {
    items: BTreeSet<OrderedTileKey>,
}

impl CoarsestTileQueue {
    pub(super) fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub(super) fn len(&self) -> usize {
        self.items.len()
    }

    pub(super) fn clear(&mut self) {
        self.items.clear();
    }

    pub(super) fn pop(&mut self) -> Option<TileKey> {
        self.items.pop_first().map(|entry| entry.0)
    }

    pub(super) fn contains(&self, key: &TileKey) -> bool {
        self.items.contains(&OrderedTileKey(*key))
    }

    /// Insert a single `TileKey` into the queue without duplicating entries.
    pub(super) fn insert_unique(&mut self, key: TileKey) -> bool {
        self.items.insert(OrderedTileKey(key))
    }

    /// Batch-insert keys (skipping keys already present).
    pub(super) fn push_unique(&mut self, keys: impl IntoIterator<Item = TileKey>) {
        self.items.extend(keys.into_iter().map(OrderedTileKey));
    }

    /// Retain only entries matching `keep` in place with zero auxiliary
    /// allocations.
    pub(super) fn retain(&mut self, mut keep: impl FnMut(&TileKey) -> bool) {
        self.items.retain(|entry| keep(&entry.0));
    }

    pub(super) fn drain(&mut self) -> impl Iterator<Item = TileKey> + use<> {
        std::mem::take(&mut self.items)
            .into_iter()
            .map(|entry| entry.0)
    }
}

#[derive(Default)]
pub(super) struct PendingQueues {
    /// Coarsest-first queue awaiting fast local disk-cache lookup.
    pub(super) disk: CoarsestTileQueue,
    /// Coarsest-first queue of disk misses awaiting an HTTP per-host slot.
    pub(super) http: CoarsestTileQueue,
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

    pub(super) fn insert_unique(&mut self, key: TileKey, retry_disk: bool) {
        if retry_disk {
            self.disk.insert_unique(key);
        } else {
            self.http.insert_unique(key);
        }
    }

    /// Retain still-wanted / active-root entries in place, keeping `http`
    /// entries out of `disk` so disk misses never bounce back.
    pub(super) fn retain(&mut self, mut keep: impl FnMut(&TileKey) -> bool) {
        self.http.retain(&mut keep);
        let http = &self.http;
        self.disk.retain(|key| !http.contains(key) && keep(key));
    }
}

#[cfg(test)]
mod tests {
    use veldmap_core::tile::{PlanetaryScheme, QuadTileId};

    use super::*;
    use crate::basemap::LiveBasemapSpec;

    fn wm_key(spec: &LiveBasemapSpec, level: u8, x: u32, y: u32) -> TileKey {
        spec.tile_key(QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level,
            x,
            y,
        })
    }

    #[test]
    fn insert_unique_and_retain_maintain_coarsest_first_order() {
        let spec = LiveBasemapSpec::eox_sentinel2_cloudless();
        let mut queue = CoarsestTileQueue::default();
        let k0 = wm_key(&spec, 0, 0, 0);
        let k1 = wm_key(&spec, 2, 1, 1);
        let k2 = wm_key(&spec, 1, 0, 0);

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
}
