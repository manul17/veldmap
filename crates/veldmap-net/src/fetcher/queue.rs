//! Coarsest-first deduplicated `TileKey` priority queue backed by a
//! reverse-sorted `Vec` so `pop()` yields the coarsest key in O(1).

use std::cmp::Reverse;
use std::collections::HashSet;

use veldmap_core::tile::TileKey;

#[derive(Default)]
pub(super) struct CoarsestTileQueue {
    items: Vec<TileKey>,
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
        self.items.pop()
    }

    pub(super) fn contains(&self, key: &TileKey) -> bool {
        let sort_key = Reverse(key.coarsest_first_key());
        self.items
            .binary_search_by_key(&sort_key, |k| Reverse(k.coarsest_first_key()))
            .is_ok()
    }

    /// Insert a single `TileKey` into the reverse-sorted queue without
    /// duplicating entries or re-sorting the entire `Vec`.
    pub(super) fn insert_unique(&mut self, key: TileKey) -> bool {
        let sort_key = Reverse(key.coarsest_first_key());
        match self
            .items
            .binary_search_by_key(&sort_key, |k| Reverse(k.coarsest_first_key()))
        {
            Ok(_) => false,
            Err(idx) => {
                self.items.insert(idx, key);
                true
            }
        }
    }

    /// Batch-insert keys (skipping keys already present) and restore
    /// coarsest-at-end ordering.
    pub(super) fn push_unique(&mut self, keys: impl IntoIterator<Item = TileKey>) -> bool {
        let mut iter = keys.into_iter().peekable();
        if iter.peek().is_none() {
            return false;
        }
        let mut existing: HashSet<TileKey> = self.items.iter().copied().collect();
        let mut added = false;
        for key in iter {
            if existing.insert(key) {
                self.items.push(key);
                added = true;
            }
        }
        if added {
            Self::sort_coarsest_at_end(&mut self.items);
        }
        added
    }

    pub(super) fn replace_sorted(&mut self, mut keys: Vec<TileKey>) {
        Self::sort_coarsest_at_end(&mut keys);
        self.items = keys;
    }

    pub(super) fn retain_and_collect(
        &mut self,
        mut keep: impl FnMut(&TileKey) -> bool,
    ) -> HashSet<TileKey> {
        let mut retained = HashSet::new();
        self.items.retain(|key| {
            let ok = keep(key);
            if ok {
                retained.insert(*key);
            }
            ok
        });
        retained
    }

    pub(super) fn restore_deferred(&mut self, deferred: Vec<TileKey>, multi_host: bool) {
        if deferred.is_empty() {
            return;
        }
        // `deferred` was popped coarsest-first from the back of the queue;
        // reversing restores ascending-priority order without a full sort for
        // single-host short-circuits, and sorting when mixed.
        if multi_host {
            self.items.extend(deferred);
            Self::sort_coarsest_at_end(&mut self.items);
        } else {
            self.items.extend(deferred.into_iter().rev());
        }
    }

    pub(super) fn drain(&mut self) -> Vec<TileKey> {
        self.items.drain(..).collect()
    }

    fn sort_coarsest_at_end(queue: &mut [TileKey]) {
        queue.sort_by_key(|k| Reverse(k.coarsest_first_key()));
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
    fn insert_unique_maintains_coarsest_at_end_order() {
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
        assert_eq!(queue.pop(), Some(k0));
        assert_eq!(queue.pop(), Some(k2));
        assert_eq!(queue.pop(), Some(k1));
        assert_eq!(queue.pop(), None);
    }
}
