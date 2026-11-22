//! Pure O(1) byte-budgeted LRU tile cache with scheme-aware ancestor touching
//! and `< 50%` per-pyramid appetite capping.

use std::collections::HashMap;
use std::hash::Hash;

use super::key::TileKey;

struct Node<K, V> {
    key: K,
    value: V,
    byte_size: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Pure O(1) doubly-linked intrusive slab LRU cache bounded by `max_bytes`.
///
/// Evictions return the evicted `(K, V)` entries so callers can inspect or
/// drop released resources deterministically.
pub struct TileCache<K: Copy + Eq + Hash, V> {
    nodes: Vec<Option<Node<K, V>>>,
    free: Vec<usize>,
    index: HashMap<K, usize>,
    head: Option<usize>, // Most recently used
    tail: Option<usize>, // Least recently used
    total_bytes: usize,
    max_bytes: usize,
}

impl<K: Copy + Eq + Hash, V> TileCache<K, V> {
    /// Create an empty cache capped at `max_bytes` bytes.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            nodes: Vec::new(),
            free: Vec::new(),
            index: HashMap::new(),
            head: None,
            tail: None,
            total_bytes: 0,
            max_bytes,
        }
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Total accounted bytes across all cached entries.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Configured maximum byte budget.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Raise the configured maximum byte budget to at least `min_bytes`.
    pub fn ensure_max_bytes(&mut self, min_bytes: usize) {
        if min_bytes > self.max_bytes {
            self.max_bytes = min_bytes;
        }
    }

    /// Return whether `key` is currently cached (without updating recency).
    pub fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    /// Read-only lookup without modifying LRU order.
    pub fn peek(&self, key: &K) -> Option<&V> {
        let &slot = self.index.get(key)?;
        self.nodes[slot].as_ref().map(|n| &n.value)
    }

    /// Mark `key` as most recently used in O(1). Returns `true` if present.
    pub fn touch(&mut self, key: &K) -> bool {
        let Some(&slot) = self.index.get(key) else {
            return false;
        };
        self.move_to_front(slot);
        true
    }

    /// Look up `key` and mark it as most recently used in O(1).
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let &slot = self.index.get(key)?;
        self.move_to_front(slot);
        self.nodes[slot].as_ref().map(|n| &n.value)
    }

    /// Insert or replace `key` with `value` occupying `byte_size` bytes,
    /// evicting least-recently-used entries until `total_bytes() <= max_bytes()`.
    ///
    /// Returns `Ok(evicted)` containing all evicted `(K, V)` pairs (including
    /// any replaced previous value for `key`), or `Err((key, value))` without
    /// modifying existing cache contents if `max_bytes == 0` or
    /// `byte_size > max_bytes`.
    pub fn insert(&mut self, key: K, value: V, byte_size: usize) -> Result<Vec<(K, V)>, (K, V)> {
        self.insert_protecting(key, value, byte_size, |_| false)
    }

    /// Return whether inserting or replacing `key` of `byte_size` can succeed
    /// within `max_bytes` without evicting any entry where `is_protected(&k)`
    /// returns `true`.
    pub fn can_admit_protecting<F>(&self, key: &K, byte_size: usize, is_protected: F) -> bool
    where
        F: Fn(&K) -> bool,
    {
        if self.max_bytes == 0 || byte_size > self.max_bytes {
            return false;
        }
        let existing_bytes = self
            .index
            .get(key)
            .and_then(|&slot| self.nodes[slot].as_ref())
            .map_or(0, |n| n.byte_size);
        let bytes_without_key = self.total_bytes.saturating_sub(existing_bytes);
        let target_before_insert = self.max_bytes - byte_size;
        if bytes_without_key <= target_before_insert {
            return true;
        }
        let need_to_free = bytes_without_key - target_before_insert;
        let mut reclaimable = 0usize;
        let mut cursor = self.tail;
        while let Some(slot) = cursor {
            if let Some(node) = self.nodes[slot].as_ref() {
                if node.key != *key && !is_protected(&node.key) {
                    reclaimable = reclaimable.saturating_add(node.byte_size);
                    if reclaimable >= need_to_free {
                        return true;
                    }
                }
                cursor = node.prev;
            } else {
                break;
            }
        }
        false
    }

    /// Insert or replace `key` with `value` occupying `byte_size` bytes while
    /// refusing to evict any cached entry for which `is_protected(&k)` returns
    /// `true` (e.g. tiles currently displayed in the active frame).
    ///
    /// Preserves `total_bytes() <= max_bytes()` strictly. If freeing space
    /// would require evicting a protected entry, returns `Err((key, value))`
    /// without modifying existing cache contents.
    pub fn insert_protecting<F>(
        &mut self,
        key: K,
        value: V,
        byte_size: usize,
        is_protected: F,
    ) -> Result<Vec<(K, V)>, (K, V)>
    where
        F: Fn(&K) -> bool,
    {
        if !self.can_admit_protecting(&key, byte_size, &is_protected) {
            return Err((key, value));
        }

        let target_before_insert = self.max_bytes - byte_size;
        let mut evicted = Vec::new();
        if let Some((old_val, _)) = self.remove(&key) {
            evicted.push((key, old_val));
        }

        while self.total_bytes > target_before_insert {
            let Some(victim) = self.pop_lru_skipping(&is_protected) else {
                break;
            };
            evicted.push(victim);
        }

        let slot = self.alloc_slot(Node {
            key,
            value,
            byte_size,
            prev: None,
            next: None,
        });
        self.attach_front(slot);
        self.index.insert(key, slot);
        self.total_bytes += byte_size;
        Ok(evicted)
    }

    /// Remove every entry matching `predicate` and return the removed `(K, V)`
    /// pairs while keeping intrusive LRU links and `total_bytes` consistent.
    pub fn remove_where<F>(&mut self, mut predicate: F) -> Vec<(K, V)>
    where
        F: FnMut(&K) -> bool,
    {
        let matching: Vec<K> = self
            .index
            .keys()
            .copied()
            .filter(|k| predicate(k))
            .collect();
        let mut removed = Vec::with_capacity(matching.len());
        for key in matching {
            if let Some((val, _)) = self.remove(&key) {
                removed.push((key, val));
            }
        }
        removed
    }

    /// Remove `key` from the cache in O(1), returning `(value, byte_size)`.
    pub fn remove(&mut self, key: &K) -> Option<(V, usize)> {
        let slot = self.index.remove(key)?;
        self.detach(slot);
        let node = self.nodes[slot].take()?;
        self.free.push(slot);
        self.total_bytes = self.total_bytes.saturating_sub(node.byte_size);
        Some((node.value, node.byte_size))
    }

    /// Clear all entries and return them.
    pub fn clear(&mut self) -> Vec<(K, V)> {
        let mut out = Vec::with_capacity(self.index.len());
        while let Some(item) = self.pop_lru() {
            out.push(item);
        }
        self.nodes.clear();
        self.free.clear();
        self.index.clear();
        self.head = None;
        self.tail = None;
        self.total_bytes = 0;
        out
    }

    /// Maximum byte appetite allowed for one active pyramid (`< 50%` of
    /// `max_bytes` divided evenly among `active_pyramids`).
    ///
    /// Keeping each visible set + in-flight request strictly below half the
    /// cache budget prevents newly arriving tiles from evicting the currently
    /// displayed frame before a step finishes.
    pub fn pyramid_byte_cap(&self, active_pyramids: usize) -> usize {
        let half_strict = (self.max_bytes / 2).saturating_sub(1);
        half_strict / active_pyramids.max(1)
    }

    /// Maximum tile count allowed for one active pyramid given `tile_bytes`,
    /// strictly honoring the `< 50%` byte appetite cap (returns `0` if even one
    /// tile would exceed the per-pyramid cap).
    pub fn max_tiles_for_pyramid(&self, active_pyramids: usize, tile_bytes: usize) -> usize {
        if tile_bytes == 0 || tile_bytes > self.max_bytes {
            return 0;
        }
        self.pyramid_byte_cap(active_pyramids) / tile_bytes
    }

    fn alloc_slot(&mut self, node: Node<K, V>) -> usize {
        if let Some(slot) = self.free.pop() {
            self.nodes[slot] = Some(node);
            slot
        } else {
            let slot = self.nodes.len();
            self.nodes.push(Some(node));
            slot
        }
    }

    fn attach_front(&mut self, slot: usize) {
        let old_head = self.head;
        if let Some(node) = self.nodes[slot].as_mut() {
            node.prev = None;
            node.next = old_head;
        }
        if let Some(h) = old_head
            && let Some(head_node) = self.nodes[h].as_mut()
        {
            head_node.prev = Some(slot);
        } else {
            self.tail = Some(slot);
        }
        self.head = Some(slot);
    }

    fn detach(&mut self, slot: usize) {
        let (prev, next) = match self.nodes[slot].as_ref() {
            Some(node) => (node.prev, node.next),
            None => return,
        };
        if let Some(p) = prev {
            if let Some(prev_node) = self.nodes[p].as_mut() {
                prev_node.next = next;
            }
        } else {
            self.head = next;
        }
        if let Some(n) = next {
            if let Some(next_node) = self.nodes[n].as_mut() {
                next_node.prev = prev;
            }
        } else {
            self.tail = prev;
        }
        if let Some(node) = self.nodes[slot].as_mut() {
            node.prev = None;
            node.next = None;
        }
    }

    fn move_to_front(&mut self, slot: usize) {
        if self.head == Some(slot) {
            return;
        }
        self.detach(slot);
        self.attach_front(slot);
    }

    fn pop_lru(&mut self) -> Option<(K, V)> {
        let slot = self.tail?;
        self.remove_slot(slot)
    }

    fn pop_lru_skipping<F>(&mut self, is_protected: &F) -> Option<(K, V)>
    where
        F: Fn(&K) -> bool,
    {
        let mut cursor = self.tail;
        while let Some(slot) = cursor {
            let (key, prev) = match self.nodes[slot].as_ref() {
                Some(node) => (node.key, node.prev),
                None => return None,
            };
            if !is_protected(&key) {
                return self.remove_slot(slot);
            }
            cursor = prev;
        }
        None
    }

    fn remove_slot(&mut self, slot: usize) -> Option<(K, V)> {
        self.detach(slot);
        let node = self.nodes[slot].take()?;
        self.index.remove(&node.key);
        self.free.push(slot);
        self.total_bytes = self.total_bytes.saturating_sub(node.byte_size);
        Some((node.key, node.value))
    }
}

impl<V> TileCache<TileKey, V> {
    /// Touch `key` and every cached coarser ancestor along
    /// [`TileKey::parent`] up to the root so coarse fallback tiles never age
    /// out while their descendants are visible.
    pub fn touch_with_ancestors(&mut self, key: &TileKey) {
        let mut cursor = Some(*key);
        while let Some(cur) = cursor {
            self.touch(&cur);
            cursor = cur.parent();
        }
    }

    fn find_carrier_slot(&self, key: &TileKey) -> Option<(TileKey, usize)> {
        let mut cursor = Some(*key);
        while let Some(cur) = cursor {
            if let Some(&slot) = self.index.get(&cur) {
                return Some((cur, slot));
            }
            cursor = cur.parent();
        }
        None
    }

    fn touch_carrier_and_ancestors(&mut self, carrier_key: TileKey, carrier_slot: usize) {
        self.move_to_front(carrier_slot);
        let mut anc = carrier_key.parent();
        while let Some(cur) = anc {
            self.touch(&cur);
            anc = cur.parent();
        }
    }

    /// Find the finest cached tile covering `key` (either `key` itself or its
    /// nearest cached coarser ancestor via [`TileKey::parent`]), touch that
    /// carrier and all of its cached ancestors up to the root, and return
    /// `(carrier_key, &value)`.
    ///
    /// Decoupled from UV sub-rectangle math so callers serving clipped or
    /// guttered `RasterPyramid` tiles can compute exact UVs from the carrier's
    /// pixel dimensions and gutter metadata.
    pub fn find_carrier(&mut self, key: &TileKey) -> Option<(TileKey, &V)> {
        let (carrier_key, carrier_slot) = self.find_carrier_slot(key)?;
        self.touch_carrier_and_ancestors(carrier_key, carrier_slot);
        let value = self.nodes[carrier_slot].as_ref().map(|n| &n.value)?;
        Some((carrier_key, value))
    }

    /// Planetary convenience wrapper around [`Self::find_carrier`] that also
    /// computes the unguttered power-of-two UV sub-rectangle via
    /// [`TileKey::sub_rect_in_ancestor`].
    pub fn carrier(&mut self, key: &TileKey) -> Option<(TileKey, [f32; 4], &V)> {
        let (carrier_key, carrier_slot) = self.find_carrier_slot(key)?;
        let uv_rect = key.sub_rect_in_ancestor(&carrier_key)?;
        self.touch_carrier_and_ancestors(carrier_key, carrier_slot);
        let value = self.nodes[carrier_slot].as_ref().map(|n| &n.value)?;
        Some((carrier_key, uv_rect, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::key::{BandKey, PixelFormat, SourceId, TileScheme};

    fn make_key(scheme: TileScheme, level: u8, x: u32, y: u32) -> TileKey {
        TileKey {
            source: SourceId::BUNDLED_BASEMAP,
            band: BandKey::Display,
            scheme,
            format: PixelFormat::Rgba8UnormSrgb,
            gutter: 0,
            decode_rev: 0,
            level,
            x,
            y,
        }
    }

    #[test]
    fn byte_cap_evicts_lru_and_preserves_total_bytes_invariant() {
        let mut cache = TileCache::new(300);
        let k1 = make_key(TileScheme::Geodetic, 1, 0, 0);
        let k2 = make_key(TileScheme::Geodetic, 1, 1, 0);
        let k3 = make_key(TileScheme::Geodetic, 1, 2, 0);
        let k4 = make_key(TileScheme::Geodetic, 1, 3, 0);

        assert!(cache.insert(k1, 1u32, 100).expect("insert k1").is_empty());
        assert!(cache.insert(k2, 2u32, 100).expect("insert k2").is_empty());
        assert!(cache.insert(k3, 3u32, 100).expect("insert k3").is_empty());
        assert_eq!(cache.total_bytes(), 300);
        assert_eq!(cache.len(), 3);

        // Touch k1 so k2 becomes the oldest.
        assert_eq!(cache.get(&k1), Some(&1));

        // Inserting k4 (100 B) must evict k2.
        let evicted = cache.insert(k4, 4u32, 100).expect("insert k4");
        assert_eq!(evicted, vec![(k2, 2)]);
        assert_eq!(cache.total_bytes(), 300);
        assert!(!cache.contains(&k2));
        assert!(cache.contains(&k1));
        assert!(cache.contains(&k3));
        assert!(cache.contains(&k4));

        // Replacing an existing key returns its previous value and updates byte count.
        let replaced = cache.insert(k3, 30u32, 150).expect("replace k3");
        assert_eq!(replaced, vec![(k3, 3), (k1, 1)]);
        assert_eq!(cache.total_bytes(), 250);
        assert_eq!(cache.peek(&k3), Some(&30));

        // Oversized single entry (> max_bytes) returns Err without corrupting cache.
        let k_big = make_key(TileScheme::Geodetic, 2, 0, 0);
        let rejected = cache.insert(k_big, 99, 500);
        assert_eq!(rejected, Err((k_big, 99)));
        assert_eq!(cache.total_bytes(), 250);
        assert_eq!(cache.len(), 2);

        // Zero-budget cache rejects even 0-byte inserts cleanly.
        let mut zero_cache: TileCache<TileKey, u32> = TileCache::new(0);
        assert_eq!(zero_cache.insert(k1, 1, 0), Err((k1, 1)));

        // usize::MAX budget evicts cleanly without overflowing `total_bytes + byte_size`.
        let mut max_cache: TileCache<TileKey, u32> = TileCache::new(usize::MAX);
        let half_plus = (usize::MAX / 2) + 1;
        assert!(max_cache.insert(k1, 1, half_plus).expect("k1").is_empty());
        let ev = max_cache.insert(k2, 2, half_plus).expect("k2");
        assert_eq!(ev, vec![(k1, 1)]);
        assert_eq!(max_cache.total_bytes(), half_plus);
    }

    #[test]
    fn remove_and_clear_maintain_intrusive_links_and_carrier_misses() {
        let mut cache = TileCache::new(400);
        let k1 = make_key(TileScheme::Geodetic, 1, 0, 0);
        let k2 = make_key(TileScheme::Geodetic, 1, 1, 0);
        let k3 = make_key(TileScheme::Geodetic, 1, 2, 0);
        assert!(cache.carrier(&k1).is_none());

        cache.insert(k1, "a", 100).expect("k1");
        cache.insert(k2, "b", 100).expect("k2");
        cache.insert(k3, "c", 100).expect("k3");

        // Remove middle, head, then singleton tail.
        assert_eq!(cache.remove(&k2), Some(("b", 100)));
        assert_eq!(cache.remove(&k3), Some(("c", 100)));
        assert_eq!(cache.remove(&k1), Some(("a", 100)));
        assert!(cache.is_empty());
        assert_eq!(cache.total_bytes(), 0);
        assert_eq!(cache.remove(&k1), None);

        // Re-insert into recycled slab slots and clear.
        cache.insert(k1, "x", 50).expect("re-insert k1");
        cache.insert(k2, "y", 50).expect("re-insert k2");
        let cleared = cache.clear();
        assert_eq!(cleared.len(), 2);
        assert!(cache.is_empty());
        assert_eq!(cache.total_bytes(), 0);
    }

    #[test]
    fn ancestor_touch_protects_root_for_both_xyz_and_raster_pyramid() {
        // 1. WebMercator / Geodetic (z = 0 coarsest):
        let mut geo_cache = TileCache::new(300);
        let root = make_key(TileScheme::WebMercator, 0, 0, 0);
        let mid = make_key(TileScheme::WebMercator, 1, 1, 0);
        let decoy = make_key(TileScheme::WebMercator, 2, 0, 0);
        geo_cache.insert(root, "root", 100).expect("root");
        geo_cache.insert(mid, "mid", 100).expect("mid");
        geo_cache.insert(decoy, "decoy", 100).expect("decoy");

        // Query a leaf under `mid`: carrier resolves to `mid` AND touches `root`!
        let leaf = make_key(TileScheme::WebMercator, 3, 4, 0);
        let (carrier_key, uv, val) = geo_cache.carrier(&leaf).expect("carrier found");
        assert_eq!(carrier_key, mid);
        assert_eq!(*val, "mid");
        assert_eq!(uv, [0.0, 0.0, 0.25, 0.25]);

        // Inserting a new tile must evict `decoy`, NOT `root`.
        let newcomer = make_key(TileScheme::WebMercator, 2, 3, 3);
        let evicted = geo_cache.insert(newcomer, "new", 100).expect("newcomer");
        assert_eq!(evicted, vec![(decoy, "decoy")]);
        assert!(geo_cache.contains(&root));
        assert!(geo_cache.contains(&mid));

        // 2. RasterPyramid (level = max_level coarsest):
        let pyr = TileScheme::RasterPyramid { max_level: 3 };
        let mut pyr_cache = TileCache::new(300);
        let p_root = make_key(pyr, 3, 0, 0);
        let p_mid = make_key(pyr, 2, 1, 1);
        let p_decoy = make_key(pyr, 1, 0, 0);
        pyr_cache.insert(p_root, "p_root", 100).expect("p_root");
        pyr_cache.insert(p_mid, "p_mid", 100).expect("p_mid");
        pyr_cache.insert(p_decoy, "p_decoy", 100).expect("p_decoy");

        let p_leaf = make_key(pyr, 0, 5, 6); // (5>>2=1, 6>>2=1) -> p_mid!
        let (p_carrier, p_uv, p_val) = pyr_cache.carrier(&p_leaf).expect("pyramid carrier");
        assert_eq!(p_carrier, p_mid);
        assert_eq!(*p_val, "p_mid");
        assert_eq!(p_uv, [0.25, 0.5, 0.25, 0.25]);

        let p_new = make_key(pyr, 1, 3, 3);
        let p_evicted = pyr_cache.insert(p_new, "p_new", 100).expect("p_new");
        assert_eq!(p_evicted, vec![(p_decoy, "p_decoy")]);
        assert!(pyr_cache.contains(&p_root));
        assert!(pyr_cache.contains(&p_mid));

        // Guttered RasterPyramid tiles resolve via `find_carrier`, while
        // unguttered planetary `carrier()` refuses them.
        let g_root = TileKey {
            gutter: 1,
            ..p_root
        };
        let g_leaf = TileKey {
            gutter: 1,
            ..p_leaf
        };
        let mut g_cache = TileCache::new(200);
        g_cache.insert(g_root, "g_root", 100).expect("g_root");
        assert!(g_cache.carrier(&g_leaf).is_none());
        let (found_key, found_val) = g_cache.find_carrier(&g_leaf).expect("guttered carrier");
        assert_eq!(found_key, g_root);
        assert_eq!(*found_val, "g_root");
    }

    #[test]
    fn appetite_cap_stays_strictly_below_half_budget() {
        let cache: TileCache<TileKey, u8> = TileCache::new(256 * 1024 * 1024);
        let cap1 = cache.pyramid_byte_cap(1);
        let cap2 = cache.pyramid_byte_cap(2);
        let cap4 = cache.pyramid_byte_cap(4);

        assert!(cap1 * 2 < cache.max_bytes());
        assert!(cap2 * 4 < cache.max_bytes());
        assert!(cap4 * 8 < cache.max_bytes());
        assert!(cache.max_tiles_for_pyramid(1, 256 * 256 * 4) < 512);

        // Small budget with multiple pyramids must never exceed < 50% total appetite.
        let small: TileCache<TileKey, u8> = TileCache::new(1000);
        let per_pyr = small.max_tiles_for_pyramid(4, 200);
        assert_eq!(per_pyr, 0);
        assert!(per_pyr * 4 * 200 * 2 < small.max_bytes());
    }

    #[test]
    fn active_frame_protection_and_remove_where_preserve_invariants() {
        let mut cache = TileCache::new(300);
        let k1 = make_key(TileScheme::WebMercator, 1, 0, 0);
        let k2 = make_key(TileScheme::WebMercator, 1, 1, 0);
        let k3 = make_key(TileScheme::WebMercator, 1, 0, 1);
        let k4 = make_key(TileScheme::WebMercator, 1, 1, 1);
        let other = TileKey {
            source: SourceId::from_stable_key("other-source"),
            ..k1
        };

        cache.insert(k1, "k1", 100).expect("k1");
        cache.insert(k2, "k2", 100).expect("k2");
        cache.insert(k3, "k3", 100).expect("k3");

        // k1 is the LRU tail, but protected (currently displayed in active frame).
        // Inserting k4 must skip k1 and evict k2 instead.
        let ev = cache
            .insert_protecting(k4, "k4", 100, |k| *k == k1)
            .expect("insert skipping protected");
        assert_eq!(ev, vec![(k2, "k2")]);
        assert!(cache.contains(&k1));
        assert!(!cache.contains(&k2));
        assert!(cache.contains(&k3));
        assert!(cache.contains(&k4));
        assert_eq!(cache.total_bytes(), 300);

        // If every cached tile is protected, admission is rejected atomically
        // without evicting protected tiles or corrupting existing entries.
        let rejected = cache.insert_protecting(other, "other", 100, |_| true);
        assert_eq!(rejected, Err((other, "other")));
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.total_bytes(), 300);
        assert!(cache.contains(&k1));

        // Replacing an existing entry with a larger payload that cannot fit
        // without evicting protected tiles also fails atomically without removing `k3`.
        let rejected_replace = cache.insert_protecting(k3, "k3_big", 200, |k| *k != k3);
        assert_eq!(rejected_replace, Err((k3, "k3_big")));
        assert_eq!(cache.peek(&k3), Some(&"k3"));
        assert_eq!(cache.total_bytes(), 300);

        // Insert a tile from another source, then remove_where clears only the bundled source.
        cache
            .insert_protecting(other, "other", 100, |k| *k == k1)
            .expect("insert other");
        let removed = cache.remove_where(|k| k.source == SourceId::BUNDLED_BASEMAP);
        assert_eq!(removed.len(), 2);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&other));
        assert_eq!(cache.total_bytes(), 100);
    }
}
