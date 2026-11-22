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
/// Evictions return the evicted `(K, V)` entries so callers (such as
/// `GpuTileCache`) can prune associated GPU bind groups deterministically.
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
    /// Returns all evicted `(K, V)` pairs (including any replaced previous
    /// value for `key`). If `byte_size > max_bytes`, the entry cannot fit and
    /// is returned directly without modifying existing cache contents.
    pub fn insert(&mut self, key: K, value: V, byte_size: usize) -> Vec<(K, V)> {
        if byte_size > self.max_bytes {
            return vec![(key, value)];
        }

        let mut evicted = Vec::new();
        if let Some((old_val, _)) = self.remove(&key) {
            evicted.push((key, old_val));
        }

        while self.total_bytes.saturating_add(byte_size) > self.max_bytes {
            let Some(victim) = self.pop_lru() else {
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
        evicted
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
    /// clamped to at least 1 when `tile_bytes <= max_bytes`.
    pub fn max_tiles_for_pyramid(&self, active_pyramids: usize, tile_bytes: usize) -> usize {
        if tile_bytes == 0 || tile_bytes > self.max_bytes {
            return 0;
        }
        (self.pyramid_byte_cap(active_pyramids) / tile_bytes).max(1)
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

    /// Find the finest cached tile covering `key` (either `key` itself or its
    /// nearest cached coarser ancestor via [`TileKey::parent`]), touch that
    /// carrier and all of its cached ancestors up to the root, and return
    /// `(carrier_key, uv_rect_in_carrier, &value)`.
    pub fn carrier(&mut self, key: &TileKey) -> Option<(TileKey, [f32; 4], &V)> {
        let mut cursor = Some(*key);
        let mut found = None;
        while let Some(cur) = cursor {
            if self.contains(&cur) {
                found = Some(cur);
                break;
            }
            cursor = cur.parent();
        }
        let carrier_key = found?;
        let uv_rect = key.sub_rect_in_ancestor(&carrier_key)?;
        self.touch_with_ancestors(&carrier_key);
        let value = self.peek(&carrier_key)?;
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

        assert!(cache.insert(k1, 1u32, 100).is_empty());
        assert!(cache.insert(k2, 2u32, 100).is_empty());
        assert!(cache.insert(k3, 3u32, 100).is_empty());
        assert_eq!(cache.total_bytes(), 300);
        assert_eq!(cache.len(), 3);

        // Touch k1 so k2 becomes the oldest.
        assert_eq!(cache.get(&k1), Some(&1));

        // Inserting k4 (100 B) must evict k2.
        let evicted = cache.insert(k4, 4u32, 100);
        assert_eq!(evicted, vec![(k2, 2)]);
        assert_eq!(cache.total_bytes(), 300);
        assert!(!cache.contains(&k2));
        assert!(cache.contains(&k1));
        assert!(cache.contains(&k3));
        assert!(cache.contains(&k4));

        // Oversized single entry (> max_bytes) is rejected without corrupting cache.
        let k_big = make_key(TileScheme::Geodetic, 2, 0, 0);
        let rejected = cache.insert(k_big, 99, 500);
        assert_eq!(rejected, vec![(k_big, 99)]);
        assert_eq!(cache.total_bytes(), 300);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn ancestor_touch_protects_root_for_both_xyz_and_raster_pyramid() {
        // 1. WebMercator / Geodetic (z = 0 coarsest):
        let mut geo_cache = TileCache::new(300);
        let root = make_key(TileScheme::WebMercator, 0, 0, 0);
        let mid = make_key(TileScheme::WebMercator, 1, 1, 0);
        let decoy = make_key(TileScheme::WebMercator, 2, 0, 0);
        geo_cache.insert(root, "root", 100);
        geo_cache.insert(mid, "mid", 100);
        geo_cache.insert(decoy, "decoy", 100);

        // Query a leaf under `mid`: carrier resolves to `mid` AND touches `root`!
        let leaf = make_key(TileScheme::WebMercator, 3, 4, 0);
        let (carrier_key, uv, val) = geo_cache.carrier(&leaf).expect("carrier found");
        assert_eq!(carrier_key, mid);
        assert_eq!(*val, "mid");
        assert_eq!(uv, [0.0, 0.0, 0.25, 0.25]);

        // Inserting a new tile must evict `decoy`, NOT `root`.
        let newcomer = make_key(TileScheme::WebMercator, 2, 3, 3);
        let evicted = geo_cache.insert(newcomer, "new", 100);
        assert_eq!(evicted, vec![(decoy, "decoy")]);
        assert!(geo_cache.contains(&root));
        assert!(geo_cache.contains(&mid));

        // 2. RasterPyramid (level = max_level coarsest):
        let pyr = TileScheme::RasterPyramid { max_level: 3 };
        let mut pyr_cache = TileCache::new(300);
        let p_root = make_key(pyr, 3, 0, 0);
        let p_mid = make_key(pyr, 2, 1, 1);
        let p_decoy = make_key(pyr, 1, 0, 0);
        pyr_cache.insert(p_root, "p_root", 100);
        pyr_cache.insert(p_mid, "p_mid", 100);
        pyr_cache.insert(p_decoy, "p_decoy", 100);

        let p_leaf = make_key(pyr, 0, 5, 6); // (5>>2=1, 6>>2=1) -> p_mid!
        let (p_carrier, p_uv, p_val) = pyr_cache.carrier(&p_leaf).expect("pyramid carrier");
        assert_eq!(p_carrier, p_mid);
        assert_eq!(*p_val, "p_mid");
        assert_eq!(p_uv, [0.25, 0.5, 0.25, 0.25]);

        let p_new = make_key(pyr, 1, 3, 3);
        let p_evicted = pyr_cache.insert(p_new, "p_new", 100);
        assert_eq!(p_evicted, vec![(p_decoy, "p_decoy")]);
        assert!(pyr_cache.contains(&p_root));
        assert!(pyr_cache.contains(&p_mid));
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
    }
}
