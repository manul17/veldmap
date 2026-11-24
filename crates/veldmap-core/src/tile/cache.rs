//! Byte-budgeted intrusive LRU tile cache with O(1) lookups/unprotected
//! inserts, optional active-frame-protected admission, scheme-aware ancestor
//! touching, and `< 50%` per-pyramid appetite capping.

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

    /// Adjust the configured maximum byte budget, returning any evicted LRU
    /// entries when shrinking below the current total.
    pub fn set_max_bytes(&mut self, max_bytes: usize) -> Vec<(K, V)> {
        self.max_bytes = max_bytes;
        let mut evicted = Vec::new();
        while self.total_bytes > self.max_bytes {
            let Some(item) = self.pop_lru() else {
                break;
            };
            evicted.push(item);
        }
        evicted
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

    fn insert_front_entry(&mut self, key: K, value: V, byte_size: usize) {
        let slot = self.insert_front_node(key, value, byte_size);
        self.index.insert(key, slot);
        self.total_bytes += byte_size;
    }

    /// Insert or replace `key` with `value` occupying `byte_size` bytes,
    /// evicting least-recently-used entries in O(1) per eviction until
    /// `total_bytes() <= max_bytes()`.
    ///
    /// Returns `Ok(evicted)` containing all evicted `(K, V)` pairs (including
    /// any replaced previous value for `key`), or `Err((key, value))` without
    /// modifying existing cache contents if `max_bytes == 0` or
    /// `byte_size > max_bytes`.
    pub fn insert(&mut self, key: K, value: V, byte_size: usize) -> Result<Vec<(K, V)>, (K, V)> {
        if self.max_bytes == 0 || byte_size > self.max_bytes {
            return Err((key, value));
        }

        let target_before_insert = self.max_bytes - byte_size;
        let mut evicted = Vec::new();
        if let Some((old_val, _)) = self.remove(&key) {
            evicted.push((key, old_val));
        }

        while self.total_bytes > target_before_insert {
            let Some(victim) = self.pop_lru() else {
                break;
            };
            evicted.push(victim);
        }

        self.insert_front_entry(key, value, byte_size);
        Ok(evicted)
    }

    fn plan_protected_evictions<F>(
        &self,
        key: &K,
        byte_size: usize,
        is_protected: &F,
        collect_slots: bool,
    ) -> Option<Vec<usize>>
    where
        F: Fn(&K) -> bool,
    {
        if self.max_bytes == 0 || byte_size > self.max_bytes {
            return None;
        }
        let existing_bytes = self
            .index
            .get(key)
            .and_then(|&slot| self.nodes[slot].as_ref())
            .map_or(0, |n| n.byte_size);
        let bytes_without_key = self.total_bytes.saturating_sub(existing_bytes);
        let target_before_insert = self.max_bytes - byte_size;
        if bytes_without_key <= target_before_insert {
            return Some(Vec::new());
        }

        let need_to_free = bytes_without_key - target_before_insert;
        let mut victim_slots = Vec::new();
        let mut reclaimable = 0usize;
        let mut cursor = self.tail;
        while let Some(slot) = cursor {
            let Some(node) = self.nodes[slot].as_ref() else {
                break;
            };
            if node.key != *key && !is_protected(&node.key) {
                if collect_slots {
                    victim_slots.push(slot);
                }
                reclaimable = reclaimable.saturating_add(node.byte_size);
                if reclaimable >= need_to_free {
                    return Some(victim_slots);
                }
            }
            cursor = node.prev;
        }
        None
    }

    /// Return whether inserting or replacing `key` of `byte_size` can succeed
    /// within `max_bytes` without evicting any entry where `is_protected(&k)`
    /// returns `true`.
    pub fn can_admit_protecting<F>(&self, key: &K, byte_size: usize, is_protected: F) -> bool
    where
        F: Fn(&K) -> bool,
    {
        self.plan_protected_evictions(key, byte_size, &is_protected, false)
            .is_some()
    }

    /// Insert or replace `key` with `value` occupying `byte_size` bytes while
    /// refusing to evict any cached entry for which `is_protected(&k)` returns
    /// `true` (e.g. tiles currently displayed in the active frame).
    ///
    /// Preserves `total_bytes() <= max_bytes()` strictly in a single O(P + E)
    /// tail-to-head scan. If freeing space would require evicting a protected
    /// entry, returns `Err((key, value))` without modifying existing cache
    /// contents.
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
        let Some(victim_slots) =
            self.plan_protected_evictions(&key, byte_size, &is_protected, true)
        else {
            return Err((key, value));
        };

        let mut evicted = Vec::with_capacity(victim_slots.len() + 1);
        if let Some((old_val, _)) = self.remove(&key) {
            evicted.push((key, old_val));
        }
        for slot in victim_slots {
            if let Some(victim) = self.remove_slot(slot) {
                evicted.push(victim);
            }
        }

        self.insert_front_entry(key, value, byte_size);
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
        let node = self.take_slot(slot)?;
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

    fn insert_front_node(&mut self, key: K, value: V, byte_size: usize) -> usize {
        let slot = self.alloc_slot(Node {
            key,
            value,
            byte_size,
            prev: None,
            next: None,
        });
        self.attach_front(slot);
        slot
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

    fn take_slot(&mut self, slot: usize) -> Option<Node<K, V>> {
        self.detach(slot);
        let node = self.nodes[slot].take()?;
        self.free.push(slot);
        self.total_bytes = self.total_bytes.saturating_sub(node.byte_size);
        Some(node)
    }

    fn remove_slot(&mut self, slot: usize) -> Option<(K, V)> {
        let node = self.take_slot(slot)?;
        self.index.remove(&node.key);
        Some((node.key, node.value))
    }
}

impl<V> TileCache<TileKey, V> {
    /// Maximum byte appetite allowed for one active pyramid weighted
    /// proportionally by its per-tile byte cost (`pyramid_weight / total_weight`)
    /// within the strict `< 50%` combined appetite ceiling.
    ///
    /// Useful for heterogeneous multi-band stacks (e.g. basemap `Rgba8` +
    /// multiple `R32Float` optical/DEM bands + `R8Uint` `SCL`).
    pub fn proportional_pyramid_byte_cap(&self, pyramid_weight: u64, total_weight: u64) -> usize {
        let half_strict = (self.max_bytes / 2).saturating_sub(1);
        if pyramid_weight == 0 || half_strict == 0 {
            return 0;
        }
        let denom = total_weight.max(pyramid_weight);
        let scaled =
            (u128::from(half_strict as u64) * u128::from(pyramid_weight)) / u128::from(denom);
        usize::try_from(scaled).unwrap_or(half_strict)
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

    fn touch_and_borrow_carrier(
        &mut self,
        carrier_key: TileKey,
        carrier_slot: usize,
    ) -> Option<&V> {
        self.move_to_front(carrier_slot);
        let mut anc = carrier_key.parent();
        while let Some(cur) = anc {
            self.touch(&cur);
            anc = cur.parent();
        }
        self.nodes[carrier_slot].as_ref().map(|n| &n.value)
    }

    /// Read-only check whether `key` or any of its coarser ancestors along
    /// [`TileKey::parent`] is currently cached (without mutating LRU order).
    pub fn has_carrier(&self, key: &TileKey) -> bool {
        self.find_carrier_slot(key).is_some()
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
        let value = self.touch_and_borrow_carrier(carrier_key, carrier_slot)?;
        Some((carrier_key, value))
    }

    /// Planetary convenience wrapper around [`Self::find_carrier`] that also
    /// computes the unguttered power-of-two UV sub-rectangle via
    /// [`TileKey::sub_rect_in_ancestor`].
    pub fn carrier(&mut self, key: &TileKey) -> Option<(TileKey, [f32; 4], &V)> {
        let (carrier_key, carrier_slot) = self.find_carrier_slot(key)?;
        let uv_rect = key.sub_rect_in_ancestor(&carrier_key)?;
        let value = self.touch_and_borrow_carrier(carrier_key, carrier_slot)?;
        Some((carrier_key, uv_rect, value))
    }
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
