//! Shared elastic active-frame mesh LRU cache and regular grid quad index
//! helper reused by planetary tile meshes ([`super::mesh`]) and raster overlay
//! warp meshes ([`super::overlay`]).

use std::hash::Hash;

use veldmap_core::tile::TileCache;

/// Default VRAM budget for cached per-tile geometry vertex buffers (`16 MiB`,
/// ~1,000 cached `16 x 16` tiles).
pub(super) const DEFAULT_MESH_CACHE_BYTES: usize = 16 * 1024 * 1024;

/// Elastic active-frame LRU cache over [`TileCache`] shared by planetary tile
/// meshes and overlay warp meshes.
///
/// Raises `max_bytes` to cover active-frame demand before touching/inserting
/// current-frame meshes so earlier draws are never evicted mid-frame, then
/// relaxes `max_bytes` back toward `default_cap_bytes` once demand subsides.
pub(super) struct ElasticMeshCache<K: Copy + Eq + Hash, V> {
    default_cap_bytes: usize,
    inner: TileCache<K, V>,
}

impl<K: Eq + Hash + Copy, V> ElasticMeshCache<K, V> {
    pub(super) fn new(default_cap_bytes: usize) -> Self {
        Self {
            default_cap_bytes,
            inner: TileCache::new(default_cap_bytes),
        }
    }

    pub(super) fn peek(&self, key: &K) -> Option<&V> {
        self.inner.peek(key)
    }

    pub(super) fn insert(&mut self, key: K, value: V, byte_size: usize) {
        let _ = self.inner.insert(key, value, byte_size);
    }

    pub(super) fn clear_and_reset_cap(&mut self) {
        let _ = self.inner.clear();
        if self.inner.max_bytes() > self.default_cap_bytes {
            let _ = self.inner.set_max_bytes(self.default_cap_bytes);
        }
    }

    pub(super) fn begin_frame<'a>(
        &mut self,
        needed_bytes: usize,
        active_keys: impl IntoIterator<Item = &'a K>,
    ) where
        K: 'a,
    {
        let active_floor = needed_bytes.max(self.default_cap_bytes);
        if active_floor > self.inner.max_bytes() {
            let _ = self.inner.set_max_bytes(active_floor);
        }
        for key in active_keys {
            self.inner.touch(key);
        }
    }

    pub(super) fn end_frame(&mut self, needed_bytes: usize) {
        let target_cap = needed_bytes.max(self.default_cap_bytes);
        if target_cap < self.inner.max_bytes() {
            let _ = self.inner.set_max_bytes(target_cap);
        }
    }
}

/// Append triangle indices for an `n x n` row-major vertex grid
/// (`(n + 1) x (n + 1)` vertices), flipping triangle winding when
/// `flip_winding` is `true` so negative-determinant bindings still face outward.
pub(super) fn push_grid_quad_indices(out: &mut Vec<u32>, n: u32, flip_winding: bool) {
    let stride = n + 1;
    for row in 0..n {
        for col in 0..n {
            let top = row * stride + col;
            let bottom = top + stride;
            if flip_winding {
                out.extend_from_slice(&[top, top + 1, bottom]);
                out.extend_from_slice(&[top + 1, bottom + 1, bottom]);
            } else {
                out.extend_from_slice(&[top, bottom, top + 1]);
                out.extend_from_slice(&[top + 1, bottom, bottom + 1]);
            }
        }
    }
}
