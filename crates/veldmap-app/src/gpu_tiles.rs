//! GPU tile ownership (`GpuTile`), VRAM-budgeted `GpuTileCache`, and
//! multi-producer coarsest-first `PendingUploads`.
//!
//! `GpuTile` owns strictly its `Texture`, `TextureView`, `PixelFormat`,
//! dimensions (`width`, `height`, `valid_w`, `valid_h`, `gutter`), exact byte
//! size (including CPU-generated mip levels), and monotonic `tile_id: u64` —
//! never pipeline-specific `BindGroup`s or `view_formats`.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use veldmap_core::tile::{
    DecodedTile, PixelFormat, PlanetaryScheme, QuadTileId, SourceId, TileCache, TileKey,
    compose_uv_rect,
};

/// Default GPU tile VRAM cache cap (`256 MiB`).
pub const DEFAULT_GPU_TILE_CACHE_BYTES: usize = 256 * 1024 * 1024;

/// Default per-frame upload byte budget (`4 MiB`).
pub const DEFAULT_UPLOAD_BYTE_BUDGET: usize = 4 * 1024 * 1024;

/// Default per-frame upload wall-clock budget (`2 ms`).
pub const DEFAULT_UPLOAD_TIME_BUDGET: Duration = Duration::from_millis(2);

/// Default CPU byte cap on decoded tiles buffered in [`PendingUploads`]
/// awaiting GPU upload (`64 MiB`).
pub(crate) const DEFAULT_MAX_QUEUED_UPLOAD_BYTES: usize = 64 * 1024 * 1024;

/// Immutable GPU-resident tile resource.
pub struct GpuTile {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub format: PixelFormat,
    pub width: u32,
    pub height: u32,
    /// Valid interior pixel width excluding gutters (`valid_w + 2*gutter <= width`).
    pub valid_w: u32,
    /// Valid interior pixel height excluding gutters (`valid_h + 2*gutter <= height`).
    pub valid_h: u32,
    /// Border gutter width in texels (`0` for standard tiles, `1` for DEM tiles).
    pub gutter: u8,
    pub byte_size: usize,
    pub tile_id: u64,
}

impl std::fmt::Debug for GpuTile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTile")
            .field("tile_id", &self.tile_id)
            .field("format", &self.format)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("valid_w", &self.valid_w)
            .field("valid_h", &self.valid_h)
            .field("gutter", &self.gutter)
            .field("byte_size", &self.byte_size)
            .finish()
    }
}

/// Convert domain [`PixelFormat`] to the corresponding `wgpu::TextureFormat`.
pub(crate) fn wgpu_format(format: PixelFormat) -> wgpu::TextureFormat {
    match format {
        PixelFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        PixelFormat::R32Float => wgpu::TextureFormat::R32Float,
        PixelFormat::R8Uint => wgpu::TextureFormat::R8Uint,
        PixelFormat::R16Uint => wgpu::TextureFormat::R16Uint,
    }
}

/// Outcome of attempting to upload and admit a decoded tile into
/// [`GpuTileCache`] under active-frame protection.
#[derive(Debug)]
pub enum UploadInsertOutcome {
    /// Tile was uploaded and admitted into VRAM.
    Inserted { tile: Arc<GpuTile>, evicted: usize },
    /// Tile is valid and fits the cache ceiling, but all evictable entries are
    /// currently protected by the active frame (caller should keep it deferred).
    RefusedByProtection,
    /// Tile failed dimension/format/gutter/buffer validation or exceeds
    /// `cache.max_bytes()` (caller should discard it).
    Rejected,
}

/// VRAM-budgeted LRU cache of immutable `Arc<GpuTile>` instances owned on
/// `VeldmapApp`.
pub struct GpuTileCache {
    cache: TileCache<TileKey, Arc<GpuTile>>,
    next_tile_id: u64,
}

impl GpuTileCache {
    /// Create an empty GPU tile cache bounded by `max_bytes`.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            cache: TileCache::new(max_bytes),
            next_tile_id: 1,
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.cache.total_bytes()
    }

    pub fn max_bytes(&self) -> usize {
        self.cache.max_bytes()
    }

    pub fn contains(&self, key: &TileKey) -> bool {
        self.cache.contains(key)
    }

    /// Read-only check whether `key` or any coarser ancestor carrier is cached.
    pub fn has_carrier(&self, key: &TileKey) -> bool {
        self.cache.has_carrier(key)
    }

    /// Remove every cached GPU tile belonging to `source`, returning how many
    /// tiles were removed.
    pub fn remove_source(&mut self, source: SourceId) -> usize {
        self.cache.remove_where(|k| k.source == source).len()
    }

    /// Remove every cached GPU tile matching `predicate`, returning how many
    /// tiles were removed (useful for multi-geocell DEM mosaics or timelines).
    pub fn remove_where<F>(&mut self, predicate: F) -> usize
    where
        F: FnMut(&TileKey) -> bool,
    {
        self.cache.remove_where(predicate).len()
    }

    /// Maximum byte appetite allowed for one active pyramid (`< 50%` of VRAM
    /// budget divided across `active_pyramids`).
    pub fn pyramid_byte_cap(&self, active_pyramids: usize) -> usize {
        self.cache.pyramid_byte_cap(active_pyramids)
    }

    /// Proportional byte appetite cap for one weighted pyramid (`< 50%`
    /// combined ceiling across heterogeneous multi-band stacks).
    pub fn proportional_pyramid_byte_cap(&self, pyramid_weight: u64, total_weight: u64) -> usize {
        self.cache
            .proportional_pyramid_byte_cap(pyramid_weight, total_weight)
    }

    /// Resolve the finest cached carrier tile covering `key` without assuming
    /// power-of-two unguttered UVs (required for clipped `RasterPyramid` edge
    /// tiles and `gutter = 1` DEM tiles).
    pub fn find_carrier(&mut self, key: &TileKey) -> Option<(TileKey, Arc<GpuTile>)> {
        let (carrier_key, tile) = self.cache.find_carrier(key)?;
        Some((carrier_key, Arc::clone(tile)))
    }

    /// Resolve the finest cached tile covering `key` (exact or nearest coarser
    /// ancestor) together with its unguttered power-of-two UV sub-rectangle,
    /// touching the carrier and all cached ancestors up to the root.
    pub fn carrier(&mut self, key: &TileKey) -> Option<(TileKey, [f32; 4], Arc<GpuTile>)> {
        let (carrier_key, uv, tile) = self.cache.carrier(key)?;
        Some((carrier_key, uv, Arc::clone(tile)))
    }

    /// Validate and upload a CPU-decoded tile (including all precomputed CPU
    /// mip levels) into GPU memory and insert it into the LRU cache.
    pub fn upload_and_insert(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: TileKey,
        decoded: &DecodedTile,
    ) -> Option<(Arc<GpuTile>, usize)> {
        match self.upload_and_insert_protecting(device, queue, key, decoded, |_| false) {
            UploadInsertOutcome::Inserted { tile, evicted } => Some((tile, evicted)),
            UploadInsertOutcome::RefusedByProtection | UploadInsertOutcome::Rejected => None,
        }
    }

    /// Validate and upload a CPU-decoded tile (with full optional mip chain)
    /// into GPU memory while refusing to evict any cached tile for which
    /// `is_protected(&k)` returns `true` (active-frame protection).
    pub fn upload_and_insert_protecting<F>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: TileKey,
        decoded: &DecodedTile,
        is_protected: F,
    ) -> UploadInsertOutcome
    where
        F: Fn(&TileKey) -> bool,
    {
        if decoded.format() != key.format || decoded.gutter() != key.gutter {
            tracing::warn!(
                ?key,
                decoded_format = ?decoded.format(),
                decoded_gutter = decoded.gutter(),
                "Skipping decoded tile whose format or gutter mismatches TileKey"
            );
            return UploadInsertOutcome::Rejected;
        }
        let max_dim = device.limits().max_texture_dimension_2d;
        if decoded.width() == 0
            || decoded.height() == 0
            || decoded.width() > max_dim
            || decoded.height() > max_dim
        {
            tracing::warn!(
                ?key,
                width = decoded.width(),
                height = decoded.height(),
                max_dim,
                "Skipping decoded tile with invalid or oversized GPU dimensions"
            );
            return UploadInsertOutcome::Rejected;
        }

        let byte_size = decoded.byte_size();
        if self.cache.max_bytes() == 0 || byte_size > self.cache.max_bytes() {
            return UploadInsertOutcome::Rejected;
        }

        if !self
            .cache
            .can_admit_protecting(&key, byte_size, &is_protected)
        {
            return UploadInsertOutcome::RefusedByProtection;
        }

        let Some((texture, view)) = upload_tile_texture(device, queue, decoded) else {
            return UploadInsertOutcome::Rejected;
        };
        let tile_id = self.next_tile_id;
        self.next_tile_id = self.next_tile_id.wrapping_add(1);
        let tile = Arc::new(GpuTile {
            texture,
            view,
            format: decoded.format(),
            width: decoded.width(),
            height: decoded.height(),
            valid_w: decoded.valid_w(),
            valid_h: decoded.valid_h(),
            gutter: decoded.gutter(),
            byte_size,
            tile_id,
        });

        match self
            .cache
            .insert_protecting(key, Arc::clone(&tile), byte_size, is_protected)
        {
            Ok(evicted) => UploadInsertOutcome::Inserted {
                tile,
                evicted: evicted.len(),
            },
            Err(_) => UploadInsertOutcome::RefusedByProtection,
        }
    }
}

fn upload_tile_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    decoded: &DecodedTile,
) -> Option<(wgpu::Texture, wgpu::TextureView)> {
    let mip_level_count = decoded.mip_level_count();
    let bpp = decoded.format().bytes_per_pixel() as u32;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("veldmap gpu tile"),
        size: wgpu::Extent3d {
            width: decoded.width(),
            height: decoded.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu_format(decoded.format()),
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for mip_level in 0..mip_level_count {
        let mip_bytes = decoded.mip_data(mip_level)?;
        let (mw, mh) = decoded.mip_dims(mip_level);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            mip_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(mw * bpp),
                rows_per_image: Some(mh),
            },
            wgpu::Extent3d {
                width: mw,
                height: mh,
                depth_or_array_layers: 1,
            },
        );
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Some((texture, view))
}

/// Parameters governing one frame's [`PendingUploads::drain_and_upload`] pass.
#[derive(Clone, Copy, Debug)]
pub struct UploadPassParams<'a> {
    /// Sources whose coarsest root tiles (`key.is_root_of(src)`) are pinned
    /// against `wanted` pruning and queue-cap eviction.
    pub pinned_root_sources: &'a [SourceId],
    /// Optional source whose largest uploaded tile byte size is reported in
    /// [`UploadBatchReport::max_active_tile_bytes`] (used for live basemap
    /// `@2x`/512 px appetite sizing).
    pub observed_source: Option<SourceId>,
    /// Currently desired tile keys across all active layers (`None` skips
    /// `wanted` pruning).
    pub wanted: Option<&'a HashSet<TileKey>>,
    /// Active-frame carrier keys protected from LRU eviction during upload.
    pub protected: Option<&'a HashSet<TileKey>>,
    pub byte_budget: usize,
    pub time_budget: Duration,
}

/// Summary of one frame's [`PendingUploads::drain_and_upload`] pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct UploadBatchReport {
    pub uploaded: usize,
    pub evicted: usize,
    /// Total tiles still queued (`remaining_ready + deferred_protected`).
    pub remaining: usize,
    /// Tiles deferred solely by per-frame byte/time budgets (should request an
    /// immediate follow-up repaint).
    pub remaining_ready: usize,
    /// Tiles kept queued because active-frame protection blocked eviction (must
    /// NOT trigger an infinite idle repaint loop on its own).
    pub deferred_protected: usize,
    /// Largest single tile byte size uploaded for `observed_source` in this
    /// pass.
    pub max_active_tile_bytes: usize,
}

fn is_pinned_root(key: &TileKey, pinned_root_sources: &[SourceId]) -> bool {
    pinned_root_sources.iter().any(|&src| key.is_root_of(src))
}

/// Multi-producer completion-channel owner and byte-capped, coarsest-first
/// budgeted GPU tile upload queue owned on `VeldmapApp`.
pub struct PendingUploads {
    tx: Option<Sender<(TileKey, DecodedTile)>>,
    rx: Option<Receiver<(TileKey, DecodedTile)>>,
    queued: HashMap<TileKey, DecodedTile>,
    queued_bytes: usize,
    max_queued_bytes: usize,
}

impl Default for PendingUploads {
    fn default() -> Self {
        Self::new()
    }
}

impl PendingUploads {
    /// Create a new upload queue owning its multi-producer channel bounded by
    /// `DEFAULT_MAX_QUEUED_UPLOAD_BYTES` (64 MiB).
    pub fn new() -> Self {
        Self::with_max_queued_bytes(DEFAULT_MAX_QUEUED_UPLOAD_BYTES)
    }

    /// Create a new upload queue with an explicit CPU queued byte ceiling.
    pub fn with_max_queued_bytes(max_queued_bytes: usize) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx: Some(tx),
            rx: Some(rx),
            queued: HashMap::new(),
            queued_bytes: 0,
            max_queued_bytes,
        }
    }

    /// Clone a producer sender for a background tile worker or decode pool.
    pub fn sender(&self) -> Option<Sender<(TileKey, DecodedTile)>> {
        self.tx.clone()
    }

    /// Drop the internal receiver and queued payloads during deterministic
    /// shutdown so background producers exit promptly on send.
    pub fn close(&mut self) {
        self.tx = None;
        self.rx = None;
        self.queued.clear();
        self.queued_bytes = 0;
    }

    /// Total CPU bytes currently buffered across queued decoded tiles.
    pub fn queued_bytes(&self) -> usize {
        self.queued_bytes
    }

    /// Return whether `key` is currently decoded and waiting for GPU upload.
    pub fn contains(&self, key: &TileKey) -> bool {
        self.queued.contains_key(key)
    }

    /// Count queued decoded tiles belonging to `source`.
    pub fn count_for_source(&self, source: SourceId) -> usize {
        self.queued.keys().filter(|k| k.source == source).count()
    }

    /// Discard any queued decoded tiles belonging to `source`.
    pub fn clear_source(&mut self, source: SourceId) {
        self.drain_channel_into_queue(None, None, &[]);
        self.retain_queued(|k| k.source != source);
    }

    /// Discard any queued decoded tiles matching `predicate`.
    pub fn clear_where<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&TileKey) -> bool,
    {
        self.drain_channel_into_queue(None, None, &[]);
        self.retain_queued(|k| !predicate(k));
    }

    fn insert_queued(&mut self, key: TileKey, decoded: DecodedTile) {
        let added = decoded.byte_size();
        if let Some(prev) = self.queued.insert(key, decoded) {
            self.queued_bytes = self.queued_bytes.saturating_sub(prev.byte_size());
        }
        self.queued_bytes = self.queued_bytes.saturating_add(added);
    }

    fn remove_queued(&mut self, key: &TileKey) -> Option<DecodedTile> {
        let removed = self.queued.remove(key)?;
        self.queued_bytes = self.queued_bytes.saturating_sub(removed.byte_size());
        Some(removed)
    }

    fn retain_queued(&mut self, mut keep: impl FnMut(&TileKey) -> bool) {
        let mut bytes = 0usize;
        self.queued.retain(|key, tile| {
            let ok = keep(key);
            if ok {
                bytes = bytes.saturating_add(tile.byte_size());
            }
            ok
        });
        self.queued_bytes = bytes;
    }

    fn drain_channel_into_queue(
        &mut self,
        cache: Option<&GpuTileCache>,
        wanted: Option<&HashSet<TileKey>>,
        pinned_root_sources: &[SourceId],
    ) {
        let Some(rx) = &self.rx else {
            return;
        };
        let mut incoming = Vec::new();
        while let Ok(item) = rx.try_recv() {
            incoming.push(item);
        }
        for (key, decoded) in incoming {
            if cache.is_some_and(|c| c.contains(&key)) {
                continue;
            }
            if let Some(wanted_set) = wanted
                && !wanted_set.contains(&key)
                && !is_pinned_root(&key, pinned_root_sources)
            {
                continue;
            }
            self.insert_queued(key, decoded);
        }
    }

    fn enforce_queued_byte_cap(&mut self, pinned_root_sources: &[SourceId]) {
        if self.queued_bytes <= self.max_queued_bytes {
            return;
        }
        // Drop finest non-pinned tiles first so coarsest carriers and pinned
        // roots always survive bursts from fast SSD/COG decode workers.
        let mut candidates: Vec<TileKey> = self
            .queued
            .keys()
            .copied()
            .filter(|k| !is_pinned_root(k, pinned_root_sources))
            .collect();
        candidates.sort_by_key(|k| Reverse(k.coarsest_first_key()));
        for key in candidates {
            if self.queued_bytes <= self.max_queued_bytes {
                break;
            }
            let _ = self.remove_queued(&key);
        }
    }

    /// Drain the completion channel, prune cached/unwanted tiles (retaining
    /// roots of `params.pinned_root_sources`), enforce `max_queued_bytes`, and
    /// upload coarsest-first within `(byte_budget, time_budget)` while
    /// protecting active-frame carriers and guaranteeing `>= 1` tile per frame
    /// whenever an admissible tile is queued.
    pub fn drain_and_upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cache: &mut GpuTileCache,
        params: UploadPassParams<'_>,
    ) -> UploadBatchReport {
        // 1. Drain the completion channel, immediately dropping cached or
        //    unwanted non-pinned tiles so stale bursts never spike RAM.
        self.drain_channel_into_queue(Some(cache), params.wanted, params.pinned_root_sources);

        // 2. Prune already-queued tiles that became cached or unwanted.
        self.retain_queued(|key| {
            if cache.contains(key) {
                return false;
            }
            match params.wanted {
                Some(wanted_set) => {
                    wanted_set.contains(key) || is_pinned_root(key, params.pinned_root_sources)
                }
                None => true,
            }
        });

        // 3. Bound total queued CPU bytes, dropping finest non-pinned tiles first.
        self.enforce_queued_byte_cap(params.pinned_root_sources);

        if self.queued.is_empty() {
            return UploadBatchReport::default();
        }

        // 4. Order remaining tiles deterministically coarsest-first and upload
        //    within budget (note steps 1 & 2 already pruned cached keys).
        let mut order: Vec<TileKey> = self.queued.keys().copied().collect();
        order.sort_by_key(TileKey::coarsest_first_key);
        self.upload_ordered_tiles(device, queue, cache, &params, order)
    }

    fn upload_ordered_tiles(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cache: &mut GpuTileCache,
        params: &UploadPassParams<'_>,
        order: Vec<TileKey>,
    ) -> UploadBatchReport {
        let start = Instant::now();
        let mut uploaded = 0usize;
        let mut evicted = 0usize;
        let mut bytes_uploaded = 0usize;
        let mut deferred_protected = 0usize;
        let mut max_active_tile_bytes = 0usize;

        for key in order {
            // Anti-starvation: always upload at least 1 admissible tile per frame.
            if uploaded >= 1
                && (bytes_uploaded >= params.byte_budget || start.elapsed() >= params.time_budget)
            {
                break;
            }
            let Some(decoded) = self.queued.get(&key) else {
                continue;
            };
            let is_protected = |k: &TileKey| params.protected.is_some_and(|set| set.contains(k));
            match cache.upload_and_insert_protecting(device, queue, key, decoded, is_protected) {
                UploadInsertOutcome::Inserted {
                    tile,
                    evicted: ev_count,
                } => {
                    bytes_uploaded = bytes_uploaded.saturating_add(tile.byte_size);
                    evicted = evicted.saturating_add(ev_count);
                    uploaded += 1;
                    if params.observed_source == Some(key.source) {
                        max_active_tile_bytes = max_active_tile_bytes.max(tile.byte_size);
                    }
                    let _ = self.remove_queued(&key);
                }
                UploadInsertOutcome::RefusedByProtection => {
                    deferred_protected += 1;
                }
                UploadInsertOutcome::Rejected => {
                    let _ = self.remove_queued(&key);
                }
            }
        }

        let remaining = self.queued.len();
        let remaining_ready = remaining.saturating_sub(deferred_protected);
        UploadBatchReport {
            uploaded,
            evicted,
            remaining,
            remaining_ready,
            deferred_protected,
            max_active_tile_bytes,
        }
    }
}

/// Pure planetary basemap carrier-routing policy shared by `DrawSnapshot::build`
/// and `BasemapManager::schedule_for_tiles`.
///
/// Enforces the explicit `PolarCap` + live-basemap contract:
/// - `PolarCap` tiles always resolve against `SourceId::BUNDLED_BASEMAP` (never
///   a live HTTP source).
/// - Non-polar live tiles fall back to cached `SourceId::BUNDLED_BASEMAP`
///   ancestors while streaming or offline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BasemapCarrierPolicy {
    pub active_source: SourceId,
}

impl BasemapCarrierPolicy {
    pub const fn new(active_source: SourceId) -> Self {
        Self { active_source }
    }

    /// Primary [`TileKey`] serving `tex_id` under this basemap policy.
    pub const fn primary_key(self, tex_id: QuadTileId) -> TileKey {
        if matches!(tex_id.scheme, PlanetaryScheme::PolarCap { .. }) {
            bundled_tile_key(tex_id)
        } else {
            source_tile_key(self.active_source, tex_id)
        }
    }

    /// Optional offline bundled fallback [`TileKey`] when `primary_key(tex_id)`
    /// has no cached carrier in VRAM yet.
    pub const fn fallback_key(self, tex_id: QuadTileId) -> Option<TileKey> {
        if matches!(tex_id.scheme, PlanetaryScheme::PolarCap { .. })
            || self.active_source.0 == SourceId::BUNDLED_BASEMAP.0
        {
            None
        } else {
            Some(bundled_tile_key(tex_id))
        }
    }

    /// Resolve the finest cached GPU carrier tile and composed UV sub-rectangle
    /// for `(tex_id, uv_rect)`.
    pub fn resolve_carrier(
        self,
        tex_id: QuadTileId,
        uv_rect: [f32; 4],
        gpu_cache: &mut GpuTileCache,
    ) -> Option<(TileKey, [f32; 4], Arc<GpuTile>)> {
        let primary = self.primary_key(tex_id);
        if let Some((ck, c_uv, tile)) = gpu_cache.carrier(&primary) {
            return Some((ck, compose_uv_rect(c_uv, uv_rect), tile));
        }
        let fallback = self.fallback_key(tex_id)?;
        gpu_cache
            .carrier(&fallback)
            .map(|(ck, c_uv, tile)| (ck, compose_uv_rect(c_uv, uv_rect), tile))
    }
}

/// Build the canonical display `TileKey` for a tile belonging to `source`.
pub const fn source_tile_key(source: SourceId, id: QuadTileId) -> TileKey {
    TileKey::display_quad(source, id)
}

/// Build the canonical display `TileKey` for a bundled offline basemap tile.
pub const fn bundled_tile_key(id: QuadTileId) -> TileKey {
    TileKey::display_quad(SourceId::BUNDLED_BASEMAP, id)
}

#[cfg(test)]
#[path = "gpu_tiles_tests.rs"]
mod tests;
