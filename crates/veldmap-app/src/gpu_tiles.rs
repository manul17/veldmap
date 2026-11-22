//! GPU tile ownership (`GpuTile`), VRAM-budgeted `GpuTileCache`, multi-producer
//! coarsest-first `PendingUploads`, and the virtual offline bundled basemap
//! worker (`BundledBasemapProvider`).
//!
//! `GpuTile` owns strictly its `Texture`, `TextureView`, `PixelFormat`,
//! dimensions, exact byte size (including CPU-generated mip levels), and
//! monotonic `tile_id: u64` — never pipeline-specific `BindGroup`s or
//! `view_formats`.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use veldmap_core::tile::{
    BandKey, DecodedTile, PixelFormat, QuadTileId, SourceId, TileCache, TileKey, TileScheme,
};
use veldmap_raster::BundledBasemap;

/// Embedded public-domain NASA Blue Marble Next Generation equirectangular
/// global basemap (`3600 x 1800` JPEG, see `assets/README.md`).
pub const BUNDLED_BASEMAP_JPEG: &[u8] =
    include_bytes!("../../../assets/basemap/blue_marble_ng.jpg");

/// Default GPU tile VRAM cache cap (`256 MiB`).
pub const DEFAULT_GPU_TILE_CACHE_BYTES: usize = 256 * 1024 * 1024;

/// Default per-frame upload byte budget (`4 MiB`).
pub const DEFAULT_UPLOAD_BYTE_BUDGET: usize = 4 * 1024 * 1024;

/// Default per-frame upload wall-clock budget (`2 ms`).
pub const DEFAULT_UPLOAD_TIME_BUDGET: Duration = Duration::from_millis(2);

/// UI repaint wakeup callback invoked by background tile workers on arrival.
pub type Wakeup = Arc<dyn Fn() + Send + Sync>;

/// Immutable GPU-resident tile resource.
pub struct GpuTile {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub format: PixelFormat,
    pub width: u32,
    pub height: u32,
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
            .field("byte_size", &self.byte_size)
            .finish()
    }
}

/// Convert domain [`PixelFormat`] to the corresponding `wgpu::TextureFormat`.
pub fn wgpu_format(format: PixelFormat) -> wgpu::TextureFormat {
    match format {
        PixelFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        PixelFormat::R32Float => wgpu::TextureFormat::R32Float,
        PixelFormat::R8Uint => wgpu::TextureFormat::R8Uint,
        PixelFormat::R16Uint => wgpu::TextureFormat::R16Uint,
    }
}

/// Compute true GPU texture byte size across `mip_level_count` mip levels.
pub fn texture_byte_size(
    width: u32,
    height: u32,
    format: PixelFormat,
    mip_level_count: u32,
) -> usize {
    let bpp = format.bytes_per_pixel();
    let mut total = 0usize;
    let mut w = width.max(1);
    let mut h = height.max(1);
    for _ in 0..mip_level_count.max(1) {
        total = total.saturating_add((w as usize) * (h as usize) * bpp);
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    total
}

/// VRAM-budgeted O(1) LRU cache of immutable `Arc<GpuTile>` instances owned on
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

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
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

    /// Remove every cached GPU tile belonging to `source`, returning how many
    /// tiles were removed.
    pub fn remove_source(&mut self, source: SourceId) -> usize {
        self.cache.remove_where(|k| k.source == source).len()
    }

    /// Remove every cached GPU tile matching `predicate`.
    pub fn remove_where<F>(&mut self, predicate: F) -> usize
    where
        F: FnMut(&TileKey) -> bool,
    {
        self.cache.remove_where(predicate).len()
    }

    /// Appetite cap in tiles for one active pyramid (`< 50%` of VRAM budget).
    pub fn max_tiles_for_pyramid(&self, active_pyramids: usize, tile_bytes: usize) -> usize {
        self.cache
            .max_tiles_for_pyramid(active_pyramids, tile_bytes)
    }

    /// Resolve the finest cached tile covering `key` (exact or nearest coarser
    /// ancestor), touching the carrier and all cached ancestors up to the root.
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
        self.upload_and_insert_protecting(device, queue, key, decoded, |_| false)
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
    ) -> Option<(Arc<GpuTile>, usize)>
    where
        F: Fn(&TileKey) -> bool,
    {
        let max_dim = device.limits().max_texture_dimension_2d;
        if decoded.width == 0
            || decoded.height == 0
            || decoded.width > max_dim
            || decoded.height > max_dim
        {
            tracing::warn!(
                ?key,
                width = decoded.width,
                height = decoded.height,
                max_dim,
                "Skipping decoded tile with invalid or oversized GPU dimensions"
            );
            return None;
        }
        if !decoded.validate() {
            tracing::warn!(
                ?key,
                width = decoded.width,
                height = decoded.height,
                mip_levels = decoded.mip_level_count(),
                actual_bytes = decoded.byte_size(),
                "Skipping decoded tile failing mip/buffer validation"
            );
            return None;
        }

        let mip_level_count = decoded.mip_level_count();
        let byte_size = texture_byte_size(
            decoded.width,
            decoded.height,
            decoded.format,
            mip_level_count,
        );
        // Pre-check cache admission before allocating GPU textures so
        // saturated active frames never waste texture allocations.
        if !self
            .cache
            .can_admit_protecting(&key, byte_size, &is_protected)
        {
            return None;
        }

        let bpp = decoded.format.bytes_per_pixel() as u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("veldmap gpu tile"),
            size: wgpu::Extent3d {
                width: decoded.width,
                height: decoded.height,
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format(decoded.format),
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
        let tile_id = self.next_tile_id;
        self.next_tile_id = self.next_tile_id.wrapping_add(1);
        let tile = Arc::new(GpuTile {
            texture,
            view,
            format: decoded.format,
            width: decoded.width,
            height: decoded.height,
            byte_size,
            tile_id,
        });

        match self
            .cache
            .insert_protecting(key, Arc::clone(&tile), byte_size, is_protected)
        {
            Ok(evicted) => Some((tile, evicted.len())),
            Err(_) => None,
        }
    }
}

/// Summary of one frame's [`PendingUploads::drain_and_upload`] pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct UploadBatchReport {
    pub uploaded: usize,
    pub evicted: usize,
    pub remaining: usize,
}

/// Multi-producer completion-channel owner and coarsest-first budgeted GPU tile
/// upload queue owned on `VeldmapApp`.
pub struct PendingUploads {
    tx: Option<Sender<(TileKey, DecodedTile)>>,
    rx: Option<Receiver<(TileKey, DecodedTile)>>,
    queued: HashMap<TileKey, DecodedTile>,
}

impl Default for PendingUploads {
    fn default() -> Self {
        Self::new()
    }
}

impl PendingUploads {
    /// Create a new upload queue owning its multi-producer channel.
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx: Some(tx),
            rx: Some(rx),
            queued: HashMap::new(),
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
    }

    /// Number of decoded tiles currently waiting to upload.
    pub fn len(&self) -> usize {
        self.queued.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queued.is_empty()
    }

    /// Return whether `key` is currently decoded and waiting for GPU upload.
    pub fn contains(&self, key: &TileKey) -> bool {
        self.queued.contains_key(key)
    }

    /// Discard any queued decoded tiles belonging to `source`.
    pub fn clear_source(&mut self, source: SourceId) {
        self.drain_channel_into_queue();
        self.queued.retain(|k, _| k.source != source);
    }

    fn drain_channel_into_queue(&mut self) {
        if let Some(rx) = &self.rx {
            while let Ok((key, decoded)) = rx.try_recv() {
                self.queued.insert(key, decoded);
            }
        }
    }

    /// Drain the entire completion channel, prune stale tiles not in `wanted`
    /// (retaining active-source root tiles where `key.parent().is_none()`), and
    /// upload coarsest-first within `(byte_budget, time_budget)` while
    /// protecting active-frame carriers and guaranteeing `>= 1` tile per frame
    /// whenever an admissible tile is queued.
    #[expect(
        clippy::too_many_arguments,
        reason = "upload pass takes explicit GPU handles, active source, wanted/protected sets, and budgets"
    )]
    pub fn drain_and_upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cache: &mut GpuTileCache,
        active_source: SourceId,
        wanted: Option<&HashSet<TileKey>>,
        protected: Option<&HashSet<TileKey>>,
        byte_budget: usize,
        time_budget: Duration,
    ) -> UploadBatchReport {
        // 1. Drain the entire completion channel every frame so background
        //    producers never back up behind stale tiles.
        self.drain_channel_into_queue();

        // 2. Prune tiles already cached or no longer wanted (retaining
        //    active-source roots and bundled offline/polar fallback roots via
        //    `key.parent().is_none()` so Frame-1 pre-seeded roots never drop).
        self.queued.retain(|key, _| {
            if cache.contains(key) {
                return false;
            }
            match wanted {
                Some(wanted_set) => {
                    wanted_set.contains(key)
                        || is_active_source_root(key, active_source)
                        || is_active_source_root(key, SourceId::BUNDLED_BASEMAP)
                }
                None => true,
            }
        });

        if self.queued.is_empty() {
            return UploadBatchReport::default();
        }

        // 3. Order remaining tiles deterministically coarsest-first.
        let mut order: Vec<TileKey> = self.queued.keys().copied().collect();
        order.sort_by_key(TileKey::coarsest_first_key);

        let start = Instant::now();
        let mut uploaded = 0usize;
        let mut evicted = 0usize;
        let mut bytes_uploaded = 0usize;

        for key in order {
            // Anti-starvation: always upload at least 1 tile per frame.
            if uploaded >= 1 && (bytes_uploaded >= byte_budget || start.elapsed() >= time_budget) {
                break;
            }
            let Some(decoded) = self.queued.remove(&key) else {
                continue;
            };
            if cache.contains(&key) {
                continue;
            }
            let is_protected = |k: &TileKey| protected.is_some_and(|set| set.contains(k));
            if let Some((_, ev_count)) =
                cache.upload_and_insert_protecting(device, queue, key, &decoded, is_protected)
            {
                bytes_uploaded = bytes_uploaded.saturating_add(decoded.byte_size());
                evicted = evicted.saturating_add(ev_count);
                uploaded += 1;
            }
        }

        UploadBatchReport {
            uploaded,
            evicted,
            remaining: self.queued.len(),
        }
    }
}

/// Build the canonical `TileKey` for a tile belonging to `source`.
pub const fn source_tile_key(source: SourceId, id: QuadTileId) -> TileKey {
    TileKey {
        source,
        band: BandKey::Display,
        scheme: id.scheme,
        format: PixelFormat::Rgba8UnormSrgb,
        gutter: 0,
        decode_rev: 0,
        level: id.level,
        x: id.x,
        y: id.y,
    }
}

/// Build the canonical `TileKey` for a bundled offline basemap tile.
pub const fn bundled_tile_key(id: QuadTileId) -> TileKey {
    source_tile_key(SourceId::BUNDLED_BASEMAP, id)
}

#[derive(Default)]
struct WorkerQueue {
    pending: Vec<TileKey>,
    failed: HashSet<TileKey>,
    shutdown: bool,
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Return whether `key` is a coarsest root tile of `active_source` (works for
/// both planetary `z = 0` and `RasterPyramid` `level == max_level`).
pub const fn is_active_source_root(key: &TileKey, active_source: SourceId) -> bool {
    key.source.0 == active_source.0 && key.parent().is_none()
}

/// Virtual offline `z = 0..3` Geodetic/WebMercator + `z = 0` PolarCap basemap
/// tile provider backed by a dedicated background worker thread, coarsest-first
/// cancellable queue, and injected [`Wakeup`].
pub struct BundledBasemapProvider {
    shared: Arc<(Mutex<WorkerQueue>, Condvar)>,
    in_flight: HashSet<TileKey>,
    worker: Option<JoinHandle<()>>,
}

impl BundledBasemapProvider {
    /// Spawn the bundled offline basemap provider pushing mipmapped decoded
    /// tiles into `done_tx` and invoking `wakeup` on each arrival.
    pub fn spawn(done_tx: Sender<(TileKey, DecodedTile)>, wakeup: Wakeup) -> Self {
        let shared = Arc::new((Mutex::new(WorkerQueue::default()), Condvar::new()));
        let worker_shared = Arc::clone(&shared);

        let worker = match thread::Builder::new()
            .name("veldmap-bundled-basemap".to_owned())
            .spawn(move || {
                let basemap = match BundledBasemap::from_encoded_bytes(BUNDLED_BASEMAP_JPEG) {
                    Ok(b) => b,
                    Err(err) => {
                        tracing::error!(error = %err, "Failed to decode bundled offline basemap");
                        let (lock, _) = &*worker_shared;
                        let mut state = lock_unpoisoned(lock);
                        state.shutdown = true;
                        state.pending.clear();
                        return;
                    }
                };
                loop {
                    let key = {
                        let (lock, cvar) = &*worker_shared;
                        let mut state = lock_unpoisoned(lock);
                        while state.pending.is_empty() && !state.shutdown {
                            state = match cvar.wait(state) {
                                Ok(guard) => guard,
                                Err(poisoned) => poisoned.into_inner(),
                            };
                        }
                        if state.shutdown {
                            break;
                        }
                        // Pop the coarsest pending tile first.
                        let best_idx = state
                            .pending
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, k)| k.coarsest_first_key())
                            .map(|(i, _)| i);
                        match best_idx {
                            Some(i) => state.pending.swap_remove(i),
                            None => continue,
                        }
                    };

                    let id = QuadTileId {
                        scheme: key.scheme,
                        level: key.level,
                        x: key.x,
                        y: key.y,
                    };
                    match basemap.render_tile_with_mipmaps(id) {
                        Ok(tile) => {
                            if done_tx.send((key, tile)).is_err() {
                                break;
                            }
                            (wakeup)();
                        }
                        Err(err) => {
                            tracing::debug!(?key, error = %err, "Skipping out-of-range bundled tile");
                            let (lock, _) = &*worker_shared;
                            lock_unpoisoned(lock).failed.insert(key);
                        }
                    }
                }
            }) {
            Ok(handle) => Some(handle),
            Err(err) => {
                tracing::error!(error = %err, "Failed to spawn bundled basemap worker thread");
                None
            }
        };

        Self {
            shared,
            in_flight: HashSet::new(),
            worker,
        }
    }

    /// Number of tiles currently tracked as in flight.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Whether `key` is already tracked as in flight.
    pub fn is_in_flight(&self, key: &TileKey) -> bool {
        self.in_flight.contains(key)
    }

    /// Enqueue `key` for background tile rendering if not already in flight or
    /// terminally failed. Returns `true` if a new request was submitted.
    pub fn request(&mut self, key: TileKey) -> bool {
        if self.worker.is_none() || self.in_flight.contains(&key) {
            return false;
        }
        let (lock, cvar) = &*self.shared;
        let mut state = lock_unpoisoned(lock);
        if state.shutdown || state.failed.contains(&key) {
            return false;
        }
        self.in_flight.insert(key);
        state.pending.push(key);
        cvar.notify_one();
        true
    }

    /// Clear in-flight markers and cancel unstarted queued requests for tiles
    /// that have already landed in `cache` or `pending_uploads`, failed
    /// terminally, or are no longer requested.
    pub fn reconcile(
        &mut self,
        cache: &GpuTileCache,
        pending_uploads: &PendingUploads,
        wanted: &HashSet<TileKey>,
    ) {
        let (lock, _) = &*self.shared;
        let mut state = lock_unpoisoned(lock);
        let WorkerQueue {
            pending,
            failed,
            shutdown,
        } = &mut *state;
        if *shutdown {
            self.in_flight.clear();
            pending.clear();
            return;
        }
        let keep = |k: &TileKey| {
            !cache.contains(k)
                && !pending_uploads.contains(k)
                && !failed.contains(k)
                && (wanted.contains(k) || is_active_source_root(k, SourceId::BUNDLED_BASEMAP))
        };
        self.in_flight.retain(keep);
        pending.retain(keep);
    }

    /// Pre-seed the coarsest root tiles (`scheme z = 0` + North/South
    /// `PolarCap z = 0`) so the entire planet renders immediately on startup
    /// and offline fallback stays warm.
    pub fn seed_roots(&mut self, scheme: TileScheme) {
        if let Some((cols, rows)) = scheme.grid_dims(0) {
            for y in 0..rows {
                for x in 0..cols {
                    self.request(bundled_tile_key(QuadTileId {
                        scheme,
                        level: 0,
                        x,
                        y,
                    }));
                }
            }
        }
        for north in [true, false] {
            self.request(bundled_tile_key(QuadTileId {
                scheme: TileScheme::PolarCap { north },
                level: 0,
                x: 0,
                y: 0,
            }));
        }
    }

    /// Signal shutdown without joining yet (paired with [`Self::join`]).
    pub fn signal_shutdown(&mut self) {
        let (lock, cvar) = &*self.shared;
        let mut state = lock_unpoisoned(lock);
        state.shutdown = true;
        state.pending.clear();
        cvar.notify_all();
    }

    /// Join the worker thread deterministically.
    pub fn join(&mut self) {
        self.signal_shutdown();
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for BundledBasemapProvider {
    fn drop(&mut self) {
        self.join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconcile_clears_queued_uploads_and_terminal_failures_from_in_flight() {
        let mut pending = PendingUploads::new();
        let tx = pending.sender().expect("sender");
        let cache = GpuTileCache::new(1024 * 1024);

        let k_queued = bundled_tile_key(QuadTileId {
            scheme: TileScheme::Geodetic,
            level: 1,
            x: 0,
            y: 0,
        });
        let k_failed = bundled_tile_key(QuadTileId {
            scheme: TileScheme::Geodetic,
            level: 9, // above BUNDLED_MAX_LEVEL
            x: 0,
            y: 0,
        });
        let k_active = bundled_tile_key(QuadTileId {
            scheme: TileScheme::Geodetic,
            level: 1,
            x: 1,
            y: 0,
        });

        // Simulate `k_queued` sitting in `PendingUploads::queued`.
        tx.send((
            k_queued,
            DecodedTile::new(1, 1, PixelFormat::Rgba8UnormSrgb, vec![0, 0, 0, 255]),
        ))
        .expect("send queued tile");
        pending.drain_channel_into_queue();
        assert!(pending.contains(&k_queued));

        let shared = Arc::new((Mutex::new(WorkerQueue::default()), Condvar::new()));
        {
            let mut st = lock_unpoisoned(&shared.0);
            st.pending.extend([k_queued, k_failed, k_active]);
            st.failed.insert(k_failed);
        }
        let mut provider = BundledBasemapProvider {
            shared,
            in_flight: HashSet::from([k_queued, k_failed, k_active]),
            worker: None,
        };

        let wanted = HashSet::from([k_queued, k_failed, k_active]);
        provider.reconcile(&cache, &pending, &wanted);

        assert!(
            !provider.is_in_flight(&k_queued),
            "queued tile must not stay in_flight (prevents appetite double-count)"
        );
        assert!(
            !provider.is_in_flight(&k_failed),
            "terminally failed tile must be cleared from in_flight"
        );
        assert!(provider.is_in_flight(&k_active));
        assert_eq!(provider.in_flight_count(), 1);
    }

    #[test]
    fn active_source_root_pinning_handles_planetary_and_raster_pyramid() {
        let live = SourceId::from_stable_key("eox");
        let wm_root = source_tile_key(
            live,
            QuadTileId {
                scheme: TileScheme::WebMercator,
                level: 0,
                x: 0,
                y: 0,
            },
        );
        let wm_child = source_tile_key(
            live,
            QuadTileId {
                scheme: TileScheme::WebMercator,
                level: 1,
                x: 0,
                y: 0,
            },
        );
        assert!(is_active_source_root(&wm_root, live));
        assert!(!is_active_source_root(&wm_root, SourceId::BUNDLED_BASEMAP));
        assert!(!is_active_source_root(&wm_child, live));

        let pyr_root = TileKey {
            source: live,
            band: BandKey::Display,
            scheme: TileScheme::RasterPyramid { max_level: 4 },
            format: PixelFormat::Rgba8UnormSrgb,
            gutter: 0,
            decode_rev: 0,
            level: 4,
            x: 0,
            y: 0,
        };
        let pyr_finest = TileKey {
            level: 0,
            ..pyr_root
        };
        assert!(is_active_source_root(&pyr_root, live));
        assert!(!is_active_source_root(&pyr_finest, live));
    }
}
