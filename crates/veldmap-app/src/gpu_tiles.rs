//! GPU tile ownership (`GpuTile`), VRAM-budgeted `GpuTileCache`, budgeted
//! coarsest-first `PendingUploads`, and the virtual offline bundled basemap
//! worker (`BundledBasemapProvider`).
//!
//! `GpuTile` owns strictly its `Texture`, `TextureView`, `PixelFormat`,
//! dimensions, exact byte size (including mip levels), and monotonic
//! `tile_id: u64` — never pipeline-specific `BindGroup`s or `view_formats`.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use veldmap_core::tile::{
    BandKey, PixelFormat, QuadTileId, SourceId, TileCache, TileKey, TileScheme,
};
use veldmap_raster::{BundledBasemap, DecodedTile};

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
///
/// Owns only the GPU texture, default view, pixel format, dimensions, true
/// byte size (including mip chain), and monotonic `tile_id: u64`.
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

    /// Validate and upload a CPU-decoded tile into GPU memory and insert it
    /// into the LRU cache. Returns `Some((tile, evicted_count))` on success,
    /// or `None` if dimensions, buffer size, or cache budget cannot admit it.
    pub fn upload_and_insert(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: TileKey,
        decoded: &DecodedTile,
    ) -> Option<(Arc<GpuTile>, usize)> {
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
        let bpp = decoded.format.bytes_per_pixel();
        let expected_len = (decoded.width as usize)
            .checked_mul(decoded.height as usize)
            .and_then(|px| px.checked_mul(bpp))?;
        if decoded.data.len() != expected_len {
            tracing::warn!(
                ?key,
                actual = decoded.data.len(),
                expected = expected_len,
                "Skipping decoded tile with mismatched pixel buffer length"
            );
            return None;
        }

        let mip_level_count = 1u32;
        let byte_size = texture_byte_size(
            decoded.width,
            decoded.height,
            decoded.format,
            mip_level_count,
        );
        if self.cache.max_bytes() == 0 || byte_size > self.cache.max_bytes() {
            return None;
        }

        let size = wgpu::Extent3d {
            width: decoded.width,
            height: decoded.height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("veldmap gpu tile"),
            size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format(decoded.format),
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &decoded.data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(decoded.width * (bpp as u32)),
                rows_per_image: Some(decoded.height),
            },
            size,
        );

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

        match self.cache.insert(key, Arc::clone(&tile), byte_size) {
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

/// Completion-channel drain and coarsest-first budgeted GPU tile upload queue
/// owned on `VeldmapApp`.
pub struct PendingUploads {
    rx: Receiver<(TileKey, DecodedTile)>,
    queued: HashMap<TileKey, DecodedTile>,
}

impl PendingUploads {
    pub fn new(rx: Receiver<(TileKey, DecodedTile)>) -> Self {
        Self {
            rx,
            queued: HashMap::new(),
        }
    }

    /// Number of decoded tiles currently waiting to upload.
    pub fn len(&self) -> usize {
        self.queued.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queued.is_empty()
    }

    /// Drain the entire completion channel, prune stale tiles not in `wanted`
    /// (unless they are root coarse fallback tiles), and upload coarsest-first
    /// within `(byte_budget, time_budget)` while guaranteeing `>= 1` tile per
    /// frame whenever the queue is non-empty.
    pub fn drain_and_upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cache: &mut GpuTileCache,
        wanted: Option<&HashSet<TileKey>>,
        byte_budget: usize,
        time_budget: Duration,
    ) -> UploadBatchReport {
        // 1. Drain the entire completion channel every frame so background
        //    producers never back up behind stale tiles.
        while let Ok((key, decoded)) = self.rx.try_recv() {
            self.queued.insert(key, decoded);
        }

        // 2. Prune tiles already cached or no longer wanted (retaining
        //    `level == 0` planetary roots so coarse fallback imagery stays warm).
        self.queued.retain(|key, _| {
            if cache.contains(key) {
                return false;
            }
            match wanted {
                Some(wanted_set) => {
                    wanted_set.contains(key) || (key.scheme.is_coarsest_zero() && key.level == 0)
                }
                None => true,
            }
        });

        if self.queued.is_empty() {
            return UploadBatchReport::default();
        }

        // 3. Order remaining tiles deterministically coarsest-first.
        let mut order: Vec<TileKey> = self.queued.keys().copied().collect();
        order.sort_by_key(|k| (k.scheme.coarseness_rank(k.level), k.scheme, k.y, k.x));

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
            bytes_uploaded = bytes_uploaded.saturating_add(decoded.byte_size());
            if let Some((_, ev_count)) = cache.upload_and_insert(device, queue, key, &decoded) {
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

/// Build the canonical `TileKey` for a bundled offline basemap tile.
pub const fn bundled_tile_key(id: QuadTileId) -> TileKey {
    TileKey {
        source: SourceId::BUNDLED_BASEMAP,
        band: BandKey::Display,
        scheme: id.scheme,
        format: PixelFormat::Rgba8UnormSrgb,
        decode_rev: 0,
        level: id.level,
        x: id.x,
        y: id.y,
    }
}

#[derive(Default)]
struct WorkerQueue {
    pending: Vec<TileKey>,
    shutdown: bool,
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn is_planetary_root(key: &TileKey) -> bool {
    key.scheme.is_coarsest_zero() && key.level == 0
}

/// Virtual offline `z = 0..3` Geodetic + `z = 0` PolarCap basemap tile
/// provider backed by a dedicated background worker thread, coarsest-first
/// cancellable queue, and injected [`Wakeup`].
pub struct BundledBasemapProvider {
    shared: Arc<(Mutex<WorkerQueue>, Condvar)>,
    in_flight: HashSet<TileKey>,
    worker: Option<JoinHandle<()>>,
}

impl BundledBasemapProvider {
    /// Spawn the bundled offline basemap provider delivering decoded tiles to
    /// the returned [`PendingUploads`] and invoking `wakeup` on each arrival.
    pub fn spawn(wakeup: Wakeup) -> (Self, PendingUploads) {
        let shared = Arc::new((Mutex::new(WorkerQueue::default()), Condvar::new()));
        let worker_shared = Arc::clone(&shared);
        let (done_tx, done_rx) = mpsc::channel::<(TileKey, DecodedTile)>();

        let worker = match thread::Builder::new()
            .name("veldmap-bundled-basemap".to_owned())
            .spawn(move || {
                let basemap = match BundledBasemap::from_encoded_bytes(BUNDLED_BASEMAP_JPEG) {
                    Ok(b) => b,
                    Err(err) => {
                        tracing::error!(error = %err, "Failed to decode bundled offline basemap");
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
                            .min_by_key(|(_, k)| {
                                (k.scheme.coarseness_rank(k.level), k.scheme, k.y, k.x)
                            })
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
                    match basemap.render_tile(id) {
                        Ok(tile) => {
                            if done_tx.send((key, tile)).is_err() {
                                break;
                            }
                            (wakeup)();
                        }
                        Err(err) => {
                            tracing::debug!(?key, error = %err, "Skipping out-of-range bundled tile");
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

        (
            Self {
                shared,
                in_flight: HashSet::new(),
                worker,
            },
            PendingUploads::new(done_rx),
        )
    }

    /// Number of tiles currently tracked as in flight.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Whether `key` is already tracked as in flight.
    pub fn is_in_flight(&self, key: &TileKey) -> bool {
        self.in_flight.contains(key)
    }

    /// Enqueue `key` for background tile rendering if not already in flight.
    /// Returns `true` if a new request was submitted.
    pub fn request(&mut self, key: TileKey) -> bool {
        if self.worker.is_none() || !self.in_flight.insert(key) {
            return false;
        }
        let (lock, cvar) = &*self.shared;
        let mut state = lock_unpoisoned(lock);
        if state.shutdown {
            self.in_flight.remove(&key);
            return false;
        }
        state.pending.push(key);
        cvar.notify_one();
        true
    }

    /// Clear in-flight markers and cancel unstarted queued requests for tiles
    /// that have already landed in `cache` or are no longer requested.
    pub fn reconcile(&mut self, cache: &GpuTileCache, wanted: &HashSet<TileKey>) {
        let keep = |k: &TileKey| !cache.contains(k) && (wanted.contains(k) || is_planetary_root(k));
        self.in_flight.retain(keep);
        let (lock, _) = &*self.shared;
        let mut state = lock_unpoisoned(lock);
        state.pending.retain(keep);
    }

    /// Pre-seed the coarsest root tiles (`Geodetic z = 0` + North/South
    /// `PolarCap z = 0`) so the entire planet renders immediately on startup.
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
}

impl Drop for BundledBasemapProvider {
    fn drop(&mut self) {
        {
            let (lock, cvar) = &*self.shared;
            let mut state = lock_unpoisoned(lock);
            state.shutdown = true;
            state.pending.clear();
            cvar.notify_all();
        }
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}
