//! Immutable GPU tile ownership (`GpuTile`) and VRAM-budgeted LRU cache
//! (`GpuTileCache`).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use veldmap_core::tile::{DecodedTile, PixelFormat, SourceId, TileCache, TileKey};

/// Process-global monotonic tile ID generator preventing cross-cache `tile_id`
/// collisions when multiple [`GpuTileCache`] instances feed a shared renderer.
static NEXT_TILE_ID: AtomicU64 = AtomicU64::new(1);

/// Immutable GPU-resident tile resource.
pub struct GpuTile {
    _texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) format: PixelFormat,
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Valid interior pixel width excluding gutters (`valid_w + 2*gutter <= width`).
    pub(crate) valid_w: u32,
    /// Valid interior pixel height excluding gutters (`valid_h + 2*gutter <= height`).
    pub(crate) valid_h: u32,
    /// Border gutter width in texels (`0` for standard tiles, `1` for DEM tiles).
    pub(crate) gutter: u8,
    /// Number of uploaded mip levels on `_texture` (`>= 1`).
    pub(crate) mip_level_count: u32,
    pub(crate) byte_size: usize,
    pub(crate) tile_id: u64,
}

impl GpuTile {
    /// Pixel width of the uploaded GPU texture.
    pub fn width(&self) -> u32 {
        self.width
    }
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
            .field("mip_level_count", &self.mip_level_count)
            .field("byte_size", &self.byte_size)
            .finish()
    }
}

/// Convert domain [`PixelFormat`] to the corresponding `wgpu::TextureFormat`.
fn wgpu_format(format: PixelFormat) -> wgpu::TextureFormat {
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
pub(super) enum UploadInsertOutcome {
    /// Tile was uploaded and admitted into VRAM.
    Inserted { tile: Arc<GpuTile>, evicted: usize },
    /// Tile is valid and fits the cache ceiling, but all evictable entries are
    /// currently protected by the active frame (caller should keep it deferred).
    RefusedByProtection,
    /// Tile failed dimension/format/gutter/buffer validation or exceeds
    /// `cache.max_bytes()` (caller should discard it).
    Rejected,
}

/// VRAM-budgeted LRU cache of immutable `Arc<GpuTile>` textures.
pub struct GpuTileCache {
    cache: TileCache<TileKey, Arc<GpuTile>>,
}

impl GpuTileCache {
    /// Create an empty GPU tile cache bounded by `max_bytes`.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            cache: TileCache::new(max_bytes),
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.cache.total_bytes()
    }

    pub fn max_bytes(&self) -> usize {
        self.cache.max_bytes()
    }

    /// Update the VRAM byte ceiling live, evicting oldest entries if needed.
    pub(crate) fn set_max_bytes(&mut self, max_bytes: usize) {
        let evicted = self.cache.set_max_bytes(max_bytes);
        drop(evicted);
    }

    pub fn contains(&self, key: &TileKey) -> bool {
        self.cache.contains(key)
    }

    /// Read-only check whether `key` or any coarser ancestor carrier is cached.
    pub(crate) fn has_carrier(&self, key: &TileKey) -> bool {
        self.cache.has_carrier(key)
    }

    /// Remove every cached GPU tile belonging to `source`, returning how many
    /// tiles were removed.
    pub fn remove_source(&mut self, source: SourceId) -> usize {
        self.remove_where(|k| k.source == source)
    }

    /// Remove every cached GPU tile matching `predicate`, returning how many
    /// tiles were removed (useful for multi-geocell DEM mosaics or timelines).
    pub fn remove_where<F>(&mut self, predicate: F) -> usize
    where
        F: FnMut(&TileKey) -> bool,
    {
        self.cache.remove_where(predicate).len()
    }

    /// Proportional byte appetite cap for one weighted pyramid (`< 50%`
    /// combined ceiling across heterogeneous multi-band stacks).
    pub(crate) fn proportional_pyramid_byte_cap(
        &self,
        pyramid_weight: u64,
        total_weight: u64,
    ) -> usize {
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
    pub(crate) fn carrier(&mut self, key: &TileKey) -> Option<(TileKey, [f32; 4], Arc<GpuTile>)> {
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
    pub(super) fn upload_and_insert_protecting<F>(
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
        let tile_id = NEXT_TILE_ID.fetch_add(1, Ordering::Relaxed);
        let tile = Arc::new(GpuTile {
            _texture: texture,
            view,
            format: decoded.format(),
            width: decoded.width(),
            height: decoded.height(),
            valid_w: decoded.valid_w(),
            valid_h: decoded.valid_h(),
            gutter: decoded.gutter(),
            mip_level_count: decoded.mip_level_count(),
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
