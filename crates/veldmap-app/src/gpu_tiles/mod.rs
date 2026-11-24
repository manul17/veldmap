//! GPU tile ownership (`GpuTile`), VRAM-budgeted [`GpuTileCache`], and
//! bounded, coarsest-first [`PendingUploads`].
//!
//! `GpuTile` owns strictly its `Texture`, `TextureView`, `PixelFormat`,
//! dimensions (`width`, `height`, `valid_w`, `valid_h`, `gutter`), exact byte
//! size (including CPU-generated mip levels), and monotonic `tile_id: u64` —
//! never pipeline-specific `BindGroup`s or `view_formats`.

mod gpu_cache;
mod upload_queue;

use veldmap_core::tile::{PixelFormat, max_mip_levels, mip_chain_byte_size};

pub use gpu_cache::{GpuTile, GpuTileCache};
pub(crate) use upload_queue::{DEFAULT_UPLOAD_BYTE_BUDGET, DEFAULT_UPLOAD_TIME_BUDGET};
pub use upload_queue::{
    PendingUploads, UploadBatchReport, UploadChannelClosed, UploadPassParams, UploadSender,
};

/// Estimated GPU byte footprint of one square `Rgba8UnormSrgb` display tile
/// (`tile_px x tile_px`) including its complete CPU-generated mip chain.
pub(crate) fn display_tile_mip_bytes(tile_px: u32) -> usize {
    let mips = max_mip_levels(tile_px, tile_px);
    mip_chain_byte_size(tile_px, tile_px, PixelFormat::Rgba8UnormSrgb, mips).max(1)
}
