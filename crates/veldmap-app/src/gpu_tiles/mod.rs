//! GPU tile ownership (`GpuTile`), VRAM-budgeted [`GpuTileCache`], and
//! bounded, coarsest-first [`PendingUploads`].
//!
//! `GpuTile` owns strictly its `Texture`, `TextureView`, `PixelFormat`,
//! dimensions (`width`, `height`, `valid_w`, `valid_h`, `gutter`), exact byte
//! size (including CPU-generated mip levels), and monotonic `tile_id: u64` —
//! never pipeline-specific `BindGroup`s or `view_formats`.

mod gpu_cache;
mod upload_queue;

pub use gpu_cache::{GpuTile, GpuTileCache};
pub(crate) use upload_queue::{DEFAULT_UPLOAD_BYTE_BUDGET, DEFAULT_UPLOAD_TIME_BUDGET};
pub use upload_queue::{
    PendingUploads, UploadBatchReport, UploadChannelClosed, UploadPassParams, UploadSender,
};
