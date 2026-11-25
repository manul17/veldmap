//! Native raster decoding, bounded CPU decode pools, linear-light mipmap
//! generation, virtual offline equirectangular basemap tiling, and analytical
//! GeoTIFF/BigTIFF/COG pyramid decoding for VeldMap.
//!
//! Keeps `veldmap-core` permanently free of image/codec dependencies and keeps
//! `veldmap-net` decoupled from raster codecs via an injected decode executor.

#![forbid(unsafe_code)]

use std::sync::Arc;

use thiserror::Error;
use veldmap_core::budget::BudgetError;
use veldmap_core::io::IoError;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::TileScheme;

mod adapters;
mod basemap;
mod color;
mod image_codec;
mod mipmap;
mod open;
mod pipeline;
mod pool;
mod sidecar;
mod tiff;
mod tile_codec;

pub use adapters::{JpegSource, PngSource};
pub use basemap::{BundledBasemap, BundledBasemapConfig, decode_map_tile_with_mipmaps};
pub use image_codec::{has_jpeg_magic, has_png_magic};
pub use mipmap::generate_tile_mipmaps;
pub use open::{OpenedRaster, SidecarReader, open_raster};
pub use pipeline::{
    AdmittedPass, AdmittedTile, BandInfo, BandSample, BandStats, DISPLAY_STREAM_SPEC,
    DecodeContext, GeoPlacement, GeoTie, HISTOGRAM_BINS, LevelPlan, MAX_SOURCE_SIDE, PlacementCrs,
    PyramidMeta, RasterDescription, RasterSource, SampleType, ServeMode, SharedChunkCache,
    TileEmit, TileFailureOutcome, TileScheduler, frame_from_description,
    frame_from_description_with_override, stream_spec_for_band,
};
pub use pool::{
    DecodeJob, DecodeJobKey, DecodePool, DecodePoolConfig, DecodePoolHandle, SubmitError,
    TwoLaneDecodePool, TwoLaneDecodePoolConfig,
};
pub use sidecar::{SidecarGeoref, load_sidecar_georef_with, parse_prj_text, parse_world_file};
pub use tiff::{MAX_IFDS, TiffSource, has_tiff_magic};
pub use tile_codec::{
    MAX_CACHED_TILE_DECODED_BYTES, MAX_LZ4_RAW_DECODED_BYTES, MAX_QOI_DECODED_BYTES,
    decode_lz4_raw_tile, decode_qoi_rgba8, encode_lz4_raw_tile, encode_qoi_rgba8,
};

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

/// Map a `std::io::Error` produced by a decoder reading from
/// [`veldmap_core::io::BufferedByteReader`] back into [`RasterError::Cancelled`]
/// or [`RasterError::Io`], preserving structured core I/O errors.
pub(crate) fn map_std_io_error(io_err: std::io::Error, cancel: &CancelToken) -> RasterError {
    if cancel.is_cancelled() {
        return RasterError::Cancelled;
    }
    match io_err.downcast::<IoError>() {
        Ok(core_err) => RasterError::from(core_err),
        Err(io_err) => RasterError::from(IoError::Io(io_err)),
    }
}

/// Errors returned by raster tile decoders and virtual tile providers.
///
/// Cancellation from `IoError::Cancelled` or `BudgetError::Cancelled` is
/// normalized into [`Self::Cancelled`] via explicit `From` implementations so
/// schedulers and caches never have to match three cancellation shapes.
#[derive(Clone, Debug, Error)]
pub enum RasterError {
    #[error("raster decode cancelled")]
    Cancelled,
    #[error("unsupported or unrecognized image format")]
    UnsupportedFormat,
    #[error("codec decoder limits exceeded: {0}")]
    DecoderLimitsExceeded(String),
    #[error("raster decoder panicked: {0}")]
    DecoderPanicked(String),
    #[error("I/O error: {0}")]
    Io(#[source] Arc<IoError>),
    #[error("memory budget exceeded: {0}")]
    Budget(BudgetError),
    #[error("raster decode failed: {0}")]
    Decode(String),
    #[error("invalid decoded image dimensions {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    #[error("image {width}x{height} exceeds decode limits ({estimated_bytes} B > {limit_bytes} B)")]
    LimitExceeded {
        width: u32,
        height: u32,
        estimated_bytes: u64,
        limit_bytes: u64,
    },
    #[error("tile request out of bounds or above max_level: {scheme:?} z={level} ({x}, {y})")]
    TileOutOfRange {
        scheme: TileScheme,
        level: u8,
        x: u32,
        y: u32,
    },
    #[error("georeference binding error: {0}")]
    Binding(String),
}

impl RasterError {
    /// Return whether this error represents task cancellation.
    pub const fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled | Self::Budget(BudgetError::Cancelled))
    }

    /// Return whether this error is deterministic and permanent for the
    /// requested tile/source, so schedulers should suppress retries
    /// immediately rather than allowing transient retry attempts.
    pub const fn is_terminal(&self) -> bool {
        match self {
            Self::UnsupportedFormat
            | Self::DecoderLimitsExceeded(_)
            | Self::Budget(BudgetError::ExceedsCap { .. })
            | Self::InvalidDimensions { .. }
            | Self::LimitExceeded { .. }
            | Self::TileOutOfRange { .. }
            | Self::Binding(_) => true,
            Self::Cancelled
            | Self::Budget(BudgetError::Cancelled)
            | Self::DecoderPanicked(_)
            | Self::Io(_)
            | Self::Decode(_) => false,
        }
    }
}

impl From<IoError> for RasterError {
    fn from(err: IoError) -> Self {
        if err.is_cancelled() {
            Self::Cancelled
        } else {
            Self::Io(Arc::new(err))
        }
    }
}

impl From<BudgetError> for RasterError {
    fn from(err: BudgetError) -> Self {
        match err {
            BudgetError::Cancelled => Self::Cancelled,
            other => Self::Budget(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_error_is_terminal_classifies_all_variants() {
        let terminal = [
            RasterError::UnsupportedFormat,
            RasterError::DecoderLimitsExceeded("limit".to_owned()),
            RasterError::Budget(BudgetError::ExceedsCap {
                requested: 10,
                cap: 5,
                detail: "cap".to_owned(),
            }),
            RasterError::InvalidDimensions {
                width: 0,
                height: 0,
            },
            RasterError::LimitExceeded {
                width: 1,
                height: 1,
                estimated_bytes: 10,
                limit_bytes: 5,
            },
            RasterError::TileOutOfRange {
                scheme: TileScheme::WebMercator,
                level: 1,
                x: 9,
                y: 9,
            },
            RasterError::Binding("bad crs".to_owned()),
        ];
        for err in &terminal {
            assert!(err.is_terminal(), "expected terminal for {err:?}");
        }

        let non_terminal = [
            RasterError::Cancelled,
            RasterError::Budget(BudgetError::Cancelled),
            RasterError::DecoderPanicked("boom".to_owned()),
            RasterError::Io(Arc::new(IoError::Io(std::io::Error::other("io")))),
            RasterError::Decode("corrupt".to_owned()),
        ];
        for err in &non_terminal {
            assert!(!err.is_terminal(), "expected non-terminal for {err:?}");
        }
    }

    #[test]
    fn raster_error_is_cancelled_covers_direct_and_budget_cancellation() {
        assert!(RasterError::Cancelled.is_cancelled());
        assert!(RasterError::Budget(BudgetError::Cancelled).is_cancelled());
        assert!(
            !RasterError::Budget(BudgetError::ExceedsCap {
                requested: 10,
                cap: 5,
                detail: "cap".to_owned(),
            })
            .is_cancelled()
        );
        assert!(!RasterError::Decode("err".to_owned()).is_cancelled());
    }
}
