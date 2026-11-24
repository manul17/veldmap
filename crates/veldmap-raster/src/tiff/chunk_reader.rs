//! Pooled `tiff::decoder::Decoder` management and positioned chunk reader
//! bridging `tiff` to [`Chunked`] and [`SharedChunkCache`].

use std::sync::{Arc, Mutex};

use tiff::decoder::{Decoder, DecodingResult, Limits};
use veldmap_core::budget::{BudgetLease, DecodeBudgetTracker};
use veldmap_core::io::{BufferedByteReader, ByteSource};
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::SourceId;

use crate::RasterError;
use crate::pipeline::chunk_cache::{ChunkKey, DecodedChunk, SharedChunkCache};
use crate::pipeline::driver::Chunked;
use crate::pipeline::grid;
use crate::pipeline::plan::DISPLAY_STREAM_SPEC;
use crate::pipeline::radiometry::DisplayMapping;

use super::TiffFormat;
use super::tags::{chunk_rgba, map_tiff_error};
use super::ycbcr::YCbCrParams;

/// Maximum idle `tiff::decoder::Decoder` handles retained per [`super::TiffSource`].
const MAX_POOLED_DECODERS: usize = 8;

/// Open a new `tiff::decoder::Decoder` backed by `byte_source` with explicit
/// `limits` and cancellation token `cancel`.
pub(super) fn open_decoder(
    byte_source: &Arc<dyn ByteSource>,
    limits: &Limits,
    cancel: &CancelToken,
) -> Result<Decoder<BufferedByteReader>, RasterError> {
    let reader = BufferedByteReader::new(Arc::clone(byte_source), cancel.clone());
    Ok(Decoder::new(reader)
        .map_err(|e| map_tiff_error(e, cancel))?
        .with_limits(limits.clone()))
}

struct DecoderEntry {
    decoder: Decoder<BufferedByteReader>,
    aimed_image: Option<usize>,
    aimed_chunk_count: Option<u64>,
}

/// Bounded pool of initialized `Decoder<BufferedByteReader>` instances so
/// pointwise tile cache misses reuse open decoders without re-reading the TIFF
/// header and re-walking the IFD chain from byte 0 on every tile.
pub(super) struct DecoderPool {
    byte_source: Arc<dyn ByteSource>,
    limits: Limits,
    idle: Mutex<Vec<DecoderEntry>>,
}

impl DecoderPool {
    pub(super) fn new(
        byte_source: Arc<dyn ByteSource>,
        limits: Limits,
        initial: Option<(Decoder<BufferedByteReader>, usize)>,
    ) -> Self {
        let mut entries = Vec::with_capacity(MAX_POOLED_DECODERS);
        if let Some((mut decoder, aimed_image)) = initial {
            decoder.inner().set_cancel(CancelToken::new());
            entries.push(DecoderEntry {
                decoder,
                aimed_image: Some(aimed_image),
                aimed_chunk_count: None,
            });
        }
        Self {
            byte_source,
            limits,
            idle: Mutex::new(entries),
        }
    }

    pub(super) fn checkout<'a>(
        &'a self,
        cancel: &CancelToken,
    ) -> Result<PooledDecoder<'a>, RasterError> {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let entry = if let Some(mut cached) = lock_unpoisoned(&self.idle).pop() {
            cached.decoder.inner().set_cancel(cancel.clone());
            cached
        } else {
            let decoder = open_decoder(&self.byte_source, &self.limits, cancel)?;
            DecoderEntry {
                decoder,
                aimed_image: Some(0),
                aimed_chunk_count: None,
            }
        };
        Ok(PooledDecoder {
            pool: self,
            entry: Some(entry),
            clean: false,
        })
    }
}

/// Decoded raw TIFF chunk held together with its active memory-budget lease so
/// callers keep working-memory reservation alive until post-processing finishes.
pub(super) struct BudgetedChunk {
    pub(super) raw: DecodingResult,
    pub(super) width: u32,
    pub(super) height: u32,
    _lease: Option<BudgetLease>,
}

/// RAII checkout guard returning a clean `Decoder` to [`DecoderPool`] on drop
/// with its `CancelToken` reset so cancelled requests never poison reuse.
///
/// Any mutating operation (`decoder_mut`, `seek_to_image`,
/// `read_budgeted_chunk`) automatically marks the checkout dirty until the
/// caller confirms success via [`Self::mark_clean`].
pub(super) struct PooledDecoder<'a> {
    pool: &'a DecoderPool,
    entry: Option<DecoderEntry>,
    clean: bool,
}

impl PooledDecoder<'_> {
    pub(super) fn mark_clean(&mut self) {
        self.clean = true;
    }

    fn entry_mut(&mut self) -> &mut DecoderEntry {
        #[expect(
            clippy::expect_used,
            reason = "INVARIANT: entry is only taken inside Drop"
        )]
        self.entry.as_mut().expect("pooled decoder present")
    }

    fn decoder_mut(&mut self) -> &mut Decoder<BufferedByteReader> {
        self.clean = false;
        &mut self.entry_mut().decoder
    }

    pub(super) fn seek_to_image(
        &mut self,
        image: usize,
        cancel: &CancelToken,
    ) -> Result<(), RasterError> {
        self.clean = false;
        let entry = self.entry_mut();
        if entry.aimed_image == Some(image) {
            return Ok(());
        }
        entry.aimed_image = None;
        entry.aimed_chunk_count = None;
        entry
            .decoder
            .seek_to_image(image)
            .map_err(|e| map_tiff_error(e, cancel))?;
        entry.aimed_image = Some(image);
        Ok(())
    }

    /// Compute the total chunk count of the currently aimed IFD, caching it on
    /// the active entry until the aimed IFD changes, and returning
    /// [`RasterError::InvalidDimensions`] when either image or chunk
    /// dimensions are zero.
    pub(super) fn ifd_chunk_layout(&mut self, cancel: &CancelToken) -> Result<u64, RasterError> {
        if let Some(count) = self.entry_mut().aimed_chunk_count {
            return Ok(count);
        }
        let (img_w, img_h) = self
            .decoder_mut()
            .dimensions()
            .map_err(|e| map_tiff_error(e, cancel))?;
        if img_w == 0 || img_h == 0 {
            return Err(RasterError::InvalidDimensions {
                width: img_w,
                height: img_h,
            });
        }
        let (cw, ch) = self.decoder_mut().chunk_dimensions();
        if cw == 0 || ch == 0 {
            return Err(RasterError::InvalidDimensions {
                width: cw,
                height: ch,
            });
        }
        let count = u64::from(img_w.div_ceil(cw)) * u64::from(img_h.div_ceil(ch));
        self.entry_mut().aimed_chunk_count = Some(count);
        Ok(count)
    }

    /// Read raw chunk `index` from the currently aimed IFD under optional
    /// working-memory budget tracking.
    ///
    /// Returns [`RasterError::InvalidDimensions`] when the chunk reports zero
    /// width or height so callers can either propagate the error (tile decode)
    /// or skip empty edge chunks (stretch sampling).
    pub(super) fn read_budgeted_chunk(
        &mut self,
        index: u32,
        depth: u32,
        includes_rgba8: bool,
        budget_tracker: Option<&DecodeBudgetTracker>,
        cancel: &CancelToken,
    ) -> Result<BudgetedChunk, RasterError> {
        let (dw, dh) = self.decoder_mut().chunk_data_dimensions(index);
        if dw == 0 || dh == 0 {
            return Err(RasterError::InvalidDimensions {
                width: dw,
                height: dh,
            });
        }
        let est_bytes = grid::chunk_working_bytes(dw, dh, depth, includes_rgba8);
        let lease = match budget_tracker {
            Some(tracker) => Some(tracker.reserve_blocking(est_bytes, cancel)?),
            None => None,
        };
        let raw = self
            .decoder_mut()
            .read_chunk(index)
            .map_err(|e| map_tiff_error(e, cancel))?;
        Ok(BudgetedChunk {
            raw,
            width: dw,
            height: dh,
            _lease: lease,
        })
    }
}

impl Drop for PooledDecoder<'_> {
    fn drop(&mut self) {
        if !self.clean {
            return;
        }
        if let Some(mut entry) = self.entry.take() {
            entry.decoder.inner().set_cancel(CancelToken::new());
            let mut idle = lock_unpoisoned(&self.pool.idle);
            if idle.len() < MAX_POOLED_DECODERS {
                idle.push(entry);
            }
        }
    }
}

/// Explicitly models the two chunk-reading modes of [`TiffChunkReader`].
pub(super) enum ChunkReadMode<'a> {
    /// Pointwise random access: deduplicates through `chunk_cache` and reserves
    /// each fresh chunk on `budget_tracker`.
    Pointwise {
        chunk_cache: &'a SharedChunkCache,
        budget_tracker: &'a DecodeBudgetTracker,
    },
    /// Sequential full-IFD pass: bypasses `SharedChunkCache` and per-chunk
    /// budget reservation because the caller already holds the full pass lease.
    SequentialPass,
}

pub(super) struct TiffChunkReader<'a> {
    decoder_pool: &'a DecoderPool,
    source_id: SourceId,
    format: TiffFormat,
    ifd_ycbcr: &'a [(usize, YCbCrParams)],
    mapping: DisplayMapping,
    mode: ChunkReadMode<'a>,
    pooled: Option<PooledDecoder<'a>>,
}

impl<'a> TiffChunkReader<'a> {
    pub(super) fn new(
        decoder_pool: &'a DecoderPool,
        source_id: SourceId,
        format: TiffFormat,
        ifd_ycbcr: &'a [(usize, YCbCrParams)],
        mapping: DisplayMapping,
        mode: ChunkReadMode<'a>,
    ) -> Self {
        Self {
            decoder_pool,
            source_id,
            format,
            ifd_ycbcr,
            mapping,
            mode,
            pooled: None,
        }
    }

    fn ycbcr_for(&self, image: usize) -> Option<YCbCrParams> {
        self.ifd_ycbcr
            .iter()
            .find_map(|&(idx, params)| (idx == image).then_some(params))
    }

    fn decode_chunk_uncached(
        &mut self,
        image: usize,
        index: u32,
        budget_tracker: Option<&DecodeBudgetTracker>,
        cancel: &CancelToken,
    ) -> Result<DecodedChunk, RasterError> {
        let pooled = match self.pooled.as_mut() {
            Some(pooled) => pooled,
            None => self.pooled.insert(self.decoder_pool.checkout(cancel)?),
        };
        pooled.seek_to_image(image, cancel)?;
        let total_chunks = pooled.ifd_chunk_layout(cancel)?;
        if u64::from(index) >= total_chunks {
            return Err(RasterError::Decode(format!(
                "chunk index {index} out of range for IFD {image} ({total_chunks} chunks)"
            )));
        }
        let chunk =
            pooled.read_budgeted_chunk(index, self.format.depth, true, budget_tracker, cancel)?;
        pooled.mark_clean();
        let rgba = chunk_rgba(
            &self.mapping,
            &chunk.raw,
            self.format.pixel_layout,
            self.ycbcr_for(image),
            chunk.width,
            chunk.height,
        )?;
        Ok(DecodedChunk {
            data: rgba,
            width: chunk.width,
            height: chunk.height,
        })
    }
}

impl Chunked for TiffChunkReader<'_> {
    fn chunk(
        &mut self,
        image: usize,
        index: u32,
        cancel: &CancelToken,
    ) -> Result<Arc<DecodedChunk>, RasterError> {
        match self.mode {
            ChunkReadMode::SequentialPass => self
                .decode_chunk_uncached(image, index, None, cancel)
                .map(Arc::new),
            ChunkReadMode::Pointwise {
                chunk_cache,
                budget_tracker,
            } => {
                let key = ChunkKey {
                    source: self.source_id,
                    spec: DISPLAY_STREAM_SPEC,
                    image,
                    chunk_index: index,
                };

                chunk_cache.get_or_decode(key, cancel, || {
                    self.decode_chunk_uncached(image, index, Some(budget_tracker), cancel)
                })
            }
        }
    }
}
