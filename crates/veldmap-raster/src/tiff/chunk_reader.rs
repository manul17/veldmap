//! Pooled `tiff::decoder::Decoder` management and positioned chunk reader
//! bridging `tiff` to [`Chunked`] and [`SharedChunkCache`].

use std::sync::{Arc, Mutex};

use tiff::decoder::{Decoder, Limits};
use veldmap_core::budget::DecodeBudgetTracker;
use veldmap_core::io::{BufferedByteReader, ByteSource};
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{BandKey, PixelFormat, SourceId, TileStreamSpec};

use crate::RasterError;
use crate::pipeline::chunk_cache::{ChunkKey, DecodedChunk, SharedChunkCache};
use crate::pipeline::grid::{self, Chunked};
use crate::pipeline::radiometry::{DisplayMapping, PixelLayout};

use super::TIFF_DECODE_REV;
use super::tags::{chunk_rgba, map_tiff_error};

/// Maximum idle `tiff::decoder::Decoder` handles retained per [`super::TiffSource`].
const MAX_POOLED_DECODERS: usize = 8;

struct DecoderEntry {
    decoder: Decoder<BufferedByteReader>,
    aimed_image: Option<usize>,
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
            let reader = BufferedByteReader::new(Arc::clone(&self.byte_source), cancel.clone());
            let decoder = Decoder::new(reader)
                .map_err(|e| map_tiff_error(e, cancel))?
                .with_limits(self.limits.clone());
            DecoderEntry {
                decoder,
                aimed_image: Some(0),
            }
        };
        Ok(PooledDecoder {
            pool: self,
            entry: Some(entry),
            clean: false,
        })
    }
}

/// RAII checkout guard returning a clean `Decoder` to [`DecoderPool`] on drop
/// with its `CancelToken` reset so cancelled requests never poison reuse.
pub(super) struct PooledDecoder<'a> {
    pool: &'a DecoderPool,
    entry: Option<DecoderEntry>,
    clean: bool,
}

impl PooledDecoder<'_> {
    pub(super) fn mark_clean(&mut self) {
        self.clean = true;
    }

    pub(super) fn decoder_mut(&mut self) -> &mut Decoder<BufferedByteReader> {
        #[expect(
            clippy::expect_used,
            reason = "INVARIANT: entry is only taken inside Drop"
        )]
        &mut self.entry.as_mut().expect("pooled decoder present").decoder
    }

    pub(super) fn seek_to_image(
        &mut self,
        image: usize,
        cancel: &CancelToken,
    ) -> Result<(), RasterError> {
        #[expect(
            clippy::expect_used,
            reason = "INVARIANT: entry is only taken inside Drop"
        )]
        let entry = self.entry.as_mut().expect("pooled decoder present");
        if entry.aimed_image == Some(image) {
            return Ok(());
        }
        entry.aimed_image = None;
        entry
            .decoder
            .seek_to_image(image)
            .map_err(|e| map_tiff_error(e, cancel))?;
        entry.aimed_image = Some(image);
        Ok(())
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
    pixel_layout: PixelLayout,
    depth: u32,
    mapping: DisplayMapping,
    mode: ChunkReadMode<'a>,
    pooled: Option<PooledDecoder<'a>>,
}

impl<'a> TiffChunkReader<'a> {
    pub(super) fn new(
        decoder_pool: &'a DecoderPool,
        source_id: SourceId,
        pixel_layout: PixelLayout,
        depth: u32,
        mapping: DisplayMapping,
        mode: ChunkReadMode<'a>,
    ) -> Self {
        Self {
            decoder_pool,
            source_id,
            pixel_layout,
            depth,
            mapping,
            mode,
            pooled: None,
        }
    }

    fn decode_chunk_uncached(
        &mut self,
        image: usize,
        index: u32,
        budget_tracker: Option<&DecodeBudgetTracker>,
        cancel: &CancelToken,
    ) -> Result<DecodedChunk, RasterError> {
        if self.pooled.is_none() {
            self.pooled = Some(self.decoder_pool.checkout(cancel)?);
        }
        let pooled = self
            .pooled
            .as_mut()
            .ok_or_else(|| RasterError::InvalidFormat("missing TIFF decoder".to_string()))?;
        pooled.clean = false;
        pooled.seek_to_image(image, cancel)?;
        let (dw, dh) = pooled.decoder_mut().chunk_data_dimensions(index);
        if dw == 0 || dh == 0 {
            return Err(RasterError::InvalidDimensions {
                width: dw,
                height: dh,
            });
        }
        let est_bytes = grid::chunk_working_bytes(dw, dh, self.depth, true);
        let _lease = match budget_tracker {
            Some(tracker) => Some(tracker.reserve_blocking(est_bytes, cancel)?),
            None => None,
        };
        let raw = pooled
            .decoder_mut()
            .read_chunk(index)
            .map_err(|e| map_tiff_error(e, cancel))?;
        pooled.mark_clean();
        let rgba = chunk_rgba(&self.mapping, &raw, self.pixel_layout, dw, dh)?;
        Ok(DecodedChunk {
            data: rgba,
            width: dw,
            height: dh,
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
                    spec: TileStreamSpec::new(
                        BandKey::Display,
                        PixelFormat::Rgba8UnormSrgb,
                        0,
                        TIFF_DECODE_REV,
                    ),
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
