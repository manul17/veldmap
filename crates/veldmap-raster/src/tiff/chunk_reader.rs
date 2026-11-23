//! Positioned chunk reader bridging `tiff::decoder::Decoder` to [`Chunked`]
//! and [`SharedChunkCache`].

use std::sync::Arc;

use tiff::decoder::{Decoder, Limits};
use veldmap_core::budget::DecodeBudgetTracker;
use veldmap_core::io::{BufferedByteReader, ByteSource, CancelToken};
use veldmap_core::tile::{BandKey, PixelFormat, SourceId};

use crate::RasterError;
use crate::chunk_cache::{ChunkKey, DecodedChunk, SharedChunkCache};
use crate::grid::{self, Chunked};
use crate::radiometry::{DisplayMapping, PixelLayout};

use super::TIFF_DECODE_REV;
use super::tags::{chunk_rgba, ensure_chunky, ensure_readable, map_tiff_error};

pub(super) struct TiffChunkReader<'a> {
    byte_source: Arc<dyn ByteSource>,
    source_id: SourceId,
    limits: Limits,
    depth: u32,
    mapping: DisplayMapping,
    chunk_cache: Option<&'a SharedChunkCache>,
    budget_tracker: &'a DecodeBudgetTracker,
    decoder: Option<Decoder<BufferedByteReader>>,
    aimed_image: Option<(usize, PixelLayout)>,
    pub(super) skip_chunk_budget: bool,
}

impl<'a> TiffChunkReader<'a> {
    pub(super) fn new(
        byte_source: Arc<dyn ByteSource>,
        source_id: SourceId,
        limits: Limits,
        depth: u32,
        mapping: DisplayMapping,
        chunk_cache: Option<&'a SharedChunkCache>,
        budget_tracker: &'a DecodeBudgetTracker,
    ) -> Self {
        Self {
            byte_source,
            source_id,
            limits,
            depth,
            mapping,
            chunk_cache,
            budget_tracker,
            decoder: None,
            aimed_image: None,
            skip_chunk_budget: false,
        }
    }

    fn ensure_decoder(
        &mut self,
        image: usize,
        cancel: &CancelToken,
    ) -> Result<(&mut Decoder<BufferedByteReader>, PixelLayout), RasterError> {
        if self.decoder.is_none() {
            let reader = BufferedByteReader::new(Arc::clone(&self.byte_source), cancel.clone());
            let dec = Decoder::new(reader)
                .map_err(|e| map_tiff_error(e, cancel))?
                .with_limits(self.limits.clone());
            self.decoder = Some(dec);
            self.aimed_image = None;
        }
        let decoder = self
            .decoder
            .as_mut()
            .ok_or_else(|| RasterError::InvalidFormat("missing TIFF decoder".to_string()))?;
        if let Some((at, layout)) = self.aimed_image
            && at == image
        {
            return Ok((decoder, layout));
        }
        self.aimed_image = None;
        decoder
            .seek_to_image(image)
            .map_err(|e| map_tiff_error(e, cancel))?;
        ensure_chunky(decoder, cancel)?;
        let (layout, _, _) = ensure_readable(decoder, cancel)?;
        self.aimed_image = Some((image, layout));
        Ok((decoder, layout))
    }

    fn decode_chunk_uncached(
        &mut self,
        image: usize,
        index: u32,
        cancel: &CancelToken,
    ) -> Result<DecodedChunk, RasterError> {
        let depth = self.depth;
        let mapping = self.mapping;
        let skip_budget = self.skip_chunk_budget;
        let budget_tracker = self.budget_tracker;
        let (decoder, layout) = self.ensure_decoder(image, cancel)?;
        let (dw, dh) = decoder.chunk_data_dimensions(index);
        if dw == 0 || dh == 0 {
            return Err(RasterError::InvalidDimensions {
                width: dw,
                height: dh,
            });
        }
        let est_bytes = grid::chunk_working_bytes(dw, dh, depth, true);
        let _lease = if skip_budget {
            None
        } else {
            Some(budget_tracker.reserve_blocking(est_bytes, cancel)?)
        };
        let raw = decoder
            .read_chunk(index)
            .map_err(|e| map_tiff_error(e, cancel))?;
        let rgba = chunk_rgba(&mapping, &raw, layout, dw, dh)?;
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
        let Some(cache) = self.chunk_cache else {
            return self
                .decode_chunk_uncached(image, index, cancel)
                .map(Arc::new);
        };
        let key = ChunkKey {
            source: self.source_id,
            band: BandKey::Display,
            format: PixelFormat::Rgba8UnormSrgb,
            gutter: 0,
            decode_rev: TIFF_DECODE_REV,
            image,
            chunk_index: index,
        };

        cache.get_or_decode(key, cancel, || {
            self.decode_chunk_uncached(image, index, cancel)
        })
    }
}
