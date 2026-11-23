//! GeoTIFF / BigTIFF / Cloud-Optimized GeoTIFF (COG) reader built over
//! positioned [`ByteSource`] with explicit `tiff::decoder::Limits`,
//! `MAX_IFDS` bounds, `GeoDoubleParamsTag`/`GeoAsciiParamsTag` parsing,
//! `Predictor = 3` (Float32) & ZSTD support, and singleflight chunk caching.

use std::sync::{Arc, Mutex, OnceLock};

use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::tags::Tag;
use veldmap_core::budget::{DecodeBudgetTracker, DecodePeak, MemoryBudget};
use veldmap_core::io::{
    BufferedByteReader, ByteSource, CancelToken, SourceFingerprint, lock_unpoisoned,
};
use veldmap_core::tile::{BandKey, DecodedTile, LevelPlan, PixelFormat, PyramidMeta, SourceId};

use crate::RasterError;
use crate::band::{
    BandInfo, GeoKeyParams, GeoPlacement, GeoRasterType, GeoTie, RasterDescription, SampleType,
    is_valid_lat_lon,
};
use crate::chunk_cache::{ChunkKey, DecodedChunk, SharedChunkCache};
use crate::grid::{self, Chunked, Grid, Overview};
use crate::pyramid::{self, RASTER_TILE_SIZE};
use crate::radiometry::{self, DisplayMapping, PixelLayout, Samples, percentile_stretch};
use crate::table;

/// BigTIFF file header signatures (`version = 43`).
pub const BIG_MAGIC: [&[u8]; 2] = [b"II\x2b\x00", b"MM\x00\x2b"];

/// Classic TIFF file header signatures (`version = 42`).
pub const CLASSIC_MAGIC: [&[u8]; 2] = [b"II\x2a\x00", b"MM\x00\x2a"];

/// Maximum number of IFD directories scanned during `open` to guard against
/// malformed or pathological multi-IFD files.
pub const MAX_IFDS: usize = 64;

/// Maximum raster side dimension accepted by the GeoTIFF reader.
pub const MAX_SOURCE_SIDE: u32 = 65_536;

/// Current display decode revision encoded into `TileKey` and `ChunkKey`.
pub const TIFF_DECODE_REV: u32 = 1;

const DECODED_COMPRESSIONS: [(u16, &str); 8] = [
    (1, "no compression"),
    (4, "Fax4"),
    (5, "LZW"),
    (7, "JPEG"),
    (8, "Deflate"),
    (0x8005, "PackBits"),
    (0x80B2, "Deflate (old code)"),
    (0xC350, "ZSTD"),
];

/// Return whether `head` begins with a classic TIFF or BigTIFF signature.
pub fn has_tiff_magic(head: &[u8]) -> bool {
    CLASSIC_MAGIC.iter().any(|m| head.starts_with(m))
        || BIG_MAGIC.iter().any(|m| head.starts_with(m))
}

fn build_tiff_limits(budget: &MemoryBudget) -> Limits {
    let mut limits = Limits::default();
    limits.decoding_buffer_size = budget.decode_working_bytes.min(256 * 1024 * 1024) as usize;
    limits.intermediate_buffer_size = 128 * 1024 * 1024;
    limits.ifd_value_size = 4 * 1024 * 1024;
    limits
}

/// Shared runtime context passed into GeoTIFF decoding calls.
pub struct DecodeContext<'a> {
    pub chunk_cache: &'a SharedChunkCache,
    pub budget_tracker: &'a DecodeBudgetTracker,
    pub budget: &'a MemoryBudget,
    pub cancel: &'a CancelToken,
}

/// Thread-safe opened GeoTIFF/BigTIFF/COG source backed by a positioned
/// [`ByteSource`].
pub struct TiffSource {
    byte_source: Arc<dyn ByteSource>,
    fingerprint: SourceFingerprint,
    source_id: SourceId,
    description: RasterDescription,
    grid: Grid,
    stats_ifd: usize,
    nodata: Option<f32>,
    sample_bits: u8,
    limits: Limits,
    mapping_lock: Mutex<()>,
    mapping: OnceLock<DisplayMapping>,
}

impl TiffSource {
    /// Open and describe a GeoTIFF/BigTIFF/COG from `byte_source` using
    /// `fingerprint`, `budget`, and `cancel` without reading base-level pixels.
    pub fn open(
        byte_source: Arc<dyn ByteSource>,
        fingerprint: SourceFingerprint,
        budget: &MemoryBudget,
        cancel: &CancelToken,
    ) -> Result<Self, RasterError> {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let limits = build_tiff_limits(budget);
        let reader = BufferedByteReader::new(Arc::clone(&byte_source), cancel.clone());
        let mut decoder = Decoder::new(reader)
            .map_err(|e| map_tiff_error(e, cancel))?
            .with_limits(limits.clone());

        let (width, height) = decoder
            .dimensions()
            .map_err(|e| map_tiff_error(e, cancel))?;
        if width == 0 || height == 0 {
            return Err(RasterError::InvalidDimensions { width, height });
        }
        if width > MAX_SOURCE_SIDE || height > MAX_SOURCE_SIDE {
            return Err(RasterError::LimitExceeded {
                width,
                height,
                estimated_bytes: u64::from(width) * u64::from(height) * 4,
                limit_bytes: u64::from(MAX_SOURCE_SIDE) * u64::from(MAX_SOURCE_SIDE) * 4,
            });
        }

        ensure_chunky(&mut decoder)?;
        let (pixel_layout, sample_bits, sample_type) = ensure_readable(&mut decoder)?;
        let tiled = decoder.get_tag_unsigned::<u32>(Tag::TileWidth).is_ok();
        let depth = depth_of(&mut decoder)?;
        let chunk = chunk_grid(&mut decoder, budget).unwrap_or((width, height));

        let nodata_f64 = decoder
            .get_tag_ascii_string(Tag::GdalNodata)
            .ok()
            .and_then(|s| s.trim().trim_end_matches('\0').parse::<f64>().ok())
            .filter(|v| v.is_finite());
        let nodata = nodata_f64.map(|v| v as f32);

        let (ties, placement, binding_trouble) = georef(&mut decoder, width, height);

        let mut overviews = Vec::new();
        let mut ifd_index = 0usize;
        while ifd_index + 1 < MAX_IFDS && decoder.more_images() {
            if cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            if let Err(err) = decoder.next_image() {
                if cancel.is_cancelled() {
                    return Err(RasterError::Cancelled);
                }
                tracing::debug!(error = %err, "Stopped scanning TIFF overviews");
                break;
            }
            ifd_index += 1;
            let subfile = decoder
                .get_tag_unsigned::<u32>(Tag::NewSubfileType)
                .unwrap_or(0);
            if !is_overview(subfile) {
                continue;
            }
            if ensure_chunky(&mut decoder).is_err() || ensure_readable(&mut decoder).is_err() {
                continue;
            }
            let Ok((ow, oh)) = decoder.dimensions() else {
                continue;
            };
            if ow == 0 || oh == 0 {
                continue;
            }
            let ochunk = chunk_grid(&mut decoder, budget).unwrap_or((ow, oh));
            overviews.push(Overview {
                image: ifd_index,
                width: ow,
                height: oh,
                chunk: ochunk,
            });
        }

        let grid = Grid {
            tiled,
            chunk,
            overviews,
            depth,
        };
        let levels = table::build_chunked_level_table(&grid, width, height, budget);
        if levels.is_empty() || levels.iter().all(|row| !row.fits) {
            let max_peak = levels.last().map_or(0, |row| row.estimated_peak_bytes);
            return Err(RasterError::LimitExceeded {
                width,
                height,
                estimated_bytes: max_peak,
                limit_bytes: budget.decode_working_bytes,
            });
        }

        let stats_ifd = grid
            .overviews
            .iter()
            .min_by_key(|o| o.width)
            .map_or(0, |o| o.image);

        let description = RasterDescription {
            width,
            height,
            bands: vec![BandInfo::display(
                sample_type,
                pixel_layout.channels,
                pixel_layout.has_alpha(),
                nodata_f64,
            )],
            ties,
            placement,
            binding_trouble,
            levels,
        };

        Ok(Self {
            byte_source,
            fingerprint,
            source_id: fingerprint.to_source_id(),
            description,
            grid,
            stats_ifd,
            nodata,
            sample_bits,
            limits,
            mapping_lock: Mutex::new(()),
            mapping: OnceLock::new(),
        })
    }

    /// Return the content fingerprint identifying this source.
    pub const fn fingerprint(&self) -> SourceFingerprint {
        self.fingerprint
    }

    /// Return the deterministic [`SourceId`] derived from [`Self::fingerprint`].
    pub const fn source_id(&self) -> SourceId {
        self.source_id
    }

    /// Return the parsed header metadata and level plan.
    pub fn description(&self) -> &RasterDescription {
        &self.description
    }

    /// Return the underlying chunk layout.
    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// Return scheduler-facing [`PyramidMeta`] for the display band.
    pub fn pyramid_meta(&self) -> PyramidMeta {
        PyramidMeta {
            source: self.source_id,
            band: BandKey::Display,
            format: PixelFormat::Rgba8UnormSrgb,
            gutter: 0,
            decode_rev: TIFF_DECODE_REV,
            width: self.description.width,
            height: self.description.height,
            tile_size: RASTER_TILE_SIZE,
            levels: self.description.levels.clone(),
        }
    }

    /// Return the computed level table.
    pub fn levels(&self) -> &[LevelPlan] {
        &self.description.levels
    }

    fn display_mapping(
        &self,
        budget_tracker: &DecodeBudgetTracker,
        cancel: &CancelToken,
    ) -> Result<DisplayMapping, RasterError> {
        if let Some(ready) = self.mapping.get() {
            return Ok(*ready);
        }
        let _guard = lock_unpoisoned(&self.mapping_lock);
        if let Some(ready) = self.mapping.get() {
            return Ok(*ready);
        }
        let computed = compute_mapping(
            &self.byte_source,
            self.limits.clone(),
            self.stats_ifd,
            self.nodata,
            self.sample_bits,
            self.grid.depth,
            budget_tracker,
            cancel,
        )?;
        let _ = self.mapping.set(computed);
        Ok(computed)
    }

    /// Produce one pointwise display tile `(level, tx, ty)` into a validated
    /// [`DecodedTile`] using `ctx`.
    pub fn decode_pointwise_tile(
        &self,
        level: u8,
        tx: u32,
        ty: u32,
        ctx: &DecodeContext<'_>,
    ) -> Result<DecodedTile, RasterError> {
        let mut produced: Option<DecodedTile> = None;
        self.produce_direct(
            level,
            &[(tx, ty)],
            ctx,
            &mut |_lvl, _x, _y, tw, th, rgba| {
                let tile = DecodedTile::try_new(tw, th, PixelFormat::Rgba8UnormSrgb, rgba)
                    .map_err(|e| RasterError::InvalidFormat(e.to_string()))?;
                produced = Some(tile);
                Ok(())
            },
        )?;
        produced.ok_or_else(|| {
            RasterError::InvalidFormat(format!("tile ({tx}, {ty}) at level {level} out of bounds"))
        })
    }

    /// Produce pointwise tiles `wants` at `level`, deduplicating source chunk
    /// reads via `ctx.chunk_cache` and bounding active decode memory via
    /// `ctx.budget_tracker`.
    pub fn produce_direct(
        &self,
        level: u8,
        wants: &[(u32, u32)],
        ctx: &DecodeContext<'_>,
        emit: &mut dyn FnMut(u8, u32, u32, u32, u32, Vec<u8>) -> Result<(), RasterError>,
    ) -> Result<(), RasterError> {
        let mapping = self.display_mapping(ctx.budget_tracker, ctx.cancel)?;
        let mut reader = TiffChunkReader::new(
            Arc::clone(&self.byte_source),
            self.source_id,
            self.limits.clone(),
            self.grid.depth,
            mapping,
            Some(ctx.chunk_cache),
            ctx.budget_tracker,
        );
        let req = grid::DirectRead {
            grid: &self.grid,
            base: (self.description.width, self.description.height),
            level,
            wants,
            budget: ctx.budget,
            cancel: ctx.cancel,
        };
        grid::direct(&mut reader, &req, emit)
    }

    /// Run a sequential top-to-bottom pass over the base IFD, emitting every
    /// pyramid tile via `emit`.
    pub fn produce_pass(
        &self,
        ctx: &DecodeContext<'_>,
        emit: &mut dyn FnMut(u8, u32, u32, u32, u32, Vec<u8>) -> Result<(), RasterError>,
    ) -> Result<(), RasterError> {
        let mapping = self.display_mapping(ctx.budget_tracker, ctx.cancel)?;
        let pass_peak = self
            .grid
            .pass_peak((self.description.width, self.description.height));
        pass_peak.admit(ctx.budget.decode_working_bytes)?;
        let _pass_lease = ctx
            .budget_tracker
            .reserve_blocking(pass_peak.total(), ctx.cancel)?;

        // Sequential passes read base-IFD chunks once top-to-bottom without
        // entering `SharedChunkCache` singleflight, avoiding both AB-BA lock
        // inversion against `_pass_lease` and pointwise chunk-cache pollution.
        let mut reader = TiffChunkReader::new(
            Arc::clone(&self.byte_source),
            self.source_id,
            self.limits.clone(),
            self.grid.depth,
            mapping,
            None,
            ctx.budget_tracker,
        );
        reader.skip_chunk_budget = true;
        grid::pass(
            &mut reader,
            &self.grid,
            (self.description.width, self.description.height),
            ctx.budget,
            ctx.cancel,
            emit,
        )
    }
}

struct TiffChunkReader<'a> {
    byte_source: Arc<dyn ByteSource>,
    source_id: SourceId,
    limits: Limits,
    depth: u32,
    mapping: DisplayMapping,
    chunk_cache: Option<&'a SharedChunkCache>,
    budget_tracker: &'a DecodeBudgetTracker,
    decoder: Option<Decoder<BufferedByteReader>>,
    aimed_image: Option<(usize, PixelLayout)>,
    skip_chunk_budget: bool,
}

impl<'a> TiffChunkReader<'a> {
    fn new(
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
        decoder
            .seek_to_image(image)
            .map_err(|e| map_tiff_error(e, cancel))?;
        ensure_chunky(decoder)?;
        let (layout, _, _) = model(decoder.colortype().map_err(|e| map_tiff_error(e, cancel))?)?;
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
        let est_bytes = u64::from(dw) * u64::from(dh) * (u64::from(depth) + 4);
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

fn map_tiff_error(err: tiff::TiffError, cancel: &CancelToken) -> RasterError {
    if cancel.is_cancelled() {
        return RasterError::Cancelled;
    }
    match &err {
        tiff::TiffError::IoError(io_err) if io_err.kind() == std::io::ErrorKind::Interrupted => {
            RasterError::Cancelled
        }
        tiff::TiffError::LimitsExceeded => RasterError::LimitExceeded {
            width: 0,
            height: 0,
            estimated_bytes: 0,
            limit_bytes: 0,
        },
        _ => RasterError::Tiff(err.to_string()),
    }
}

fn is_overview(subfile_type: u32) -> bool {
    const REDUCED: u32 = 1;
    const MASK: u32 = 4;
    subfile_type & REDUCED != 0 && subfile_type & MASK == 0
}

fn ensure_chunky(decoder: &mut Decoder<BufferedByteReader>) -> Result<(), RasterError> {
    let planar = decoder
        .get_tag_unsigned::<u16>(Tag::PlanarConfiguration)
        .unwrap_or(1);
    if planar != 1 {
        return Err(RasterError::Tiff(
            "planar sample layout (PlanarConfiguration != 1) is not supported yet".to_string(),
        ));
    }
    Ok(())
}

fn model(color: tiff::ColorType) -> Result<(PixelLayout, u8, SampleType), RasterError> {
    match color {
        tiff::ColorType::Gray(bits) => {
            Ok((PixelLayout::named(1), bits, sample_type_from_bits(bits)?))
        }
        tiff::ColorType::GrayA(bits) => {
            Ok((PixelLayout::named(2), bits, sample_type_from_bits(bits)?))
        }
        tiff::ColorType::RGB(bits) => {
            Ok((PixelLayout::named(3), bits, sample_type_from_bits(bits)?))
        }
        tiff::ColorType::RGBA(bits) => {
            Ok((PixelLayout::named(4), bits, sample_type_from_bits(bits)?))
        }
        tiff::ColorType::Multiband {
            bit_depth,
            num_samples,
        } => Ok((
            PixelLayout::stack(usize::from(num_samples)),
            bit_depth,
            sample_type_from_bits(bit_depth)?,
        )),
        other => Err(RasterError::Tiff(format!(
            "color model {other:?} is not supported"
        ))),
    }
}

fn sample_type_from_bits(bits: u8) -> Result<SampleType, RasterError> {
    match bits {
        8 => Ok(SampleType::U8),
        16 => Ok(SampleType::U16),
        32 => Ok(SampleType::F32),
        other => Err(RasterError::Tiff(format!(
            "unsupported sample bit depth {other}"
        ))),
    }
}

fn depth_of(decoder: &mut Decoder<BufferedByteReader>) -> Result<u32, RasterError> {
    let (layout, bits, _) = model(
        decoder
            .colortype()
            .map_err(|e| RasterError::Tiff(e.to_string()))?,
    )?;
    Ok(layout.channels as u32 * u32::from(bits.div_ceil(8)))
}

fn chunk_grid(
    decoder: &mut Decoder<BufferedByteReader>,
    budget: &MemoryBudget,
) -> Result<(u32, u32), RasterError> {
    ensure_chunky(decoder)?;
    let (cw, ch) = decoder.chunk_dimensions();
    if cw == 0 || ch == 0 {
        return Err(RasterError::Tiff("zero TIFF chunk dimensions".to_string()));
    }
    let depth = depth_of(decoder)?;
    DecodePeak::new()
        .with(
            "fresh chunk",
            u64::from(cw) * u64::from(ch) * (u64::from(depth) + 4),
        )
        .admit(budget.decode_working_bytes)?;
    Ok((cw, ch))
}

fn ensure_readable(
    decoder: &mut Decoder<BufferedByteReader>,
) -> Result<(PixelLayout, u8, SampleType), RasterError> {
    let color = decoder
        .colortype()
        .map_err(|e| RasterError::Tiff(e.to_string()))?;
    let (layout, bits, mut stype) = model(color)?;
    if bits < 8 {
        return Err(RasterError::Tiff(format!(
            "{bits} bits per sample cannot unpack directly into pixels"
        )));
    }
    ensure_decodable(decoder)?;
    let sample_format = ensure_sampled(decoder, bits)?;
    if sample_format == 2 && bits == 16 {
        stype = SampleType::I16;
    }
    Ok((layout, bits, stype))
}

fn ensure_decodable(decoder: &mut Decoder<BufferedByteReader>) -> Result<(), RasterError> {
    let code = decoder
        .get_tag_unsigned::<u16>(Tag::Compression)
        .unwrap_or(1);
    if DECODED_COMPRESSIONS.iter().any(|(known, _)| *known == code) {
        return Ok(());
    }
    Err(RasterError::Tiff(format!(
        "unsupported TIFF compression code {code}"
    )))
}

fn ensure_sampled(decoder: &mut Decoder<BufferedByteReader>, bits: u8) -> Result<u16, RasterError> {
    let format = decoder
        .get_tag_u16_vec(Tag::SampleFormat)
        .ok()
        .and_then(|formats| formats.first().copied())
        .or_else(|| decoder.get_tag_unsigned::<u16>(Tag::SampleFormat).ok())
        .unwrap_or(1);
    if matches!((format, bits), (1, 8) | (1, 16) | (2, 16) | (3, 32)) {
        return Ok(format);
    }
    Err(RasterError::Tiff(format!(
        "unsupported TIFF sample format ({format}, {bits}-bit)"
    )))
}

fn typed_samples(data: &DecodingResult) -> Result<Samples<'_>, RasterError> {
    match data {
        DecodingResult::U8(v) => Ok(Samples::U8(v)),
        DecodingResult::U16(v) => Ok(Samples::U16(v)),
        DecodingResult::I16(v) => Ok(Samples::I16(v)),
        DecodingResult::F32(v) => Ok(Samples::F32(v)),
        other => Err(RasterError::Tiff(format!(
            "unsupported TIFF decoded buffer variant ({other:?})"
        ))),
    }
}

fn chunk_rgba(
    mapping: &DisplayMapping,
    data: &DecodingResult,
    pixel: PixelLayout,
    dw: u32,
    dh: u32,
) -> Result<Vec<u8>, RasterError> {
    let pixels = (dw as usize) * (dh as usize);
    let rgba = mapping.rgba(&typed_samples(data)?, pixel, pixels);
    if rgba.len() == pixels * 4 {
        Ok(rgba)
    } else {
        Err(RasterError::Tiff(format!(
            "chunk {dw}x{dh} decoded {} pixels instead of {pixels}",
            rgba.len() / 4
        )))
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Cold-path helper called only once from TiffSource::display_mapping"
)]
fn compute_mapping(
    byte_source: &Arc<dyn ByteSource>,
    limits: Limits,
    stats_ifd: usize,
    nodata: Option<f32>,
    sample_bits: u8,
    depth: u32,
    budget_tracker: &DecodeBudgetTracker,
    cancel: &CancelToken,
) -> Result<DisplayMapping, RasterError> {
    if sample_bits <= 8 {
        return Ok(DisplayMapping::identity(nodata));
    }
    let reader = BufferedByteReader::new(Arc::clone(byte_source), cancel.clone());
    let mut decoder = Decoder::new(reader)
        .map_err(|e| map_tiff_error(e, cancel))?
        .with_limits(limits);
    decoder
        .seek_to_image(stats_ifd)
        .map_err(|e| map_tiff_error(e, cancel))?;
    let (w, h) = decoder
        .dimensions()
        .map_err(|e| map_tiff_error(e, cancel))?;
    let (cw, ch) = decoder.chunk_dimensions();
    if cw == 0 || ch == 0 {
        return Ok(DisplayMapping::identity(nodata));
    }
    let (pixel, _, _) = model(decoder.colortype().map_err(|e| map_tiff_error(e, cancel))?)?;

    let total = u64::from(w.div_ceil(cw)) * u64::from(h.div_ceil(ch));
    let mut picks = Vec::new();
    for raw_idx in [0, total / 3, (2 * total) / 3, total.saturating_sub(1)] {
        if let Ok(idx) = u32::try_from(raw_idx) {
            picks.push(idx);
        }
    }
    picks.dedup();

    let mut values = Vec::new();
    for index in picks {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let (dw, dh) = decoder.chunk_data_dimensions(index);
        let est_bytes = u64::from(dw) * u64::from(dh) * (u64::from(depth) + 4);
        let _lease = budget_tracker.reserve_blocking(est_bytes, cancel)?;
        let raw = decoder
            .read_chunk(index)
            .map_err(|e| map_tiff_error(e, cancel))?;
        let samples = typed_samples(&raw)?;
        let pixels = samples.len() / pixel.channels.max(1);
        let step = (pixels * 4 / radiometry::STRETCH_SAMPLES).max(1);
        for at in (0..pixels).step_by(step) {
            for channel in 0..pixel.colors() {
                let v = samples.get(at * pixel.channels + channel);
                if radiometry::is_data(v, nodata) {
                    values.push(v);
                }
            }
        }
    }

    Ok(match percentile_stretch(&mut values) {
        Some((lo, hi)) => DisplayMapping::stretched(lo, hi, nodata),
        None => DisplayMapping::identity(nodata),
    })
}

fn georef(
    decoder: &mut Decoder<BufferedByteReader>,
    width: u32,
    height: u32,
) -> (Vec<GeoTie>, Option<GeoPlacement>, Option<String>) {
    let raw_keys = decoder
        .get_tag_u16_vec(Tag::GeoKeyDirectoryTag)
        .unwrap_or_default();
    let double_params = decoder
        .get_tag_f64_vec(Tag::GeoDoubleParamsTag)
        .unwrap_or_default();
    let ascii_params = decoder
        .get_tag_ascii_string(Tag::GeoAsciiParamsTag)
        .unwrap_or_default();
    let geo_params = GeoKeyParams::parse(&raw_keys, &double_params, &ascii_params);

    let points = decoder
        .get_tag_f64_vec(Tag::ModelTiepointTag)
        .unwrap_or_default();
    let scale = decoder
        .get_tag_f64_vec(Tag::ModelPixelScaleTag)
        .unwrap_or_default();
    let matrix = decoder
        .get_tag_f64_vec(Tag::ModelTransformationTag)
        .unwrap_or_default();

    let placement = geo_placement(&geo_params, &points, &scale, &matrix, width, height);
    let ties = geo_ties(&geo_params, &points, &scale, &matrix, width, height);

    let carries = !points.is_empty() || !scale.is_empty() || !matrix.is_empty();
    let taken = !ties.is_empty() || placement.is_some();
    let trouble = (carries && !taken).then(|| {
        let named = |opt: Option<u16>| opt.map_or_else(|| "unnamed".to_string(), |v| v.to_string());
        format!(
            "raster carries georeference tags, but could not be bound: model {}, projected {}, tie points {}",
            named(geo_params.model_type),
            named(geo_params.projected_crs),
            points.len() / 6
        )
    });

    (ties, placement, trouble)
}

fn foreign_datum(geo_params: &GeoKeyParams) -> Option<u16> {
    const WGS84: [u16; 3] = [4326, 4979, 4055];
    const UNSAID: [u16; 2] = [0, 32767];
    geo_params
        .geographic_crs
        .filter(|code| !WGS84.contains(code) && !UNSAID.contains(code))
}

fn usable_step(scale: &[f64]) -> bool {
    matches!(scale.get(..2), Some([x, y]) if x.is_finite() && y.is_finite() && *x != 0.0 && *y != 0.0)
}

fn affine_from_matrix(matrix: &[f64], half: f64) -> Option<[f64; 6]> {
    let a = matrix.get(..8)?;
    if !a.iter().all(|v| v.is_finite()) || a[0] * a[5] - a[1] * a[4] == 0.0 {
        return None;
    }
    Some([
        a[0],
        a[1],
        a[3] - (a[0] + a[1]) * half,
        a[4],
        a[5],
        a[7] - (a[4] + a[5]) * half,
    ])
}

fn geo_placement(
    geo_params: &GeoKeyParams,
    points: &[f64],
    scale: &[f64],
    matrix: &[f64],
    width: u32,
    height: u32,
) -> Option<GeoPlacement> {
    if geo_params.model_type != Some(1) || points.len() > 6 {
        return None;
    }
    let epsg = match geo_params.projected_crs {
        Some(code) if code != 0 && code != 32767 => Some(u32::from(code)),
        _ => return None,
    };

    let half = geo_params.raster_type.half_pixel_shift();
    let affine = match (points.get(..6), usable_step(scale)) {
        (Some(tie), true) => [
            scale[0],
            0.0,
            tie[3] - (tie[0] + half) * scale[0],
            0.0,
            -scale[1],
            tie[4] + (tie[1] + half) * scale[1],
        ],
        _ => affine_from_matrix(matrix, half)?,
    };

    let (right, bottom) = (f64::from(width), f64::from(height));
    let corner = [
        affine[0] * right + affine[1] * bottom + affine[2],
        affine[3] * right + affine[4] * bottom + affine[5],
    ];
    if !affine.iter().chain(corner.iter()).all(|v| v.is_finite()) {
        return None;
    }

    Some(GeoPlacement {
        epsg,
        affine,
        foreign_datum_epsg: foreign_datum(geo_params),
        geo_params: geo_params.clone(),
    })
}

fn geo_ties(
    geo_params: &GeoKeyParams,
    points: &[f64],
    scale: &[f64],
    matrix: &[f64],
    width: u32,
    height: u32,
) -> Vec<GeoTie> {
    if geo_params.model_type != Some(2) {
        return Vec::new();
    }
    let half = geo_params.raster_type.half_pixel_shift();
    let finite_ties = |ties: Vec<GeoTie>| -> Vec<GeoTie> {
        let valid = ties.iter().all(|t| {
            t.px.is_finite() && t.py.is_finite() && t.lat.is_finite() && t.lon.is_finite()
        });
        if valid { ties } else { Vec::new() }
    };

    if points.len() > 6 {
        let ties: Vec<GeoTie> = points
            .chunks_exact(6)
            .map(|tie| GeoTie {
                px: tie[0] + half,
                py: tie[1] + half,
                lon: tie[3],
                lat: tie[4],
            })
            .collect();
        if ties.iter().all(|t| is_valid_lat_lon(t.lat, t.lon)) {
            return finite_ties(ties);
        }
        return Vec::new();
    }

    if let (Some(tie), true) = (points.get(..6), usable_step(scale)) {
        let x0 = tie[3] - (tie[0] + half) * scale[0];
        let y0 = tie[4] + (tie[1] + half) * scale[1];
        let (right, bottom) = (f64::from(width), f64::from(height));
        return finite_ties(vec![
            GeoTie {
                px: 0.0,
                py: 0.0,
                lon: x0,
                lat: y0,
            },
            GeoTie {
                px: right,
                py: 0.0,
                lon: x0 + right * scale[0],
                lat: y0,
            },
            GeoTie {
                px: 0.0,
                py: bottom,
                lon: x0,
                lat: y0 - bottom * scale[1],
            },
            GeoTie {
                px: right,
                py: bottom,
                lon: x0 + right * scale[0],
                lat: y0 - bottom * scale[1],
            },
        ]);
    }

    if let Some(a) = affine_from_matrix(matrix, half) {
        let (right, bottom) = (f64::from(width), f64::from(height));
        let place = |px: f64, py: f64| GeoTie {
            px,
            py,
            lon: a[0] * px + a[1] * py + a[2],
            lat: a[3] * px + a[4] * py + a[5],
        };
        return finite_ties(vec![
            place(0.0, 0.0),
            place(right, 0.0),
            place(0.0, bottom),
            place(right, bottom),
        ]);
    }

    Vec::new()
}

/// Convert `GeoRasterType` to boolean `PixelIsPoint` for callers.
pub const fn is_pixel_is_point(raster_type: GeoRasterType) -> bool {
    matches!(raster_type, GeoRasterType::PixelIsPoint)
}

/// Helper verifying pyramid sizing matches `pyramid::level_count`.
pub fn pyramid_level_count(width: u32, height: u32) -> u32 {
    pyramid::level_count(width, height)
}
