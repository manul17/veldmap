//! GeoTIFF / BigTIFF / Cloud-Optimized GeoTIFF (COG) reader built over
//! positioned [`ByteSource`] with explicit `tiff::decoder::Limits`,
//! `MAX_IFDS` bounds, `GeoDoubleParamsTag`/`GeoAsciiParamsTag` parsing,
//! ZSTD & `Predictor = 3` (via `tiff`) support, pooled `Decoder` handles, and
//! singleflight chunk caching.

mod chunk_reader;
mod georef;
mod stretch;
mod tags;

use std::sync::Arc;

use tiff::decoder::{Decoder, Limits};
use tiff::tags::Tag;
use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::{BufferedByteReader, ByteSource};
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{
    BandKey, DecodedTile, PixelFormat, SourceFingerprint, SourceId, TileStreamSpec,
};

use crate::RasterError;
use crate::pipeline::DecodeContext;
use crate::pipeline::band::{BandInfo, RasterDescription, SampleType};
use crate::pipeline::cascade::TileEmit;
use crate::pipeline::driver::{self, DirectRead};
use crate::pipeline::grid::{Grid, Overview};
use crate::pipeline::plan::{LevelPlan, PyramidMeta, build_chunked_level_table};
use crate::pipeline::pyramid::RASTER_TILE_SIZE;
use crate::pipeline::radiometry::{DisplayMapping, PixelLayout};

use chunk_reader::{ChunkReadMode, DecoderPool, TiffChunkReader};
use georef::{ParsedGeoref, georef, parse_gdal_nodata};
use stretch::{StretchCache, StretchRequest};
use tags::{chunk_grid, depth_of_layout, ensure_readable, is_overview, map_tiff_error, opt_tag};

/// BigTIFF file header signatures (`version = 43`).
const BIG_MAGIC: [&[u8]; 2] = [b"II\x2b\x00", b"MM\x00\x2b"];

/// Classic TIFF file header signatures (`version = 42`).
const CLASSIC_MAGIC: [&[u8]; 2] = [b"II\x2a\x00", b"MM\x00\x2a"];

/// Maximum number of IFD directories scanned during `open` to guard against
/// malformed or pathological multi-IFD files.
pub const MAX_IFDS: usize = 64;

/// Maximum raster side dimension accepted for untiled/stripped TIFF sources.
///
/// Tiled TIFFs/COGs may exceed this limit provided their chunk grid and at
/// least one pyramid level fit within [`MemoryBudget`].
pub const MAX_SOURCE_SIDE: u32 = 65_536;

/// Current display decode revision encoded into `TileKey` and `ChunkKey`.
pub const TIFF_DECODE_REV: u32 = 1;

/// Canonical display stream spec shared across [`TiffSource::pyramid_meta`] and
/// [`TiffChunkReader`].
pub(super) const DISPLAY_STREAM_SPEC: TileStreamSpec = TileStreamSpec::new(
    BandKey::Display,
    PixelFormat::Rgba8UnormSrgb,
    0,
    TIFF_DECODE_REV,
);

/// Return whether `head` begins with a classic TIFF or BigTIFF signature.
pub fn has_tiff_magic(head: &[u8]) -> bool {
    CLASSIC_MAGIC.iter().any(|m| head.starts_with(m))
        || BIG_MAGIC.iter().any(|m| head.starts_with(m))
}

fn build_tiff_limits(budget: &MemoryBudget) -> Limits {
    let mut limits = Limits::default();
    limits.decoding_buffer_size = budget.decode_working_bytes.min(256 * 1024 * 1024) as usize;
    limits.intermediate_buffer_size = budget
        .decode_working_bytes
        .clamp(4 * 1024 * 1024, 128 * 1024 * 1024) as usize;
    limits
}

struct BaseFormat {
    pixel_layout: PixelLayout,
    sample_bits: u8,
    sample_type: SampleType,
    depth: u32,
}

fn scan_overviews(
    decoder: &mut Decoder<BufferedByteReader>,
    base: &BaseFormat,
    budget: &MemoryBudget,
    cancel: &CancelToken,
) -> Result<(Vec<Overview>, usize, bool), RasterError> {
    let mut overviews = Vec::new();
    let mut ifd_index = 0usize;
    let mut decoder_clean = true;
    while ifd_index + 1 < MAX_IFDS && decoder.more_images() {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        if let Err(err) = decoder.next_image() {
            if cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            tracing::debug!(error = %err, "Stopped scanning TIFF overviews");
            decoder_clean = false;
            break;
        }
        ifd_index += 1;
        let subfile = match opt_tag(decoder.get_tag_unsigned::<u32>(Tag::NewSubfileType), cancel) {
            Ok(v) => v.unwrap_or(0),
            Err(RasterError::Cancelled) => return Err(RasterError::Cancelled),
            Err(_) => continue,
        };
        if !is_overview(subfile) {
            continue;
        }
        match ensure_readable(decoder, cancel) {
            Ok((o_layout, o_bits, o_type)) => {
                if o_layout != base.pixel_layout
                    || o_bits != base.sample_bits
                    || o_type != base.sample_type
                {
                    continue;
                }
            }
            Err(RasterError::Cancelled) => return Err(RasterError::Cancelled),
            Err(_) => {
                if cancel.is_cancelled() {
                    return Err(RasterError::Cancelled);
                }
                continue;
            }
        }
        let Ok((ow, oh)) = decoder.dimensions() else {
            if cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            continue;
        };
        if ow == 0 || oh == 0 {
            continue;
        }
        // `chunk_grid` checks `ensure_chunky` once per overview.
        let ochunk = match chunk_grid(decoder, budget, base.depth, cancel) {
            Ok(c) => c,
            Err(RasterError::Cancelled) => return Err(RasterError::Cancelled),
            Err(_) => continue,
        };
        overviews.push(Overview {
            image: ifd_index,
            width: ow,
            height: oh,
            chunk: ochunk,
        });
    }
    Ok((overviews, ifd_index, decoder_clean))
}

/// Thread-safe opened GeoTIFF/BigTIFF/COG source backed by a positioned
/// [`ByteSource`] with pooled `tiff::decoder::Decoder` instances.
pub struct TiffSource {
    decoder_pool: DecoderPool,
    fingerprint: SourceFingerprint,
    source_id: SourceId,
    description: RasterDescription,
    grid: Grid,
    pixel_layout: PixelLayout,
    stats_ifd: usize,
    nodata: Option<f32>,
    sample_bits: u8,
    stretch: StretchCache,
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

        let (pixel_layout, sample_bits, sample_type) = ensure_readable(&mut decoder, cancel)?;
        let tiled = decoder.get_tag_unsigned::<u32>(Tag::TileWidth).is_ok();
        if !tiled && (width > MAX_SOURCE_SIDE || height > MAX_SOURCE_SIDE) {
            return Err(RasterError::LimitExceeded {
                width,
                height,
                estimated_bytes: u64::from(width)
                    .saturating_mul(u64::from(height))
                    .saturating_mul(4),
                limit_bytes: u64::from(MAX_SOURCE_SIDE)
                    .saturating_mul(u64::from(MAX_SOURCE_SIDE))
                    .saturating_mul(4),
            });
        }
        let depth = depth_of_layout(pixel_layout, sample_bits);
        let chunk = chunk_grid(&mut decoder, budget, depth, cancel)?;

        let nodata_f64 = parse_gdal_nodata(&mut decoder, cancel)?;
        let nodata = nodata_f64.and_then(|v| {
            if v.is_nan() {
                None
            } else {
                Some(v.clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32)
            }
        });

        let ParsedGeoref {
            ties,
            placement,
            foreign_datum_epsg,
            trouble: binding_trouble,
        } = georef(&mut decoder, width, height, cancel)?;

        let base_fmt = BaseFormat {
            pixel_layout,
            sample_bits,
            sample_type,
            depth,
        };
        let (overviews, ifd_index, decoder_clean) =
            scan_overviews(&mut decoder, &base_fmt, budget, cancel)?;

        let grid = Grid {
            tiled,
            chunk,
            overviews,
            depth,
        };
        let levels = build_chunked_level_table(&grid, width, height, budget);
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
            foreign_datum_epsg,
            binding_trouble,
            levels,
        };

        let initial_mapping = (sample_bits <= 8).then(|| DisplayMapping::identity(nodata));
        let seed_decoder = decoder_clean.then_some((decoder, ifd_index));

        Ok(Self {
            decoder_pool: DecoderPool::new(byte_source, limits, seed_decoder),
            fingerprint,
            source_id: SourceId::from_fingerprint(&fingerprint),
            description,
            grid,
            pixel_layout,
            stats_ifd,
            nodata,
            sample_bits,
            stretch: StretchCache::new(initial_mapping),
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

    /// Return the number of compatible reduced-resolution overviews discovered
    /// during [`Self::open`].
    pub fn overview_count(&self) -> usize {
        self.grid.overviews.len()
    }

    /// Return whether the base IFD uses tiled TIFF storage (`TileWidth`).
    pub const fn is_tiled(&self) -> bool {
        self.grid.tiled
    }

    /// Return the estimated peak working bytes required by [`Self::produce_pass`].
    pub fn pass_peak_bytes(&self) -> u64 {
        self.grid
            .pass_peak((self.description.width, self.description.height))
            .total()
    }

    /// Return scheduler-facing [`PyramidMeta`] for the display band.
    pub fn pyramid_meta(&self) -> PyramidMeta {
        PyramidMeta {
            source: self.source_id,
            spec: DISPLAY_STREAM_SPEC,
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

    fn display_mapping(&self, ctx: &DecodeContext<'_>) -> Result<DisplayMapping, RasterError> {
        self.stretch.get_or_compute(&StretchRequest {
            decoder_pool: &self.decoder_pool,
            pixel_layout: self.pixel_layout,
            stats_ifd: self.stats_ifd,
            nodata: self.nodata,
            sample_bits: self.sample_bits,
            depth: self.grid.depth,
            budget_tracker: ctx.budget_tracker,
            cancel: ctx.cancel,
        })
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
        produced.ok_or_else(|| RasterError::TileOutOfRange {
            scheme: self.pyramid_meta().scheme(),
            level,
            x: tx,
            y: ty,
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
        emit: TileEmit<'_>,
    ) -> Result<(), RasterError> {
        let mapping = self.display_mapping(ctx)?;
        let mut reader = TiffChunkReader::new(
            &self.decoder_pool,
            self.source_id,
            self.pixel_layout,
            self.grid.depth,
            mapping,
            ChunkReadMode::Pointwise {
                chunk_cache: ctx.chunk_cache,
                budget_tracker: ctx.budget_tracker,
            },
        );
        let req = DirectRead {
            grid: &self.grid,
            base: (self.description.width, self.description.height),
            level,
            wants,
            budget: ctx.budget,
            budget_tracker: ctx.budget_tracker,
            cancel: ctx.cancel,
        };
        driver::direct(&mut reader, &req, emit)
    }

    /// Run a sequential top-to-bottom pass over the base IFD, emitting every
    /// pyramid tile via `emit`.
    pub fn produce_pass(
        &self,
        ctx: &DecodeContext<'_>,
        emit: TileEmit<'_>,
    ) -> Result<(), RasterError> {
        let mapping = self.display_mapping(ctx)?;
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
            &self.decoder_pool,
            self.source_id,
            self.pixel_layout,
            self.grid.depth,
            mapping,
            ChunkReadMode::SequentialPass,
        );
        driver::pass(
            &mut reader,
            &self.grid,
            (self.description.width, self.description.height),
            ctx.cancel,
            emit,
        )
    }
}

#[cfg(test)]
#[path = "tiff_tests.rs"]
mod tests;
