//! GeoTIFF / BigTIFF / Cloud-Optimized GeoTIFF (COG) reader built over
//! positioned [`ByteSource`] with explicit `tiff::decoder::Limits`,
//! `MAX_IFDS` bounds, `GeoDoubleParamsTag`/`GeoAsciiParamsTag` parsing,
//! ZSTD & `Predictor = 3` (via `tiff`) support, and singleflight chunk caching.

mod chunk_reader;
mod georef;
mod table;
mod tags;

use std::sync::{Arc, Condvar, Mutex};

use tiff::decoder::{Decoder, Limits};
use tiff::tags::Tag;
use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::io::{
    BufferedByteReader, ByteSource, CancelToken, SourceFingerprint, lock_unpoisoned,
};
use veldmap_core::tile::{BandKey, DecodedTile, LevelPlan, PixelFormat, PyramidMeta, SourceId};

use crate::RasterError;
use crate::band::{BandInfo, RasterDescription};
use crate::chunk_cache::SharedChunkCache;
use crate::grid::{self, Grid, Overview};
use crate::pyramid::RASTER_TILE_SIZE;
use crate::radiometry::{self, DisplayMapping, percentile_stretch};

use chunk_reader::TiffChunkReader;
use georef::{ParsedGeoref, georef, parse_gdal_nodata};
use tags::{
    chunk_grid, depth_of_layout, ensure_chunky, ensure_readable, is_overview, map_tiff_error,
    typed_samples,
};

/// BigTIFF file header signatures (`version = 43`).
pub const BIG_MAGIC: [&[u8]; 2] = [b"II\x2b\x00", b"MM\x00\x2b"];

/// Classic TIFF file header signatures (`version = 42`).
pub const CLASSIC_MAGIC: [&[u8]; 2] = [b"II\x2a\x00", b"MM\x00\x2a"];

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

/// Return whether `head` begins with a classic TIFF or BigTIFF signature.
pub fn has_tiff_magic(head: &[u8]) -> bool {
    CLASSIC_MAGIC.iter().any(|m| head.starts_with(m))
        || BIG_MAGIC.iter().any(|m| head.starts_with(m))
}

fn build_tiff_limits(budget: &MemoryBudget) -> Limits {
    let mut limits = Limits::default();
    limits.decoding_buffer_size = budget.decode_working_bytes.min(256 * 1024 * 1024) as usize;
    limits.intermediate_buffer_size = 128 * 1024 * 1024;
    limits
}

/// Shared runtime context passed into GeoTIFF decoding calls.
pub struct DecodeContext<'a> {
    pub chunk_cache: &'a SharedChunkCache,
    pub budget_tracker: &'a DecodeBudgetTracker,
    pub budget: &'a MemoryBudget,
    pub cancel: &'a CancelToken,
}

enum MappingState {
    Uncomputed,
    InFlight,
    Ready(DisplayMapping),
}

struct MappingFlightGuard<'a> {
    state: &'a Mutex<MappingState>,
    cv: &'a Condvar,
    armed: bool,
}

impl Drop for MappingFlightGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            let mut guard = lock_unpoisoned(self.state);
            if matches!(*guard, MappingState::InFlight) {
                *guard = MappingState::Uncomputed;
            }
            self.cv.notify_all();
        }
    }
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
    mapping_state: Arc<Mutex<MappingState>>,
    mapping_cv: Arc<Condvar>,
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
            match (
                ensure_chunky(&mut decoder, cancel),
                ensure_readable(&mut decoder, cancel),
            ) {
                (Ok(()), Ok((o_layout, o_bits, o_type))) => {
                    if o_layout != pixel_layout || o_bits != sample_bits || o_type != sample_type {
                        continue;
                    }
                }
                (Err(RasterError::Cancelled), _) | (_, Err(RasterError::Cancelled)) => {
                    return Err(RasterError::Cancelled);
                }
                _ => {
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
            let ochunk = match chunk_grid(&mut decoder, budget, depth, cancel) {
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
            foreign_datum_epsg,
            binding_trouble,
            levels,
        };

        let initial_mapping = if sample_bits <= 8 {
            MappingState::Ready(DisplayMapping::identity(nodata))
        } else {
            MappingState::Uncomputed
        };

        Ok(Self {
            byte_source,
            fingerprint,
            source_id: SourceId::from_fingerprint(&fingerprint),
            description,
            grid,
            stats_ifd,
            nodata,
            sample_bits,
            limits,
            mapping_state: Arc::new(Mutex::new(initial_mapping)),
            mapping_cv: Arc::new(Condvar::new()),
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
        loop {
            let mut state = lock_unpoisoned(&self.mapping_state);
            match *state {
                MappingState::Ready(ready) => return Ok(ready),
                MappingState::Uncomputed => {
                    if cancel.is_cancelled() {
                        return Err(RasterError::Cancelled);
                    }
                    *state = MappingState::InFlight;
                    drop(state);

                    let mut guard = MappingFlightGuard {
                        state: &self.mapping_state,
                        cv: &self.mapping_cv,
                        armed: true,
                    };
                    let computed = compute_mapping(MappingRequest {
                        byte_source: &self.byte_source,
                        limits: self.limits.clone(),
                        stats_ifd: self.stats_ifd,
                        nodata: self.nodata,
                        sample_bits: self.sample_bits,
                        depth: self.grid.depth,
                        budget_tracker,
                        cancel,
                    })?;
                    let mut state = lock_unpoisoned(&self.mapping_state);
                    *state = MappingState::Ready(computed);
                    guard.armed = false;
                    self.mapping_cv.notify_all();
                    return Ok(computed);
                }
                MappingState::InFlight => {
                    // Drop `state` before registering `on_cancel_guarded`: if
                    // `cancel` triggers right now, `on_cancel_guarded` runs the
                    // callback synchronously on this thread and locks
                    // `mapping_state`, which would self-deadlock if held.
                    drop(state);
                    if cancel.is_cancelled() {
                        return Err(RasterError::Cancelled);
                    }
                    let state_arc = Arc::clone(&self.mapping_state);
                    let cv_arc = Arc::clone(&self.mapping_cv);
                    let _cancel_guard = cancel.on_cancel_guarded(move || {
                        let _guard = lock_unpoisoned(&state_arc);
                        cv_arc.notify_all();
                    });
                    let mut state = lock_unpoisoned(&self.mapping_state);
                    while matches!(*state, MappingState::InFlight) {
                        if cancel.is_cancelled() {
                            return Err(RasterError::Cancelled);
                        }
                        state = self
                            .mapping_cv
                            .wait(state)
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                    }
                    if let MappingState::Ready(ready) = *state {
                        return Ok(ready);
                    }
                }
            }
        }
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

struct MappingRequest<'a> {
    byte_source: &'a Arc<dyn ByteSource>,
    limits: Limits,
    stats_ifd: usize,
    nodata: Option<f32>,
    sample_bits: u8,
    depth: u32,
    budget_tracker: &'a DecodeBudgetTracker,
    cancel: &'a CancelToken,
}

fn compute_mapping(req: MappingRequest<'_>) -> Result<DisplayMapping, RasterError> {
    if req.sample_bits <= 8 {
        return Ok(DisplayMapping::identity(req.nodata));
    }
    let reader = BufferedByteReader::new(Arc::clone(req.byte_source), req.cancel.clone());
    let mut decoder = Decoder::new(reader)
        .map_err(|e| map_tiff_error(e, req.cancel))?
        .with_limits(req.limits);
    decoder
        .seek_to_image(req.stats_ifd)
        .map_err(|e| map_tiff_error(e, req.cancel))?;
    let (w, h) = decoder
        .dimensions()
        .map_err(|e| map_tiff_error(e, req.cancel))?;
    let (cw, ch) = decoder.chunk_dimensions();
    if cw == 0 || ch == 0 {
        return Ok(DisplayMapping::identity(req.nodata));
    }
    let (pixel, _, _) = ensure_readable(&mut decoder, req.cancel)?;

    let total = u64::from(w.div_ceil(cw)) * u64::from(h.div_ceil(ch));
    let mut picks = Vec::new();
    if total > 0 {
        let count = total.min(9);
        if count == 1 {
            picks.push(0);
        } else {
            for i in 0..count {
                let raw_idx = (i * (total - 1)) / (count - 1);
                if let Ok(idx) = u32::try_from(raw_idx) {
                    picks.push(idx);
                }
            }
        }
        picks.dedup();
    }

    let target_per_chunk = (radiometry::STRETCH_SAMPLES / picks.len().max(1)).max(1);
    let mut values = Vec::new();
    for index in picks {
        if req.cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let (dw, dh) = decoder.chunk_data_dimensions(index);
        let est_bytes = grid::chunk_working_bytes(dw, dh, req.depth, false);
        let _lease = req.budget_tracker.reserve_blocking(est_bytes, req.cancel)?;
        let raw = decoder
            .read_chunk(index)
            .map_err(|e| map_tiff_error(e, req.cancel))?;
        let samples = typed_samples(&raw)?;
        let pixels = samples.len() / pixel.channels.max(1);
        let step = (pixels * pixel.colors().max(1))
            .div_ceil(target_per_chunk)
            .max(1);
        for at in (0..pixels).step_by(step) {
            for channel in 0..pixel.colors() {
                let v = samples.get(at * pixel.channels + channel);
                if radiometry::is_data(v, req.nodata) {
                    values.push(v);
                }
            }
        }
    }
    values.truncate(radiometry::STRETCH_SAMPLES);

    Ok(match percentile_stretch(&mut values) {
        Some((lo, hi)) => DisplayMapping::stretched(lo, hi, req.nodata),
        None => DisplayMapping::identity(req.nodata),
    })
}

#[cfg(test)]
#[path = "tiff_tests.rs"]
mod tests;
