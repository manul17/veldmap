//! Live `WebMercator` polar-rim sampler (`PolarRimTracker`) capturing
//! `±85.05112878°` edge texels from decoded live tiles so `PolarCap` tiles
//! blend seamlessly against the active live basemap with zero extra HTTP
//! requests.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use veldmap_core::sync::lock_unpoisoned;
use veldmap_core::tile::{DecodedTile, PlanetaryScheme, QuadTileId, SourceId};
use veldmap_raster::{POLAR_RIM_BINS, PolarRimPyramid, extract_srgb_rim_row};

/// Minimum interval between rebuilding a pole's harmonic [`PolarRimPyramid`]
/// after initial rim data has been committed, preventing polar-tile churn while
/// bursts of high-zoom rim tiles stream in.
const RIM_REBUILD_DEBOUNCE: Duration = Duration::from_millis(250);

/// Maximum `level_tag` recorded per rim bin (`z = 11` -> `cols = 2048 ==
/// POLAR_RIM_BINS`). Higher zoom tiles (`z >= 12`) can still populate empty or
/// coarser bins whose center lies inside the tile, but never overwrite a full
/// `z = 11` bin with a sub-bin fragment.
const MAX_RIM_LEVEL_TAG: u8 = 12;

/// Minimum linear-light RGB difference required to mark an equal-level rim bin
/// update dirty (suppressing redundant rebuilds when LRU-evicted tiles re-decode
/// from disk cache).
const EQUAL_LEVEL_DIFF_EPS: f32 = 1e-4;

struct PolarRimSide {
    /// Per-longitude-bin `(linear_rgb, level_tag)` where `level_tag == 0` means
    /// unpopulated and `level_tag == (z + 1).min(MAX_RIM_LEVEL_TAG)` records
    /// the finest live zoom level observed for that bin.
    bins: Vec<([f32; 3], u8)>,
    dirty: bool,
    /// Currently active revision (`0` when unpopulated).
    revision: u32,
    /// Monotonic revision allocator shared across source switches (never reset
    /// to `0` so cross-source ABA collisions are impossible).
    next_revision: u32,
    last_commit: Option<Instant>,
    pyramid: Option<(u32, Arc<PolarRimPyramid>)>,
}

impl Default for PolarRimSide {
    fn default() -> Self {
        Self {
            bins: vec![([0.0; 3], 0); POLAR_RIM_BINS],
            dirty: false,
            revision: 0,
            next_revision: 1,
            last_commit: None,
            pyramid: None,
        }
    }
}

impl PolarRimSide {
    fn clear(&mut self) {
        self.bins.fill(([0.0; 3], 0));
        self.dirty = false;
        self.revision = 0;
        self.last_commit = None;
        self.pyramid = None;
    }

    fn alloc_revision(&mut self) -> u32 {
        let rev = self.next_revision;
        self.next_revision = self.next_revision.wrapping_add(1);
        if self.next_revision == 0 {
            self.next_revision = 1;
        }
        rev
    }

    fn ingest_row(&mut self, id: QuadTileId, cols: u32, row: &[Option<[f32; 3]>]) -> bool {
        if row.is_empty() || cols == 0 {
            return false;
        }
        let level_tag = id.level.saturating_add(1).min(MAX_RIM_LEVEL_TAG);
        let bins_f = POLAR_RIM_BINS as f64;
        let cols_f = f64::from(cols);
        let start_f = (f64::from(id.x) / cols_f) * bins_f;
        let end_f = (f64::from(id.x + 1) / cols_f) * bins_f;
        let start_bin = (start_f.floor() as usize).min(POLAR_RIM_BINS - 1);
        let end_bin = (end_f.ceil() as usize).clamp(start_bin + 1, POLAR_RIM_BINS);
        let row_len = row.len();
        let row_len_f = row_len as f64;
        let span_f = (end_f - start_f).max(1e-9);

        let mut updated = false;
        for bin in start_bin..end_bin {
            let bin_center = (bin as f64) + 0.5;
            // Strictly require the bin center to fall within this tile's
            // longitudinal span so high-zoom sub-bin tiles (`z >= 12`) never
            // extrapolate out of range into neighboring bins.
            if bin_center < start_f || bin_center >= end_f {
                continue;
            }
            let (prev_rgb, prev_tag) = self.bins[bin];
            if level_tag < prev_tag {
                continue;
            }

            // Box-average all opaque tile pixels falling within `[bin, bin + 1)`
            // in linear light (preventing single-pixel aliasing at z = 4..=11).
            let u0 = (((bin as f64) - start_f) / span_f).clamp(0.0, 1.0);
            let u1 = (((bin as f64 + 1.0) - start_f) / span_f).clamp(0.0, 1.0);
            let px0 = ((u0 * row_len_f).floor() as usize).min(row_len - 1);
            let px1 = ((u1 * row_len_f).ceil() as usize).clamp(px0 + 1, row_len);

            let mut acc = [0.0_f32; 3];
            let mut count = 0u32;
            for rgb in row[px0..px1].iter().flatten() {
                acc[0] += rgb[0];
                acc[1] += rgb[1];
                acc[2] += rgb[2];
                count += 1;
            }
            if count == 0 {
                continue;
            }
            let inv = 1.0 / (count as f32);
            let new_rgb = [acc[0] * inv, acc[1] * inv, acc[2] * inv];

            let diff = (new_rgb[0] - prev_rgb[0])
                .abs()
                .max((new_rgb[1] - prev_rgb[1]).abs())
                .max((new_rgb[2] - prev_rgb[2]).abs());
            if level_tag > prev_tag || diff > EQUAL_LEVEL_DIFF_EPS {
                self.bins[bin] = (new_rgb, level_tag);
                updated = true;
            }
        }
        if updated {
            self.dirty = true;
        }
        updated
    }

    fn next_flush_delay(&self, now: Instant) -> Option<Duration> {
        if !self.dirty {
            return None;
        }
        if self.revision == 0 {
            return Some(Duration::ZERO);
        }
        let Some(last) = self.last_commit else {
            return Some(Duration::ZERO);
        };
        Some(RIM_REBUILD_DEBOUNCE.saturating_sub(now.saturating_duration_since(last)))
    }

    fn poll_revision(&mut self, now: Instant) -> u32 {
        if !self.dirty {
            return self.revision;
        }
        let debounced = self.revision > 0
            && self
                .last_commit
                .is_some_and(|t| now.saturating_duration_since(t) < RIM_REBUILD_DEBOUNCE);
        if debounced {
            return self.revision;
        }
        let populated: Vec<([f32; 3], bool)> =
            self.bins.iter().map(|&(rgb, tag)| (rgb, tag > 0)).collect();
        if let Some(pyr) = PolarRimPyramid::from_bins(&populated) {
            let rev = self.alloc_revision();
            self.pyramid = Some((rev, Arc::new(pyr)));
            self.revision = rev;
            self.last_commit = Some(now);
            self.dirty = false;
        }
        self.revision
    }
}

#[derive(Default)]
struct PolarRimInner {
    active_source: Option<SourceId>,
    sides: [PolarRimSide; 2], // [0] = north, [1] = south
}

/// Thread-safe tracker capturing `±85.05112878°` rim pixels from decoded live
/// `WebMercator` basemap tiles for harmonic polar-cap synthesis.
#[derive(Default)]
pub(super) struct PolarRimTracker {
    inner: Mutex<PolarRimInner>,
}

impl PolarRimTracker {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Reset all captured rim samples when switching active basemap sources
    /// while preserving monotonic revision IDs to prevent cross-source ABA.
    pub(super) fn reset_for_source(&self, source: Option<SourceId>) {
        let mut st = lock_unpoisoned(&self.inner);
        st.active_source = source;
        for side in &mut st.sides {
            side.clear();
        }
    }

    /// Inspect a freshly decoded live tile; if it belongs to the active
    /// `WebMercator` source and touches the North (`y == 0`) or South
    /// (`y == rows - 1`) polar boundary, record its edge pixels.
    pub(super) fn ingest_live_tile(
        &self,
        source: SourceId,
        id: QuadTileId,
        tile: &DecodedTile,
    ) -> bool {
        if id.scheme != PlanetaryScheme::WebMercator {
            return false;
        }
        let Some((cols, rows)) = id.checked_grid_dims() else {
            return false;
        };
        let touches_north = id.y == 0;
        let touches_south = id.y + 1 == rows;
        if !touches_north && !touches_south {
            return false;
        }

        // Fast check under lock before copying edge texels.
        {
            let st = lock_unpoisoned(&self.inner);
            if st.active_source != Some(source) {
                return false;
            }
        }

        let north_row = touches_north.then(|| extract_srgb_rim_row(tile, true));
        let south_row = touches_south.then(|| extract_srgb_rim_row(tile, false));

        let mut st = lock_unpoisoned(&self.inner);
        if st.active_source != Some(source) {
            return false;
        }
        let mut changed = false;
        if let Some(row) = north_row.as_deref() {
            changed |= st.sides[0].ingest_row(id, cols, row);
        }
        if let Some(row) = south_row.as_deref() {
            changed |= st.sides[1].ingest_row(id, cols, row);
        }
        changed
    }

    /// Remaining debounce delay if either pole has uncommitted rim updates so
    /// an idle UI loop schedules a follow-up frame to flush them.
    pub(super) fn next_flush_delay(&self, now: Instant) -> Option<Duration> {
        let st = lock_unpoisoned(&self.inner);
        let d0 = st.sides[0].next_flush_delay(now);
        let d1 = st.sides[1].next_flush_delay(now);
        match (d0, d1) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// Commit any pending rim updates (subject to debounce) and return the
    /// latest revision (`0` when no live rim data has been captured yet).
    pub(super) fn poll_revision(&self, north: bool, now: Instant) -> u32 {
        let mut st = lock_unpoisoned(&self.inner);
        let idx = usize::from(!north);
        st.sides[idx].poll_revision(now)
    }

    /// Borrow the committed harmonic rim pyramid strictly matching
    /// `expected_rev` (returning `None` if superseded or reset).
    pub(super) fn pyramid_for_revision(
        &self,
        north: bool,
        expected_rev: u32,
    ) -> Option<Arc<PolarRimPyramid>> {
        let st = lock_unpoisoned(&self.inner);
        let idx = usize::from(!north);
        st.sides[idx]
            .pyramid
            .as_ref()
            .filter(|(rev, _)| *rev == expected_rev)
            .map(|(_, pyr)| Arc::clone(pyr))
    }
}
