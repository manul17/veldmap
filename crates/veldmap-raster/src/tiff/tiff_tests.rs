use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use ::tiff::encoder::{TiffEncoder, colortype};
use ::tiff::tags::Tag;
use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::io::testing::MemoryByteSource;
use veldmap_core::io::{ByteSource, IoError};
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::SourceFingerprint;

use super::georef::{affine_from_matrix, foreign_datum, geo_placement, geo_ties};
use super::tags::{is_overview, resolve_sample_type};
use super::{TiffSource, build_tiff_limits};
use crate::RasterError;
use crate::pipeline::DecodeContext;
use crate::pipeline::chunk_cache::SharedChunkCache;
use crate::pipeline::grid::Grid;
use crate::pipeline::metadata::{GeoKeyParams, GeoRasterType, SampleType};
use crate::pipeline::plan::{self, ServeMode};

fn tiny_u16_tiff_bytes() -> Vec<u8> {
    let mut file = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut file).expect("tiff encoder");
        let mut dir = enc.image_directory().expect("dir");
        dir.write_tag(Tag::ImageWidth, 256u32).expect("w");
        dir.write_tag(Tag::ImageLength, 256u32).expect("h");
        dir.write_tag(Tag::BitsPerSample, 16u16).expect("bps");
        dir.write_tag(Tag::Compression, 1u16).expect("comp");
        dir.write_tag(Tag::PhotometricInterpretation, 1u16)
            .expect("photo");
        dir.write_tag(Tag::SamplesPerPixel, 1u16).expect("spp");
        dir.write_tag(Tag::PlanarConfiguration, 1u16)
            .expect("planar");
        dir.write_tag(Tag::TileWidth, 256u32).expect("tw");
        dir.write_tag(Tag::TileLength, 256u32).expect("th");
        let samples: Vec<u16> = (0..65_536u32).map(|v| (v % 4096) as u16).collect();
        let at = dir.write_data(&samples[..]).expect("write");
        dir.write_tag(Tag::TileOffsets, &[at as u32][..])
            .expect("offs");
        dir.write_tag(Tag::TileByteCounts, &[(65_536u32 * 2)][..])
            .expect("cnts");
        dir.finish().expect("finish");
    }
    file.into_inner()
}

#[test]
fn display_mapping_distinguishes_leader_cancel_from_permanent_failure() {
    struct ControlledTailSource {
        inner: MemoryByteSource,
        threshold: u64,
        armed: AtomicBool,
        hit: AtomicBool,
        fail_tail: AtomicBool,
        tail_reads: AtomicUsize,
        release: CancelToken,
    }

    impl ByteSource for ControlledTailSource {
        fn len(&self) -> u64 {
            self.inner.len()
        }

        fn read_at(
            &self,
            offset: u64,
            buf: &mut [u8],
            cancel: &CancelToken,
        ) -> Result<usize, IoError> {
            if self.armed.load(Ordering::Acquire)
                && offset.saturating_add(buf.len() as u64) > self.threshold
            {
                self.tail_reads.fetch_add(1, Ordering::AcqRel);
                self.hit.store(true, Ordering::Release);
                while !self.release.is_cancelled() && !cancel.is_cancelled() {
                    thread::sleep(Duration::from_millis(2));
                }
                if cancel.is_cancelled() {
                    return Err(IoError::Cancelled);
                }
                if self.fail_tail.load(Ordering::Acquire) {
                    return Err(IoError::Io(std::io::Error::other("simulated disk fault")));
                }
            }
            self.inner.read_at(offset, buf, cancel)
        }
    }

    let raw = tiny_u16_tiff_bytes();
    let budget = MemoryBudget::default();
    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);

    // 1. Permanent leader error fans out to current-round follower without a serial retry storm.
    let fail_release = CancelToken::new();
    let fail_gate = Arc::new(ControlledTailSource {
        inner: MemoryByteSource::from_vec(raw.clone()),
        threshold: 64 * 1024,
        armed: AtomicBool::new(false),
        hit: AtomicBool::new(false),
        fail_tail: AtomicBool::new(true),
        tail_reads: AtomicUsize::new(0),
        release: fail_release.clone(),
    });
    let fail_src = Arc::new(
        TiffSource::open(
            Arc::clone(&fail_gate) as Arc<dyn ByteSource>,
            SourceFingerprint::from_digest([7u8; 32]),
            &budget,
            &CancelToken::new(),
        )
        .expect("open u16"),
    );
    fail_gate.armed.store(true, Ordering::Release);

    let f_leader = {
        let s = Arc::clone(&fail_src);
        let t = tracker.clone();
        let c_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
        thread::spawn(move || {
            let cancel = CancelToken::new();
            let ctx = DecodeContext {
                chunk_cache: &c_cache,
                budget_tracker: &t,
                budget: &budget,
                cancel: &cancel,
            };
            s.display_mapping(&ctx)
        })
    };
    while !fail_gate.hit.load(Ordering::Acquire) {
        thread::sleep(Duration::from_millis(2));
    }
    let f_follower = {
        let s = Arc::clone(&fail_src);
        let t = tracker.clone();
        let c_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
        thread::spawn(move || {
            let cancel = CancelToken::new();
            let ctx = DecodeContext {
                chunk_cache: &c_cache,
                budget_tracker: &t,
                budget: &budget,
                cancel: &cancel,
            };
            s.display_mapping(&ctx)
        })
    };
    thread::sleep(Duration::from_millis(15));
    fail_release.cancel();
    let leader_err = f_leader
        .join()
        .expect("join leader")
        .expect_err("leader fails");
    let follower_err = f_follower
        .join()
        .expect("join follower")
        .expect_err("follower receives error directly");
    assert!(!leader_err.is_cancelled());
    assert!(!follower_err.is_cancelled());
    assert_eq!(
        fail_gate.tail_reads.load(Ordering::Acquire),
        1,
        "permanent error must not trigger serial follower retries"
    );

    // 2. Leader cancellation allows an uncancelled follower to elect itself and succeed.
    let cancel_release = CancelToken::new();
    let cancel_gate = Arc::new(ControlledTailSource {
        inner: MemoryByteSource::from_vec(raw),
        threshold: 64 * 1024,
        armed: AtomicBool::new(false),
        hit: AtomicBool::new(false),
        fail_tail: AtomicBool::new(false),
        tail_reads: AtomicUsize::new(0),
        release: cancel_release.clone(),
    });
    let cancel_src = Arc::new(
        TiffSource::open(
            Arc::clone(&cancel_gate) as Arc<dyn ByteSource>,
            SourceFingerprint::from_digest([8u8; 32]),
            &budget,
            &CancelToken::new(),
        )
        .expect("open u16 for cancel retry"),
    );
    cancel_gate.armed.store(true, Ordering::Release);

    let leader_cancel = CancelToken::new();
    let c_leader = {
        let s = Arc::clone(&cancel_src);
        let t = tracker.clone();
        let lc = leader_cancel.clone();
        let c_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
        thread::spawn(move || {
            let ctx = DecodeContext {
                chunk_cache: &c_cache,
                budget_tracker: &t,
                budget: &budget,
                cancel: &lc,
            };
            s.display_mapping(&ctx)
        })
    };
    while !cancel_gate.hit.load(Ordering::Acquire) {
        thread::sleep(Duration::from_millis(2));
    }
    let barrier = Arc::new(Barrier::new(2));
    let c_follower = {
        let s = Arc::clone(&cancel_src);
        let t = tracker.clone();
        let b = Arc::clone(&barrier);
        let c_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
        thread::spawn(move || {
            b.wait();
            let cancel = CancelToken::new();
            let ctx = DecodeContext {
                chunk_cache: &c_cache,
                budget_tracker: &t,
                budget: &budget,
                cancel: &cancel,
            };
            s.display_mapping(&ctx)
        })
    };
    barrier.wait();
    thread::sleep(Duration::from_millis(15));
    // Cancel the leader while releasing the gate for the uncancelled follower's retry.
    cancel_gate.armed.store(false, Ordering::Release);
    leader_cancel.cancel();
    assert!(matches!(
        c_leader.join().expect("join cancelled leader"),
        Err(RasterError::Cancelled)
    ));
    let mapping = c_follower
        .join()
        .expect("join retried follower")
        .expect("uncancelled follower succeeds after leader cancel");
    assert_ne!(
        mapping,
        crate::pipeline::radiometry::DisplayMapping::identity(None)
    );

    // 3. Cached Ready mapping is served immediately for subsequent calls.
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &CancelToken::new(),
    };
    assert_eq!(cancel_src.display_mapping(&ctx).expect("cached"), mapping);
}

#[test]
fn georef_tag_limits_and_table_helpers_work_consistently() {
    let mut file = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut file).expect("tiff encoder");
        let img = enc.new_image::<colortype::RGB8>(4, 4).expect("new_image");
        img.write_data(&[128u8; 4 * 4 * 3]).expect("write");
    }
    let tiny_budget = MemoryBudget::for_host_ram(1024 * 1024 * 1024);
    let limits = build_tiff_limits(&tiny_budget);
    assert!(limits.intermediate_buffer_size >= 4 * 1024 * 1024);
    assert!(limits.intermediate_buffer_size <= 128 * 1024 * 1024);

    assert!(!is_overview(0));
    assert!(is_overview(1));
    assert!(!is_overview(5)); // mask bit set
    assert_eq!(resolve_sample_type(1, 8).unwrap(), SampleType::U8);
    assert_eq!(resolve_sample_type(1, 16).unwrap(), SampleType::U16);
    assert_eq!(resolve_sample_type(2, 16).unwrap(), SampleType::I16);
    assert_eq!(resolve_sample_type(3, 32).unwrap(), SampleType::F32);
    assert!(resolve_sample_type(1, 12).is_err());
    assert!(resolve_sample_type(1, 32).is_err());

    // Foreign datum detection from geographic CRS, datum key, or Pulkovo-42 zone.
    let mut keys = GeoKeyParams {
        projected_crs: Some(28407),
        ..Default::default()
    };
    assert_eq!(foreign_datum(&keys), Some(4284));
    keys.geographic_crs = Some(4326);
    assert_eq!(foreign_datum(&keys), Some(4284));
    keys.geographic_crs = Some(4267);
    assert_eq!(foreign_datum(&keys), Some(4267));

    // PixelIsArea vs PixelIsPoint shifts affine origin by half a pixel.
    let area_keys = GeoKeyParams {
        model_type: Some(1),
        projected_crs: Some(32637),
        raster_type: GeoRasterType::PixelIsArea,
        ..Default::default()
    };
    let tie = [0.0, 0.0, 0.0, 500_000.0, 6_000_000.0, 0.0];
    let scale = [10.0, 10.0, 0.0];
    let p_area = geo_placement(&area_keys, &tie, &scale, &[], 100, 100).expect("area placement");
    assert_eq!(
        p_area.affine,
        [10.0, 0.0, 500_000.0, 0.0, -10.0, 6_000_000.0]
    );

    let mut point_keys = area_keys.clone();
    point_keys.raster_type = GeoRasterType::PixelIsPoint;
    let p_point = geo_placement(&point_keys, &tie, &scale, &[], 100, 100).expect("point placement");
    assert_eq!(
        p_point.affine,
        [10.0, 0.0, 499_995.0, 0.0, -10.0, 6_000_005.0]
    );

    // Singular matrix is rejected cleanly.
    assert!(affine_from_matrix(&[0.0; 8], 0.0).is_none());

    // Geographic tie points produce 4 corners cleanly.
    let geo_keys = GeoKeyParams {
        model_type: Some(2),
        ..Default::default()
    };
    let ties = geo_ties(
        &geo_keys,
        &[0.0, 0.0, 0.0, 30.0, 60.0, 0.0],
        &[0.01, 0.01, 0.0],
        &[],
        10,
        10,
    );
    assert_eq!(ties.len(), 4);
    assert!((ties[3].lon - 30.1).abs() < 1e-12);
    assert!((ties[3].lat - 59.9).abs() < 1e-12);

    // Table builder assigns Pass when chunk held pixels exceed decoded_chunk_cache_bytes.
    let grid = Grid {
        tiled: false,
        chunk: (8192, 4096), // 33.5M px * 4 B = 134 MB > 64 MiB chunk cache
        overviews: Vec::new(),
        depth: 3,
    };
    let tbl = plan::build_chunked_level_table(&grid, 8192, 8192, &tiny_budget);
    assert!(!tbl.is_empty());
    assert!(matches!(tbl[0].serve, ServeMode::Pass { from: 0 }));
}
