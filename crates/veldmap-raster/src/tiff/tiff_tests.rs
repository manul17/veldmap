use std::io::Cursor;
use std::sync::{Arc, Condvar, Mutex};

use ::tiff::encoder::{TiffEncoder, colortype};
use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::io::{CancelToken, MemoryByteSource, SourceFingerprint, lock_unpoisoned};
use veldmap_core::tile::ServeMode;

use super::georef::{affine_from_matrix, foreign_datum, geo_placement, geo_ties};
use super::tags::{is_overview, sample_type_from_bits};
use super::{MappingState, TiffSource, table};
use crate::RasterError;
use crate::band::{GeoKeyParams, GeoRasterType, SampleType};
use crate::grid::Grid;

fn tiny_rgb_tiff_bytes() -> Vec<u8> {
    let mut file = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut file).expect("tiff encoder");
        let img = enc.new_image::<colortype::RGB8>(4, 4).expect("new_image");
        img.write_data(&[128u8; 4 * 4 * 3]).expect("write");
    }
    file.into_inner()
}

#[test]
fn display_mapping_cancel_while_inflight_never_self_deadlocks() {
    let bytes = tiny_rgb_tiff_bytes();
    let budget = MemoryBudget::detect();
    let open_cancel = CancelToken::new();
    let src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(bytes)),
        SourceFingerprint::from_digest([7u8; 32]),
        &budget,
        &open_cancel,
    )
    .expect("open tiny tiff");

    // Simulate an in-flight leader and a follower cancelled right as it waits.
    *lock_unpoisoned(&src.mapping_state) = MappingState::InFlight;

    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let pre_cancelled = CancelToken::new();
    pre_cancelled.cancel();
    let err = src
        .display_mapping(&tracker, &pre_cancelled)
        .expect_err("pre-cancelled follower returns Cancelled");
    assert!(matches!(err, RasterError::Cancelled));

    // Also stress the race where cancel fires concurrently with InFlight waiting.
    let src_arc = Arc::new(src);
    for _ in 0..64 {
        let cancel = CancelToken::new();
        let s = Arc::clone(&src_arc);
        let t = tracker.clone();
        let c1 = cancel.clone();
        let c2 = cancel.clone();
        let h_wait = std::thread::spawn(move || s.display_mapping(&t, &c1));
        let h_cancel = std::thread::spawn(move || c2.cancel());
        h_cancel.join().expect("join cancel");
        let res = h_wait.join().expect("join waiter without deadlock");
        assert!(matches!(res, Err(RasterError::Cancelled)));
    }

    // Ensure MappingState::Ready still unblocks cleanly if leader finishes.
    let mut src_mut = TiffSource {
        byte_source: Arc::clone(&src_arc.byte_source),
        fingerprint: src_arc.fingerprint,
        source_id: src_arc.source_id,
        description: src_arc.description.clone(),
        grid: src_arc.grid.clone(),
        stats_ifd: src_arc.stats_ifd,
        nodata: src_arc.nodata,
        sample_bits: src_arc.sample_bits,
        limits: src_arc.limits.clone(),
        mapping_state: Arc::new(Mutex::new(MappingState::InFlight)),
        mapping_cv: Arc::new(Condvar::new()),
    };
    *lock_unpoisoned(&src_mut.mapping_state) = MappingState::InFlight;
    let state_w = Arc::clone(&src_mut.mapping_state);
    let cv_w = Arc::clone(&src_mut.mapping_cv);
    let h_fin = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(10));
        *lock_unpoisoned(&state_w) =
            MappingState::Ready(crate::radiometry::DisplayMapping::identity(None));
        cv_w.notify_all();
    });
    let ready = src_mut
        .display_mapping(&tracker, &CancelToken::new())
        .expect("wakes on Ready");
    h_fin.join().expect("join finisher");
    assert_eq!(ready, crate::radiometry::DisplayMapping::identity(None));
    let _ = &mut src_mut;
}

#[test]
fn georef_and_tag_helpers_handle_area_vs_point_matrix_and_foreign_datums() {
    assert!(!is_overview(0));
    assert!(is_overview(1));
    assert!(!is_overview(5)); // mask bit set
    assert_eq!(sample_type_from_bits(8).unwrap(), SampleType::U8);
    assert_eq!(sample_type_from_bits(16).unwrap(), SampleType::U16);
    assert_eq!(sample_type_from_bits(32).unwrap(), SampleType::F32);
    assert!(sample_type_from_bits(12).is_err());

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
    let p_area =
        geo_placement(&area_keys, None, &tie, &scale, &[], 100, 100).expect("area placement");
    assert_eq!(
        p_area.affine,
        [10.0, 0.0, 500_000.0, 0.0, -10.0, 6_000_000.0]
    );

    let mut point_keys = area_keys.clone();
    point_keys.raster_type = GeoRasterType::PixelIsPoint;
    let p_point =
        geo_placement(&point_keys, None, &tie, &scale, &[], 100, 100).expect("point placement");
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
    let budget = MemoryBudget::for_host_ram(1024 * 1024 * 1024); // 64 MiB chunk cache
    let grid = Grid {
        tiled: false,
        chunk: (8192, 4096), // 33.5M px * 4 B = 134 MB > 64 MiB chunk cache
        overviews: Vec::new(),
        depth: 3,
    };
    let table = table::build_chunked_level_table(&grid, 8192, 8192, &budget);
    assert!(!table.is_empty());
    assert!(matches!(table[0].serve, ServeMode::Pass { from: 0 }));
}
