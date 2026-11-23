//! GPU upload budgeting, `Weak<GpuTile>` bind-group sweeping, CPU linear-light
//! mipmaps, PolarCap fallback, and active-frame cache protection tests.

use std::collections::HashSet;
use std::time::Duration;

use veldmap_app::gpu_tiles::{
    GpuTileCache, PendingUploads, UploadPassParams, bundled_tile_key, source_tile_key,
};
use veldmap_app::render::GlobeRenderer;
use veldmap_core::geo::camera::Camera;
use veldmap_core::tile::{
    DecodedTile, PixelFormat, PlanetaryScheme, QuadTileId, QuadtreeConfig, SourceId,
    mip_chain_byte_size,
};
use veldmap_raster::{BUNDLED_MAX_LEVEL, generate_linear_mipmaps};

use crate::common::{
    build_snapshot, pixel_at, render_srgb, shared_bundled_basemap, shared_headless_device,
};

#[test]
fn pending_uploads_budget_and_end_to_end_weak_bind_group_sweeping() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let basemap = shared_bundled_basemap();
    let cfg = QuadtreeConfig::geodetic(BUNDLED_MAX_LEVEL);

    let mut pending = PendingUploads::new();
    let tx = pending.sender().expect("pending sender");

    let fine_id = QuadTileId {
        scheme: PlanetaryScheme::Geodetic,
        level: 2,
        x: 4,
        y: 1,
    };
    let root_west_id = QuadTileId {
        scheme: PlanetaryScheme::Geodetic,
        level: 0,
        x: 0,
        y: 0,
    };
    let root_east_id = QuadTileId {
        scheme: PlanetaryScheme::Geodetic,
        level: 0,
        x: 1,
        y: 0,
    };
    let south_cap_id = QuadTileId {
        scheme: PlanetaryScheme::PolarCap { north: false },
        level: 0,
        x: 0,
        y: 0,
    };

    // Send finer tile BEFORE coarser roots to verify coarsest-first priority.
    for id in [fine_id, root_west_id, root_east_id, south_cap_id] {
        tx.send((
            bundled_tile_key(id),
            basemap.render_tile_sized(id, 64).expect("render tile"),
        ))
        .expect("send tile");
    }

    // Cap cache to exactly 2 tiles (32 KiB) so uploading 4 tiles triggers real LRU eviction.
    let tile_bytes = 64 * 64 * 4;
    let mut cache = GpuTileCache::new(tile_bytes * 2);
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Budget of 1 byte must STILL upload 1 coarsest tile (anti-starvation).
    let r1 = pending.drain_and_upload(
        device,
        queue,
        &mut cache,
        UploadPassParams {
            pinned_root_sources: &[SourceId::BUNDLED_BASEMAP],
            observed_source: Some(SourceId::BUNDLED_BASEMAP),
            wanted: None,
            protected: None,
            byte_budget: 1,
            time_budget: Duration::from_millis(0),
        },
    );
    assert_eq!(r1.uploaded, 1);
    assert_eq!(r1.remaining, 3);
    assert!(
        !cache.contains(&bundled_tile_key(fine_id)),
        "coarsest z=0 must upload before z=2"
    );

    // Render using the first uploaded tile so GlobeRenderer caches its BindGroup.
    {
        let snap1 = build_snapshot(
            Camera::at(0.0, -90.0, 4_000_000.0),
            65,
            65,
            false,
            true,
            SourceId::BUNDLED_BASEMAP,
            1,
            &cfg,
            Some(&mut cache),
        );
        let _ = render_srgb(&mut renderer, device, queue, 65, 65, &snap1);
        assert_eq!(renderer.cached_tile_bind_group_count(), 1);
    }

    // Upload the remaining 3 tiles -> evicts earlier tiles from GpuTileCache.
    let r2 = pending.drain_and_upload(
        device,
        queue,
        &mut cache,
        UploadPassParams {
            pinned_root_sources: &[SourceId::BUNDLED_BASEMAP],
            observed_source: Some(SourceId::BUNDLED_BASEMAP),
            wanted: None,
            protected: None,
            byte_budget: 1024 * 1024,
            time_budget: Duration::from_secs(1),
        },
    );
    assert_eq!(r2.uploaded, 3);
    assert_eq!(r2.remaining, 0);
    assert!(r2.evicted >= 2);
    assert!(cache.total_bytes() <= cache.max_bytes());

    // Re-sending an already-cached key must be skipped (0 duplicate GPU uploads).
    let cached_id = if cache.contains(&bundled_tile_key(fine_id)) {
        fine_id
    } else {
        south_cap_id
    };
    tx.send((
        bundled_tile_key(cached_id),
        basemap.render_tile_sized(cached_id, 64).expect("dup tile"),
    ))
    .expect("send dup");
    let r_dup = pending.drain_and_upload(
        device,
        queue,
        &mut cache,
        UploadPassParams {
            pinned_root_sources: &[SourceId::BUNDLED_BASEMAP],
            observed_source: Some(SourceId::BUNDLED_BASEMAP),
            wanted: None,
            protected: None,
            byte_budget: 1024 * 1024,
            time_budget: Duration::from_secs(1),
        },
    );
    assert_eq!(r_dup.uploaded, 0);

    // Sweeping stale Weak<GpuTile> entries drops the evicted tile's BindGroup automatically.
    renderer.sweep_stale_bind_groups();
    assert_eq!(
        renderer.cached_tile_bind_group_count(),
        0,
        "evicted tile BindGroup was not swept via Weak<GpuTile>"
    );
}

#[test]
fn live_source_mipmaps_polar_fallback_and_active_frame_protection() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let live_source = SourceId::from_stable_key("eox-s2cloudless-test");
    let merc_cfg = QuadtreeConfig::web_mercator(14);
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // 1. CPU linear-light mipmaps upload into GpuTileCache with exact mip byte accounting.
    let mut bundled_wm = DecodedTile::try_new(
        64,
        64,
        PixelFormat::Rgba8UnormSrgb,
        [40, 80, 210, 255].repeat(64 * 64),
    )
    .expect("valid bundled wm tile");
    generate_linear_mipmaps(&mut bundled_wm);
    assert_eq!(bundled_wm.mip_level_count(), 7);
    let expected_bytes = mip_chain_byte_size(64, 64, PixelFormat::Rgba8UnormSrgb, 7);
    assert_eq!(bundled_wm.byte_size(), expected_bytes);

    let mut bundled_polar = DecodedTile::try_new(
        64,
        64,
        PixelFormat::Rgba8UnormSrgb,
        [235, 235, 235, 255].repeat(64 * 64),
    )
    .expect("valid bundled polar tile");
    generate_linear_mipmaps(&mut bundled_polar);

    let mut live_tile = DecodedTile::try_new(
        64,
        64,
        PixelFormat::Rgba8UnormSrgb,
        [40, 200, 70, 255].repeat(64 * 64),
    )
    .expect("valid live tile");
    generate_linear_mipmaps(&mut live_tile);

    // Budget holds exactly 2 mipmapped 64x64 tiles so Step 4 genuinely
    // exercises active-frame eviction protection and refusal deferral.
    let mut cache = GpuTileCache::new(expected_bytes * 2);
    let wm_bundled_root = bundled_tile_key(QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 0,
        x: 0,
        y: 0,
    });
    let south_polar_root = bundled_tile_key(QuadTileId {
        scheme: PlanetaryScheme::PolarCap { north: false },
        level: 0,
        x: 0,
        y: 0,
    });
    cache
        .upload_and_insert(device, queue, wm_bundled_root, &bundled_wm)
        .expect("upload bundled wm fallback");
    cache
        .upload_and_insert(device, queue, south_polar_root, &bundled_polar)
        .expect("upload bundled polar cap");
    assert_eq!(cache.total_bytes(), expected_bytes * 2);

    // 2. While live tiles have not arrived yet, DrawSnapshot parameterized by
    //    `live_source` falls back cleanly to bundled WebMercator blue.
    let (w, h) = (65, 65);
    let fallback_snap = build_snapshot(
        Camera::at(0.0, 0.0, 2_000_000.0),
        w,
        h,
        false,
        true,
        live_source,
        1,
        &merc_cfg,
        Some(&mut cache),
    );
    assert!(fallback_snap.carrier_keys.contains(&wm_bundled_root));
    let fb_px = pixel_at(
        &render_srgb(&mut renderer, device, queue, w, h, &fallback_snap),
        w,
        w / 2,
        h / 2,
    );
    assert!(
        fb_px[2] > fb_px[1] + 50,
        "expected blue bundled offline fallback, got {fb_px:?}"
    );

    // 3. PolarCap under active `live_source` ALWAYS resolves to bundled polar
    //    white (never live_source). This touches `south_polar_root`, making
    //    `wm_bundled_root` the LRU tail.
    let polar_snap = build_snapshot(
        Camera::at(-89.5, 0.0, 400_000.0),
        w,
        h,
        false,
        true,
        live_source,
        2,
        &merc_cfg,
        Some(&mut cache),
    );
    assert!(polar_snap.carrier_keys.contains(&south_polar_root));
    let polar_px = pixel_at(
        &render_srgb(&mut renderer, device, queue, w, h, &polar_snap),
        w,
        w / 2,
        h / 2,
    );
    assert!(
        polar_px[0] > 180 && polar_px[1] > 180 && polar_px[2] > 180,
        "expected bundled polar white under live source, got {polar_px:?}"
    );

    // 4. Upload live WebMercator root into the FULL 2-tile cache while
    //    `wm_bundled_root` (currently LRU tail!) is protected by `fallback_snap`.
    //    Active-frame protection MUST skip `wm_bundled_root` and evict
    //    `south_polar_root` instead.
    let live_root = source_tile_key(
        live_source,
        QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 0,
            x: 0,
            y: 0,
        },
    );
    let mut pending = PendingUploads::new();
    let tx = pending.sender().expect("sender");
    tx.send((live_root, live_tile.clone())).expect("send live");
    let pinned_live = [live_source, SourceId::BUNDLED_BASEMAP];
    let rep = pending.drain_and_upload(
        device,
        queue,
        &mut cache,
        UploadPassParams {
            pinned_root_sources: &pinned_live,
            observed_source: Some(live_source),
            wanted: Some(&HashSet::from([live_root])),
            protected: Some(&fallback_snap.carrier_keys),
            byte_budget: 1024 * 1024,
            time_budget: Duration::from_secs(1),
        },
    );
    assert_eq!(rep.uploaded, 1);
    assert_eq!(rep.evicted, 1);
    assert!(
        cache.contains(&wm_bundled_root),
        "protected LRU tail must survive"
    );
    assert!(
        !cache.contains(&south_polar_root),
        "unprotected tile must be evicted"
    );
    assert!(cache.contains(&live_root));

    // 4b. When ALL cached tiles are protected, an incoming tile is deferred
    //     (NOT silently discarded) with `remaining_ready == 0, deferred_protected == 1`,
    //     and uploads cleanly once protection lifts without re-sending.
    let all_protected = HashSet::from([wm_bundled_root, live_root]);
    tx.send((south_polar_root, bundled_polar))
        .expect("re-send polar cap");
    let deferred_rep = pending.drain_and_upload(
        device,
        queue,
        &mut cache,
        UploadPassParams {
            pinned_root_sources: &pinned_live,
            observed_source: Some(live_source),
            wanted: Some(&HashSet::from([south_polar_root])),
            protected: Some(&all_protected),
            byte_budget: 1024 * 1024,
            time_budget: Duration::from_secs(1),
        },
    );
    assert_eq!(deferred_rep.uploaded, 0);
    assert_eq!(deferred_rep.remaining_ready, 0);
    assert_eq!(deferred_rep.deferred_protected, 1);
    assert!(
        pending.contains(&south_polar_root),
        "protected-refused tile must stay deferred in queue"
    );

    let live_snap = build_snapshot(
        Camera::at(0.0, 0.0, 2_000_000.0),
        w,
        h,
        false,
        true,
        live_source,
        3,
        &merc_cfg,
        Some(&mut cache),
    );
    let live_px = pixel_at(
        &render_srgb(&mut renderer, device, queue, w, h, &live_snap),
        w,
        w / 2,
        h / 2,
    );
    assert!(
        live_px[1] > live_px[2] + 50,
        "expected green live tile override, got {live_px:?}"
    );

    // Now `live_snap` protects only `live_root` (unprotecting `wm_bundled_root`),
    // so the deferred `south_polar_root` uploads immediately without re-sending.
    let resumed_rep = pending.drain_and_upload(
        device,
        queue,
        &mut cache,
        UploadPassParams {
            pinned_root_sources: &pinned_live,
            observed_source: Some(live_source),
            wanted: Some(&HashSet::from([south_polar_root])),
            protected: Some(&live_snap.carrier_keys),
            byte_budget: 1024 * 1024,
            time_budget: Duration::from_secs(1),
        },
    );
    assert_eq!(resumed_rep.uploaded, 1);
    assert_eq!(resumed_rep.evicted, 1);
    assert!(cache.contains(&south_polar_root));
    assert!(cache.contains(&live_root));

    // Verify pre-3A `find_carrier` and `remove_where` hooks on real GpuTileCache.
    let live_child = source_tile_key(
        live_source,
        QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 2,
            x: 1,
            y: 1,
        },
    );
    let (found_key, found_tile) = cache.find_carrier(&live_child).expect("ancestor carrier");
    assert_eq!(found_key, live_root);
    assert_eq!(found_tile.width, 64);

    // 5. Removing `live_source` clears only live tiles and preserves bundled fallback.
    assert_eq!(cache.remove_where(|k| k.source == live_source), 1);
    assert!(!cache.contains(&live_root));
    assert!(cache.contains(&south_polar_root));
}
