//! Headless GPU smoke tests for Pass-2 depth-less draped GeoJSON vector
//! rendering: polygons with interior holes, zoom-invariant instanced point
//! markers, polyline layer opacity blending, and exact sRGB-to-linear color
//! conversion into the `Rgba8UnormSrgb` offscreen target.

use std::sync::Arc;

use veldmap_app::render::{DrawVectorId, DrawVectorLayer};
use veldmap_core::geo::{Camera, DEFAULT_VECTOR_MAX_CHORD_DEG, VectorMesh, parse_geojson};
use veldmap_core::tile::QuadtreeConfig;

use crate::common::{
    TestGlobe, build_empty_snapshot, pixel_at, render_srgb, shared_headless_device,
};

#[test]
fn draped_geojson_polygon_with_hole_renders_fill_and_excludes_hole_in_pass2() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let raw = r#"{
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "Donut"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[38.0, 46.0], [42.0, 46.0], [42.0, 50.0], [38.0, 50.0], [38.0, 46.0]],
                    [[39.2, 47.2], [39.2, 48.8], [40.8, 48.8], [40.8, 47.2], [39.2, 47.2]]
                ]
            }
        }]
    }"#;
    let fc = parse_geojson(raw).expect("parse donut geojson");
    let mesh = Arc::new(VectorMesh::from_collection(
        &fc,
        DEFAULT_VECTOR_MAX_CHORD_DEG,
    ));

    let (width, height) = (128u32, 128u32);
    let mut renderer = TestGlobe::new_srgb(device);
    let cam = Camera::at(48.0, 40.0, 450_000.0);
    let config = QuadtreeConfig::default();

    let baseline = render_srgb(
        &mut renderer,
        device,
        queue,
        width,
        height,
        &Arc::new(build_empty_snapshot(cam, width, height, &config)),
    );

    let mut snap = build_empty_snapshot(cam, width, height, &config);
    snap.vectors.push(DrawVectorLayer {
        id: DrawVectorId::Layer(1),
        rev: 1,
        opacity: 1.0,
        fill_color: [1.0, 0.0, 0.0, 0.9],
        line_color: [1.0, 1.0, 0.0, 1.0],
        stroke_width_px: 1.5,
        hatch_spacing_px: 0.0,
        mesh,
    });
    let with_vec = render_srgb(&mut renderer, device, queue, width, height, &Arc::new(snap));

    // Center pixel lies inside the hole: must match the baseline ellipsoid (not painted red).
    let hole_px = pixel_at(&with_vec, width, width / 2, height / 2);
    let base_hole = pixel_at(&baseline, width, width / 2, height / 2);
    assert_eq!(hole_px, base_hole, "interior hole must stay unfilled");

    // Ring pixel midway between hole edge (48.8 deg) and exterior edge (50.0 deg) must be red
    // (linear-light 0.9 alpha blend over gray ellipsoid gives sRGB ~[255, 86, 83]).
    let ring_px = pixel_at(&with_vec, width, width / 2, height / 2 - 36);
    assert!(
        ring_px[0] > 230
            && ring_px[0] > ring_px[1].saturating_add(120)
            && ring_px[0] > ring_px[2].saturating_add(120),
        "polygon exterior ring must render bright red fill, got {ring_px:?}"
    );
}

#[test]
fn draped_geojson_point_only_layer_renders_visible_markers_in_pass2() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let raw = r#"{
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "Site A"},
            "geometry": { "type": "Point", "coordinates": [40.0, 48.0] }
        }]
    }"#;
    let fc = parse_geojson(raw).expect("parse point geojson");
    let mesh = Arc::new(VectorMesh::from_collection(
        &fc,
        DEFAULT_VECTOR_MAX_CHORD_DEG,
    ));
    assert_eq!(mesh.point_count(), 1);
    assert_eq!(mesh.fill_triangle_count(), 0);
    assert_eq!(mesh.line_segment_count(), 0);

    let (width, height) = (128u32, 128u32);
    let mut renderer = TestGlobe::new_srgb(device);
    let cam = Camera::at(48.0, 40.0, 200_000.0);
    let config = QuadtreeConfig::default();

    let mut snap = build_empty_snapshot(cam, width, height, &config);
    snap.vectors.push(DrawVectorLayer {
        id: DrawVectorId::Layer(7),
        rev: 1,
        opacity: 1.0,
        fill_color: [0.0, 0.0, 1.0, 0.3],
        line_color: [0.0, 1.0, 0.0, 1.0],
        stroke_width_px: 1.5,
        hatch_spacing_px: 0.0,
        mesh: Arc::clone(&mesh),
    });
    let pixels = render_srgb(&mut renderer, device, queue, width, height, &Arc::new(snap));

    let center_px = pixel_at(&pixels, width, width / 2, height / 2);
    assert!(
        center_px[1] > 235 && center_px[0] < 20 && center_px[2] < 20,
        "GeoJSON Point marker at screen center must render bright green diamond, got {center_px:?}"
    );

    // Zooming in by 20x must keep the instanced point marker visible at the same
    // screen-space scale without rebuilding the CPU VectorMesh.
    let close_cam = Camera::at(48.0, 40.0, 10_000.0);
    let mut close_snap = build_empty_snapshot(close_cam, width, height, &config);
    close_snap.vectors.push(DrawVectorLayer {
        id: DrawVectorId::Layer(7),
        rev: 1,
        opacity: 1.0,
        fill_color: [0.0, 0.0, 1.0, 0.3],
        line_color: [0.0, 1.0, 0.0, 1.0],
        stroke_width_px: 1.5,
        hatch_spacing_px: 0.0,
        mesh,
    });
    let close_pixels = render_srgb(
        &mut renderer,
        device,
        queue,
        width,
        height,
        &Arc::new(close_snap),
    );
    let close_center = pixel_at(&close_pixels, width, width / 2, height / 2);
    assert!(
        close_center[1] > 235 && close_center[0] < 20 && close_center[2] < 20,
        "Zoomed-in instanced Point marker must remain crisp green at screen center, got {close_center:?}"
    );
}

#[test]
fn draped_geojson_linestring_respects_layer_opacity_in_pass2() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let raw = r#"{
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "Track"},
            "geometry": { "type": "LineString", "coordinates": [[38.0, 48.0], [42.0, 48.0]] }
        }]
    }"#;
    let fc = parse_geojson(raw).expect("parse linestring geojson");
    let mesh = Arc::new(VectorMesh::from_collection(
        &fc,
        DEFAULT_VECTOR_MAX_CHORD_DEG,
    ));
    assert_eq!(mesh.fill_triangle_count(), 0);
    assert!(mesh.line_segment_count() >= 4);
    assert_eq!(mesh.point_count(), 0);

    let (width, height) = (128u32, 128u32);
    let mut renderer = TestGlobe::new_srgb(device);
    let cam = Camera::at(48.0, 40.0, 250_000.0);
    let config = QuadtreeConfig::default();

    let baseline = render_srgb(
        &mut renderer,
        device,
        queue,
        width,
        height,
        &Arc::new(build_empty_snapshot(cam, width, height, &config)),
    );

    let mut full_snap = build_empty_snapshot(cam, width, height, &config);
    full_snap.vectors.push(DrawVectorLayer {
        id: DrawVectorId::Layer(9),
        rev: 1,
        opacity: 1.0,
        fill_color: [0.0, 0.0, 0.0, 0.0],
        line_color: [0.0, 0.2, 1.0, 1.0],
        stroke_width_px: 2.0,
        hatch_spacing_px: 0.0,
        mesh: Arc::clone(&mesh),
    });
    let full_px = render_srgb(
        &mut renderer,
        device,
        queue,
        width,
        height,
        &Arc::new(full_snap),
    );

    let mut half_snap = build_empty_snapshot(cam, width, height, &config);
    half_snap.vectors.push(DrawVectorLayer {
        id: DrawVectorId::Layer(9),
        rev: 1,
        opacity: 0.35,
        fill_color: [0.0, 0.0, 0.0, 0.0],
        line_color: [0.0, 0.2, 1.0, 1.0],
        stroke_width_px: 2.0,
        hatch_spacing_px: 0.0,
        mesh,
    });
    let half_px = render_srgb(
        &mut renderer,
        device,
        queue,
        width,
        height,
        &Arc::new(half_snap),
    );

    // Search a small vertical window around center in case rasterization lands on row 63 or 64.
    // The default ellipsoid is bright off-white (~[251, 248, 240]), so a blue line [0, 0.2, 1]
    // drops the red channel sharply.
    let best_row = (height / 2 - 2..=height / 2 + 2)
        .min_by_key(|&y| pixel_at(&full_px, width, width / 2, y)[0])
        .expect("non-empty row window");
    let base_c = pixel_at(&baseline, width, width / 2, best_row);
    let full_c = pixel_at(&full_px, width, width / 2, best_row);
    let half_c = pixel_at(&half_px, width, width / 2, best_row);
    assert!(
        base_c[0] > full_c[0].saturating_add(180) && full_c[2] > 230,
        "full-opacity blue line must stand out clearly over baseline: base={base_c:?}, full={full_c:?}"
    );
    assert!(
        half_c[0] < base_c[0].saturating_sub(20) && half_c[0] > full_c[0].saturating_add(20),
        "partial-opacity line must blend between baseline and full opacity: base={base_c:?}, half={half_c:?}, full={full_c:?}"
    );
}

#[test]
fn draped_geojson_midtone_srgb_color_reads_back_without_double_gamma_washout() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let raw = r#"{
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "Midtone"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [38.0, 46.0], [42.0, 46.0], [42.0, 50.0], [38.0, 50.0], [38.0, 46.0]
                ]]
            }
        }]
    }"#;
    let fc = parse_geojson(raw).expect("parse midtone geojson");
    let mesh = Arc::new(VectorMesh::from_collection(
        &fc,
        DEFAULT_VECTOR_MAX_CHORD_DEG,
    ));

    let (width, height) = (128u32, 128u32);
    let mut renderer = TestGlobe::new_srgb(device);
    let cam = Camera::at(48.0, 40.0, 350_000.0);
    let config = QuadtreeConfig::default();

    let mut snap = build_empty_snapshot(cam, width, height, &config);
    snap.vectors.push(DrawVectorLayer {
        id: DrawVectorId::Layer(42),
        rev: 1,
        opacity: 1.0,
        fill_color: [0.28, 0.78, 1.0, 1.0],
        line_color: [0.0, 0.0, 0.0, 0.0],
        stroke_width_px: 1.5,
        hatch_spacing_px: 0.0,
        mesh,
    });

    let pixels = render_srgb(&mut renderer, device, queue, width, height, &Arc::new(snap));
    let center = pixel_at(&pixels, width, width / 2, height / 2);
    assert!(
        (70..=73).contains(&center[0])
            && (198..=200).contains(&center[1])
            && center[2] == 255
            && center[3] == 255,
        "mid-tone sRGB vector fill [0.28, 0.78, 1.0, 1.0] must read back [70..=73, 198..=200, 255, 255], got {center:?}"
    );
}

#[test]
fn draped_stac_highlight_renders_hidpi_ribbon_outline_and_diagonal_hatch_fill() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let raw = r#"{
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "STAC Scene"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [38.0, 46.0], [42.0, 46.0], [42.0, 50.0], [38.0, 50.0], [38.0, 46.0]
                ]]
            }
        }]
    }"#;
    let fc = parse_geojson(raw).expect("parse stac footprint");
    let mesh = Arc::new(VectorMesh::from_collection(
        &fc,
        DEFAULT_VECTOR_MAX_CHORD_DEG,
    ));

    let (width, height) = (128u32, 128u32);
    let mut renderer = TestGlobe::new_srgb(device);
    let cam = Camera::at(48.0, 40.0, 450_000.0);
    let config = QuadtreeConfig::default();

    let mut snap_2x = build_empty_snapshot(cam, width, height, &config);
    snap_2x.pixels_per_point = 2.0;
    let vector_layer = DrawVectorLayer {
        id: DrawVectorId::StacHighlight,
        rev: 1,
        opacity: 1.0,
        fill_color: [1.0, 0.0, 0.0, 0.85],
        line_color: [0.0, 0.0, 1.0, 1.0],
        stroke_width_px: 3.0,
        hatch_spacing_px: 8.0,
        mesh,
    };
    snap_2x.vectors.push(vector_layer.clone());

    let pixels_2x = render_srgb(
        &mut renderer,
        device,
        queue,
        width,
        height,
        &Arc::new(snap_2x),
    );
    // Diagonal hatch inside the polygon interior should produce alternating
    // stripe/gap green-channel intensities across a 16px horizontal span.
    let mut min_g = u8::MAX;
    let mut max_g = 0u8;
    for x in (width / 2 - 8)..=(width / 2 + 8) {
        let g = pixel_at(&pixels_2x, width, x, height / 2)[1];
        min_g = min_g.min(g);
        max_g = max_g.max(g);
    }
    assert!(
        max_g > min_g.saturating_add(40),
        "hatched STAC highlight fill should modulate interior intensity across stripes (min_g={min_g}, max_g={max_g})"
    );

    // Count dominant-blue ribbon outline pixels at ppp=2.0 vs ppp=1.0 to verify
    // both outline rendering and HiDPI width scaling.
    let count_blue = |buf: &[u8]| -> usize {
        buf.chunks_exact(4)
            .filter(|px| px[2] > 180 && px[0] < 80)
            .count()
    };
    let blue_2x = count_blue(&pixels_2x);
    assert!(
        blue_2x > 100,
        "expected visible blue ribbon outline at ppp=2.0, got {blue_2x} pixels"
    );

    let mut snap_1x = build_empty_snapshot(cam, width, height, &config);
    snap_1x.pixels_per_point = 1.0;
    snap_1x.vectors.push(vector_layer);
    let pixels_1x = render_srgb(
        &mut renderer,
        device,
        queue,
        width,
        height,
        &Arc::new(snap_1x),
    );
    let blue_1x = count_blue(&pixels_1x);
    assert!(
        blue_1x > 40 && blue_2x > blue_1x,
        "HiDPI ppp=2.0 ribbon outline must cover more physical pixels than ppp=1.0 (1x={blue_1x}, 2x={blue_2x})"
    );
}
