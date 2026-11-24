use super::*;
use crate::geo::projection::System;

#[test]
fn projected_and_axial_tie_grid_fraction_inverse_round_trip() {
    let utm37 = Projection::from(System::utm(37, true));
    let frame = Frame::projected_bounds(utm37, 500_000.0, 5_200_000.0, 510_000.0, 5_210_000.0)
        .expect("valid projected frame");
    let (lat, lon) = frame.geodetic(0.37, 0.64);
    let (fx, fy) = frame.fraction_at(lat, lon).expect("inside projected frame");
    assert!((fx - 0.37).abs() < 1e-7);
    assert!((fy - 0.64).abs() < 1e-7);
    assert!(frame.fraction_at(0.0, 0.0).is_none());

    let ties = [
        Tie {
            fx: 0.0,
            fy: 0.0,
            lat: 48.0,
            lon: 39.0,
        },
        Tie {
            fx: 1.0,
            fy: 0.0,
            lat: 48.0,
            lon: 41.0,
        },
        Tie {
            fx: 0.0,
            fy: 1.0,
            lat: 46.0,
            lon: 39.0,
        },
        Tie {
            fx: 1.0,
            fy: 1.0,
            lat: 46.0,
            lon: 41.0,
        },
    ];
    let grid = TieGrid::new(&ties).expect("valid axial grid");
    assert!(grid.is_axial());
    let gframe = Frame::from_grid(grid);
    let (glat, glon) = gframe.geodetic(0.25, 0.75);
    let (gfx, gfy) = gframe.fraction_at(glat, glon).expect("inside tie grid");
    assert!((gfx - 0.25).abs() < 1e-9);
    assert!((gfy - 0.75).abs() < 1e-9);
    assert!(gframe.fraction_at(50.0, 40.0).is_none());
}

#[test]
fn non_axial_tie_grid_extrapolates_pixel_is_point_borders_and_unwinds_antimeridian() {
    // 3x3 rotated / skewed swath straddling the antimeridian with inset ties
    // (simulating PixelIsPoint half-pixel inset).
    let mut ties = Vec::new();
    let fxs = [0.05, 0.5, 0.95];
    let fys = [0.05, 0.5, 0.95];
    for &fy in &fys {
        for &fx in &fxs {
            let lat = 10.0 - 4.0 * fy + 0.4 * fx;
            let raw_lon = 178.0 + 4.0 * fx + 0.5 * fy;
            let wrapped_lon = if raw_lon > 180.0 {
                raw_lon - 360.0
            } else {
                raw_lon
            };
            ties.push(Tie {
                fx,
                fy,
                lat,
                lon: wrapped_lon,
            });
        }
    }
    let grid = TieGrid::new(&ties).expect("valid non-axial antimeridian grid");
    assert!(grid.is_dense());
    assert!(!grid.is_axial());
    let frame = Frame::from_grid(grid);

    // Interior and extrapolated border points (fx = 0.01, fy = 0.98) must round-trip.
    for (fx, fy) in [(0.5, 0.5), (0.2, 0.8), (0.01, 0.02), (0.99, 0.98)] {
        let (lat, lon) = frame.geodetic(fx, fy);
        let wrapped = geodesy::unwind(0.0, lon);
        let (inv_x, inv_y) = frame
            .fraction_at(lat, wrapped)
            .unwrap_or_else(|| panic!("expected hit at ({fx}, {fy}) -> ({lat}, {wrapped})"));
        assert!((inv_x - fx).abs() < 1e-5, "fx={fx}, inv_x={inv_x}");
        assert!((inv_y - fy).abs() < 1e-5, "fy={fy}, inv_y={inv_y}");
    }
}

#[test]
fn global_quad_preserves_360_span_and_centers_footprint_away_from_antimeridian() {
    let global = Frame::quad([
        (80.0, -180.0),
        (80.0, 180.0),
        (-80.0, 180.0),
        (-80.0, -180.0),
    ])
    .expect("global 360 quad");
    assert!(global.axial());
    assert!(global.measurable());
    let (lat_mid, lon_mid) = global.geodetic(0.5, 0.5);
    assert!(lat_mid.abs() < 1e-9);
    assert!(lon_mid.abs() < 1e-9);

    let bounds = global.footprint_bounds().expect("global bounds");
    assert!(
        bounds.lon.abs() < 1.0,
        "global footprint should center near lon=0, got {}",
        bounds.lon
    );
    assert!(bounds.radius_deg > 120.0);

    // Round-trip at +90 deg and -90 deg.
    let (fx_e, fy_e) = global.fraction_at(0.0, 90.0).expect("east hemisphere");
    assert!((fx_e - 0.75).abs() < 1e-6 && (fy_e - 0.5).abs() < 1e-6);
    let (fx_w, fy_w) = global.fraction_at(0.0, -90.0).expect("west hemisphere");
    assert!((fx_w - 0.25).abs() < 1e-6 && (fy_w - 0.5).abs() < 1e-6);
}

#[test]
fn web_mercator_antimeridian_fraction_at_and_anisotropic_sharpest_level() {
    // WebMercator raster spanning [+170 deg, +190 deg] easting.
    let (x0, y1) = Projection::WebMercator.from_geodetic(10.0, 170.0);
    let (x1, y0) = Projection::WebMercator.from_geodetic(-10.0, 190.0);
    let wm = Frame::projected_bounds(Projection::WebMercator, x0, y0, x1, y1)
        .expect("antimeridian WebMercator");
    assert!(wm.axial());

    // Probe at -175 deg (equivalent to +185 deg -> fx = 0.75).
    let (fx, fy) = wm.fraction_at(0.0, -175.0).expect("antimeridian hit");
    assert!((fx - 0.75).abs() < 1e-6, "fx={fx}");
    assert!((fy - 0.5).abs() < 1e-6, "fy={fy}");

    // High-latitude geographic raster where vertical m/px is ~5x coarser
    // than horizontal m/px: sharpest_level must respect the finer axis.
    let polar_band =
        Frame::quad([(82.0, 0.0), (82.0, 4.0), (78.0, 4.0), (78.0, 0.0)]).expect("polar band");
    let (mpp_x, mpp_y) = polar_band.pixel_size_m(1000, 1000).expect("pixel size");
    assert!(mpp_y > mpp_x * 4.0);
    // Viewport GSD between mpp_x and mpp_y should pick level 0 based on finer mpp_x.
    assert_eq!(polar_band.sharpest_level(1000, 1000, mpp_x * 1.1, 4), 0);
}

#[test]
fn invalid_epsg_and_degenerate_frames_return_typed_errors() {
    assert_eq!(
        Frame::from_epsg_pixel_affine(999_999, [1.0, 0.0, 0.0, 0.0, -1.0, 0.0], 64, 64),
        Err(BindingError::UnsupportedEpsg(999_999))
    );
    assert_eq!(
        Frame::from_epsg_pixel_affine(
            32637,
            [0.0, 0.0, 500_000.0, 0.0, -10.0, 5_000_000.0],
            64,
            64
        ),
        Err(BindingError::InvalidExtent(32637))
    );
    assert!(
        Frame::quad([(10.0, 0.0), (10.0, 5.0), (10.0, 5.0), (10.0, 0.0)]).is_none(),
        "zero-height quad must be rejected"
    );
    assert!(
        TieGrid::new(&[
            Tie {
                fx: 0.0,
                fy: 0.0,
                lat: 95.0,
                lon: 0.0,
            },
            Tie {
                fx: 1.0,
                fy: 0.0,
                lat: 95.0,
                lon: 10.0,
            },
            Tie {
                fx: 0.0,
                fy: 1.0,
                lat: 85.0,
                lon: 0.0,
            },
            Tie {
                fx: 1.0,
                fy: 1.0,
                lat: 85.0,
                lon: 10.0,
            },
        ])
        .is_none(),
        "out-of-range latitude ties must be rejected"
    );
}

#[test]
fn web_mercator_fraction_at_rejects_polar_caps() {
    let (x0, y1) = Projection::WebMercator.from_geodetic(85.05112878, -180.0);
    let (x1, y0) = Projection::WebMercator.from_geodetic(-85.05112878, 180.0);
    let global_wm =
        Frame::projected_bounds(Projection::WebMercator, x0, y0, x1, y1).expect("global 3857");

    // Interior points stay valid.
    assert!(global_wm.fraction_at(0.0, 0.0).is_some());
    assert!(global_wm.fraction_at(85.0, 10.0).is_some());
    assert!(global_wm.fraction_at(-85.0, -10.0).is_some());

    // Polar caps beyond the Web Mercator extent must never produce phantom hits.
    for lat in [85.1, 88.0, 89.9, 90.0, -85.1, -88.0, -90.0] {
        assert!(
            global_wm.fraction_at(lat, 0.0).is_none(),
            "expected no WebMercator hit at lat={lat}"
        );
    }
}

#[test]
fn pole_touching_pixel_is_point_tie_grid_is_measurable_and_clamps_outer_half_pixel() {
    // 3601x1801 global PixelIsPoint lattice with pixel centers at ±90 deg and ±180 deg.
    let (w, h) = (3601.0_f64, 1801.0_f64);
    let (fx0, fx1) = (0.5 / w, (w - 0.5) / w);
    let (fy0, fy1) = (0.5 / h, (h - 0.5) / h);
    let ties = [
        Tie {
            fx: fx0,
            fy: fy0,
            lat: 90.0,
            lon: -180.0,
        },
        Tie {
            fx: fx1,
            fy: fy0,
            lat: 90.0,
            lon: 180.0,
        },
        Tie {
            fx: fx0,
            fy: fy1,
            lat: -90.0,
            lon: -180.0,
        },
        Tie {
            fx: fx1,
            fy: fy1,
            lat: -90.0,
            lon: 180.0,
        },
    ];
    let grid = TieGrid::new(&ties).expect("pole-touching PixelIsPoint grid");
    assert!(
        grid.widest_spans_m().0 > 39_000_000.0,
        "midpoint row check should measure equatorial width, got {}",
        grid.widest_spans_m().0
    );
    let frame = Frame::from_grid(grid);
    assert!(frame.measurable());

    // Outer extrapolated corners clamp cleanly to ±90 deg instead of folding over the pole.
    assert_eq!(frame.geodetic(0.0, 0.0).0, 90.0);
    assert_eq!(frame.geodetic(1.0, 1.0).0, -90.0);

    // Every pixel center retains exact 0.0 deg placement error.
    let (lat_top, lon_left) = frame.geodetic(0.5 / w, 0.5 / h);
    assert!((lat_top - 90.0).abs() < 1e-12);
    assert!((lon_left - -180.0).abs() < 1e-12);
    let (lat_mid, lon_mid) = frame.geodetic(1800.5 / w, 900.5 / h);
    assert!(lat_mid.abs() < 1e-12);
    assert!(lon_mid.abs() < 1e-12);
}

#[test]
fn foreign_datum_geographic_frame_shifts_to_wgs84_and_inverts_cleanly() {
    use super::super::projection::PULKOVO_1942;

    let ties = [
        Tie {
            fx: 0.0,
            fy: 0.0,
            lat: 47.5,
            lon: 39.5,
        },
        Tie {
            fx: 1.0,
            fy: 0.0,
            lat: 47.5,
            lon: 40.0,
        },
        Tie {
            fx: 0.0,
            fy: 1.0,
            lat: 47.0,
            lon: 39.5,
        },
        Tie {
            fx: 1.0,
            fy: 1.0,
            lat: 47.0,
            lon: 40.0,
        },
    ];
    let grid = TieGrid::new(&ties).expect("pulkovo geographic grid");
    let frame = Frame::from_grid_with_datum(grid, PULKOVO_1942, "EPSG:4284");
    assert!(!frame.axial(), "datum-shifted grid is not WGS84-axial");

    let (wgs_lat, wgs_lon) = frame.geodetic(0.5, 0.5);
    let expected = PULKOVO_1942.to_wgs84(47.25, 39.75);
    assert!((wgs_lat - expected.0).abs() < 1e-12);
    assert!((wgs_lon - expected.1).abs() < 1e-12);

    // Must differ from unshifted WGS84 by ~100-130 m
    let unshifted_deg = (wgs_lat - 47.25).hypot(wgs_lon - 39.75);
    assert!(
        unshifted_deg > 5e-4,
        "expected datum shift, got {unshifted_deg} deg"
    );

    // Inverse lookup maps shifted WGS84 coordinates back to (0.5, 0.5).
    let (fx, fy) = frame
        .fraction_at(wgs_lat, wgs_lon)
        .expect("inverse hit on Pulkovo geographic frame");
    assert!((fx - 0.5).abs() < 1e-6);
    assert!((fy - 0.5).abs() < 1e-6);
}

#[test]
fn crs_pixel_affine_geographic_handles_pole_touching_pixel_is_point_and_360_domain() {
    use super::super::crs::CrsSpec;

    let wgs84 = CrsSpec::from_epsg(4326).expect("EPSG:4326");

    // 3601x1801 PixelIsPoint [-180, 180] x [-90, 90] (0.1 deg spacing):
    // top-left pixel edge sits at (-180.05, +90.05) so pixel (0, 0) center is (-180, +90).
    let affine_pm180 = [0.1, 0.0, -180.05, 0.0, -0.1, 90.05];
    let frame_pm180 = Frame::from_crs_pixel_affine(&wgs84, affine_pm180, 3601, 1801)
        .expect("pole-touching [-180, 180] PixelIsPoint frame");
    assert!(frame_pm180.axial());
    assert!(frame_pm180.measurable());
    let (lat_c0, lon_c0) = frame_pm180.geodetic(0.5 / 3601.0, 0.5 / 1801.0);
    assert!(
        (lat_c0 - 90.0).abs() < 1e-12,
        "top pixel center lat={lat_c0}"
    );
    assert!(
        (lon_c0 - -180.0).abs() < 1e-12,
        "left pixel center lon={lon_c0}"
    );
    // Interior row (row 10 -> lat = +89.0) must not be compressed by pole clamping.
    let (lat_r10, _) = frame_pm180.geodetic(1800.5 / 3601.0, 10.5 / 1801.0);
    assert!((lat_r10 - 89.0).abs() < 1e-12, "row 10 lat={lat_r10}");
    assert_eq!(frame_pm180.geodetic(0.0, 0.0).0, 90.0);
    assert_eq!(frame_pm180.geodetic(1.0, 1.0).0, -90.0);

    // 3601x1801 PixelIsPoint [0, 360] x [-90, 90] (top-left edge lon = -0.05, right edge = 360.05).
    let affine_360 = [0.1, 0.0, -0.05, 0.0, -0.1, 90.05];
    let frame_360 = Frame::from_crs_pixel_affine(&wgs84, affine_360, 3601, 1801)
        .expect("pole-touching [0, 360] PixelIsPoint frame");
    let (lat_r, lon_r) = frame_360.geodetic(3600.5 / 3601.0, 900.5 / 1801.0);
    assert!(lat_r.abs() < 1e-12);
    assert!(
        (lon_r - 360.0).abs() < 1e-12,
        "right pixel center lon={lon_r}"
    );

    // Standard 3600x1800 PixelIsArea [-180, 180] x [-90, 90] retains exact corners.
    let affine_area = [0.1, 0.0, -180.0, 0.0, -0.1, 90.0];
    let frame_area = Frame::from_crs_pixel_affine(&wgs84, affine_area, 3600, 1800)
        .expect("global PixelIsArea frame");
    let (lat_tl, lon_tl) = frame_area.geodetic(0.0, 0.0);
    let (lat_br, lon_br) = frame_area.geodetic(1.0, 1.0);
    assert!((lat_tl - 90.0).abs() < 1e-12 && (lon_tl - -180.0).abs() < 1e-12);
    assert!((lat_br - -90.0).abs() < 1e-12 && (lon_br - 180.0).abs() < 1e-12);
}
