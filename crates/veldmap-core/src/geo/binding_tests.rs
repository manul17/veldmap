use super::*;
use crate::geo::projection::System;

#[test]
fn projected_and_axial_tie_grid_fraction_inverse_round_trip() {
    let utm37 = Projection::Transverse(System::utm(37, true));
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
}
