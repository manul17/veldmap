//! MGRS grid indexing: Sentinel-2 tile codes and general MGRS grid references
//! mapped to UTM frames and WGS84 coordinates.

use thiserror::Error;

use super::geodesy::unwind;
use super::projection::System;

/// Sentinel-2 granule frame in its UTM zone (`y1` is the northern edge).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub zone: u32,
    pub south: bool,
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

/// Errors produced when parsing an MGRS tile code or grid reference.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MgrsError {
    #[error("MGRS code '{0}' has invalid length or format")]
    InvalidFormat(String),
    #[error("MGRS zone in '{0}' is not a valid integer")]
    InvalidZoneNumber(String),
    #[error("MGRS UTM zone {0} is out of range (expected 1..=60)")]
    ZoneOutOfRange(u32),
    #[error("MGRS latitude band '{0}' is invalid")]
    InvalidBand(char),
    #[error("MGRS zone {zone}{band} does not exist")]
    InvalidZoneBand { zone: u32, band: char },
    #[error("MGRS 100 km column '{col}' is invalid in zone {zone}")]
    InvalidColumn { col: char, zone: u32 },
    #[error("MGRS 100 km row '{0}' is invalid")]
    InvalidRow(char),
    #[error("MGRS numerical grid coordinates '{0}' must have an even number of 2..=10 digits")]
    InvalidDigits(String),
}

/// Sentinel-2 granule side in metres (10,980 pixels at 10 m).
const TILE_SIDE_M: f64 = 109_800.0;
/// Side of an MGRS 100 km square in metres.
const SQUARE_M: f64 = 100_000.0;
/// Repetition period of a row letter in metres.
const ROW_CYCLE_M: f64 = 2_000_000.0;
/// Standard latitude band height in degrees.
const BAND_DEG: f64 = 8.0;
/// Height of Arctic band X in degrees (`72..=84`).
const LAST_BAND_DEG: f64 = 12.0;

/// Latitude band letters south-to-north (`-80..=84`, omitting I and O).
const BANDS: &[u8] = b"CDEFGHJKLMNPQRSTUVWX";
/// Row letters of 100 km squares (cycle of 20, omitting I and O).
const ROWS: &[u8] = b"ABCDEFGHJKLMNPQRSTUV";

fn band_bounds_deg(band: usize) -> (f64, f64) {
    let south_edge = -80.0 + band as f64 * BAND_DEG;
    let height = if band == BANDS.len() - 1 {
        LAST_BAND_DEG
    } else {
        BAND_DEG
    };
    (south_edge, south_edge + height)
}

#[cfg(test)]
fn band_middle_deg(band: usize) -> f64 {
    let (south_edge, north_edge) = band_bounds_deg(band);
    (south_edge + north_edge) * 0.5
}

/// Extract a 5-character MGRS tile code (`_TxxABC_`) from a Sentinel-2 product
/// name.
pub fn tile_of(product_name: &str) -> Option<&str> {
    product_name.split('_').find_map(|part| {
        let tile = part.strip_prefix('T')?;
        let bytes = tile.as_bytes();
        let ok = bytes.len() == 5
            && bytes[..2].iter().all(u8::is_ascii_digit)
            && bytes[2..].iter().all(u8::is_ascii_uppercase);
        ok.then_some(tile)
    })
}

struct SquareOrigin {
    zone: u32,
    south: bool,
    easting: f64,
    northing: f64,
    south_edge: f64,
    north_edge: f64,
    west_lon: f64,
    east_lon: f64,
}

/// Longitude bounds `(west_deg, east_deg)` of an MGRS zone-band cell,
/// accounting for the Norway (`31V`/`32V`) and Svalbard (`31X`/`33X`/`35X`/`37X`)
/// exceptions.
fn zone_lon_bounds_deg(zone: u32, band_char: u8) -> (f64, f64) {
    match (zone, band_char) {
        (31, b'V') => (0.0, 3.0),
        (32, b'V') => (3.0, 12.0),
        (31, b'X') => (0.0, 9.0),
        (33, b'X') => (9.0, 21.0),
        (35, b'X') => (21.0, 33.0),
        (37, b'X') => (33.0, 42.0),
        _ => {
            let west = f64::from(zone) * 6.0 - 186.0;
            (west, west + 6.0)
        }
    }
}

/// Resolve the southwest UTM corner and latitude/longitude bounds of an MGRS
/// 100 km grid square from a zone (`1..=60`), band letter, column letter, and
/// row letter.
fn square_origin(
    zone: u32,
    band_char: u8,
    col_char: u8,
    row_char: u8,
) -> Result<SquareOrigin, MgrsError> {
    if !(1..=60).contains(&zone) {
        return Err(MgrsError::ZoneOutOfRange(zone));
    }
    let band = BANDS
        .iter()
        .position(|&b| b == band_char)
        .ok_or(MgrsError::InvalidBand(band_char as char))?;
    if band_char == b'X' && matches!(zone, 32 | 34 | 36) {
        return Err(MgrsError::InvalidZoneBand {
            zone,
            band: band_char as char,
        });
    }

    let column_set: &[u8] = match (zone - 1) % 3 {
        0 => b"ABCDEFGH",
        1 => b"JKLMNPQR",
        _ => b"STUVWXYZ",
    };
    let column =
        column_set
            .iter()
            .position(|&b| b == col_char)
            .ok_or(MgrsError::InvalidColumn {
                col: col_char as char,
                zone,
            })?;
    let easting = (column as f64 + 1.0) * SQUARE_M;

    let row = ROWS
        .iter()
        .position(|&b| b == row_char)
        .ok_or(MgrsError::InvalidRow(row_char as char))?;
    let shift = if zone.is_multiple_of(2) { 5 } else { 0 };
    let base = ((row + ROWS.len() - shift) % ROWS.len()) as f64 * SQUARE_M;

    let (south_edge, north_edge) = band_bounds_deg(band);
    let (west_lon, east_lon) = zone_lon_bounds_deg(zone, band_char);
    let band_mid_deg = (south_edge + north_edge) * 0.5;
    let south = band_mid_deg < 0.0;
    let approx = band_mid_deg.abs() * 110_946.0;
    let target = if south { 10_000_000.0 - approx } else { approx };
    let mut northing = base;
    while northing + ROW_CYCLE_M * 0.5 < target {
        northing += ROW_CYCLE_M;
    }

    // Verify that the 100 km square actually overlaps the stated latitude band
    // and UTM zone longitude span (rejects out-of-band rows like `37MAA` and
    // out-of-zone high-latitude columns like `37XAA`).
    let system = System::utm(zone, south);
    let mut min_lat = f64::INFINITY;
    let mut max_lat = f64::NEG_INFINITY;
    let mut min_lon = f64::INFINITY;
    let mut max_lon = f64::NEG_INFINITY;
    for (x, y) in [
        (easting, northing),
        (easting + SQUARE_M, northing),
        (easting, northing + SQUARE_M),
        (easting + SQUARE_M, northing + SQUARE_M),
    ] {
        let (lat, lon) = system.to_geodetic(x, y);
        min_lat = min_lat.min(lat);
        max_lat = max_lat.max(lat);
        min_lon = min_lon.min(lon);
        max_lon = max_lon.max(lon);
    }
    if max_lat <= south_edge || min_lat >= north_edge {
        return Err(MgrsError::InvalidRow(row_char as char));
    }
    if max_lon <= west_lon || min_lon >= east_lon {
        return Err(MgrsError::InvalidColumn {
            col: col_char as char,
            zone,
        });
    }

    Ok(SquareOrigin {
        zone,
        south,
        easting,
        northing,
        south_edge,
        north_edge,
        west_lon,
        east_lon,
    })
}

/// Compute the Sentinel-2 109.8 km granule frame in UTM metres for a
/// 5-character tile code such as `"40WFC"`.
pub fn frame(tile: &str) -> Result<Frame, MgrsError> {
    let bytes = tile.as_bytes();
    if !tile.is_ascii() || bytes.len() != 5 {
        return Err(MgrsError::InvalidFormat(tile.to_owned()));
    }
    let zone: u32 = tile[..2]
        .parse()
        .map_err(|_| MgrsError::InvalidZoneNumber(tile.to_owned()))?;
    let origin = square_origin(zone, bytes[2], bytes[3], bytes[4])?;

    let y1 = origin.northing + SQUARE_M;
    Ok(Frame {
        zone: origin.zone,
        south: origin.south,
        x0: origin.easting,
        y0: y1 - TILE_SIDE_M,
        x1: origin.easting + TILE_SIDE_M,
        y1,
    })
}

/// Parse an MGRS grid reference (for example `"37UDQ"`, `"T40WFC"`,
/// `"4Q FJ 12345 67890"`, or `"37UDQ5050"`) into WGS84
/// `(lat_deg, lon_deg, suggested_radius_deg)` centered on the addressed cell.
pub fn parse_to_wgs84(input: &str) -> Result<(f64, f64, f64), MgrsError> {
    let trimmed = input.trim();
    if trimmed.is_empty() || !trimmed.is_ascii() {
        return Err(MgrsError::InvalidFormat(input.to_owned()));
    }

    // Track non-whitespace byte positions so we can validate whitespace-separated
    // easting/northing digit halves without silently gluing unequal halves.
    let mut compact = Vec::with_capacity(trimmed.len());
    let mut orig_positions = Vec::with_capacity(trimmed.len());
    for (idx, b) in trimmed.bytes().enumerate() {
        if !b.is_ascii_whitespace() {
            compact.push(b.to_ascii_uppercase());
            orig_positions.push(idx);
        }
    }
    if compact.is_empty() {
        return Err(MgrsError::InvalidFormat(input.to_owned()));
    }

    // Allow Sentinel-2 tile notation with leading `T` (e.g. `T40WFC`).
    let start = usize::from(compact[0] == b'T' && compact.get(1).is_some_and(u8::is_ascii_digit));
    let bytes = &compact[start..];

    let zone_digits = bytes.iter().take_while(|b| b.is_ascii_digit()).count();
    if !(1..=2).contains(&zone_digits) || bytes.len() < zone_digits + 3 {
        return Err(MgrsError::InvalidFormat(input.to_owned()));
    }
    let zone_str = std::str::from_utf8(&bytes[..zone_digits])
        .map_err(|_| MgrsError::InvalidZoneNumber(input.to_owned()))?;
    let zone: u32 = zone_str
        .parse()
        .map_err(|_| MgrsError::InvalidZoneNumber(input.to_owned()))?;

    let band = bytes[zone_digits];
    let col = bytes[zone_digits + 1];
    let row = bytes[zone_digits + 2];
    let digits_start = start + zone_digits + 3;

    let origin = square_origin(zone, band, col, row)?;

    let has_digits = digits_start < compact.len();
    let (easting, northing, cell_side_m) = if !has_digits {
        (
            origin.easting + SQUARE_M * 0.5,
            origin.northing + SQUARE_M * 0.5,
            SQUARE_M,
        )
    } else {
        let raw_digits = &trimmed[orig_positions[digits_start]..];
        let digit_parts: Vec<&str> = raw_digits.split_whitespace().collect();
        let (e_str, n_str) = match digit_parts.as_slice() {
            [combined] => {
                if !combined.bytes().all(|b| b.is_ascii_digit())
                    || !combined.len().is_multiple_of(2)
                    || combined.len() > 10
                {
                    return Err(MgrsError::InvalidDigits(raw_digits.to_owned()));
                }
                combined.split_at(combined.len() / 2)
            }
            [e_part, n_part] => {
                if e_part.len() != n_part.len()
                    || !(1..=5).contains(&e_part.len())
                    || !e_part.bytes().all(|b| b.is_ascii_digit())
                    || !n_part.bytes().all(|b| b.is_ascii_digit())
                {
                    return Err(MgrsError::InvalidDigits(raw_digits.to_owned()));
                }
                (*e_part, *n_part)
            }
            _ => return Err(MgrsError::InvalidDigits(raw_digits.to_owned())),
        };
        let step_m = 10_f64.powi(5 - e_str.len() as i32);
        let e_val: f64 = e_str
            .parse()
            .map_err(|_| MgrsError::InvalidDigits(raw_digits.to_owned()))?;
        let n_val: f64 = n_str
            .parse()
            .map_err(|_| MgrsError::InvalidDigits(raw_digits.to_owned()))?;
        (
            origin.easting + (e_val + 0.5) * step_m,
            origin.northing + (n_val + 0.5) * step_m,
            step_m,
        )
    };

    let system = System::utm(origin.zone, origin.south);
    let (lat_deg, lon_deg) = system.to_geodetic(easting, northing);
    if has_digits {
        if !(origin.south_edge - 1e-6..=origin.north_edge + 1e-6).contains(&lat_deg) {
            return Err(MgrsError::InvalidRow(row as char));
        }
        if !(origin.west_lon - 1e-6..=origin.east_lon + 1e-6).contains(&lon_deg) {
            return Err(MgrsError::InvalidColumn {
                col: col as char,
                zone: origin.zone,
            });
        }
    }
    let lon_deg = unwind(0.0, lon_deg);
    let radius_deg = (cell_side_m / 111_320.0).clamp(0.002, 1.0);
    Ok((lat_deg, lon_deg, radius_deg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_band_fits_the_row_window_from_both_sides() {
        for (band, &letter) in BANDS.iter().enumerate() {
            let height = if band == BANDS.len() - 1 {
                LAST_BAND_DEG
            } else {
                BAND_DEG
            };
            let south = -80.0 + band as f64 * BAND_DEG;
            let target = band_middle_deg(band).abs() * 110_946.0;

            for (edge, side) in [(south, "south"), (south + height, "north")] {
                let margin = ROW_CYCLE_M / 2.0 - (edge.abs() * 110_946.0 - target).abs();
                assert!(
                    margin > 250_000.0,
                    "band {} {} margin {:.0} m",
                    letter as char,
                    side,
                    margin
                );
            }
        }
    }

    #[test]
    fn the_x_band_is_twelve_degrees_wide() {
        assert_eq!(band_middle_deg(BANDS.len() - 1), 78.0);
        assert_eq!(band_middle_deg(0), -76.0);
    }

    #[test]
    fn tile_is_read_from_product_name() {
        assert_eq!(
            tile_of("S2C_MSIL2A_20260812T081601_N0512_R121_T40WFC_20260812T121315.SAFE"),
            Some("40WFC")
        );
        assert_eq!(
            tile_of("S1C_IW_GRDH_1SDV_20260810T031955_011C6E_EDC8.SAFE"),
            None
        );
    }

    #[test]
    fn short_and_odd_segments_are_refused_not_sliced() {
        assert_eq!(tile_of("LC09_L2SP_02_T1"), None);
        assert_eq!(tile_of("A_T_B"), None);
        assert_eq!(tile_of("T"), None);
        assert_eq!(tile_of("T40W\u{00E9}C"), None);
        assert_eq!(tile_of("T40wfc"), None);
    }

    #[test]
    fn reference_granule_frame_matches_its_xml() {
        let frame = frame("40WFC").unwrap();
        assert_eq!((frame.zone, frame.south), (40, false));
        assert_eq!((frame.x0, frame.y1), (600_000.0, 7_800_000.0));
        assert_eq!(
            (frame.x1 - frame.x0, frame.y1 - frame.y0),
            (109_800.0, 109_800.0)
        );
    }

    #[test]
    fn odd_zone_rows_start_at_a() {
        let frame = frame("31NAA").unwrap();
        assert_eq!((frame.x0, frame.y1), (100_000.0, 100_000.0));
        assert!(!frame.south);
    }

    #[test]
    fn southern_band_stays_under_false_northing() {
        let frame = frame("33HVB").unwrap();
        assert!(frame.south);
        assert!(frame.y1 < 10_000_000.0 && frame.y0 > 5_000_000.0);
    }

    #[test]
    fn garbage_codes_are_refused() {
        assert!(frame("40WIC").is_err());
        assert!(frame("00WFC").is_err());
        assert!(frame("4WFC").is_err());
        assert!(frame("40ZFC").is_err());
        // Non-ASCII 5-byte and multi-byte UTF-8 inputs must error, never panic on char boundaries.
        assert!(frame("a\u{00E9}bc").is_err());
        assert!(parse_to_wgs84("37UD\u{0419}").is_err());
        assert!(parse_to_wgs84("37UD\u{00E9}").is_err());
        // Out-of-band row letter wrapping into the wrong hemisphere, unequal digit halves,
        // nonexistent Svalbard zones (32X/34X/36X), and out-of-zone high-latitude columns.
        assert!(parse_to_wgs84("37MAA").is_err());
        assert!(parse_to_wgs84("37UDQ 12 3456").is_err());
        assert!(matches!(
            parse_to_wgs84("32XAA"),
            Err(MgrsError::InvalidZoneBand {
                zone: 32,
                band: 'X'
            })
        ));
        assert!(parse_to_wgs84("34XAA").is_err());
        assert!(parse_to_wgs84("36XAA").is_err());
        assert!(matches!(
            parse_to_wgs84("37XAA"),
            Err(MgrsError::InvalidColumn { col: 'A', zone: 37 })
        ));
    }

    #[test]
    fn parse_mgrs_square_and_full_grid_to_wgs84() {
        // 31N AA 66021 00000 is the exact equator / prime meridian (0 N, 0 E) to ~1 m.
        let (lat0, lon0, radius0) = parse_to_wgs84("31N AA 66021 00000").unwrap();
        assert!(lat0.abs() < 1e-4, "lat0={lat0}");
        assert!(lon0.abs() < 1e-4, "lon0={lon0}");
        assert!(radius0 <= 0.003);

        // Sentinel-2 reference granule T40WFC center (650 km E, 7750 km N in UTM 40N).
        let (lat, lon, radius) = parse_to_wgs84("T40WFC").unwrap();
        assert!((lat - 69.8117).abs() < 0.01, "lat={lat}");
        assert!((lon - 60.9036).abs() < 0.01, "lon={lon}");
        assert!(radius > 0.5);

        // 37U DQ 50000 50000: UTM 37N E=450,000.5 m, N=5,450,000.5 m.
        let (lat_m, lon_m, radius_m) = parse_to_wgs84("37U DQ 50000 50000").unwrap();
        assert!((lat_m - 49.2015).abs() < 0.001, "lat={lat_m}");
        assert!((lon_m - 38.3132).abs() < 0.001, "lon={lon_m}");
        assert!(radius_m < 0.01);

        // Single-digit zone (Hawaii 4Q FJ 1234 5678: UTM 4N E=612,345 m, N=2,356,785 m).
        let (lat_h, lon_h, _) = parse_to_wgs84("4Q FJ 1234 5678").unwrap();
        assert!((lat_h - 21.309_478).abs() < 1e-4, "lat={lat_h}");
        assert!((lon_h + 157.916_819).abs() < 1e-4, "lon={lon_h}");

        // Norway widened 32V (Bergen ~60.391 N, 5.322 E -> 32V KN 973 010).
        let (lat_no, lon_no, _) = parse_to_wgs84("32V KN 973 010").unwrap();
        assert!((lat_no - 60.395).abs() < 0.01, "lat_no={lat_no}");
        assert!((lon_no - 5.322).abs() < 0.01, "lon_no={lon_no}");
    }
}
