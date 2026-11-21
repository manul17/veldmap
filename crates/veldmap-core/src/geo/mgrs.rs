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

fn band_middle_deg(band: usize) -> f64 {
    let south_edge = -80.0 + band as f64 * BAND_DEG;
    let height = if band == BANDS.len() - 1 {
        LAST_BAND_DEG
    } else {
        BAND_DEG
    };
    south_edge + height * 0.5
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

/// Resolve the southwest UTM corner `(zone, south, easting_m, northing_m)` of
/// an MGRS 100 km grid square from a zone (`1..=60`), band letter, column
/// letter, and row letter.
fn square_origin(
    zone: u32,
    band_char: u8,
    col_char: u8,
    row_char: u8,
) -> Result<(u32, bool, f64, f64), MgrsError> {
    if !(1..=60).contains(&zone) {
        return Err(MgrsError::ZoneOutOfRange(zone));
    }
    let band = BANDS
        .iter()
        .position(|&b| b == band_char)
        .ok_or(MgrsError::InvalidBand(band_char as char))?;

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

    let band_mid_deg = band_middle_deg(band);
    let south = band_mid_deg < 0.0;
    let approx = band_mid_deg.abs() * 110_946.0;
    let target = if south { 10_000_000.0 - approx } else { approx };
    let mut northing = base;
    while northing + ROW_CYCLE_M * 0.5 < target {
        northing += ROW_CYCLE_M;
    }

    Ok((zone, south, easting, northing))
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
    let (zone, south, x0, northing) = square_origin(zone, bytes[2], bytes[3], bytes[4])?;

    let y1 = northing + SQUARE_M;
    Ok(Frame {
        zone,
        south,
        x0,
        y0: y1 - TILE_SIDE_M,
        x1: x0 + TILE_SIDE_M,
        y1,
    })
}

/// Parse an MGRS grid reference (for example `"37UDQ"`, `"T40WFC"`,
/// `"4Q FJ 12345 67890"`, or `"37UDQ5050"`) into WGS84
/// `(lat_deg, lon_deg, suggested_radius_deg)` centered on the addressed cell.
pub fn parse_to_wgs84(input: &str) -> Result<(f64, f64, f64), MgrsError> {
    let compact: String = input
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_ascii_uppercase())
        .collect();
    if compact.is_empty() {
        return Err(MgrsError::InvalidFormat(input.to_owned()));
    }

    // Allow Sentinel-2 tile notation with leading `T` (e.g. `T40WFC`).
    let stripped = compact
        .strip_prefix('T')
        .filter(|rest| rest.bytes().next().is_some_and(|b| b.is_ascii_digit()))
        .unwrap_or(&compact);

    if !stripped.is_ascii() {
        return Err(MgrsError::InvalidFormat(input.to_owned()));
    }

    let bytes = stripped.as_bytes();
    let zone_digits = bytes.iter().take_while(|b| b.is_ascii_digit()).count();
    if !(1..=2).contains(&zone_digits) || bytes.len() < zone_digits + 3 {
        return Err(MgrsError::InvalidFormat(input.to_owned()));
    }
    let zone: u32 = stripped[..zone_digits]
        .parse()
        .map_err(|_| MgrsError::InvalidZoneNumber(input.to_owned()))?;

    let band = bytes[zone_digits];
    let col = bytes[zone_digits + 1];
    let row = bytes[zone_digits + 2];
    let rest = &stripped[zone_digits + 3..];

    let (zone, south, x0, y0) = square_origin(zone, band, col, row)?;

    let (easting, northing, cell_side_m) = if rest.is_empty() {
        (x0 + SQUARE_M * 0.5, y0 + SQUARE_M * 0.5, SQUARE_M)
    } else {
        if !rest.bytes().all(|b| b.is_ascii_digit())
            || !rest.len().is_multiple_of(2)
            || rest.len() > 10
        {
            return Err(MgrsError::InvalidDigits(rest.to_owned()));
        }
        let half = rest.len() / 2;
        let step_m = 10_f64.powi(5 - half as i32);
        let e_val: f64 = rest[..half]
            .parse()
            .map_err(|_| MgrsError::InvalidDigits(rest.to_owned()))?;
        let n_val: f64 = rest[half..]
            .parse()
            .map_err(|_| MgrsError::InvalidDigits(rest.to_owned()))?;
        (
            x0 + (e_val + 0.5) * step_m,
            y0 + (n_val + 0.5) * step_m,
            step_m,
        )
    };

    let system = System::utm(zone, south);
    let (lat_deg, lon_deg) = system.to_geodetic(easting, northing);
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
    }

    #[test]
    fn parse_mgrs_square_and_full_grid_to_wgs84() {
        let (lat, lon, radius) = parse_to_wgs84("T40WFC").unwrap();
        assert!((69.0..70.5).contains(&lat), "lat={lat}");
        assert!((59.0..62.0).contains(&lon), "lon={lon}");
        assert!(radius > 0.5);

        // 37UDQ: UTM zone 37N, band U (48..56 N), square DQ (E 400..500 km, N 5400..5500 km)
        let (lat_m, lon_m, radius_m) = parse_to_wgs84("37U DQ 50000 50000").unwrap();
        assert!((48.5..50.0).contains(&lat_m), "lat={lat_m}");
        assert!((37.5..39.5).contains(&lon_m), "lon={lon_m}");
        assert!(radius_m < 0.05);

        // Single-digit zone (e.g. Hawaii zone 4Q)
        let (lat_h, lon_h, _) = parse_to_wgs84("4Q FJ 1234 5678").unwrap();
        assert!((16.0..24.0).contains(&lat_h), "lat={lat_h}");
        assert!((-162.0..-154.0).contains(&lon_h), "lon={lon_h}");
    }
}
