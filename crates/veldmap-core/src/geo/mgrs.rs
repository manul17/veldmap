//! MGRS grid indexing: Sentinel-2 tile codes and general MGRS grid references
//! mapped to UTM frames and WGS84 coordinates.

use thiserror::Error;

use super::geodesy::unwind;
use super::projection::System;

/// Sentinel-2 granule frame in its UTM zone (`y1` is the northern edge).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MgrsGranuleFrame {
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
    min_lat: f64,
    max_lat: f64,
    min_lon: f64,
    max_lon: f64,
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
    let (min_lat, max_lat, min_lon, max_lon) = square_geo_bounds(&system, easting, northing);
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
        min_lat,
        max_lat,
        min_lon,
        max_lon,
    })
}

/// Geodetic bounding box `(min_lat, max_lat, min_lon, max_lon)` of the 4
/// corners of a 100 km MGRS square rooted at `(easting, northing)`.
fn square_geo_bounds(system: &System, easting: f64, northing: f64) -> (f64, f64, f64, f64) {
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
    (min_lat, max_lat, min_lon, max_lon)
}

/// Compute the Sentinel-2 109.8 km granule frame in UTM metres for a
/// 5-character tile code such as `"40WFC"`.
pub fn frame(tile: &str) -> Result<MgrsGranuleFrame, MgrsError> {
    let bytes = tile.as_bytes();
    if !tile.is_ascii() || bytes.len() != 5 {
        return Err(MgrsError::InvalidFormat(tile.to_owned()));
    }
    let zone: u32 = tile[..2]
        .parse()
        .map_err(|_| MgrsError::InvalidZoneNumber(tile.to_owned()))?;
    let origin = square_origin(zone, bytes[2], bytes[3], bytes[4])?;

    let y1 = origin.northing + SQUARE_M;
    Ok(MgrsGranuleFrame {
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
    let is_sentinel_tile = compact[0] == b'T' && compact.get(1).is_some_and(u8::is_ascii_digit);
    let start = usize::from(is_sentinel_tile);
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
        let (de, dn, step_m) = parse_grid_digits(&trimmed[orig_positions[digits_start]..])?;
        (origin.easting + de, origin.northing + dn, step_m)
    };

    let system = System::utm(origin.zone, origin.south);
    let (mut lat_deg, mut lon_deg) = system.to_geodetic(easting, northing);
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
    } else if !is_sentinel_tile
        && (!(origin.south_edge..=origin.north_edge).contains(&lat_deg)
            || !(origin.west_lon..=origin.east_lon).contains(&lon_deg))
    {
        // Standard MGRS 5-character squares clipped by a UTM zone or band
        // boundary (for example `31N AA`): center on the portion inside the
        // requested zone-band cell. Sentinel-2 granules (`T40WFC`) keep their
        // unclipped 100 km UTM square center.
        let clipped_south = origin.min_lat.max(origin.south_edge);
        let clipped_north = origin.max_lat.min(origin.north_edge);
        let clipped_west = origin.min_lon.max(origin.west_lon);
        let clipped_east = origin.max_lon.min(origin.east_lon);
        lat_deg = (clipped_south + clipped_north) * 0.5;
        lon_deg = (clipped_west + clipped_east) * 0.5;
    }
    let lon_deg = unwind(0.0, lon_deg);
    let radius_deg = (cell_side_m / 111_320.0).clamp(0.002, 1.0);
    Ok((lat_deg, lon_deg, radius_deg))
}

/// Parse MGRS easting/northing numeric digits into `(delta_easting_m,
/// delta_northing_m, cell_side_m)` centered on the addressed sub-cell.
fn parse_grid_digits(raw_digits: &str) -> Result<(f64, f64, f64), MgrsError> {
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
    Ok(((e_val + 0.5) * step_m, (n_val + 0.5) * step_m, step_m))
}

#[cfg(test)]
#[path = "mgrs_tests.rs"]
mod tests;
