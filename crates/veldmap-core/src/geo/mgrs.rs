//! MGRS grid reference parsing and formatting for WGS84 coordinates.

use super::crs::System;
use super::geodesy::unwind;

/// Errors produced when parsing an MGRS grid reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MgrsError {
    InvalidFormat,
    InvalidZoneNumber,
    ZoneOutOfRange(u32),
    InvalidBand(char),
    InvalidZoneBand { zone: u32, band: char },
    InvalidColumn { col: char, zone: u32 },
    InvalidRow(char),
    InvalidDigits,
}

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

/// Column-letter alphabet for a UTM zone (`1..=60`, cycling every 3 zones).
fn column_set(zone: u32) -> &'static [u8] {
    match (zone - 1) % 3 {
        0 => b"ABCDEFGH",
        1 => b"JKLMNPQR",
        _ => b"STUVWXYZ",
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

    let column =
        column_set(zone)
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
    let (min_lat, max_lat, min_lon, max_lon) =
        square_geo_bounds(&system, easting, northing, SQUARE_M);
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
/// corners of a UTM square of width `side_m` rooted at southwest corner
/// `(easting, northing)`.
fn square_geo_bounds(
    system: &System,
    easting: f64,
    northing: f64,
    side_m: f64,
) -> (f64, f64, f64, f64) {
    let mut min_lat = f64::INFINITY;
    let mut max_lat = f64::NEG_INFINITY;
    let mut min_lon = f64::INFINITY;
    let mut max_lon = f64::NEG_INFINITY;
    for (x, y) in [
        (easting, northing),
        (easting + side_m, northing),
        (easting, northing + side_m),
        (easting + side_m, northing + side_m),
    ] {
        let (lat, lon) = system.to_geodetic(x, y);
        min_lat = min_lat.min(lat);
        max_lat = max_lat.max(lat);
        min_lon = min_lon.min(lon);
        max_lon = max_lon.max(lon);
    }
    (min_lat, max_lat, min_lon, max_lon)
}

/// Parse an MGRS grid reference (for example `"37UDQ"`, `"T40WFC"`,
/// `"4Q FJ 12345 67890"`, or `"37UDQ5050"`) into WGS84
/// `(lat_deg, lon_deg, suggested_radius_deg)` centered on the addressed cell.
pub(super) fn parse_to_wgs84(input: &str) -> Result<(f64, f64, f64), MgrsError> {
    let trimmed = input.trim();
    if trimmed.is_empty() || !trimmed.is_ascii() {
        return Err(MgrsError::InvalidFormat);
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
        return Err(MgrsError::InvalidFormat);
    }

    // Allow Sentinel-2 tile notation with leading `T` (e.g. `T40WFC`).
    let is_sentinel_tile = compact[0] == b'T' && compact.get(1).is_some_and(u8::is_ascii_digit);
    let start = usize::from(is_sentinel_tile);
    let bytes = &compact[start..];

    let zone_digits = bytes.iter().take_while(|b| b.is_ascii_digit()).count();
    if !(1..=2).contains(&zone_digits) || bytes.len() < zone_digits + 3 {
        return Err(MgrsError::InvalidFormat);
    }
    let zone_str =
        std::str::from_utf8(&bytes[..zone_digits]).map_err(|_| MgrsError::InvalidZoneNumber)?;
    let zone: u32 = zone_str.parse().map_err(|_| MgrsError::InvalidZoneNumber)?;

    let band = bytes[zone_digits];
    let col = bytes[zone_digits + 1];
    let row = bytes[zone_digits + 2];
    let digits_start = start + zone_digits + 3;

    let origin = square_origin(zone, band, col, row)?;

    let has_digits = digits_start < compact.len();
    let (sw_e, sw_n, cell_side_m) = if !has_digits {
        (origin.easting, origin.northing, SQUARE_M)
    } else {
        let (de, dn, step_m) = parse_grid_digits(&trimmed[orig_positions[digits_start]..])?;
        (origin.easting + de, origin.northing + dn, step_m)
    };
    let easting = sw_e + cell_side_m * 0.5;
    let northing = sw_n + cell_side_m * 0.5;

    let system = System::utm(origin.zone, origin.south);
    let (mut lat_deg, mut lon_deg) = system.to_geodetic(easting, northing);
    if has_digits {
        // Verify that the 4-corner geodetic bounding box of the addressed
        // sub-cell overlaps the stated zone-band cell. Using the exact UTM
        // corners handles grid convergence cleanly across all precisions and
        // high-latitude Norway/Svalbard zones without rejecting valid boundary
        // cells.
        const GEO_EPS_DEG: f64 = 1e-5;
        let (min_lat, max_lat, min_lon, max_lon) =
            square_geo_bounds(&system, sw_e, sw_n, cell_side_m);
        if max_lat < origin.south_edge - GEO_EPS_DEG || min_lat > origin.north_edge + GEO_EPS_DEG {
            return Err(MgrsError::InvalidRow(row as char));
        }
        if max_lon < origin.west_lon - GEO_EPS_DEG || min_lon > origin.east_lon + GEO_EPS_DEG {
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

/// Parse MGRS easting/northing numeric digits into southwest-corner offsets
/// `(sw_delta_easting_m, sw_delta_northing_m, cell_side_m)`.
fn parse_grid_digits(raw_digits: &str) -> Result<(f64, f64, f64), MgrsError> {
    let digit_parts: Vec<&str> = raw_digits.split_whitespace().collect();
    let (e_str, n_str) = match digit_parts.as_slice() {
        [combined] => {
            if !combined.bytes().all(|b| b.is_ascii_digit())
                || !combined.len().is_multiple_of(2)
                || combined.len() > 10
            {
                return Err(MgrsError::InvalidDigits);
            }
            combined.split_at(combined.len() / 2)
        }
        [e_part, n_part] => {
            if e_part.len() != n_part.len()
                || !(1..=5).contains(&e_part.len())
                || !e_part.bytes().all(|b| b.is_ascii_digit())
                || !n_part.bytes().all(|b| b.is_ascii_digit())
            {
                return Err(MgrsError::InvalidDigits);
            }
            (*e_part, *n_part)
        }
        _ => return Err(MgrsError::InvalidDigits),
    };
    let step_m = 10_f64.powi(5 - e_str.len() as i32);
    let e_val: f64 = e_str.parse().map_err(|_| MgrsError::InvalidDigits)?;
    let n_val: f64 = n_str.parse().map_err(|_| MgrsError::InvalidDigits)?;
    Ok((e_val * step_m, n_val * step_m, step_m))
}

/// Format a WGS84 geodetic coordinate `(lat_deg, lon_deg)` as an MGRS grid
/// reference string with `precision` digits per axis (`0..=5`, corresponding to
/// 100 km, 10 km, 1 km, 100 m, 10 m, and 1 m resolution; values above `5` are
/// clamped to `5`).
///
/// Covers UTM latitude bands `C`–`X` (`-80°..=84°`) including the Norway
/// (`31V`/`32V`) and Svalbard (`31X`/`33X`/`35X`/`37X`) grid exceptions.
/// Returns `None` for non-finite coordinates or UPS polar latitudes
/// (`lat_deg < -80.0` or `lat_deg > 84.0`).
pub fn format_wgs84_to_mgrs(lat_deg: f64, lon_deg: f64, precision: u8) -> Option<String> {
    if !lat_deg.is_finite() || !lon_deg.is_finite() || !(-80.0..=84.0).contains(&lat_deg) {
        return None;
    }

    let mut lon = unwind(0.0, lon_deg);
    if lon >= 180.0 {
        lon = -180.0;
    }

    let mut band_idx = if lat_deg >= 72.0 {
        BANDS.len() - 1
    } else {
        (((lat_deg + 80.0) / BAND_DEG).floor() as usize).min(BANDS.len() - 2)
    };
    // Sub-ULP negative latitudes (e.g. `-1e-16`) satisfy `lat + 80.0 == 80.0`
    // in IEEE-754 f64; keep them strictly in southern band `M` (index 9) so
    // `band_char` and `south` never disagree.
    if lat_deg < 0.0 && band_idx > 9 {
        band_idx = 9;
    }
    let band_char = BANDS[band_idx];

    let mut zone = (((lon + 180.0) / 6.0).floor() as i32 + 1).clamp(1, 60) as u32;
    if band_char == b'V' && (0.0..12.0).contains(&lon) {
        zone = if lon < 3.0 { 31 } else { 32 };
    } else if band_char == b'X' && (0.0..42.0).contains(&lon) {
        zone = if lon < 9.0 {
            31
        } else if lon < 21.0 {
            33
        } else if lon < 33.0 {
            35
        } else {
            37
        };
    }

    let south = band_idx < 10;
    let system = System::utm(zone, south);
    let (easting, northing) = system.from_geodetic(lat_deg, lon);
    if !easting.is_finite() || !northing.is_finite() {
        return None;
    }

    let easting = easting.max(SQUARE_M);
    // Tiny negative latitudes can round UTM false northing to exact
    // `10_000_000.0`, which would wrap into row `A`/`F` (`n_km = 100`) instead
    // of southern band `M`'s top row `V`/`E` (`n_km = 99`).
    let northing = if south && northing >= 10_000_000.0 {
        10_000_000.0 - 1e-3
    } else {
        northing.max(0.0)
    };

    let e_km = (easting / SQUARE_M).floor() as u64;
    let col_idx = e_km.saturating_sub(1).min(7) as usize;
    let col_char = column_set(zone)[col_idx] as char;

    let n_km = (northing / SQUARE_M).floor() as u64;
    let shift: u64 = if zone.is_multiple_of(2) { 5 } else { 0 };
    let row_idx = ((n_km + shift) % (ROWS.len() as u64)) as usize;
    let row_char = ROWS[row_idx] as char;

    let prec = usize::from(precision.min(5));
    if prec == 0 {
        return Some(format!(
            "{zone:02}{} {col_char}{row_char}",
            band_char as char
        ));
    }

    let rem_e = (easting - (e_km as f64) * SQUARE_M).clamp(0.0, SQUARE_M - 1e-6);
    let rem_n = (northing - (n_km as f64) * SQUARE_M).clamp(0.0, SQUARE_M - 1e-6);
    let step_m = 10_f64.powi(5 - prec as i32);
    let max_val = 10_u32.pow(prec as u32) - 1;
    let e_digits = ((rem_e / step_m).floor() as u32).min(max_val);
    let n_digits = ((rem_n / step_m).floor() as u32).min(max_val);

    Some(format!(
        "{zone:02}{} {col_char}{row_char} {e_digits:0prec$} {n_digits:0prec$}",
        band_char as char
    ))
}

#[cfg(test)]
#[path = "mgrs_tests.rs"]
mod tests;
