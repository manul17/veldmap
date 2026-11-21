//! Human-readable coordinate parser for Go-To navigation:
//! Decimal Degrees (DD), Degrees-Minutes(-Seconds) (DM/DMS), and MGRS.

use thiserror::Error;

use super::mgrs;

/// Format recognized by [`parse_location`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocationKind {
    DecimalDegrees,
    DegreesMinutesSeconds,
    Mgrs,
}

/// Parsed geographic destination with a suggested framing angular radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParsedLocation {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub radius_deg: f64,
    pub kind: LocationKind,
}

/// Error returned when a Go-To input string cannot be parsed.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseLocationError {
    #[error("enter coordinates (DD/DMS) or an MGRS grid code")]
    Empty,
    #[error("latitude {0} is out of range (-90..=90)")]
    LatitudeOutOfRange(String),
    #[error("longitude {0} is out of range (-180..=180)")]
    LongitudeOutOfRange(String),
    #[error("could not parse '{0}' as DD, DMS, or MGRS")]
    Unrecognized(String),
}

/// Parse a Go-To location string expressed in DD, DM/DMS, or MGRS.
pub fn parse_location(input: &str) -> Result<ParsedLocation, ParseLocationError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(ParseLocationError::Empty);
    }

    // 1. Try MGRS first when the string contains no lat/lon separators and starts
    // with digits or `T` + digits (e.g. `37UDQ`, `T40WFC`, `4Q FJ 12345 67890`).
    if let Ok((lat_deg, lon_deg, radius_deg)) = mgrs::parse_to_wgs84(trimmed) {
        return Ok(ParsedLocation {
            lat_deg,
            lon_deg,
            radius_deg,
            kind: LocationKind::Mgrs,
        });
    }

    // 2. Try DD / DM / DMS coordinate pair.
    parse_lat_lon(trimmed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hemisphere {
    North,
    South,
    East,
    West,
}

impl Hemisphere {
    fn is_lat(self) -> bool {
        matches!(self, Self::North | Self::South)
    }

    fn sign(self) -> f64 {
        match self {
            Self::North | Self::East => 1.0,
            Self::South | Self::West => -1.0,
        }
    }
}

fn parse_lat_lon(input: &str) -> Result<ParsedLocation, ParseLocationError> {
    // Split on comma or semicolon first; if absent, split into two coordinate
    // clauses by hemisphere letters or whitespace.
    let (first_raw, second_raw) = if let Some((a, b)) = input.split_once([',', ';']) {
        (a.trim(), b.trim())
    } else if let Some((a, b)) = split_by_hemisphere(input) {
        (a, b)
    } else {
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.len() == 2 {
            (parts[0], parts[1])
        } else {
            return Err(ParseLocationError::Unrecognized(input.to_owned()));
        }
    };

    if first_raw.is_empty() || second_raw.is_empty() {
        return Err(ParseLocationError::Unrecognized(input.to_owned()));
    }

    let (v1, h1, dms1) = parse_angle(first_raw, input)?;
    let (v2, h2, dms2) = parse_angle(second_raw, input)?;

    let (lat_deg, lon_deg) = match (h1, h2) {
        (Some(a), Some(b)) if a.is_lat() != b.is_lat() => {
            if a.is_lat() {
                (v1, v2)
            } else {
                (v2, v1)
            }
        }
        (Some(a), None) => {
            if a.is_lat() {
                (v1, v2)
            } else {
                (v2, v1)
            }
        }
        (None, Some(b)) => {
            if b.is_lat() {
                (v2, v1)
            } else {
                (v1, v2)
            }
        }
        (None, None) => (v1, v2),
        (Some(_), Some(_)) => return Err(ParseLocationError::Unrecognized(input.to_owned())),
    };

    if !lat_deg.is_finite() || !(-90.0..=90.0).contains(&lat_deg) {
        return Err(ParseLocationError::LatitudeOutOfRange(format!(
            "{lat_deg:.6}"
        )));
    }
    if !lon_deg.is_finite() || !(-180.0..=180.0).contains(&lon_deg) {
        return Err(ParseLocationError::LongitudeOutOfRange(format!(
            "{lon_deg:.6}"
        )));
    }

    Ok(ParsedLocation {
        lat_deg,
        lon_deg,
        radius_deg: 0.25,
        kind: if dms1 || dms2 {
            LocationKind::DegreesMinutesSeconds
        } else {
            LocationKind::DecimalDegrees
        },
    })
}

fn split_by_hemisphere(input: &str) -> Option<(&str, &str)> {
    for (idx, ch) in input.char_indices() {
        if matches!(ch, 'N' | 'S' | 'E' | 'W' | 'n' | 's' | 'e' | 'w') {
            let after = idx + ch.len_utf8();
            let left = input[..after].trim();
            let right = input[after..].trim();
            if !left.is_empty() && !right.is_empty() {
                return Some((left, right));
            }
        }
    }
    None
}

fn parse_angle(
    clause: &str,
    original: &str,
) -> Result<(f64, Option<Hemisphere>, bool), ParseLocationError> {
    let mut hemisphere = None;
    let mut cleaned = String::with_capacity(clause.len());
    let mut has_dms_markers = false;

    for ch in clause.chars() {
        match ch {
            'N' | 'n' => {
                if hemisphere.replace(Hemisphere::North).is_some() {
                    return Err(ParseLocationError::Unrecognized(original.to_owned()));
                }
                cleaned.push(' ');
            }
            'S' | 's' if !cleaned.trim_end().ends_with(|c: char| c.is_ascii_digit()) => {
                if hemisphere.replace(Hemisphere::South).is_some() {
                    return Err(ParseLocationError::Unrecognized(original.to_owned()));
                }
                cleaned.push(' ');
            }
            'E' | 'e' => {
                if hemisphere.replace(Hemisphere::East).is_some() {
                    return Err(ParseLocationError::Unrecognized(original.to_owned()));
                }
                cleaned.push(' ');
            }
            'W' | 'w' => {
                if hemisphere.replace(Hemisphere::West).is_some() {
                    return Err(ParseLocationError::Unrecognized(original.to_owned()));
                }
                cleaned.push(' ');
            }
            '°' | 'º' | 'd' | 'D' | '\'' | '′' | 'm' | 'M' | '"' | '″' | ':' => {
                has_dms_markers = true;
                cleaned.push(' ');
            }
            's' | 'S' => {
                // `s` directly following a digit is either seconds (`8.5s`) or
                // hemisphere (`33.8688S`). If there are already DMS markers or
                // at least 2 numeric tokens before it, treat as seconds;
                // otherwise treat as South hemisphere.
                let token_count = cleaned.split_whitespace().count();
                if has_dms_markers && token_count >= 2 {
                    cleaned.push(' ');
                } else {
                    if hemisphere.replace(Hemisphere::South).is_some() {
                        return Err(ParseLocationError::Unrecognized(original.to_owned()));
                    }
                    cleaned.push(' ');
                }
            }
            _ => cleaned.push(ch),
        }
    }

    let nums: Vec<f64> = cleaned
        .split_whitespace()
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| ParseLocationError::Unrecognized(original.to_owned()))?;

    if nums.is_empty() || nums.len() > 3 {
        return Err(ParseLocationError::Unrecognized(original.to_owned()));
    }

    let sign_from_deg = if nums[0].is_sign_negative() {
        -1.0
    } else {
        1.0
    };
    let deg = nums[0].abs();
    let min = nums.get(1).copied().unwrap_or(0.0);
    let sec = nums.get(2).copied().unwrap_or(0.0);
    if !(0.0..60.0).contains(&min) || !(0.0..60.0).contains(&sec) {
        return Err(ParseLocationError::Unrecognized(original.to_owned()));
    }

    let mag = deg + min / 60.0 + sec / 3600.0;
    let sign = match hemisphere {
        Some(h) => {
            h.sign()
                * if nums[0].is_sign_negative() {
                    -1.0
                } else {
                    1.0
                }
        }
        None => sign_from_deg,
    };

    let is_dms = nums.len() > 1 || has_dms_markers;
    Ok((sign * mag, hemisphere, is_dms))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_decimal_degrees() {
        let loc = parse_location("47.2357, 39.7015").unwrap();
        assert_eq!(loc.kind, LocationKind::DecimalDegrees);
        assert!((loc.lat_deg - 47.2357).abs() < 1e-9);
        assert!((loc.lon_deg - 39.7015).abs() < 1e-9);

        let loc2 = parse_location("-33.8688 151.2093").unwrap();
        assert!((loc2.lat_deg + 33.8688).abs() < 1e-9);
        assert!((loc2.lon_deg - 151.2093).abs() < 1e-9);

        let loc3 = parse_location("33.8688S, 151.2093E").unwrap();
        assert!((loc3.lat_deg + 33.8688).abs() < 1e-9);
        assert!((loc3.lon_deg - 151.2093).abs() < 1e-9);
    }

    #[test]
    fn parses_dms_formats() {
        let loc = parse_location("47°14'08.5\"N 39°42'05.4\"E").unwrap();
        assert_eq!(loc.kind, LocationKind::DegreesMinutesSeconds);
        assert!((loc.lat_deg - (47.0 + 14.0 / 60.0 + 8.5 / 3600.0)).abs() < 1e-7);
        assert!((loc.lon_deg - (39.0 + 42.0 / 60.0 + 5.4 / 3600.0)).abs() < 1e-7);

        let loc2 = parse_location("47d 14m 8.5s N, 39d 42m 5.4s E").unwrap();
        assert!((loc2.lat_deg - loc.lat_deg).abs() < 1e-7);
        assert!((loc2.lon_deg - loc.lon_deg).abs() < 1e-7);

        let loc3 = parse_location("-33° 52' 07.7\", 151° 12' 33.5\"").unwrap();
        assert!((loc3.lat_deg + (33.0 + 52.0 / 60.0 + 7.7 / 3600.0)).abs() < 1e-7);
        assert!((loc3.lon_deg - (151.0 + 12.0 / 60.0 + 33.5 / 3600.0)).abs() < 1e-7);
    }

    #[test]
    fn parses_mgrs_codes() {
        let loc = parse_location("T40WFC").unwrap();
        assert_eq!(loc.kind, LocationKind::Mgrs);
        assert!((69.0..70.5).contains(&loc.lat_deg));
        assert!((59.0..62.0).contains(&loc.lon_deg));
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(parse_location("   "), Err(ParseLocationError::Empty));
        assert!(matches!(
            parse_location("95.0, 10.0"),
            Err(ParseLocationError::LatitudeOutOfRange(_))
        ));
        assert!(matches!(
            parse_location("10.0, 190.0"),
            Err(ParseLocationError::LongitudeOutOfRange(_))
        ));
        assert!(matches!(
            parse_location("not a coordinate"),
            Err(ParseLocationError::Unrecognized(_))
        ));
    }
}
