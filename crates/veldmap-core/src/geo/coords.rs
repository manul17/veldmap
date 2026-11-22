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

    // 1. Try MGRS first (e.g. `37UDQ`, `T40WFC`, `4Q FJ 12345 67890`).
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

fn parse_hemisphere_char(ch: char) -> Option<Hemisphere> {
    match ch {
        'N' | 'n' => Some(Hemisphere::North),
        'S' | 's' => Some(Hemisphere::South),
        'E' | 'e' => Some(Hemisphere::East),
        'W' | 'w' => Some(Hemisphere::West),
        _ => None,
    }
}

fn contains_dms_marker(s: &str) -> bool {
    s.chars().any(|c| {
        matches!(
            c,
            '°' | 'º' | '\'' | '′' | '"' | '″' | ':' | 'd' | 'D' | 'm' | 'M'
        )
    })
}

fn is_prefix_hemisphere_start(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if parse_hemisphere_char(first).is_none() {
        return false;
    }
    chars.next().is_some_and(|next| {
        next.is_whitespace() || next.is_ascii_digit() || matches!(next, '+' | '-' | '.')
    })
}

fn split_by_hemisphere(input: &str) -> Option<(&str, &str)> {
    // Case 1: Leading hemisphere prefixes, e.g. `N47.2357 E39.7015` or
    // `S 33°52'07.7" E 151°12'33.5"`. Split before the second prefix.
    if is_prefix_hemisphere_start(input) {
        let mut prev_ws = false;
        for (idx, ch) in input.char_indices().skip(1) {
            if prev_ws && is_prefix_hemisphere_start(&input[idx..]) {
                let left = input[..idx].trim();
                let right = input[idx..].trim();
                if !left.is_empty() && !right.is_empty() {
                    return Some((left, right));
                }
            }
            prev_ws = ch.is_whitespace();
        }
        return None;
    }

    // Case 2: Trailing hemisphere suffixes, e.g. `47°14'08.5"N 39°42'05.4"E`,
    // `47d 14m 8.5s N 39d 42m 5.4s E`, or `33.8688S 151.2093E`.
    for (idx, ch) in input.char_indices() {
        if parse_hemisphere_char(ch).is_none() {
            continue;
        }
        let after = idx + ch.len_utf8();
        let rest = &input[after..];
        if !rest.starts_with(char::is_whitespace) {
            continue;
        }
        let before = &input[..idx];
        let right = rest.trim();
        if before.trim().is_empty() || right.is_empty() {
            continue;
        }
        // Do not split on a seconds suffix (`8.5s`) before a real hemisphere or
        // inside a DMS clause.
        if matches!(ch, 's' | 'S')
            && before.ends_with(|c: char| c.is_ascii_digit())
            && contains_dms_marker(before)
            && (ch == 's'
                || right
                    .chars()
                    .next()
                    .and_then(parse_hemisphere_char)
                    .is_some())
        {
            continue;
        }
        return Some((input[..after].trim(), right));
    }
    None
}

fn is_seconds_suffix(last: char, before: &str) -> bool {
    matches!(last, 's' | 'S')
        && before.ends_with(|c: char| c.is_ascii_digit())
        && contains_dms_marker(before)
        && (last == 's' || before.chars().any(|c| matches!(c, 'd' | 'D' | 'm' | 'M')))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AngleUnit {
    Deg,
    Min,
    Sec,
    Colon,
}

fn parse_angle(
    clause: &str,
    original: &str,
) -> Result<(f64, Option<Hemisphere>, bool), ParseLocationError> {
    let mut rest = clause.trim();
    let mut hemisphere = None;

    // 1. Optional leading hemisphere (`N 47.2`, `S33.8688`).
    if is_prefix_hemisphere_start(rest)
        && let Some(first) = rest.chars().next()
        && let Some(h) = parse_hemisphere_char(first)
    {
        hemisphere = Some(h);
        rest = rest[first.len_utf8()..].trim();
    }

    // 2. Optional trailing hemisphere (`33°52'07.7"S`, `33d 52m 7.7s S`, `33.8688S`).
    // Always check even if a prefix hemisphere was found so conflicting forms
    // like `"N 45S"` are rejected rather than having `S` treated as seconds.
    if let Some(last) = rest.chars().next_back()
        && let Some(h) = parse_hemisphere_char(last)
    {
        let before = &rest[..rest.len() - last.len_utf8()];
        if !is_seconds_suffix(last, before) {
            if hemisphere.is_some() {
                return Err(ParseLocationError::Unrecognized(original.to_owned()));
            }
            hemisphere = Some(h);
            rest = before.trim();
        }
    }

    let mut slots: [Option<f64>; 3] = [None; 3];
    let mut next_slot = 0usize;
    let mut has_dms_markers = false;
    let mut count = 0usize;

    let mut chars = rest.char_indices().peekable();
    while let Some(&(start, ch)) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        let mut end = start;
        while let Some(&(idx, c)) = chars.peek() {
            if c.is_ascii_digit() || matches!(c, '.' | '+' | '-') {
                end = idx + c.len_utf8();
                chars.next();
            } else {
                break;
            }
        }
        if end == start {
            return Err(ParseLocationError::Unrecognized(original.to_owned()));
        }
        let val: f64 = rest[start..end]
            .parse()
            .map_err(|_| ParseLocationError::Unrecognized(original.to_owned()))?;
        if !val.is_finite() {
            return Err(ParseLocationError::Unrecognized(original.to_owned()));
        }

        let unit = match chars.peek().map(|&(_, c)| c) {
            Some('°' | 'º' | 'd' | 'D') => {
                chars.next();
                has_dms_markers = true;
                Some(AngleUnit::Deg)
            }
            Some('\'' | '′' | 'm' | 'M') => {
                chars.next();
                has_dms_markers = true;
                Some(AngleUnit::Min)
            }
            Some('"' | '″' | 's' | 'S') => {
                chars.next();
                has_dms_markers = true;
                Some(AngleUnit::Sec)
            }
            Some(':') => {
                chars.next();
                has_dms_markers = true;
                Some(AngleUnit::Colon)
            }
            Some(c) if c.is_whitespace() => None,
            None => None,
            Some(_) => return Err(ParseLocationError::Unrecognized(original.to_owned())),
        };

        let slot = match unit {
            Some(AngleUnit::Deg) => 0,
            Some(AngleUnit::Min) => 1,
            Some(AngleUnit::Sec) => 2,
            Some(AngleUnit::Colon) | None => next_slot,
        };
        if slot < next_slot || slot >= 3 {
            return Err(ParseLocationError::Unrecognized(original.to_owned()));
        }
        slots[slot] = Some(val);
        next_slot = slot + 1;
        count += 1;
    }

    let Some(deg_raw) = slots[0] else {
        return Err(ParseLocationError::Unrecognized(original.to_owned()));
    };

    if hemisphere.is_some() && deg_raw.is_sign_negative() {
        return Err(ParseLocationError::Unrecognized(original.to_owned()));
    }

    let sign_from_deg = if deg_raw.is_sign_negative() {
        -1.0
    } else {
        1.0
    };
    let deg = deg_raw.abs();
    let min = slots[1].unwrap_or(0.0);
    let sec = slots[2].unwrap_or(0.0);
    if !(0.0..60.0).contains(&min) || !(0.0..60.0).contains(&sec) {
        return Err(ParseLocationError::Unrecognized(original.to_owned()));
    }

    let mag = deg + min / 60.0 + sec / 3600.0;
    let sign = hemisphere.map_or(sign_from_deg, Hemisphere::sign);
    let is_dms = count > 1 || has_dms_markers;
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

        // Space-separated suffix and prefix hemispheres, plus lon-first order.
        let loc4 = parse_location("33.8688S 151.2093E").unwrap();
        assert!((loc4.lat_deg + 33.8688).abs() < 1e-9);
        assert!((loc4.lon_deg - 151.2093).abs() < 1e-9);

        let loc5 = parse_location("N47.2357 E39.7015").unwrap();
        assert!((loc5.lat_deg - 47.2357).abs() < 1e-9);
        assert!((loc5.lon_deg - 39.7015).abs() < 1e-9);

        let loc6 = parse_location("39.7015E 47.2357N").unwrap();
        assert!((loc6.lat_deg - 47.2357).abs() < 1e-9);
        assert!((loc6.lon_deg - 39.7015).abs() < 1e-9);
    }

    #[test]
    fn parses_dms_formats_in_both_hemispheres() {
        let loc = parse_location("47°14'08.5\"N 39°42'05.4\"E").unwrap();
        assert_eq!(loc.kind, LocationKind::DegreesMinutesSeconds);
        assert!((loc.lat_deg - (47.0 + 14.0 / 60.0 + 8.5 / 3600.0)).abs() < 1e-7);
        assert!((loc.lon_deg - (39.0 + 42.0 / 60.0 + 5.4 / 3600.0)).abs() < 1e-7);

        let loc2 = parse_location("47d 14m 8.5s N, 39d 42m 5.4s E").unwrap();
        assert!((loc2.lat_deg - loc.lat_deg).abs() < 1e-7);
        assert!((loc2.lon_deg - loc.lon_deg).abs() < 1e-7);

        // Space-separated `d/m/s` without comma.
        let loc_space = parse_location("47d 14m 8.5s N 39d 42m 5.4s E").unwrap();
        assert!((loc_space.lat_deg - loc.lat_deg).abs() < 1e-7);
        assert!((loc_space.lon_deg - loc.lon_deg).abs() < 1e-7);

        // Southern hemisphere DMS and DM with `S` must produce negative latitude.
        let sydney_lat = -(33.0 + 52.0 / 60.0 + 7.7 / 3600.0);
        let sydney_lon = 151.0 + 12.0 / 60.0 + 33.5 / 3600.0;
        for input in [
            "33°52'07.7\"S, 151°12'33.5\"E",
            "33°52'07.7\"S 151°12'33.5\"E",
            "33d 52m 7.7s S, 151d 12m 33.5s E",
            "33d 52m 7.7s S 151d 12m 33.5s E",
            "33d52m7.7sS 151d12m33.5sE",
            "S 33°52'07.7\" E 151°12'33.5\"",
            "-33° 52' 07.7\", 151° 12' 33.5\"",
        ] {
            let parsed = parse_location(input).unwrap_or_else(|e| panic!("{input}: {e}"));
            assert_eq!(parsed.kind, LocationKind::DegreesMinutesSeconds, "{input}");
            assert!(
                (parsed.lat_deg - sydney_lat).abs() < 1e-7,
                "{input} -> lat {}",
                parsed.lat_deg
            );
            assert!(
                (parsed.lon_deg - sydney_lon).abs() < 1e-7,
                "{input} -> lon {}",
                parsed.lon_deg
            );
        }

        let dm_south = parse_location("33° 52' S, 151° 12' E").unwrap();
        assert!((dm_south.lat_deg + (33.0 + 52.0 / 60.0)).abs() < 1e-7);
        assert!((dm_south.lon_deg - (151.0 + 12.0 / 60.0)).abs() < 1e-7);

        // Negative zero degrees must preserve negative sign on minutes/seconds.
        let neg_zero = parse_location("-0° 30' 00\", 10° 00' 00\"").unwrap();
        assert!((neg_zero.lat_deg + 0.5).abs() < 1e-9);
        assert!((neg_zero.lon_deg - 10.0).abs() < 1e-9);

        // Degrees + seconds with omitted minutes must treat `10s` as seconds, not minutes.
        let deg_sec = parse_location("45d 10s, 10 E").unwrap();
        assert_eq!(deg_sec.kind, LocationKind::DegreesMinutesSeconds);
        assert!((deg_sec.lat_deg - (45.0 + 10.0 / 3600.0)).abs() < 1e-9);
        assert!((deg_sec.lon_deg - 10.0).abs() < 1e-9);

        // Colon-separated DMS with semicolon delimiter and exact ±90/±180 boundaries.
        let colon = parse_location("47:14:08.5N; 39:42:05.4E").unwrap();
        assert!((colon.lat_deg - loc.lat_deg).abs() < 1e-7);
        assert!((colon.lon_deg - loc.lon_deg).abs() < 1e-7);
        let bounds = parse_location("-90.0, 180.0").unwrap();
        assert_eq!((bounds.lat_deg, bounds.lon_deg), (-90.0, 180.0));
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
        for bad in [
            "not a coordinate",
            "-33.8688S, 151.2093E",
            "N 45 S, 10 E",
            "N 45S, 10 E",
            "N 45° 10' 5\" S, 10 E",
            "47° 60' 0\", 10°",
            "47° 10' 60\", 10°",
            "47m 14d, 10 E",
            "47d 14d, 10 E",
            "NaN, 10.0",
            "inf, 10.0",
            "37MAA",
            "32XAA",
            "37UD\u{0419}",
            "37UD\u{00E9}",
        ] {
            assert!(
                matches!(
                    parse_location(bad),
                    Err(ParseLocationError::Unrecognized(_))
                ),
                "expected Unrecognized for '{bad}'"
            );
        }
    }
}
