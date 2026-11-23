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

fn strip_hemisphere<'a>(
    clause: &'a str,
    original: &str,
) -> Result<(&'a str, Option<Hemisphere>), ParseLocationError> {
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

    Ok((rest, hemisphere))
}

struct ScannedAngleSlots {
    slots: [Option<f64>; 3],
    has_dms_markers: bool,
    count: usize,
}

fn scan_dms_slots(rest: &str, original: &str) -> Result<ScannedAngleSlots, ParseLocationError> {
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

    Ok(ScannedAngleSlots {
        slots,
        has_dms_markers,
        count,
    })
}

fn assemble_angle(
    scanned: ScannedAngleSlots,
    hemisphere: Option<Hemisphere>,
    original: &str,
) -> Result<(f64, Option<Hemisphere>, bool), ParseLocationError> {
    let Some(deg_raw) = scanned.slots[0] else {
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
    let min = scanned.slots[1].unwrap_or(0.0);
    let sec = scanned.slots[2].unwrap_or(0.0);
    if !(0.0..60.0).contains(&min) || !(0.0..60.0).contains(&sec) {
        return Err(ParseLocationError::Unrecognized(original.to_owned()));
    }

    let mag = deg + min / 60.0 + sec / 3600.0;
    let sign = hemisphere.map_or(sign_from_deg, Hemisphere::sign);
    let is_dms = scanned.count > 1 || scanned.has_dms_markers;
    Ok((sign * mag, hemisphere, is_dms))
}

fn parse_angle(
    clause: &str,
    original: &str,
) -> Result<(f64, Option<Hemisphere>, bool), ParseLocationError> {
    let (rest, hemisphere) = strip_hemisphere(clause, original)?;
    let scanned = scan_dms_slots(rest, original)?;
    assemble_angle(scanned, hemisphere, original)
}

#[cfg(test)]
#[path = "coords_tests.rs"]
mod tests;
