//! Lightweight zero-dependency parser for GeoTIFF `GDAL_METADATA` (tag `42112`)
//! XML items (`SCALE`, `OFFSET`, `UNITTYPE`, `DESCRIPTION`), supporting both
//! file-wide and per-band (`sample="N"`) attributes.

use std::collections::HashMap;

/// Per-band metadata parsed from `GDAL_METADATA` (tag 42112).
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct GdalBandMeta {
    pub(super) scale: Option<f64>,
    pub(super) offset: Option<f64>,
    pub(super) units: Option<String>,
    pub(super) description: Option<String>,
}

/// Parsed file-wide and per-band `GDAL_METADATA` fields.
#[derive(Clone, Debug, Default)]
pub(super) struct GdalMetadata {
    defaults: GdalBandMeta,
    per_band: HashMap<u16, GdalBandMeta>,
}

impl GdalMetadata {
    /// Resolve metadata for 0-based band `index`, falling back to file-wide
    /// defaults when a per-band item is absent.
    pub(super) fn for_band(&self, index: u16) -> GdalBandMeta {
        let specific = self.per_band.get(&index);
        GdalBandMeta {
            scale: specific.and_then(|m| m.scale).or(self.defaults.scale),
            offset: specific.and_then(|m| m.offset).or(self.defaults.offset),
            units: specific
                .and_then(|m| m.units.clone())
                .or_else(|| self.defaults.units.clone()),
            description: specific
                .and_then(|m| m.description.clone())
                .or_else(|| self.defaults.description.clone()),
        }
    }
}

fn extract_attr<'a>(attrs: &'a str, key: &str) -> Option<&'a str> {
    let mut rest = attrs;
    while let Some(pos) = rest.find(key) {
        let after_key = &rest[pos + key.len()..];
        // Ensure key is preceded by whitespace or start of attributes.
        let valid_start = pos == 0
            || rest[..pos]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_whitespace());
        if !valid_start {
            rest = after_key;
            continue;
        }
        let after_eq = after_key.trim_start().strip_prefix('=')?;
        let trimmed = after_eq.trim_start();
        let quote = trimmed.as_bytes().first().copied()?;
        if quote != b'"' && quote != b'\'' {
            rest = after_key;
            continue;
        }
        let val_start = &trimmed[1..];
        let end = val_start.find(quote as char)?;
        return Some(&val_start[..end]);
    }
    None
}

fn unescape_xml(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Parse a `GDAL_METADATA` XML string into per-band and default metadata.
///
/// Malformed or non-matching XML is ignored cleanly without failing TIFF open.
pub(super) fn parse_gdal_metadata(xml: &str) -> GdalMetadata {
    let mut meta = GdalMetadata::default();
    let mut cursor = xml;

    while let Some(start) = cursor.find("<Item") {
        let after_tag = &cursor[start + "<Item".len()..];
        let Some(gt) = after_tag.find('>') else {
            break;
        };
        let attrs = &after_tag[..gt];
        if attrs.ends_with('/') {
            cursor = &after_tag[gt + 1..];
            continue;
        }
        let body_start = &after_tag[gt + 1..];
        let Some(end_item) = body_start.find("</Item>") else {
            break;
        };
        let raw_val = unescape_xml(body_start[..end_item].trim());
        cursor = &body_start[end_item + "</Item>".len()..];

        let Some(name) = extract_attr(attrs, "name") else {
            continue;
        };
        let sample = extract_attr(attrs, "sample").and_then(|s| s.parse::<u16>().ok());
        let target = match sample {
            Some(idx) => meta.per_band.entry(idx).or_default(),
            None => &mut meta.defaults,
        };

        if name.eq_ignore_ascii_case("SCALE") {
            if let Ok(v) = raw_val.parse::<f64>()
                && v.is_finite()
                && v != 0.0
            {
                target.scale = Some(v);
            }
        } else if name.eq_ignore_ascii_case("OFFSET") {
            if let Ok(v) = raw_val.parse::<f64>()
                && v.is_finite()
            {
                target.offset = Some(v);
            }
        } else if name.eq_ignore_ascii_case("UNITTYPE") && !raw_val.is_empty() {
            target.units = Some(raw_val);
        } else if (name.eq_ignore_ascii_case("DESCRIPTION")
            || name.eq_ignore_ascii_case("BANDNAME"))
            && !raw_val.is_empty()
        {
            target.description = Some(raw_val);
        }
    }

    meta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_per_band_and_default_gdal_metadata_items() {
        let xml = r#"<GDALMetadata>
  <Item name="UNITTYPE">reflectance</Item>
  <Item name="SCALE" sample="0">0.0001</Item>
  <Item name="OFFSET" sample="0">-0.1</Item>
  <Item name="DESCRIPTION" sample="0">B04 Red</Item>
  <Item name="DESCRIPTION" sample="1">B08 NIR &amp; SWIR</Item>
</GDALMetadata>"#;
        let parsed = parse_gdal_metadata(xml);
        let b0 = parsed.for_band(0);
        assert_eq!(b0.scale, Some(0.0001));
        assert_eq!(b0.offset, Some(-0.1));
        assert_eq!(b0.units.as_deref(), Some("reflectance"));
        assert_eq!(b0.description.as_deref(), Some("B04 Red"));

        let b1 = parsed.for_band(1);
        assert_eq!(b1.scale, None);
        assert_eq!(b1.units.as_deref(), Some("reflectance"));
        assert_eq!(b1.description.as_deref(), Some("B08 NIR & SWIR"));
    }
}
