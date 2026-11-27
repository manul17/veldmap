//! Lightweight zero-dependency parsers for GeoTIFF `GDAL_METADATA` (tag
//! `42112`) XML items (`SCALE`, `OFFSET`, `NODATA`, `UNITTYPE`, `DESCRIPTION`,
//! `BANDNAME`) and `GDAL_NODATA` (tag `42113`) ASCII values.

use std::collections::HashMap;

/// Parse an ASCII `GDAL_NODATA` (tag 42113) string into a numeric nodata
/// sentinel (`Some(f64::NAN)` for `"nan"`, or finite `f64`), returning `None`
/// when empty or unparseable.
pub(super) fn parse_gdal_nodata(raw: &str) -> Option<f64> {
    let trimmed = raw.trim_matches(|c: char| c == '\0' || c.is_whitespace());
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.eq_ignore_ascii_case("nan") {
        return Some(f64::NAN);
    }
    trimmed.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Per-band metadata parsed from `GDAL_METADATA` (tag 42112).
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct GdalBandMeta {
    pub(super) scale: Option<f64>,
    pub(super) offset: Option<f64>,
    pub(super) nodata: Option<f64>,
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
            nodata: specific.and_then(|m| m.nodata).or(self.defaults.nodata),
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
    let bytes = attrs.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] == b'/' {
            break;
        }
        let name_start = i;
        while i < bytes.len()
            && !bytes[i].is_ascii_whitespace()
            && bytes[i] != b'='
            && bytes[i] != b'/'
        {
            i += 1;
        }
        let name = &attrs[name_start..i];
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            continue;
        }
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let Some(&quote) = bytes.get(i) else {
            break;
        };
        if quote != b'"' && quote != b'\'' {
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            continue;
        }
        i += 1;
        let val_start = i;
        while i < bytes.len() && bytes[i] != quote {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let val = &attrs[val_start..i];
        i += 1;
        if name == key {
            return Some(val);
        }
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
        let is_item_elem = after_tag
            .as_bytes()
            .first()
            .is_some_and(|&b| b.is_ascii_whitespace() || b == b'>' || b == b'/');
        if !is_item_elem {
            cursor = after_tag;
            continue;
        }
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

        // Only read items in the default GDAL metadata domain (absent or empty `domain`).
        if let Some(domain) = extract_attr(attrs, "domain")
            && !domain.is_empty()
        {
            continue;
        }
        let Some(name) = extract_attr(attrs, "name") else {
            continue;
        };
        let target = match extract_attr(attrs, "sample") {
            Some(s) => match s.parse::<u16>() {
                Ok(idx) => meta.per_band.entry(idx).or_default(),
                Err(_) => continue,
            },
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
        } else if name.eq_ignore_ascii_case("NODATA") {
            if let Some(v) = parse_gdal_nodata(&raw_val) {
                target.nodata = Some(v);
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
  <Item name="NODATA">-9999</Item>
  <Item name="SCALE" sample="0">0.0001</Item>
  <Item name="OFFSET" sample="0">-0.1</Item>
  <Item name="NODATA" sample="0">65535</Item>
  <Item name="DESCRIPTION" sample="0">B04 Red</Item>
  <Item name="DESCRIPTION" sample="1">B08 NIR &amp; SWIR</Item>
</GDALMetadata>"#;
        let parsed = parse_gdal_metadata(xml);
        let b0 = parsed.for_band(0);
        assert_eq!(b0.scale, Some(0.0001));
        assert_eq!(b0.offset, Some(-0.1));
        assert_eq!(b0.nodata, Some(65535.0));
        assert_eq!(b0.units.as_deref(), Some("reflectance"));
        assert_eq!(b0.description.as_deref(), Some("B04 Red"));

        let b1 = parsed.for_band(1);
        assert_eq!(b1.scale, None);
        assert_eq!(b1.nodata, Some(-9999.0));
        assert_eq!(b1.units.as_deref(), Some("reflectance"));
        assert_eq!(b1.description.as_deref(), Some("B08 NIR & SWIR"));
    }

    #[test]
    fn gdal_metadata_handles_attr_prefixes_quoted_text_domains_and_bad_sample() {
        let xml = r#"<GDALMetadata>
  <ItemList><Item namespace="gdal" name="SCALE" sample="0">0.25</Item></ItemList>
  <Item domain="IMAGE_STRUCTURE" name="SCALE">999.0</Item>
  <Item name="SCALE" sample="not_a_number">888.0</Item>
  <Item role="sample='1' name='WRONG'" name="DESCRIPTION" sample="0">Valid Band</Item>
</GDALMetadata>"#;
        let parsed = parse_gdal_metadata(xml);
        let b0 = parsed.for_band(0);
        assert_eq!(b0.scale, Some(0.25));
        assert_eq!(b0.description.as_deref(), Some("Valid Band"));
        // Neither domain="IMAGE_STRUCTURE" nor sample="not_a_number" may pollute band 1 defaults.
        assert_eq!(parsed.for_band(1).scale, None);
    }
}
