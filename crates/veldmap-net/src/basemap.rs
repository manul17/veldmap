//! Legally clean live planetary basemap specifications (NASA GIBS
//! `GoogleMapsCompatible`, EOX Sentinel-2 Cloudless, and Custom XYZ/TMS)
//! enforcing zero HTTP requests above `max_level` and zero HTTP requests for
//! `PolarCap` keys.

use std::fmt;

use veldmap_core::tile::{GlobeScheme, PlanetaryScheme, QuadTileId, SourceId, TileKey, fnv1a_64};

use crate::NetError;
use crate::http::{parse_url_authority, sanitize_url_for_display, sanitize_urls_in_text};

/// Minimum allowed custom XYZ/TMS maximum zoom level.
pub const CUSTOM_MAX_LEVEL_MIN: u8 = 1;

/// Maximum allowed custom XYZ/TMS maximum zoom level.
pub const CUSTOM_MAX_LEVEL_MAX: u8 = 20;

/// NASA GIBS Blue Marble Shaded Relief + Bathymetry (`GoogleMapsCompatible_Level8`,
/// static WebMercator layer, Public Domain).
const GIBS_BLUE_MARBLE_URL_TEMPLATE: &str = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpg";
const GIBS_BLUE_MARBLE_MAX_LEVEL: u8 = 8;

/// NASA GIBS VIIRS SNPP Corrected Reflectance True Color
/// (`GoogleMapsCompatible_Level9`, explicit `{time}` WebMercator layer, Public Domain).
const GIBS_VIIRS_URL_TEMPLATE: &str = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/{time}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg";
const GIBS_VIIRS_MAX_LEVEL: u8 = 9;

/// EOX Sentinel-2 Cloudless 2023 (`WebMercator`, CC BY 4.0, `max_level = 14`).
const EOX_S2_URL_TEMPLATE: &str =
    "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg";
const EOX_S2_MAX_LEVEL: u8 = 14;

/// Standard `{s}` subdomain pool for custom XYZ/TMS templates.
const SUBDOMAINS: [&str; 3] = ["a", "b", "c"];

/// Validate an explicit calendar-valid `YYYY-MM-DD` UTC date string for dated
/// GIBS layers (enforcing exact month lengths and leap-year February).
pub fn is_valid_iso_date(date: &str) -> bool {
    let b = date.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return false;
    }
    let digits = |slice: &[u8]| slice.iter().all(u8::is_ascii_digit);
    if !digits(&b[0..4]) || !digits(&b[5..7]) || !digits(&b[8..10]) {
        return false;
    }
    let Ok(year) = date[0..4].parse::<u32>() else {
        return false;
    };
    let Ok(month) = date[5..7].parse::<u32>() else {
        return false;
    };
    let Ok(day) = date[8..10].parse::<u32>() else {
        return false;
    };
    if !(2000..=2100).contains(&year) || !(1..=12).contains(&month) || day == 0 {
        return false;
    }
    let is_leap = (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400);
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap {
                29
            } else {
                28
            }
        }
        _ => 0,
    };
    day <= max_day
}

fn percent_encode_variants(secret: &str) -> (String, String) {
    let mut upper = String::with_capacity(secret.len());
    let mut lower = String::with_capacity(secret.len());
    for &b in secret.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            upper.push(char::from(b));
            lower.push(char::from(b));
        } else {
            upper.push_str(&format!("%{b:02X}"));
            lower.push_str(&format!("%{b:02x}"));
        }
    }
    (upper, lower)
}

/// Pure networking configuration for constructing a Custom XYZ / TMS
/// [`LiveBasemapSpec`] without coupling `veldmap-net` to application settings.
#[derive(Clone, PartialEq, Eq)]
pub struct CustomBasemapConfig<'a> {
    pub name: &'a str,
    pub url_template: &'a str,
    pub api_key: &'a str,
    pub tms_flip_y: bool,
    pub max_level: u8,
    pub attribution: &'a str,
}

impl fmt::Debug for CustomBasemapConfig<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomBasemapConfig")
            .field("name", &self.name)
            .field("url_template", &sanitize_url_for_display(self.url_template))
            .field(
                "api_key",
                &if self.api_key.is_empty() {
                    "<empty>"
                } else {
                    "<redacted>"
                },
            )
            .field("tms_flip_y", &self.tms_flip_y)
            .field("max_level", &self.max_level)
            .field("attribution", &self.attribution)
            .finish()
    }
}

/// Validated, immutable specification of an HTTP(S) live tile basemap source.
///
/// All fields are encapsulated behind read-only accessors so contract
/// invariants (non-`PolarCap` scheme, clamped `max_level`, secret-free ASCII
/// `cache_key`, validated `url_template`) cannot be broken after construction.
#[derive(Clone, PartialEq, Eq)]
pub struct LiveBasemapSpec {
    source_id: SourceId,
    /// Safe ASCII key used inside [`crate::TileBlobStore`] (never contains `api_key`).
    cache_key: String,
    attribution: String,
    scheme: GlobeScheme,
    max_level: u8,
    url_template: String,
    tms_flip_y: bool,
    api_key: Option<String>,
}

impl fmt::Debug for LiveBasemapSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiveBasemapSpec")
            .field("source_id", &self.source_id)
            .field("cache_key", &self.cache_key)
            .field("scheme", &self.scheme)
            .field("max_level", &self.max_level)
            .field(
                "url_template",
                &sanitize_url_for_display(&self.url_template),
            )
            .field("tms_flip_y", &self.tms_flip_y)
            .field(
                "api_key",
                &if self.api_key.as_deref().is_some_and(|k| !k.is_empty()) {
                    "<redacted>"
                } else {
                    "<none>"
                },
            )
            .finish()
    }
}

impl LiveBasemapSpec {
    /// Stable [`SourceId`] derived from [`Self::cache_key`].
    pub const fn source_id(&self) -> SourceId {
        self.source_id
    }

    /// Safe ASCII disk-cache key (never contains secrets or `max_level`).
    pub fn cache_key(&self) -> &str {
        &self.cache_key
    }

    /// Required data-provider attribution string.
    pub fn attribution(&self) -> &str {
        &self.attribution
    }

    /// Active non-polar planetary scheme (`GlobeScheme::WebMercator`).
    pub const fn scheme(&self) -> GlobeScheme {
        self.scheme
    }

    /// Maximum zoom level offered by this source (`0` HTTP requests above this).
    pub const fn max_level(&self) -> u8 {
        self.max_level
    }

    /// Whether the URL template expands `{s}` across multiple subdomain hosts.
    pub(crate) fn uses_subdomains(&self) -> bool {
        self.url_template.contains("{s}")
    }

    /// NASA GIBS Blue Marble Shaded Relief + Bathymetry (`GoogleMapsCompatible_Level8`,
    /// static, Public Domain).
    pub fn nasa_gibs_blue_marble() -> Self {
        let cache_key = "nasa-gibs-bluemarble-l8".to_owned();
        Self {
            source_id: SourceId::from_stable_key(&cache_key),
            cache_key,
            attribution: "Imagery: NASA GIBS / Blue Marble (Public Domain)".to_owned(),
            scheme: GlobeScheme::WebMercator,
            max_level: GIBS_BLUE_MARBLE_MAX_LEVEL,
            url_template: GIBS_BLUE_MARBLE_URL_TEMPLATE.to_owned(),
            tms_flip_y: false,
            api_key: None,
        }
    }

    /// NASA GIBS VIIRS SNPP Corrected Reflectance True Color
    /// (`GoogleMapsCompatible_Level9`, explicit `YYYY-MM-DD` UTC date, Public Domain).
    pub fn nasa_gibs_viirs(date: &str) -> Result<Self, NetError> {
        let trimmed = date.trim();
        if !is_valid_iso_date(trimmed) {
            return Err(NetError::InvalidDate(trimmed.to_owned()));
        }
        let cache_key = format!("nasa-gibs-viirs-l9-{trimmed}");
        Ok(Self {
            source_id: SourceId::from_stable_key(&cache_key),
            cache_key,
            attribution: format!("Imagery: NASA GIBS / VIIRS SNPP {trimmed} (Public Domain)"),
            scheme: GlobeScheme::WebMercator,
            max_level: GIBS_VIIRS_MAX_LEVEL,
            url_template: GIBS_VIIRS_URL_TEMPLATE.replace("{time}", trimmed),
            tms_flip_y: false,
            api_key: None,
        })
    }

    /// EOX Sentinel-2 Cloudless 2023 (`WebMercator`, CC BY 4.0, `max_level = 14`).
    pub fn eox_sentinel2_cloudless() -> Self {
        let cache_key = "eox-s2cloudless-2023-3857".to_owned();
        Self {
            source_id: SourceId::from_stable_key(&cache_key),
            cache_key,
            attribution: "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH (CC BY 4.0, modified Copernicus Sentinel data)".to_owned(),
            scheme: GlobeScheme::WebMercator,
            max_level: EOX_S2_MAX_LEVEL,
            url_template: EOX_S2_URL_TEMPLATE.to_owned(),
            tms_flip_y: false,
            api_key: None,
        }
    }

    /// Build a custom XYZ/TMS live source, validating URL scheme (`http://` or
    /// `https://`), non-empty host (including bracketed IPv6 literals), absence
    /// of whitespace/control characters, and required coordinate placeholders
    /// (`{z}`, `{x}`, `{y}` or `{-y}`).
    pub fn custom(cfg: CustomBasemapConfig<'_>) -> Result<Self, NetError> {
        let template = cfg.url_template.trim();
        if template.is_empty() {
            return Err(NetError::InvalidUrlTemplate(
                "URL template must not be empty".to_owned(),
            ));
        }
        if template
            .bytes()
            .any(|b| b.is_ascii_whitespace() || b.is_ascii_control())
        {
            return Err(NetError::InvalidUrlTemplate(
                "URL template must not contain whitespace or control characters".to_owned(),
            ));
        }
        if !template.starts_with("https://") && !template.starts_with("http://") {
            return Err(NetError::InvalidUrlTemplate(
                "URL template must start with https:// or http://".to_owned(),
            ));
        }
        let authority = parse_url_authority(template).ok_or_else(|| {
            NetError::InvalidUrlTemplate("URL template must include a valid hostname".to_owned())
        })?;
        let host = authority.host;
        if host.contains("{z}")
            || host.contains("{x}")
            || host.contains("{y}")
            || host.contains("{-y}")
        {
            return Err(NetError::InvalidUrlTemplate(
                "URL template must include a valid hostname".to_owned(),
            ));
        }
        let has_z = template.contains("{z}");
        let has_x = template.contains("{x}");
        let has_y = template.contains("{y}") || template.contains("{-y}");
        if !has_z || !has_x || !has_y {
            return Err(NetError::InvalidUrlTemplate(
                "URL template must include {z}, {x}, and {y} (or {-y})".to_owned(),
            ));
        }
        let max_level = cfg
            .max_level
            .clamp(CUSTOM_MAX_LEVEL_MIN, CUSTOM_MAX_LEVEL_MAX);
        // Hash template + tms_flip_y (NEVER the secret API key or max_level)
        // into a safe ASCII disk cache key and SourceId so adjusting max_level
        // does not invalidate cached tiles.
        let identity = format!("custom|{template}|{}", cfg.tms_flip_y);
        let hash = fnv1a_64(identity.as_bytes());
        let cache_key = format!("custom-xyz-{hash:016x}");
        let display_name = if cfg.name.trim().is_empty() {
            "Custom XYZ / TMS"
        } else {
            cfg.name.trim()
        };
        let attribution = if cfg.attribution.trim().is_empty() {
            format!("Custom tile source: {display_name}")
        } else {
            cfg.attribution.trim().to_owned()
        };
        let api_key = (!cfg.api_key.trim().is_empty()).then(|| cfg.api_key.trim().to_owned());

        Ok(Self {
            source_id: SourceId::from_stable_key(&cache_key),
            cache_key,
            attribution,
            scheme: GlobeScheme::WebMercator,
            max_level,
            url_template: template.to_owned(),
            tms_flip_y: cfg.tms_flip_y,
            api_key,
        })
    }

    /// Build the canonical display `TileKey` for `id` under this live source.
    pub const fn tile_key(&self, id: QuadTileId) -> TileKey {
        TileKey::display_quad(self.source_id, id)
    }

    /// Validate that `id` is eligible for live fetching (never `PolarCap`,
    /// matches `self.scheme`, `level <= self.max_level`, valid `(x, y)`).
    pub fn validate_tile(&self, id: QuadTileId) -> Result<(u32, u32), NetError> {
        if matches!(id.scheme, PlanetaryScheme::PolarCap { .. }) {
            return Err(NetError::PolarCapForbidden);
        }
        let expected = self.scheme.into_planetary();
        if id.scheme != expected {
            return Err(NetError::SchemeMismatch {
                requested: id.scheme,
                expected,
            });
        }
        if id.level > self.max_level {
            return Err(NetError::LevelAboveMax {
                level: id.level,
                max_level: self.max_level,
            });
        }
        let (cols, rows) = id
            .scheme
            .grid_dims(id.level)
            .ok_or(NetError::TileOutOfRange {
                scheme: id.scheme,
                level: id.level,
                x: id.x,
                y: id.y,
            })?;
        if id.x >= cols || id.y >= rows {
            return Err(NetError::TileOutOfRange {
                scheme: id.scheme,
                level: id.level,
                x: id.x,
                y: id.y,
            });
        }
        Ok((cols, rows))
    }

    /// Resolve the concrete HTTP(S) tile URL for `id`, rejecting `PolarCap`
    /// keys and any `level > self.max_level`.
    pub fn resolve_url(&self, id: QuadTileId) -> Result<String, NetError> {
        let (_cols, rows) = self.validate_tile(id)?;
        let tms_y = rows - 1 - id.y;
        let effective_y = if self.tms_flip_y { tms_y } else { id.y };
        let sub_idx = ((u64::from(id.x) + u64::from(id.y)) % (SUBDOMAINS.len() as u64)) as usize;

        let mut url = self
            .url_template
            .replace("{s}", SUBDOMAINS[sub_idx])
            .replace("{z}", &id.level.to_string())
            .replace("{x}", &id.x.to_string())
            .replace("{-y}", &tms_y.to_string())
            .replace("{y}", &effective_y.to_string());

        if let Some(key) = self.api_key.as_deref() {
            let (encoded_key, _) = percent_encode_variants(key);
            url = url
                .replace("{api_key}", &encoded_key)
                .replace("{key}", &encoded_key);
        } else {
            url = url.replace("{api_key}", "").replace("{key}", "");
        }
        Ok(url)
    }

    /// Redact any configured API key (raw or percent-encoded) as well as any
    /// inline URL `user:pass@` authority credentials and `?query` tokens from
    /// `text` before logging or displaying error messages in the UI.
    pub fn redact_secrets(&self, text: &str) -> String {
        let mut out = sanitize_urls_in_text(text);
        if let Some(secret) = self.api_key.as_deref().filter(|k| !k.is_empty()) {
            out = out.replace(secret, "<redacted>");
            let (enc_upper, enc_lower) = percent_encode_variants(secret);
            if enc_upper != secret {
                out = out.replace(&enc_upper, "<redacted>");
            }
            if enc_lower != secret && enc_lower != enc_upper {
                out = out.replace(&enc_lower, "<redacted>");
            }
        }
        out
    }
}
