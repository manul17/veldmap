//! Legally clean live planetary basemap specifications (NASA GIBS
//! `GoogleMapsCompatible`, EOX Sentinel-2 Cloudless, and Custom XYZ/TMS)
//! enforcing zero HTTP requests above `max_level` and zero HTTP requests for
//! `PolarCap` keys.

use std::fmt::{self, Write as _};

use thiserror::Error;
use veldmap_core::tile::{GlobeScheme, PlanetaryScheme, QuadTileId, SourceId, TileKey, fnv1a_64};

use crate::NetError;
use crate::redact::{EncodedSecret, redact_secret_label, redact_text, sanitize_url_for_display};
use crate::url::{extract_host, has_invalid_url_bytes, parse_http_authority, strip_http_scheme};

/// Internal validation error returned when checking or resolving a tile key
/// against a [`LiveBasemapSpec`].
#[derive(Debug, Error)]
pub(crate) enum TileValidationError {
    #[error(
        "live HTTP requests for PolarCap tiles are forbidden (route to offline polar fallback)"
    )]
    PolarCapForbidden,
    #[error("tile scheme mismatch: requested {requested:?}, expected {expected:?}")]
    SchemeMismatch {
        requested: PlanetaryScheme,
        expected: PlanetaryScheme,
    },
    #[error("tile level {level} exceeds source max_level {max_level}")]
    LevelAboveMax { level: u8, max_level: u8 },
    #[error("tile coordinates ({x}, {y}) out of range for {scheme:?} z={level}")]
    TileOutOfRange {
        scheme: PlanetaryScheme,
        level: u8,
        x: u32,
        y: u32,
    },
}

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

#[inline]
fn subdomain_index(id: QuadTileId, count: usize) -> usize {
    ((u64::from(id.x) + u64::from(id.y)) % (count as u64)) as usize
}

/// Validate an explicit calendar-valid `YYYY-MM-DD` UTC date string within the
/// NASA GIBS epoch (`2000..=2100`, enforcing exact month lengths and leap-year
/// February).
pub fn is_valid_gibs_date(date: &str) -> bool {
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
                &redact_secret_label(Some(self.api_key), "<empty>"),
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
    /// Precomputed lowercased `host[:port]` entries (1 entry normally, or 3
    /// entries for `{s}` subdomain templates) so HTTP dispatch avoids per-tile
    /// URL expansion and string allocations under `PipelineState`'s mutex.
    hosts: Vec<String>,
    tms_flip_y: bool,
    api_key: Option<EncodedSecret>,
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
                &redact_secret_label(self.api_key.as_ref().map(EncodedSecret::raw), "<none>"),
            )
            .finish()
    }
}

/// Validate a custom XYZ/TMS URL template (scheme, non-empty host, absence of
/// whitespace/control chars, and `{z}`, `{x}`, `{y}`/`{-y}` placeholders) and
/// return the trimmed template slice on success.
pub fn validate_custom_url_template(url_template: &str) -> Result<&str, NetError> {
    let template = url_template.trim();
    if template.is_empty() {
        return Err(NetError::InvalidUrlTemplate(
            "URL template must not be empty".to_owned(),
        ));
    }
    if has_invalid_url_bytes(template) {
        return Err(NetError::InvalidUrlTemplate(
            "URL template must not contain whitespace or control characters".to_owned(),
        ));
    }
    if strip_http_scheme(template).is_none() {
        return Err(NetError::InvalidUrlTemplate(
            "URL template must start with https:// or http://".to_owned(),
        ));
    }
    let authority = parse_http_authority(template).ok_or_else(|| {
        NetError::InvalidUrlTemplate("URL template must include a valid hostname".to_owned())
    })?;
    let host = authority.host;
    if host.contains("{z}") || host.contains("{x}") || host.contains("{y}") || host.contains("{-y}")
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
    Ok(template)
}

fn precompute_hosts(url_template: &str) -> Vec<String> {
    if url_template.contains("{s}") {
        let mut hosts: Vec<String> = SUBDOMAINS
            .iter()
            .map(|sub| extract_host(&url_template.replace("{s}", sub)))
            .collect();
        if hosts.iter().all(|h| h == &hosts[0]) {
            hosts.truncate(1);
        }
        hosts
    } else {
        vec![extract_host(url_template)]
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

    /// Precomputed lowercased `host[:port]` key for `id` (zero allocations).
    pub(crate) fn host_for(&self, id: QuadTileId) -> &str {
        if self.hosts.len() == 1 {
            &self.hosts[0]
        } else {
            &self.hosts[subdomain_index(id, self.hosts.len())]
        }
    }

    /// Number of distinct precomputed host entries (1 normally, or 3 for `{s}`).
    pub(crate) fn host_count(&self) -> usize {
        self.hosts.len()
    }

    /// Precomputed lowercased `host[:port]` entries owned by this basemap spec.
    pub(crate) fn hosts(&self) -> &[String] {
        &self.hosts
    }

    fn preset(cache_key: String, attribution: String, max_level: u8, url_template: String) -> Self {
        let hosts = precompute_hosts(&url_template);
        Self {
            source_id: SourceId::from_stable_key(&cache_key),
            cache_key,
            attribution,
            scheme: GlobeScheme::WebMercator,
            max_level,
            url_template,
            hosts,
            tms_flip_y: false,
            api_key: None,
        }
    }

    /// NASA GIBS Blue Marble Shaded Relief + Bathymetry (`GoogleMapsCompatible_Level8`,
    /// static, Public Domain).
    pub fn nasa_gibs_blue_marble() -> Self {
        Self::preset(
            "nasa-gibs-bluemarble-l8".to_owned(),
            "Imagery: NASA GIBS / Blue Marble (Public Domain)".to_owned(),
            GIBS_BLUE_MARBLE_MAX_LEVEL,
            GIBS_BLUE_MARBLE_URL_TEMPLATE.to_owned(),
        )
    }

    /// NASA GIBS VIIRS SNPP Corrected Reflectance True Color
    /// (`GoogleMapsCompatible_Level9`, explicit `YYYY-MM-DD` UTC date, Public Domain).
    pub fn nasa_gibs_viirs(date: &str) -> Result<Self, NetError> {
        let trimmed = date.trim();
        if !is_valid_gibs_date(trimmed) {
            return Err(NetError::InvalidDate(trimmed.to_owned()));
        }
        Ok(Self::preset(
            format!("nasa-gibs-viirs-l9-{trimmed}"),
            format!("Imagery: NASA GIBS / VIIRS SNPP {trimmed} (Public Domain)"),
            GIBS_VIIRS_MAX_LEVEL,
            GIBS_VIIRS_URL_TEMPLATE.replace("{time}", trimmed),
        ))
    }

    /// EOX Sentinel-2 Cloudless 2023 (`WebMercator`, CC BY 4.0, `max_level = 14`).
    pub fn eox_sentinel2_cloudless() -> Self {
        Self::preset(
            "eox-s2cloudless-2023-3857".to_owned(),
            "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH (CC BY 4.0, modified Copernicus Sentinel data)".to_owned(),
            EOX_S2_MAX_LEVEL,
            EOX_S2_URL_TEMPLATE.to_owned(),
        )
    }

    /// Build a custom XYZ/TMS live source, validating URL scheme (`http://` or
    /// `https://`), non-empty host (including bracketed IPv6 literals), absence
    /// of whitespace/control characters, and required coordinate placeholders
    /// (`{z}`, `{x}`, `{y}` or `{-y}`).
    pub fn custom(cfg: CustomBasemapConfig<'_>) -> Result<Self, NetError> {
        let template = validate_custom_url_template(cfg.url_template)?;
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
        let api_key = (!cfg.api_key.trim().is_empty())
            .then(|| EncodedSecret::new(cfg.api_key.trim().to_owned()));
        let url_template = template.to_owned();
        let hosts = precompute_hosts(&url_template);

        Ok(Self {
            source_id: SourceId::from_stable_key(&cache_key),
            cache_key,
            attribution,
            scheme: GlobeScheme::WebMercator,
            max_level,
            url_template,
            hosts,
            tms_flip_y: cfg.tms_flip_y,
            api_key,
        })
    }

    /// Build the canonical display `TileKey` for `id` under this live source.
    pub const fn tile_key(&self, id: QuadTileId) -> TileKey {
        TileKey::display_quad(self.source_id, id)
    }

    #[cfg(test)]
    pub(crate) const fn wm_key(&self, level: u8, x: u32, y: u32) -> TileKey {
        self.tile_key(QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level,
            x,
            y,
        })
    }

    /// Validate that `id` is eligible for live fetching (never `PolarCap`,
    /// matches `self.scheme`, `level <= self.max_level`, valid `(x, y)`).
    pub(crate) fn validate_tile(&self, id: QuadTileId) -> Result<(u32, u32), TileValidationError> {
        if matches!(id.scheme, PlanetaryScheme::PolarCap { .. }) {
            return Err(TileValidationError::PolarCapForbidden);
        }
        let expected = self.scheme.into_planetary();
        if id.scheme != expected {
            return Err(TileValidationError::SchemeMismatch {
                requested: id.scheme,
                expected,
            });
        }
        if id.level > self.max_level {
            return Err(TileValidationError::LevelAboveMax {
                level: id.level,
                max_level: self.max_level,
            });
        }
        let (cols, rows) = id
            .checked_grid_dims()
            .ok_or(TileValidationError::TileOutOfRange {
                scheme: id.scheme,
                level: id.level,
                x: id.x,
                y: id.y,
            })?;
        Ok((cols, rows))
    }

    /// Resolve the concrete HTTP(S) tile URL for `id`, rejecting `PolarCap`
    /// keys and any `level > self.max_level`.
    pub(crate) fn resolve_url(&self, id: QuadTileId) -> Result<String, TileValidationError> {
        let (_cols, rows) = self.validate_tile(id)?;
        let tms_y = rows - 1 - id.y;
        let effective_y = if self.tms_flip_y { tms_y } else { id.y };
        let subdomain = SUBDOMAINS[subdomain_index(id, SUBDOMAINS.len())];
        let api_key = self
            .api_key
            .as_ref()
            .map_or("", EncodedSecret::encoded_upper);

        let extra_cap = api_key.len().saturating_add(16);
        let mut url = String::with_capacity(self.url_template.len().saturating_add(extra_cap));
        let mut rest = self.url_template.as_str();

        while let Some(brace) = rest.find('{') {
            url.push_str(&rest[..brace]);
            let tail = &rest[brace..];
            if let Some(after) = tail.strip_prefix("{s}") {
                url.push_str(subdomain);
                rest = after;
            } else if let Some(after) = tail.strip_prefix("{z}") {
                let _ = write!(url, "{}", id.level);
                rest = after;
            } else if let Some(after) = tail.strip_prefix("{x}") {
                let _ = write!(url, "{}", id.x);
                rest = after;
            } else if let Some(after) = tail.strip_prefix("{-y}") {
                let _ = write!(url, "{tms_y}");
                rest = after;
            } else if let Some(after) = tail.strip_prefix("{y}") {
                let _ = write!(url, "{effective_y}");
                rest = after;
            } else if let Some(after) = tail.strip_prefix("{api_key}") {
                url.push_str(api_key);
                rest = after;
            } else if let Some(after) = tail.strip_prefix("{key}") {
                url.push_str(api_key);
                rest = after;
            } else {
                url.push('{');
                rest = &tail[1..];
            }
        }
        url.push_str(rest);
        Ok(url)
    }

    /// Redact any configured API key (raw or percent-encoded) as well as any
    /// inline URL `user:pass@` authority credentials and `?query` tokens from
    /// `text` before logging or displaying error messages in the UI.
    pub(crate) fn redact_secrets(&self, text: &str) -> String {
        redact_text(text, self.api_key.as_ref())
    }
}

#[cfg(test)]
#[path = "basemap_tests.rs"]
mod tests;
