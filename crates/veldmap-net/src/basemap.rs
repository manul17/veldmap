//! Legally clean live planetary basemap specifications (NASA GIBS
//! `GoogleMapsCompatible`, EOX Sentinel-2 Cloudless, and Custom XYZ/TMS)
//! enforcing zero HTTP requests above `max_level` and zero HTTP requests for
//! `PolarCap` keys.

use std::fmt;

use veldmap_core::tile::{BandKey, PixelFormat, QuadTileId, SourceId, TileKey, TileScheme};
use veldmap_storage::{
    AppSettings, BasemapSelection, CustomXyzSettings, DEFAULT_GIBS_DATE, DiskTileKey,
};

use crate::NetError;

/// NASA GIBS Blue Marble Shaded Relief + Bathymetry (`GoogleMapsCompatible_Level8`,
/// static WebMercator layer, Public Domain).
pub const GIBS_BLUE_MARBLE_URL_TEMPLATE: &str = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpg";
pub const GIBS_BLUE_MARBLE_MAX_LEVEL: u8 = 8;

/// NASA GIBS VIIRS SNPP Corrected Reflectance True Color
/// (`GoogleMapsCompatible_Level9`, explicit `{time}` WebMercator layer, Public Domain).
pub const GIBS_VIIRS_URL_TEMPLATE: &str = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/{time}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg";
pub const GIBS_VIIRS_MAX_LEVEL: u8 = 9;

/// EOX Sentinel-2 Cloudless 2023 (`WebMercator`, CC BY 4.0, `max_level = 14`).
pub const EOX_S2_URL_TEMPLATE: &str =
    "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg";
pub const EOX_S2_MAX_LEVEL: u8 = 14;

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

fn fnv1a_hex(input: &str) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for &b in input.as_bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

/// Complete specification of an HTTP(S) live tile basemap source.
#[derive(Clone, PartialEq, Eq)]
pub struct LiveBasemapSpec {
    pub source_id: SourceId,
    /// Safe ASCII key used inside `TileDiskCache` (never contains `api_key`).
    pub cache_key: String,
    pub label: String,
    pub attribution: String,
    pub scheme: TileScheme,
    pub max_level: u8,
    pub tile_px: u32,
    pub url_template: String,
    pub tms_flip_y: bool,
    pub api_key: Option<String>,
}

impl fmt::Debug for LiveBasemapSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiveBasemapSpec")
            .field("source_id", &self.source_id)
            .field("cache_key", &self.cache_key)
            .field("label", &self.label)
            .field("scheme", &self.scheme)
            .field("max_level", &self.max_level)
            .field("tile_px", &self.tile_px)
            .field("url_template", &self.url_template)
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
    /// NASA GIBS Blue Marble Shaded Relief + Bathymetry (`GoogleMapsCompatible_Level8`,
    /// static, Public Domain).
    pub fn nasa_gibs_blue_marble() -> Self {
        let cache_key = "nasa-gibs-bluemarble-l8".to_owned();
        Self {
            source_id: SourceId::from_stable_key(&cache_key),
            cache_key,
            label: "NASA GIBS Blue Marble".to_owned(),
            attribution: "Imagery: NASA GIBS / Blue Marble (Public Domain)".to_owned(),
            scheme: TileScheme::WebMercator,
            max_level: GIBS_BLUE_MARBLE_MAX_LEVEL,
            tile_px: 256,
            url_template: GIBS_BLUE_MARBLE_URL_TEMPLATE.to_owned(),
            tms_flip_y: false,
            api_key: None,
        }
    }

    /// NASA GIBS VIIRS SNPP Corrected Reflectance True Color
    /// (`GoogleMapsCompatible_Level9`, explicit `YYYY-MM-DD` date, Public Domain).
    pub fn nasa_gibs_viirs(date: &str) -> Self {
        let trimmed = date.trim();
        let valid_date = if is_valid_iso_date(trimmed) {
            trimmed
        } else {
            DEFAULT_GIBS_DATE
        };
        let cache_key = format!("nasa-gibs-viirs-l9-{valid_date}");
        Self {
            source_id: SourceId::from_stable_key(&cache_key),
            cache_key,
            label: format!("NASA GIBS VIIRS ({valid_date})"),
            attribution: format!("Imagery: NASA GIBS / VIIRS SNPP {valid_date} (Public Domain)"),
            scheme: TileScheme::WebMercator,
            max_level: GIBS_VIIRS_MAX_LEVEL,
            tile_px: 256,
            url_template: GIBS_VIIRS_URL_TEMPLATE.replace("{time}", valid_date),
            tms_flip_y: false,
            api_key: None,
        }
    }

    /// EOX Sentinel-2 Cloudless 2023 (`WebMercator`, CC BY 4.0, `max_level = 14`).
    pub fn eox_sentinel2_cloudless() -> Self {
        let cache_key = "eox-s2cloudless-2023-3857".to_owned();
        Self {
            source_id: SourceId::from_stable_key(&cache_key),
            cache_key,
            label: "EOX Sentinel-2 Cloudless".to_owned(),
            attribution: "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH (CC BY 4.0, modified Copernicus Sentinel data)".to_owned(),
            scheme: TileScheme::WebMercator,
            max_level: EOX_S2_MAX_LEVEL,
            tile_px: 256,
            url_template: EOX_S2_URL_TEMPLATE.to_owned(),
            tms_flip_y: false,
            api_key: None,
        }
    }

    /// Build a custom XYZ/TMS live source from user settings, validating the
    /// URL scheme (`http://` or `https://`) and required coordinate tokens.
    pub fn from_custom(cfg: &CustomXyzSettings) -> Result<Self, NetError> {
        let template = cfg.url_template.trim();
        if template.is_empty() {
            return Err(NetError::InvalidUrlTemplate(
                "URL template must not be empty".to_owned(),
            ));
        }
        if !template.starts_with("https://") && !template.starts_with("http://") {
            return Err(NetError::InvalidUrlTemplate(
                "URL template must start with https:// or http://".to_owned(),
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
        let max_level = cfg.max_level.clamp(1, 20);
        // Hash template + tms_flip_y (NEVER the secret API key) into a safe
        // ASCII disk cache key and SourceId.
        let identity = format!("custom|{}|{}|{}", template, cfg.tms_flip_y, max_level);
        let cache_key = format!("custom-xyz-{}", fnv1a_hex(&identity));
        let label = if cfg.name.trim().is_empty() {
            "Custom XYZ / TMS".to_owned()
        } else {
            cfg.name.trim().to_owned()
        };
        let attribution = if cfg.attribution.trim().is_empty() {
            format!("Custom tile source: {label}")
        } else {
            cfg.attribution.trim().to_owned()
        };
        let api_key = (!cfg.api_key.trim().is_empty()).then(|| cfg.api_key.trim().to_owned());

        Ok(Self {
            source_id: SourceId::from_stable_key(&cache_key),
            cache_key,
            label,
            attribution,
            scheme: TileScheme::WebMercator,
            max_level,
            tile_px: 256,
            url_template: template.to_owned(),
            tms_flip_y: cfg.tms_flip_y,
            api_key,
        })
    }

    /// Resolve the active [`LiveBasemapSpec`] from [`AppSettings`], returning
    /// `Ok(None)` for offline (`BundledBlueMarble`) or `EllipsoidOnly` modes.
    pub fn from_settings(settings: &AppSettings) -> Result<Option<Self>, NetError> {
        match settings.active_basemap {
            BasemapSelection::BundledBlueMarble | BasemapSelection::EllipsoidOnly => Ok(None),
            BasemapSelection::NasaGibsBlueMarble => Ok(Some(Self::nasa_gibs_blue_marble())),
            BasemapSelection::NasaGibsViirs => Ok(Some(Self::nasa_gibs_viirs(&settings.gibs_date))),
            BasemapSelection::EoxSentinel2Cloudless => Ok(Some(Self::eox_sentinel2_cloudless())),
            BasemapSelection::CustomXyz => Self::from_custom(&settings.custom_xyz).map(Some),
        }
    }

    /// Build the canonical `TileKey` for `id` under this live source.
    pub const fn tile_key(&self, id: QuadTileId) -> TileKey {
        TileKey {
            source: self.source_id,
            band: BandKey::Display,
            scheme: id.scheme,
            format: PixelFormat::Rgba8UnormSrgb,
            gutter: 0,
            decode_rev: 0,
            level: id.level,
            x: id.x,
            y: id.y,
        }
    }

    /// Validate that `id` is eligible for live fetching (never `PolarCap`,
    /// matches `self.scheme`, `level <= self.max_level`, valid `(x, y)`).
    pub fn validate_tile(&self, id: QuadTileId) -> Result<(u32, u32), NetError> {
        if matches!(id.scheme, TileScheme::PolarCap { .. }) {
            return Err(NetError::PolarCapForbidden);
        }
        if id.scheme != self.scheme {
            return Err(NetError::SchemeMismatch {
                requested: id.scheme,
                expected: self.scheme,
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

    /// Build the disk cache key for `id` after verifying the live-basemap
    /// contract.
    pub fn disk_key(&self, id: QuadTileId) -> Result<DiskTileKey, NetError> {
        self.validate_tile(id)?;
        Ok(DiskTileKey::new(
            self.cache_key.clone(),
            id.scheme,
            id.level,
            id.x,
            id.y,
        ))
    }

    /// Resolve the concrete HTTP(S) tile URL for `id`, rejecting `PolarCap`
    /// keys and any `level > self.max_level`.
    pub fn resolve_url(&self, id: QuadTileId) -> Result<String, NetError> {
        let (_cols, rows) = self.validate_tile(id)?;
        let tms_y = rows - 1 - id.y;
        let effective_y = if self.tms_flip_y { tms_y } else { id.y };

        let mut url = self
            .url_template
            .replace("{z}", &id.level.to_string())
            .replace("{x}", &id.x.to_string())
            .replace("{-y}", &tms_y.to_string())
            .replace("{y}", &effective_y.to_string());

        if let Some(key) = self.api_key.as_deref() {
            url = url.replace("{api_key}", key).replace("{key}", key);
        } else {
            url = url.replace("{api_key}", "").replace("{key}", "");
        }
        Ok(url)
    }

    /// Redact any configured API key (raw or percent-encoded) from `text`
    /// before logging or displaying error messages in the UI.
    pub fn redact_secrets(&self, text: &str) -> String {
        match self.api_key.as_deref().filter(|k| !k.is_empty()) {
            Some(secret) => {
                let mut out = text.replace(secret, "<redacted>");
                let (enc_upper, enc_lower) = percent_encode_variants(secret);
                if enc_upper != secret {
                    out = out.replace(&enc_upper, "<redacted>");
                }
                if enc_lower != secret && enc_lower != enc_upper {
                    out = out.replace(&enc_lower, "<redacted>");
                }
                out
            }
            None => text.to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gibs_and_eox_specs_enforce_polar_cap_and_max_level_contracts() {
        let gibs = LiveBasemapSpec::nasa_gibs_viirs("2024-07-01");
        assert_eq!(gibs.scheme, TileScheme::WebMercator);
        assert_eq!(gibs.max_level, 9);
        let url = gibs
            .resolve_url(QuadTileId {
                scheme: TileScheme::WebMercator,
                level: 3,
                x: 5,
                y: 2,
            })
            .expect("valid gibs tile");
        assert!(url.contains("/2024-07-01/GoogleMapsCompatible_Level9/3/2/5.jpg"));

        // PolarCap is strictly forbidden (0 live HTTP requests).
        assert!(matches!(
            gibs.resolve_url(QuadTileId {
                scheme: TileScheme::PolarCap { north: true },
                level: 0,
                x: 0,
                y: 0,
            }),
            Err(NetError::PolarCapForbidden)
        ));

        // Levels above max_level are strictly forbidden (0 requests above max_level).
        assert!(matches!(
            gibs.resolve_url(QuadTileId {
                scheme: TileScheme::WebMercator,
                level: 10,
                x: 0,
                y: 0,
            }),
            Err(NetError::LevelAboveMax {
                level: 10,
                max_level: 9
            })
        ));

        let eox = LiveBasemapSpec::eox_sentinel2_cloudless();
        assert_eq!(eox.max_level, 14);
        assert!(eox.attribution.contains("CC BY 4.0"));
        assert!(
            eox.resolve_url(QuadTileId {
                scheme: TileScheme::WebMercator,
                level: 14,
                x: 0,
                y: 0,
            })
            .is_ok()
        );
        assert!(matches!(
            eox.resolve_url(QuadTileId {
                scheme: TileScheme::WebMercator,
                level: 15,
                x: 0,
                y: 0,
            }),
            Err(NetError::LevelAboveMax { .. })
        ));
    }

    #[test]
    fn custom_xyz_tms_handles_flip_y_and_redacts_api_key() {
        let cfg = CustomXyzSettings {
            name: "Custom".to_owned(),
            url_template: "https://tiles.example.org/{z}/{x}/{y}.png?token={key}".to_owned(),
            api_key: "secret-abc-999".to_owned(),
            tms_flip_y: true,
            max_level: 6,
            attribution: "Example".to_owned(),
        };
        let spec = LiveBasemapSpec::from_custom(&cfg).expect("build custom");
        let url = spec
            .resolve_url(QuadTileId {
                scheme: TileScheme::WebMercator,
                level: 2,
                x: 1,
                y: 0,
            })
            .expect("resolve custom url");
        // At z=2, rows=4, flipped y for y=0 is 3.
        assert_eq!(
            url,
            "https://tiles.example.org/2/1/3.png?token=secret-abc-999"
        );
        assert!(!format!("{spec:?}").contains("secret-abc-999"));
        assert!(!spec.cache_key.contains("secret-abc-999"));
        assert_eq!(
            spec.redact_secrets(&url),
            "https://tiles.example.org/2/1/3.png?token=<redacted>"
        );

        // Invalid templates are rejected.
        assert!(matches!(
            LiveBasemapSpec::from_custom(&CustomXyzSettings {
                url_template: "ftp://example.com/{z}/{x}/{y}.png".to_owned(),
                ..cfg
            }),
            Err(NetError::InvalidUrlTemplate(_))
        ));
    }

    #[test]
    fn iso_date_validation_accepts_valid_and_rejects_malformed() {
        assert!(is_valid_iso_date("2024-07-01"));
        assert!(is_valid_iso_date("2024-02-29")); // leap year
        assert!(!is_valid_iso_date("2023-02-29")); // non-leap year
        assert!(!is_valid_iso_date("2024-02-31")); // impossible calendar date
        assert!(!is_valid_iso_date("2024-13-01"));
        assert!(!is_valid_iso_date("not-a-date"));
        // Invalid date falls back cleanly to DEFAULT_GIBS_DATE.
        let gibs = LiveBasemapSpec::nasa_gibs_viirs("bad-date");
        assert!(gibs.url_template.contains(DEFAULT_GIBS_DATE));
    }
}
