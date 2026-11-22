//! Persistent user application settings (active basemap, NASA GIBS date,
//! Custom XYZ/TMS template & API key, disk cache budget) stored via durable
//! atomic JSON writes.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{StorageError, StoragePaths, read_json_or_default, write_json_durable_private};

/// Current JSON schema version for [`AppSettings`].
pub const SETTINGS_SCHEMA_VERSION: u32 = 1;

/// Default explicit UTC date (`YYYY-MM-DD`) for NASA GIBS daily reflectance
/// layers.
pub const DEFAULT_GIBS_DATE: &str = "2024-07-01";

/// Default disk cache cap in mebibytes (`512 MiB`).
pub const DEFAULT_DISK_CACHE_MAX_MB: u64 = 512;

/// Minimum configurable disk cache cap in mebibytes (`16 MiB`).
pub const MIN_DISK_CACHE_MAX_MB: u64 = 16;

/// Maximum configurable disk cache cap in mebibytes (`16 GiB`).
pub const MAX_DISK_CACHE_MAX_MB: u64 = 16 * 1024;

/// Active planetary basemap selection persisted across sessions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BasemapSelection {
    /// Bundled public-domain NASA Blue Marble (`z = 0..3` Geodetic + `z = 0` PolarCap).
    #[default]
    BundledBlueMarble,
    /// Live NASA GIBS Blue Marble Shaded Relief + Bathymetry (`GoogleMapsCompatible_Level8`).
    NasaGibsBlueMarble,
    /// Live NASA GIBS VIIRS SNPP Corrected Reflectance True Color (`GoogleMapsCompatible_Level9`, dated).
    NasaGibsViirs,
    /// Live EOX Sentinel-2 Cloudless (`WebMercator`, CC BY 4.0, `max_level = 14`).
    EoxSentinel2Cloudless,
    /// User-supplied Custom XYZ or TMS tile template.
    CustomXyz,
    /// Untextured WGS84 ellipsoid rendered through the exact same quadtree.
    EllipsoidOnly,
}

impl BasemapSelection {
    /// Short human-readable label for toolbar dropdowns and settings UI.
    pub const fn label(self) -> &'static str {
        match self {
            Self::BundledBlueMarble => "Offline Blue Marble",
            Self::NasaGibsBlueMarble => "NASA GIBS Blue Marble",
            Self::NasaGibsViirs => "NASA GIBS VIIRS True Color",
            Self::EoxSentinel2Cloudless => "EOX Sentinel-2 Cloudless",
            Self::CustomXyz => "Custom XYZ / TMS",
            Self::EllipsoidOnly => "Ellipsoid-only",
        }
    }

    /// Whether this selection streams tiles over HTTP(S).
    pub const fn is_live(self) -> bool {
        matches!(
            self,
            Self::NasaGibsBlueMarble
                | Self::NasaGibsViirs
                | Self::EoxSentinel2Cloudless
                | Self::CustomXyz
        )
    }
}

/// User-configured Custom XYZ / TMS basemap settings.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CustomXyzSettings {
    pub name: String,
    pub url_template: String,
    pub api_key: String,
    pub tms_flip_y: bool,
    pub max_level: u8,
    pub attribution: String,
}

impl Default for CustomXyzSettings {
    fn default() -> Self {
        Self {
            name: "Custom Basemap".to_owned(),
            url_template: String::new(),
            api_key: String::new(),
            tms_flip_y: false,
            max_level: 14,
            attribution: String::new(),
        }
    }
}

impl fmt::Debug for CustomXyzSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomXyzSettings")
            .field("name", &self.name)
            .field("url_template", &self.url_template)
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

/// Durable user application settings saved at `StoragePaths::settings_path()`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub version: u32,
    pub active_basemap: BasemapSelection,
    pub gibs_date: String,
    pub custom_xyz: CustomXyzSettings,
    pub disk_cache_max_mb: u64,
    pub show_graticule: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            version: SETTINGS_SCHEMA_VERSION,
            active_basemap: BasemapSelection::BundledBlueMarble,
            gibs_date: DEFAULT_GIBS_DATE.to_owned(),
            custom_xyz: CustomXyzSettings::default(),
            disk_cache_max_mb: DEFAULT_DISK_CACHE_MAX_MB,
            show_graticule: true,
        }
    }
}

impl AppSettings {
    /// Clamp numeric settings and sanitize fields loaded from disk.
    pub fn normalize(&mut self) {
        self.version = SETTINGS_SCHEMA_VERSION;
        self.custom_xyz.max_level = self.custom_xyz.max_level.clamp(1, 20);
        self.disk_cache_max_mb = self
            .disk_cache_max_mb
            .clamp(MIN_DISK_CACHE_MAX_MB, MAX_DISK_CACHE_MAX_MB);
        if self.gibs_date.trim().is_empty() {
            self.gibs_date = DEFAULT_GIBS_DATE.to_owned();
        }
    }

    /// Configured disk cache cap in bytes.
    pub fn disk_cache_max_bytes(&self) -> u64 {
        self.disk_cache_max_mb
            .clamp(MIN_DISK_CACHE_MAX_MB, MAX_DISK_CACHE_MAX_MB)
            .saturating_mul(1024 * 1024)
    }

    /// Load settings from `paths.settings_path()`, falling back to defaults if
    /// missing or malformed (logging a warning on parse errors).
    pub fn load_or_default(paths: &StoragePaths) -> Self {
        let path = paths.settings_path();
        match read_json_or_default::<Self>(&path) {
            Ok(mut settings) => {
                settings.normalize();
                settings
            }
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "Failed to load user settings; using defaults"
                );
                Self::default()
            }
        }
    }

    /// Persist settings durably via `write_atomic_durable_private` (`0600` on
    /// Unix so optional Custom XYZ API keys are owner-readable only).
    pub fn save(&self, paths: &StoragePaths) -> Result<(), StorageError> {
        let mut normalized = self.clone();
        normalized.normalize();
        write_json_durable_private(&paths.settings_path(), &normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn settings_round_trip_and_redact_api_key_in_debug() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let root = std::env::temp_dir().join(format!(
            "veldmap-settings-test-{}-{nanos}",
            std::process::id()
        ));
        let paths = StoragePaths::with_root(&root);

        let mut settings = AppSettings {
            active_basemap: BasemapSelection::CustomXyz,
            gibs_date: "2024-08-15".to_owned(),
            custom_xyz: CustomXyzSettings {
                name: "My Tile Server".to_owned(),
                url_template: "https://tiles.example.com/{z}/{x}/{y}.png?key={key}".to_owned(),
                api_key: "super-secret-token-123".to_owned(),
                tms_flip_y: true,
                max_level: 99, // clamped to 20
                attribution: "Example".to_owned(),
            },
            disk_cache_max_mb: 256,
            show_graticule: false,
            ..AppSettings::default()
        };

        let dbg = format!("{settings:?}");
        assert!(!dbg.contains("super-secret-token-123"));
        assert!(dbg.contains("<redacted>"));

        settings.save(&paths).expect("save settings");
        let loaded = AppSettings::load_or_default(&paths);
        settings.normalize();
        assert_eq!(loaded, settings);
        assert_eq!(loaded.custom_xyz.max_level, 20);

        let _ = std::fs::remove_dir_all(&root);
    }
}
