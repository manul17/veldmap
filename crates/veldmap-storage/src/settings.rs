//! Persistent user application settings (active basemap, NASA GIBS date,
//! Custom XYZ/TMS template & API key, disk cache budget) stored via durable
//! atomic JSON writes.

use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Deserializer, Serialize};

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
///
/// Unknown string variants encountered when loading settings written by a newer
/// version fall back to [`Self::BundledBlueMarble`] with a warning instead of
/// discarding the entire `settings.json` file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
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

impl<'de> Deserialize<'de> for BasemapSelection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        match raw.as_str() {
            "bundled_blue_marble" => Ok(Self::BundledBlueMarble),
            "nasa_gibs_blue_marble" => Ok(Self::NasaGibsBlueMarble),
            "nasa_gibs_viirs" => Ok(Self::NasaGibsViirs),
            "eox_sentinel2_cloudless" => Ok(Self::EoxSentinel2Cloudless),
            "custom_xyz" => Ok(Self::CustomXyz),
            "ellipsoid_only" => Ok(Self::EllipsoidOnly),
            other => {
                tracing::warn!(
                    variant = other,
                    "Unknown active_basemap variant in settings.json; falling back to bundled_blue_marble"
                );
                Ok(Self::BundledBlueMarble)
            }
        }
    }
}

impl BasemapSelection {
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

/// Redact query-string values from a URL template before debug formatting so
/// inline tokens (`?token=...`) are never printed verbatim in logs.
fn redact_url_query(url: &str) -> String {
    match url.split_once('?') {
        Some((base, _)) => format!("{base}?<redacted>"),
        None => url.to_owned(),
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
            .field("url_template", &redact_url_query(&self.url_template))
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
        let trimmed = self.gibs_date.trim();
        if trimmed.is_empty() {
            self.gibs_date = DEFAULT_GIBS_DATE.to_owned();
        } else if trimmed.len() != self.gibs_date.len() {
            self.gibs_date = trimmed.to_owned();
        }
    }

    /// Configured disk cache cap in bytes.
    pub fn disk_cache_max_bytes(&self) -> u64 {
        self.disk_cache_max_mb
            .clamp(MIN_DISK_CACHE_MAX_MB, MAX_DISK_CACHE_MAX_MB)
            .saturating_mul(1024 * 1024)
    }

    /// Load settings from `paths.settings_path()`, falling back to defaults if
    /// missing or malformed. On JSON parse failure, preserves the corrupt file
    /// as `settings.json.corrupt` (preserving permissions via rename) so a
    /// subsequent save never permanently destroys a hand-edited config.
    pub fn load_or_default(paths: &StoragePaths) -> Self {
        let path = paths.settings_path();
        match read_json_or_default::<Self>(&path) {
            Ok(mut settings) => {
                settings.normalize();
                settings
            }
            Err(StorageError::Json(err)) => {
                let mut backup_os = path.as_os_str().to_os_string();
                backup_os.push(".corrupt");
                let backup = PathBuf::from(backup_os);
                let _ = std::fs::rename(&path, &backup);
                tracing::warn!(
                    path = %path.display(),
                    backup = %backup.display(),
                    error = %err,
                    "Failed to parse user settings JSON; backed up corrupt file and using defaults"
                );
                Self::default()
            }
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "Failed to read user settings; using defaults"
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
    use crate::TestTempDir;

    #[test]
    fn settings_round_trip_and_redact_api_key_and_query_in_debug() {
        let dir = TestTempDir::new("settings-rt");
        let paths = StoragePaths::with_root(dir.path());

        let mut settings = AppSettings {
            active_basemap: BasemapSelection::CustomXyz,
            gibs_date: " 2024-08-15 ".to_owned(),
            custom_xyz: CustomXyzSettings {
                name: "My Tile Server".to_owned(),
                url_template: "https://tiles.example.com/{z}/{x}/{y}.png?inline=leak-me&key={key}"
                    .to_owned(),
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
        assert!(!dbg.contains("leak-me"));
        assert!(dbg.contains("<redacted>"));

        settings.save(&paths).expect("save settings");
        let loaded = AppSettings::load_or_default(&paths);
        settings.normalize();
        assert_eq!(loaded, settings);
        assert_eq!(loaded.gibs_date, "2024-08-15");
        assert_eq!(loaded.custom_xyz.max_level, 20);
    }

    #[test]
    fn corrupt_json_backs_up_file_and_unknown_basemap_variant_is_forward_compatible() {
        let dir = TestTempDir::new("settings-recovery");
        let paths = StoragePaths::with_root(dir.path());
        let settings_path = paths.settings_path();
        std::fs::create_dir_all(settings_path.parent().expect("parent")).expect("mkdir");

        // 1. Corrupt JSON is backed up to settings.json.corrupt before returning defaults.
        std::fs::write(&settings_path, b"{not valid json").expect("write corrupt");
        let loaded = AppSettings::load_or_default(&paths);
        assert_eq!(loaded, AppSettings::default());
        let backup = settings_path.with_extension("json.corrupt");
        assert_eq!(
            std::fs::read(&backup).expect("backup preserved"),
            b"{not valid json"
        );

        // 2. Partial JSON + unknown future basemap variant keeps all other user settings.
        std::fs::write(
            &settings_path,
            br#"{"active_basemap":"future_provider_v9","disk_cache_max_mb":999999,"gibs_date":"   "}"#,
        )
        .expect("write forward-compat json");
        let compat = AppSettings::load_or_default(&paths);
        assert_eq!(compat.active_basemap, BasemapSelection::BundledBlueMarble);
        assert_eq!(compat.disk_cache_max_mb, MAX_DISK_CACHE_MAX_MB);
        assert_eq!(compat.gibs_date, DEFAULT_GIBS_DATE);
        assert!(compat.show_graticule);
    }
}
