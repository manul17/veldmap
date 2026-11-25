//! Persistent user application settings (active basemap, NASA GIBS date,
//! Custom XYZ/TMS template & API key, disk cache budget, graticule toggle)
//! stored via `veldmap-storage`'s durable private atomic JSON I/O.
//!
//! Kept in `veldmap-app` rather than `veldmap-storage` so the low-level
//! persistence crate stays free of UI and basemap domain schemas while
//! settings normalization and redacted `Debug` formatting delegate directly to
//! `veldmap-net`'s canonical constants and [`CustomBasemapConfig`].

use std::fmt;
use std::ops::RangeInclusive;

use serde::{Deserialize, Deserializer, Serialize};
use veldmap_net::{
    CUSTOM_MAX_LEVEL_MAX, CUSTOM_MAX_LEVEL_MIN, CustomBasemapConfig, is_valid_gibs_date,
};
use veldmap_storage::{
    StorageError, StoragePaths, read_json_or_default_with_backup, write_json_durable_private,
};

/// Default disk cache cap in mebibytes (`512 MiB`).
const DEFAULT_DISK_CACHE_MAX_MB: u64 = 512;

/// Minimum configurable disk cache cap in mebibytes (`16 MiB`).
const MIN_DISK_CACHE_MAX_MB: u64 = 16;

/// Maximum configurable disk cache cap in mebibytes (`16 GiB`).
const MAX_DISK_CACHE_MAX_MB: u64 = 16 * 1024;

/// Allowed slider range for Custom XYZ / TMS maximum zoom levels.
pub(crate) const CUSTOM_MAX_LEVEL_RANGE: RangeInclusive<u8> =
    CUSTOM_MAX_LEVEL_MIN..=CUSTOM_MAX_LEVEL_MAX;

/// Allowed slider range for the SQLite WAL tile disk cache budget in MiB.
pub(crate) const DISK_CACHE_MAX_MB_RANGE: RangeInclusive<u64> =
    MIN_DISK_CACHE_MAX_MB..=MAX_DISK_CACHE_MAX_MB;

/// Validate an explicit `YYYY-MM-DD` UTC date string for NASA GIBS daily layers.
pub(crate) fn validate_gibs_date(date: &str) -> bool {
    is_valid_gibs_date(date.trim())
}

/// Validate a user-supplied Custom XYZ / TMS URL template without exposing
/// `veldmap-net` directly to UI widgets.
pub(crate) fn validate_custom_url_template(template: &str) -> Result<(), String> {
    veldmap_net::validate_custom_url_template(template)
        .map(|_| ())
        .map_err(|err| err.to_string())
}

/// Current JSON schema version for [`AppSettings`].
const SETTINGS_SCHEMA_VERSION: u32 = 1;

/// Default explicit UTC date (`YYYY-MM-DD`) for NASA GIBS daily reflectance
/// layers.
const DEFAULT_GIBS_DATE: &str = "2024-07-01";

/// Active planetary basemap selection persisted across sessions.
///
/// Unknown string variants encountered when loading settings written by a newer
/// version fall back to [`Self::BundledBlueMarble`] with a warning instead of
/// discarding the entire `settings.json` file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BasemapSelection {
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
    /// All selectable basemap presets in toolbar display order.
    pub(crate) const ALL: [Self; 6] = [
        Self::BundledBlueMarble,
        Self::NasaGibsBlueMarble,
        Self::NasaGibsViirs,
        Self::EoxSentinel2Cloudless,
        Self::CustomXyz,
        Self::EllipsoidOnly,
    ];

    /// Short human-readable label for toolbar dropdowns and settings UI.
    pub(crate) const fn label(self) -> &'static str {
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
    pub(crate) const fn is_live(self) -> bool {
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
pub(crate) struct CustomXyzSettings {
    pub(crate) name: String,
    pub(crate) url_template: String,
    pub(crate) api_key: String,
    pub(crate) tms_flip_y: bool,
    pub(crate) max_level: u8,
    pub(crate) attribution: String,
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

impl CustomXyzSettings {
    /// Borrow fields as a `veldmap-net` [`CustomBasemapConfig`].
    pub(crate) fn as_net_config(&self) -> CustomBasemapConfig<'_> {
        CustomBasemapConfig {
            name: &self.name,
            url_template: &self.url_template,
            api_key: &self.api_key,
            tms_flip_y: self.tms_flip_y,
            max_level: self.max_level,
            attribution: &self.attribution,
        }
    }
}

impl fmt::Debug for CustomXyzSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.as_net_config(), f)
    }
}

/// Scoped editable slice of [`AppSettings`] for the Basemap & Tile Cache
/// Settings window.
///
/// Keeping this slice and its merge/validation rules alongside [`AppSettings`]
/// guarantees saving basemap settings can never silently clobber unrelated
/// fields (such as `show_graticule`) or overwrite valid saved values with
/// half-typed invalid inputs while keeping `ui/` free of persistence rules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BasemapSettingsDraft {
    pub(crate) active_basemap: BasemapSelection,
    pub(crate) gibs_date: String,
    pub(crate) custom_xyz: CustomXyzSettings,
    pub(crate) disk_cache_max_mb: u64,
}

impl BasemapSettingsDraft {
    pub(crate) fn from_settings(settings: &AppSettings) -> Self {
        Self {
            active_basemap: settings.active_basemap,
            gibs_date: settings.gibs_date.clone(),
            custom_xyz: settings.custom_xyz.clone(),
            disk_cache_max_mb: settings.disk_cache_max_mb,
        }
    }

    pub(crate) fn apply_into(&self, target: &mut AppSettings) {
        target.active_basemap = self.active_basemap;
        // Never overwrite a valid persisted GIBS date with a half-typed invalid
        // string when saving unrelated settings (e.g. disk cache cap or Custom XYZ).
        if validate_gibs_date(&self.gibs_date)
            || self.active_basemap == BasemapSelection::NasaGibsViirs
        {
            target.gibs_date = self.gibs_date.clone();
        }
        // Never overwrite a valid persisted Custom XYZ configuration with a
        // half-typed invalid URL when saving unrelated settings, while still
        // allowing an intentionally cleared empty URL to be persisted.
        if self.active_basemap == BasemapSelection::CustomXyz
            || self.custom_xyz.url_template.trim().is_empty()
            || validate_custom_url_template(&self.custom_xyz.url_template).is_ok()
        {
            target.custom_xyz = self.custom_xyz.clone();
        }
        target.disk_cache_max_mb = self.disk_cache_max_mb;
    }
}

/// Durable user application settings saved at `StoragePaths::settings_path()`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct AppSettings {
    pub(crate) version: u32,
    pub(crate) active_basemap: BasemapSelection,
    pub(crate) gibs_date: String,
    pub(crate) custom_xyz: CustomXyzSettings,
    pub(crate) disk_cache_max_mb: u64,
    pub(crate) show_graticule: bool,
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
    pub(crate) fn normalize(&mut self) {
        self.version = SETTINGS_SCHEMA_VERSION;
        self.custom_xyz.max_level = self
            .custom_xyz
            .max_level
            .clamp(CUSTOM_MAX_LEVEL_MIN, CUSTOM_MAX_LEVEL_MAX);
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
    pub(crate) fn disk_cache_max_bytes(&self) -> u64 {
        self.disk_cache_max_mb
            .clamp(MIN_DISK_CACHE_MAX_MB, MAX_DISK_CACHE_MAX_MB)
            .saturating_mul(1024 * 1024)
    }

    /// Load settings from `paths.settings_path()`, falling back to defaults if
    /// missing or malformed (backing up corrupt JSON to `settings.json.corrupt`).
    pub(crate) fn load_or_default(paths: &StoragePaths) -> Self {
        let mut settings: Self = read_json_or_default_with_backup(&paths.settings_path());
        settings.normalize();
        settings
    }

    /// Persist settings durably via `write_json_durable_private` (`0600` on
    /// Unix so optional Custom XYZ API keys are owner-readable only).
    pub(crate) fn save(&self, paths: &StoragePaths) -> Result<(), StorageError> {
        let mut normalized = self.clone();
        normalized.normalize();
        write_json_durable_private(&paths.settings_path(), &normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use veldmap_storage::test_support::TestTempDir;

    #[test]
    fn settings_round_trip_and_redact_api_key_userinfo_and_query_in_debug() {
        let dir = TestTempDir::new("settings-rt");
        let paths = StoragePaths::with_root(dir.path());

        let mut settings = AppSettings {
            active_basemap: BasemapSelection::CustomXyz,
            gibs_date: " 2024-08-15 ".to_owned(),
            custom_xyz: CustomXyzSettings {
                name: "My Tile Server".to_owned(),
                url_template:
                    "https://alice:pw@tiles.example.com/{z}/{x}/{y}.png?inline=leak-me&key={key}"
                        .to_owned(),
                api_key: "super-secret-token-123".to_owned(),
                tms_flip_y: true,
                max_level: 99, // clamped to CUSTOM_MAX_LEVEL_MAX (20)
                attribution: "Example".to_owned(),
            },
            disk_cache_max_mb: 256,
            show_graticule: false,
            ..AppSettings::default()
        };

        let dbg = format!("{settings:?}");
        assert!(!dbg.contains("super-secret-token-123"));
        assert!(!dbg.contains("alice:pw"));
        assert!(!dbg.contains("leak-me"));
        assert!(dbg.contains("<redacted>"));

        settings.save(&paths).expect("save settings");
        let loaded = AppSettings::load_or_default(&paths);
        settings.normalize();
        assert_eq!(loaded, settings);
        assert_eq!(loaded.gibs_date, "2024-08-15");
        assert_eq!(loaded.custom_xyz.max_level, CUSTOM_MAX_LEVEL_MAX);
    }

    #[test]
    fn corrupt_json_backs_up_file_and_unknown_basemap_variant_is_forward_compatible() {
        let dir = TestTempDir::new("settings-recovery");
        let paths = StoragePaths::with_root(dir.path());
        let settings_path = paths.settings_path();
        std::fs::create_dir_all(settings_path.parent().expect("parent")).expect("mkdir");

        std::fs::write(&settings_path, b"{not valid json").expect("write corrupt");
        let loaded = AppSettings::load_or_default(&paths);
        assert_eq!(loaded, AppSettings::default());
        let backup = settings_path.with_extension("json.corrupt");
        assert_eq!(
            std::fs::read(&backup).expect("backup preserved"),
            b"{not valid json"
        );

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

    #[test]
    fn basemap_draft_preserves_unrelated_app_settings_and_validates_partial_edits() {
        let mut settings = AppSettings {
            show_graticule: false,
            ..AppSettings::default()
        };
        let mut draft = BasemapSettingsDraft::from_settings(&settings);
        draft.active_basemap = BasemapSelection::EoxSentinel2Cloudless;
        draft.disk_cache_max_mb = 1024;

        // Applying basemap draft must leave unrelated `show_graticule` intact.
        draft.apply_into(&mut settings);
        assert!(!settings.show_graticule);
        assert_eq!(
            settings.active_basemap,
            BasemapSelection::EoxSentinel2Cloudless
        );
        assert_eq!(settings.disk_cache_max_mb, 1024);

        // Applying draft with an invalid GIBS date while GIBS VIIRS is inactive
        // preserves the existing valid GIBS date instead of persisting a typo.
        let prev_date = settings.gibs_date.clone();
        draft.gibs_date = "not-a-date".to_owned();
        draft.disk_cache_max_mb = 2048;
        draft.apply_into(&mut settings);
        assert_eq!(settings.gibs_date, prev_date);
        assert_eq!(settings.disk_cache_max_mb, 2048);

        // Saving a valid Custom XYZ URL while CustomXyz is inactive persists it.
        draft.custom_xyz.url_template = "https://tiles.example.org/{z}/{x}/{y}.png".to_owned();
        draft.apply_into(&mut settings);
        assert_eq!(
            settings.custom_xyz.url_template,
            "https://tiles.example.org/{z}/{x}/{y}.png"
        );

        // A half-typed invalid URL while CustomXyz is inactive must preserve
        // the existing saved Custom XYZ configuration.
        let saved_custom = settings.custom_xyz.clone();
        draft.custom_xyz.url_template = "https://incomplete-no-placeholders".to_owned();
        draft.custom_xyz.name = "Half-typed".to_owned();
        draft.apply_into(&mut settings);
        assert_eq!(settings.custom_xyz, saved_custom);

        // Intentionally clearing the URL to empty while CustomXyz is inactive
        // IS applied so the user can clear a saved Custom XYZ source.
        draft.custom_xyz.url_template = "   ".to_owned();
        draft.apply_into(&mut settings);
        assert_eq!(settings.custom_xyz.url_template, "   ");
    }
}
