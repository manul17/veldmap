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

/// Minimum optional user override for the GPU tile cache cap (`128 MiB`).
pub(crate) const MIN_GPU_CACHE_MAX_MB: u64 = 128;

/// Maximum optional user override for the GPU tile cache cap (`4096 MiB`).
pub(crate) const MAX_GPU_CACHE_MAX_MB: u64 = 4096;

/// Default value pre-filled when the user enables a custom GPU VRAM cap (`512 MiB`).
pub(crate) const DEFAULT_CUSTOM_GPU_CACHE_MAX_MB: u64 = 512;

/// Allowed slider range for optional user GPU tile VRAM cap in MiB.
pub(crate) const GPU_CACHE_MAX_MB_RANGE: RangeInclusive<u64> =
    MIN_GPU_CACHE_MAX_MB..=MAX_GPU_CACHE_MAX_MB;

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
    pub(crate) gpu_tile_cache_max_mb: Option<u64>,
}

impl BasemapSettingsDraft {
    pub(crate) fn from_settings(settings: &AppSettings) -> Self {
        Self {
            active_basemap: settings.active_basemap,
            gibs_date: settings.gibs_date.clone(),
            custom_xyz: settings.custom_xyz.clone(),
            disk_cache_max_mb: settings.disk_cache_max_mb,
            gpu_tile_cache_max_mb: settings.gpu_tile_cache_max_mb,
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
        target.gpu_tile_cache_max_mb = self.gpu_tile_cache_max_mb;
    }
}

/// Borrowed basemap-specific view of [`AppSettings`] passed into
/// `BasemapManager` so the basemap coordinator stays decoupled from unrelated
/// UI and cache-budget preferences.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BasemapConfigView<'a> {
    pub(crate) active_basemap: BasemapSelection,
    pub(crate) gibs_date: &'a str,
    pub(crate) custom_xyz: &'a CustomXyzSettings,
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
    /// Optional user GPU tile cache cap in MiB (`None` = host-RAM-scaled auto cap).
    pub(crate) gpu_tile_cache_max_mb: Option<u64>,
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
            gpu_tile_cache_max_mb: None,
            show_graticule: true,
        }
    }
}

impl AppSettings {
    /// Borrow the basemap-specific subset of settings for `BasemapManager`.
    pub(crate) fn basemap_config(&self) -> BasemapConfigView<'_> {
        BasemapConfigView {
            active_basemap: self.active_basemap,
            gibs_date: &self.gibs_date,
            custom_xyz: &self.custom_xyz,
        }
    }

    /// Clamp numeric settings and sanitize fields loaded from disk.
    pub(super) fn normalize(&mut self) {
        self.version = SETTINGS_SCHEMA_VERSION;
        self.custom_xyz.max_level = self
            .custom_xyz
            .max_level
            .clamp(CUSTOM_MAX_LEVEL_MIN, CUSTOM_MAX_LEVEL_MAX);
        self.disk_cache_max_mb = self
            .disk_cache_max_mb
            .clamp(MIN_DISK_CACHE_MAX_MB, MAX_DISK_CACHE_MAX_MB);
        self.gpu_tile_cache_max_mb = self
            .gpu_tile_cache_max_mb
            .map(|mb| mb.clamp(MIN_GPU_CACHE_MAX_MB, MAX_GPU_CACHE_MAX_MB));
        let trimmed = self.gibs_date.trim();
        if trimmed.is_empty() {
            self.gibs_date = DEFAULT_GIBS_DATE.to_owned();
        } else if trimmed.len() != self.gibs_date.len() {
            self.gibs_date = trimmed.to_owned();
        }
    }

    /// Resolve effective GPU tile cache byte cap from user override or auto
    /// host-RAM budget.
    pub(crate) fn effective_gpu_tile_cache_bytes(&self, default_bytes: usize) -> usize {
        match self.gpu_tile_cache_max_mb {
            Some(mb) => usize::try_from(
                mb.clamp(MIN_GPU_CACHE_MAX_MB, MAX_GPU_CACHE_MAX_MB)
                    .saturating_mul(1024 * 1024),
            )
            .unwrap_or(usize::MAX),
            None => default_bytes,
        }
    }

    /// Configured disk cache cap in bytes.
    pub(super) fn disk_cache_max_bytes(&self) -> u64 {
        self.disk_cache_max_mb
            .clamp(MIN_DISK_CACHE_MAX_MB, MAX_DISK_CACHE_MAX_MB)
            .saturating_mul(1024 * 1024)
    }

    /// Load settings from `paths.settings_path()`, falling back to defaults if
    /// missing or malformed (backing up corrupt JSON to `settings.json.corrupt`).
    pub(super) fn load_or_default(paths: &StoragePaths) -> Self {
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
#[path = "settings_tests.rs"]
mod tests;
