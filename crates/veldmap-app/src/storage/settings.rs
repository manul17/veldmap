//! Persistent user application settings (active basemap, NASA GIBS date,
//! Custom XYZ/TMS template & API key, disk/GPU cache budgets, graticule toggle,
//! and Custom STAC endpoint & auth credentials) stored via `veldmap-storage`'s
//! durable private atomic JSON I/O.
//!
//! Kept in `veldmap-app` rather than `veldmap-storage` so the low-level
//! persistence crate stays free of UI and basemap domain schemas while
//! settings normalization and redacted `Debug` formatting delegate directly to
//! `veldmap-net`'s canonical constants and validators.

use std::fmt;
use std::ops::RangeInclusive;

use serde::{Deserialize, Deserializer, Serialize};
use veldmap_net::{
    CUSTOM_MAX_LEVEL_MAX, CUSTOM_MAX_LEVEL_MIN, CustomBasemapConfig, StacAuth, StacCatalogConfig,
    is_valid_custom_header_name, is_valid_gibs_date, is_valid_http_endpoint,
    sanitize_url_for_display,
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
const MIN_GPU_CACHE_MAX_MB: u64 = 128;

/// Maximum optional user override for the GPU tile cache cap (`4096 MiB`).
const MAX_GPU_CACHE_MAX_MB: u64 = 4096;

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

/// Forward-compatible deserializer for string-encoded settings enums: unknown
/// string variants written by newer versions warn and fall back to
/// `T::default()`, while non-string JSON still errors so corrupt
/// `settings.json` files trigger `.corrupt` backup recovery.
fn deserialize_lossy_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + for<'a> Deserialize<'a>,
{
    let raw = String::deserialize(deserializer)?;
    let de = serde::de::value::StrDeserializer::<serde::de::value::Error>::new(&raw);
    Ok(T::deserialize(de).unwrap_or_else(|_| {
        tracing::warn!(
            variant = %raw,
            "Unknown settings enum variant in settings.json; falling back to default"
        );
        T::default()
    }))
}

/// Active planetary basemap selection persisted across sessions.
///
/// Unknown string variants encountered when loading settings written by a newer
/// version fall back to [`Self::BundledBlueMarble`] with a warning instead of
/// discarding the entire `settings.json` file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// Authentication mechanism for custom STAC API catalogs and their COG assets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StacAuthKind {
    #[default]
    None,
    Bearer,
    ApiKey,
    SasQuery,
}

impl StacAuthKind {
    pub(crate) const ALL: [Self; 4] = [Self::None, Self::Bearer, Self::ApiKey, Self::SasQuery];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::None => "None (Public)",
            Self::Bearer => "Bearer Token",
            Self::ApiKey => "API Key Header",
            Self::SasQuery => "SAS Query Token",
        }
    }
}

/// Maximum number of named Custom STAC profiles persisted in `settings.json`.
const MAX_STAC_PROFILES: usize = 16;

/// Single named Custom STAC API profile (endpoint, default collection, and
/// optional authentication credentials). Secrets and URL credentials/query
/// parameters are redacted in `Debug`.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct StacProfile {
    pub(crate) name: String,
    pub(crate) endpoint: String,
    pub(crate) collection: String,
    #[serde(deserialize_with = "deserialize_lossy_default")]
    pub(crate) auth_kind: StacAuthKind,
    pub(crate) auth_header_name: String,
    pub(crate) auth_secret: String,
}

impl Default for StacProfile {
    fn default() -> Self {
        Self {
            name: "Custom STAC".to_owned(),
            endpoint: String::new(),
            collection: String::new(),
            auth_kind: StacAuthKind::None,
            auth_header_name: "x-api-key".to_owned(),
            auth_secret: String::new(),
        }
    }
}

impl StacProfile {
    /// Switch the authentication mode, scrubbing any previous secret whenever
    /// the mode changes so credentials never leak across auth schemes.
    pub(crate) fn set_auth_kind(&mut self, kind: StacAuthKind) {
        if self.auth_kind != kind {
            self.auth_kind = kind;
            self.auth_secret.clear();
        }
    }

    pub(crate) fn normalize(&mut self, fallback_index: usize) {
        self.name = self.name.trim().to_owned();
        if self.name.is_empty() {
            self.name = format!("Custom Profile {}", fallback_index + 1);
        }
        self.endpoint = self.endpoint.trim().to_owned();
        self.collection = self.collection.trim().to_owned();
        self.auth_header_name = self.auth_header_name.trim().to_owned();
        if self.auth_kind == StacAuthKind::None {
            self.auth_secret.clear();
        } else {
            self.auth_secret = self.auth_secret.trim().to_owned();
        }
    }

    pub(crate) fn is_configured(&self) -> bool {
        !self.endpoint.trim().is_empty()
    }

    pub(crate) fn validation_error(&self) -> Option<String> {
        let ep = self.endpoint.trim();
        if !ep.is_empty() && !is_valid_http_endpoint(ep) {
            return Some(format!(
                "Profile '{}': STAC API URL must be a valid http:// or https:// URL.",
                self.name
            ));
        }
        if self.auth_kind == StacAuthKind::ApiKey {
            let hdr = self.auth_header_name.trim();
            if hdr.is_empty() || !is_valid_custom_header_name(hdr) {
                return Some(format!(
                    "Profile '{}': API key header name must be a valid non-protocol HTTP header (e.g. x-api-key).",
                    self.name
                ));
            }
        }
        None
    }

    pub(crate) fn to_stac_auth(&self) -> StacAuth {
        let secret = self.auth_secret.trim();
        if secret.is_empty() {
            return StacAuth::None;
        }
        match self.auth_kind {
            StacAuthKind::None => StacAuth::None,
            StacAuthKind::Bearer => StacAuth::Bearer(secret.to_owned()),
            StacAuthKind::ApiKey => StacAuth::ApiKey {
                header_name: self.auth_header_name.trim().to_owned(),
                value: secret.to_owned(),
            },
            StacAuthKind::SasQuery => StacAuth::SasQuery(secret.to_owned()),
        }
    }

    pub(crate) fn to_catalog_config(&self) -> StacCatalogConfig {
        StacCatalogConfig {
            endpoint: self.endpoint.trim().to_owned(),
            collection: self.collection.trim().to_owned(),
            auth: self.to_stac_auth(),
        }
    }
}

impl fmt::Debug for StacProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secret_label = if self.auth_secret.trim().is_empty() {
            "<empty>"
        } else {
            "<redacted>"
        };
        f.debug_struct("StacProfile")
            .field("name", &self.name)
            .field("endpoint", &sanitize_url_for_display(&self.endpoint))
            .field("collection", &self.collection)
            .field("auth_kind", &self.auth_kind)
            .field("auth_header_name", &self.auth_header_name)
            .field("auth_secret", &secret_label)
            .finish()
    }
}

/// Persisted Custom STAC API profiles and optional authentication settings.
/// Transparently migrates legacy v0.5 single-endpoint fields on load.
/// Secrets and URL credentials/query parameters are redacted in `Debug`.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct StacSettings {
    pub(crate) profiles: Vec<StacProfile>,
    pub(crate) active_profile: usize,
    #[serde(default, skip_serializing, rename = "custom_endpoint")]
    legacy_custom_endpoint: String,
    #[serde(default, skip_serializing, rename = "custom_collection")]
    legacy_custom_collection: String,
    #[serde(
        default,
        skip_serializing,
        rename = "auth_kind",
        deserialize_with = "deserialize_lossy_default"
    )]
    legacy_auth_kind: StacAuthKind,
    #[serde(default, skip_serializing, rename = "auth_header_name")]
    legacy_auth_header_name: String,
    #[serde(default, skip_serializing, rename = "auth_secret")]
    legacy_auth_secret: String,
}

impl Default for StacSettings {
    fn default() -> Self {
        Self {
            profiles: Vec::new(),
            active_profile: 0,
            legacy_custom_endpoint: String::new(),
            legacy_custom_collection: String::new(),
            legacy_auth_kind: StacAuthKind::None,
            legacy_auth_header_name: "x-api-key".to_owned(),
            legacy_auth_secret: String::new(),
        }
    }
}

impl StacSettings {
    #[cfg(test)]
    pub(crate) fn from_single_profile(mut profile: StacProfile) -> Self {
        profile.normalize(0);
        Self {
            profiles: vec![profile],
            active_profile: 0,
            ..Self::default()
        }
    }

    /// Migrate any legacy single-endpoint fields into `profiles`, trim
    /// whitespace, scrub unused secret material, and clamp `active_profile`.
    pub(crate) fn normalize(&mut self) {
        let has_legacy = !self.legacy_custom_endpoint.trim().is_empty()
            || !self.legacy_custom_collection.trim().is_empty()
            || self.legacy_auth_kind != StacAuthKind::None
            || !self.legacy_auth_secret.trim().is_empty();
        if self.profiles.is_empty() && has_legacy {
            self.profiles.push(StacProfile {
                name: "Custom STAC".to_owned(),
                endpoint: std::mem::take(&mut self.legacy_custom_endpoint),
                collection: std::mem::take(&mut self.legacy_custom_collection),
                auth_kind: self.legacy_auth_kind,
                auth_header_name: if self.legacy_auth_header_name.trim().is_empty() {
                    "x-api-key".to_owned()
                } else {
                    std::mem::take(&mut self.legacy_auth_header_name)
                },
                auth_secret: std::mem::take(&mut self.legacy_auth_secret),
            });
        }
        self.legacy_custom_endpoint.clear();
        self.legacy_custom_collection.clear();
        self.legacy_auth_kind = StacAuthKind::None;
        self.legacy_auth_header_name = "x-api-key".to_owned();
        self.legacy_auth_secret.clear();

        self.profiles.truncate(MAX_STAC_PROFILES);
        for (idx, profile) in self.profiles.iter_mut().enumerate() {
            profile.normalize(idx);
        }
        if self.profiles.is_empty() {
            self.active_profile = 0;
        } else if self.active_profile >= self.profiles.len() {
            self.active_profile = self.profiles.len() - 1;
        }
    }

    pub(crate) fn active_profile(&self) -> Option<&StacProfile> {
        self.profiles
            .get(self.active_profile)
            .or_else(|| self.profiles.first())
    }

    pub(crate) fn profile(&self, index: usize) -> Option<&StacProfile> {
        self.profiles.get(index)
    }

    /// Ensure at least one editable profile exists (used by the Settings UI)
    /// and return a mutable reference to the active profile.
    pub(crate) fn ensure_active_profile_mut(&mut self) -> &mut StacProfile {
        if self.profiles.is_empty() {
            self.profiles.push(StacProfile::default());
            self.active_profile = 0;
        } else if self.active_profile >= self.profiles.len() {
            self.active_profile = self.profiles.len() - 1;
        }
        &mut self.profiles[self.active_profile]
    }

    /// Add a new empty custom STAC profile (up to `MAX_STAC_PROFILES`) and
    /// select it.
    pub(crate) fn add_profile(&mut self) -> bool {
        if self.profiles.len() >= MAX_STAC_PROFILES {
            return false;
        }
        let mut next_num = 1usize;
        while self
            .profiles
            .iter()
            .any(|p| p.name.trim() == format!("Custom STAC {next_num}"))
        {
            next_num += 1;
        }
        self.profiles.push(StacProfile {
            name: format!("Custom STAC {next_num}"),
            ..StacProfile::default()
        });
        self.active_profile = self.profiles.len() - 1;
        true
    }

    /// Remove the currently selected profile, keeping `active_profile` valid.
    pub(crate) fn remove_active_profile(&mut self) -> bool {
        if self.profiles.is_empty() {
            return false;
        }
        let idx = self.active_profile.min(self.profiles.len() - 1);
        self.profiles.remove(idx);
        if self.profiles.is_empty() {
            self.active_profile = 0;
        } else if self.active_profile >= self.profiles.len() {
            self.active_profile = self.profiles.len() - 1;
        }
        true
    }

    /// Find the first configured profile whose endpoint shares the exact HTTP
    /// origin with `url` (checking `active_profile` first).
    pub(crate) fn profile_matching_origin(&self, url: &str) -> Option<&StacProfile> {
        if let Some(active) = self.active_profile()
            && active.is_configured()
            && veldmap_net::same_http_origin(&active.endpoint, url)
        {
            return Some(active);
        }
        self.profiles
            .iter()
            .find(|p| p.is_configured() && veldmap_net::same_http_origin(&p.endpoint, url))
    }

    /// Return a human-readable validation error if any configured Custom STAC
    /// profile has an invalid endpoint URL or API-key header name.
    pub(crate) fn validation_error(&self) -> Option<String> {
        self.profiles.iter().find_map(StacProfile::validation_error)
    }

    pub(crate) fn to_stac_auth(&self) -> StacAuth {
        self.active_profile()
            .map_or(StacAuth::None, StacProfile::to_stac_auth)
    }
}

impl fmt::Debug for StacSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StacSettings")
            .field("active_profile", &self.active_profile)
            .field("profiles", &self.profiles)
            .finish()
    }
}

/// Scoped editable slice of [`AppSettings`] for the unified Settings modal
/// (Basemap, Cache & GPU, and STAC / Cloud tabs).
///
/// Keeping this slice and its merge/validation rules alongside [`AppSettings`]
/// guarantees saving settings can never silently clobber unrelated fields
/// (such as `show_graticule`) or overwrite valid saved values with half-typed
/// invalid inputs while keeping `ui/` free of persistence rules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AppSettingsDraft {
    pub(crate) active_basemap: BasemapSelection,
    pub(crate) gibs_date: String,
    pub(crate) custom_xyz: CustomXyzSettings,
    pub(crate) disk_cache_max_mb: u64,
    pub(crate) gpu_tile_cache_max_mb: Option<u64>,
    pub(crate) stac: StacSettings,
}

impl AppSettingsDraft {
    pub(crate) fn from_settings(settings: &AppSettings) -> Self {
        Self {
            active_basemap: settings.active_basemap,
            gibs_date: settings.gibs_date.clone(),
            custom_xyz: settings.custom_xyz.clone(),
            disk_cache_max_mb: settings.disk_cache_max_mb,
            gpu_tile_cache_max_mb: settings.gpu_tile_cache_max_mb,
            stac: settings.stac.clone(),
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
        if self.stac.validation_error().is_none() {
            let mut stac = self.stac.clone();
            stac.normalize();
            target.stac = stac;
        }
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
    #[serde(deserialize_with = "deserialize_lossy_default")]
    pub(crate) active_basemap: BasemapSelection,
    pub(crate) gibs_date: String,
    pub(crate) custom_xyz: CustomXyzSettings,
    pub(crate) disk_cache_max_mb: u64,
    /// Optional user GPU tile cache cap in MiB (`None` = host-RAM-scaled auto cap).
    pub(crate) gpu_tile_cache_max_mb: Option<u64>,
    pub(crate) show_graticule: bool,
    pub(crate) stac: StacSettings,
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
            stac: StacSettings::default(),
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
        self.stac.normalize();
    }

    /// Optional user GPU tile cache cap override in bytes (`None` = auto
    /// host-RAM budget owned by `SceneState`).
    pub(crate) fn gpu_tile_cache_override_bytes(&self) -> Option<usize> {
        self.gpu_tile_cache_max_mb.map(|mb| {
            usize::try_from(
                mb.clamp(MIN_GPU_CACHE_MAX_MB, MAX_GPU_CACHE_MAX_MB)
                    .saturating_mul(1024 * 1024),
            )
            .unwrap_or(usize::MAX)
        })
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
    /// Unix so optional Custom XYZ API keys and Custom STAC auth secrets are
    /// owner-readable only).
    pub(super) fn save(&self, paths: &StoragePaths) -> Result<(), StorageError> {
        let mut normalized = self.clone();
        normalized.normalize();
        write_json_durable_private(&paths.settings_path(), &normalized)
    }

    #[cfg(test)]
    pub(crate) fn save_for_test(&self, paths: &StoragePaths) -> Result<(), StorageError> {
        self.save(paths)
    }
}

#[cfg(test)]
#[path = "settings_tests.rs"]
mod tests;
