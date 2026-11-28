//! Durable workstation session, recent-files, camera bookmarks, swipe layout,
//! and sanitized local/cloud/`BandMath` workspace persistence
//! (`config_dir/session.json` via `write_json_durable_private()`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use veldmap_core::geo::Camera;
use veldmap_net::sanitize_url_for_persistence;
use veldmap_raster::ExternalBandMeta;
use veldmap_storage::{
    StorageError, StoragePaths, read_json_or_default_with_backup, write_json_durable_private,
};

use crate::clamp_opacity;
use crate::layers::{
    BandAssetPersistEntry, CrsOverride, LayerStyleSpec, OverlayPersistEntry, StacCogAuthMode,
};

use super::wire::{
    SavedBandMathStyle, SavedCrsOverride, SavedExternalBandMeta, SavedLayerStyle,
    WireStacCogAuthMode, deserialize_lossy_opt,
};

/// Current on-disk schema version for `session.json` (tagged `open_layers`,
/// sanitized remote COGs & `BandMath`, vector `outline_only`, and `swipe`).
pub(super) const SESSION_SCHEMA_VERSION: u32 = 2;

/// Maximum number of recent files kept in `session.json`.
const MAX_RECENT_FILES: usize = 16;

/// Maximum number of camera bookmarks kept in `session.json`.
pub(crate) const MAX_BOOKMARKS: usize = 64;

fn default_true() -> bool {
    true
}

fn default_one_f32() -> f32 {
    1.0
}

fn default_half_f32() -> f32 {
    0.5
}

/// Saved local single-raster entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SavedLocalRasterEntry {
    pub(crate) path: PathBuf,
    #[serde(default = "default_true")]
    pub(crate) visible: bool,
    #[serde(default = "default_one_f32")]
    pub(crate) opacity: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    crs_override: Option<SavedCrsOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    style: Option<SavedLayerStyle>,
}

impl Default for SavedLocalRasterEntry {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            visible: true,
            opacity: 1.0,
            crs_override: None,
            style: None,
        }
    }
}

impl SavedLocalRasterEntry {
    pub(crate) fn crs_override(&self) -> Option<CrsOverride> {
        self.crs_override.as_ref().map(CrsOverride::from)
    }

    pub(crate) fn style(&self) -> Option<LayerStyleSpec> {
        self.style.as_ref().map(LayerStyleSpec::from)
    }
}

/// Saved sanitized remote Cloud-Optimized GeoTIFF entry (never stores query
/// strings, presigned tokens, `user:pass@`, or custom auth headers).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SavedRemoteCogEntry {
    pub(crate) url: String,
    pub(crate) title: String,
    #[serde(default, deserialize_with = "deserialize_lossy_opt")]
    auth_mode: Option<WireStacCogAuthMode>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) external_bands: Vec<SavedExternalBandMeta>,
    #[serde(default)]
    pub(crate) is_scl: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) stac_item_href: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) asset_key: Option<String>,
    #[serde(default = "default_true")]
    pub(crate) visible: bool,
    #[serde(default = "default_one_f32")]
    pub(crate) opacity: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    style: Option<SavedLayerStyle>,
}

impl Default for SavedRemoteCogEntry {
    fn default() -> Self {
        Self {
            url: String::new(),
            title: String::new(),
            auth_mode: Some(WireStacCogAuthMode::None),
            external_bands: Vec::new(),
            is_scl: false,
            stac_item_href: None,
            asset_key: None,
            visible: true,
            opacity: 1.0,
            style: None,
        }
    }
}

impl SavedRemoteCogEntry {
    pub(crate) fn auth_mode(&self) -> StacCogAuthMode {
        self.auth_mode
            .map_or(StacCogAuthMode::None, StacCogAuthMode::from)
    }

    pub(crate) fn style(&self) -> Option<LayerStyleSpec> {
        self.style.as_ref().map(LayerStyleSpec::from)
    }

    pub(crate) fn external_band_metas(&self) -> Vec<ExternalBandMeta> {
        self.external_bands
            .iter()
            .map(ExternalBandMeta::from)
            .collect()
    }

    fn sanitize(&mut self) -> bool {
        let Some(clean) = sanitize_url_for_persistence(&self.url) else {
            return false;
        };
        self.url = clean;
        self.stac_item_href = self
            .stac_item_href
            .as_deref()
            .and_then(sanitize_url_for_persistence);
        self.opacity = clamp_opacity(self.opacity);
        if self.title.trim().is_empty() {
            self.title = self.url.clone();
        }
        true
    }
}

/// Constituent local or sanitized remote band asset inside
/// [`SavedBandMathEntry`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub(crate) enum SavedBandAssetSource {
    Local {
        label: String,
        path: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        crs_override: Option<SavedCrsOverride>,
        #[serde(default)]
        band_index: u16,
    },
    Remote {
        label: String,
        url: String,
        #[serde(default, deserialize_with = "deserialize_lossy_opt")]
        auth_mode: Option<WireStacCogAuthMode>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        external_bands: Vec<SavedExternalBandMeta>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stac_item_href: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        asset_key: Option<String>,
        #[serde(default)]
        band_index: u16,
    },
}

impl SavedBandAssetSource {
    pub(crate) fn from_persist(entry: &BandAssetPersistEntry) -> Self {
        match entry {
            BandAssetPersistEntry::Local {
                label,
                path,
                crs_override,
                band_index,
            } => Self::Local {
                label: label.clone(),
                path: path.clone(),
                crs_override: crs_override.as_ref().map(SavedCrsOverride::from),
                band_index: *band_index,
            },
            BandAssetPersistEntry::Remote {
                label,
                canonical_url,
                auth_mode,
                external_bands,
                stac_item_href,
                asset_key,
                band_index,
            } => Self::Remote {
                label: label.clone(),
                url: canonical_url.clone(),
                auth_mode: Some(WireStacCogAuthMode::from(*auth_mode)),
                external_bands: external_bands
                    .iter()
                    .map(SavedExternalBandMeta::from)
                    .collect(),
                stac_item_href: stac_item_href.clone(),
                asset_key: asset_key.clone(),
                band_index: *band_index,
            },
        }
    }

    pub(crate) fn remote_auth_mode(&self) -> StacCogAuthMode {
        match self {
            Self::Remote { auth_mode, .. } => {
                auth_mode.map_or(StacCogAuthMode::None, StacCogAuthMode::from)
            }
            Self::Local { .. } => StacCogAuthMode::None,
        }
    }

    pub(crate) fn local_crs_override(&self) -> Option<CrsOverride> {
        match self {
            Self::Local { crs_override, .. } => crs_override.as_ref().map(CrsOverride::from),
            Self::Remote { .. } => None,
        }
    }

    fn sanitize(&mut self) -> bool {
        match self {
            Self::Local { path, .. } => !path.as_os_str().is_empty(),
            Self::Remote {
                url,
                stac_item_href,
                ..
            } => {
                let Some(clean) = sanitize_url_for_persistence(url) else {
                    return false;
                };
                *url = clean;
                *stac_item_href = stac_item_href
                    .as_deref()
                    .and_then(sanitize_url_for_persistence);
                true
            }
        }
    }
}

/// Saved multi-asset Band Math / RGB Composite recipe (supporting local,
/// remote, and heterogeneous local + remote assets).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SavedBandMathEntry {
    pub(crate) title: String,
    #[serde(default = "default_true")]
    pub(crate) visible: bool,
    #[serde(default = "default_one_f32")]
    pub(crate) opacity: f32,
    pub(crate) bands: Vec<SavedBandAssetSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) scl: Option<SavedBandAssetSource>,
    pub(crate) style: SavedBandMathStyle,
}

impl Default for SavedBandMathEntry {
    fn default() -> Self {
        Self {
            title: String::new(),
            visible: true,
            opacity: 1.0,
            bands: Vec::new(),
            scl: None,
            style: SavedBandMathStyle::default(),
        }
    }
}

impl SavedBandMathEntry {
    fn sanitize(&mut self) -> bool {
        self.opacity = clamp_opacity(self.opacity);
        self.style.sanitize();
        let spec = self.style.to_spec();
        if self.bands.len() != spec.formula.required_bands() {
            return false;
        }
        if !self.bands.iter_mut().all(SavedBandAssetSource::sanitize) {
            return false;
        }
        if let Some(scl) = &mut self.scl
            && !scl.sanitize()
        {
            self.scl = None;
        }
        if self.title.trim().is_empty() {
            self.title = spec.recipe_label;
        }
        true
    }
}

/// Tagged persisted raster layer entry in painter's order (bottom-to-top).
///
/// Serializes strictly as tagged JSON (`"kind": "local_raster" | "remote_cog" | "band_math"`),
/// while deserializing both the tagged v2 schema and legacy v1 untagged local
/// raster entries seamlessly.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SavedLayerEntry {
    LocalRaster(SavedLocalRasterEntry),
    RemoteCog(SavedRemoteCogEntry),
    BandMath(SavedBandMathEntry),
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SavedLayerEntryRef<'a> {
    LocalRaster(&'a SavedLocalRasterEntry),
    RemoteCog(&'a SavedRemoteCogEntry),
    BandMath(&'a SavedBandMathEntry),
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SavedLayerEntryTagged {
    LocalRaster(SavedLocalRasterEntry),
    RemoteCog(SavedRemoteCogEntry),
    BandMath(SavedBandMathEntry),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SavedLayerEntryWire {
    Tagged(SavedLayerEntryTagged),
    LegacyLocal(SavedLocalRasterEntry),
}

impl Serialize for SavedLayerEntry {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let tagged = match self {
            Self::LocalRaster(e) => SavedLayerEntryRef::LocalRaster(e),
            Self::RemoteCog(e) => SavedLayerEntryRef::RemoteCog(e),
            Self::BandMath(e) => SavedLayerEntryRef::BandMath(e),
        };
        tagged.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SavedLayerEntry {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = SavedLayerEntryWire::deserialize(deserializer)?;
        Ok(match wire {
            SavedLayerEntryWire::Tagged(SavedLayerEntryTagged::LocalRaster(e))
            | SavedLayerEntryWire::LegacyLocal(e) => Self::LocalRaster(e),
            SavedLayerEntryWire::Tagged(SavedLayerEntryTagged::RemoteCog(e)) => Self::RemoteCog(e),
            SavedLayerEntryWire::Tagged(SavedLayerEntryTagged::BandMath(e)) => Self::BandMath(e),
        })
    }
}

impl SavedLayerEntry {
    #[cfg(test)]
    pub(crate) fn new(
        path: PathBuf,
        visible: bool,
        opacity: f32,
        crs_override: Option<&CrsOverride>,
    ) -> Self {
        Self::with_style(path, visible, opacity, crs_override, None)
    }

    #[cfg(test)]
    pub(crate) fn with_style(
        path: PathBuf,
        visible: bool,
        opacity: f32,
        crs_override: Option<&CrsOverride>,
        style: Option<&LayerStyleSpec>,
    ) -> Self {
        Self::LocalRaster(SavedLocalRasterEntry {
            path,
            visible,
            opacity,
            crs_override: crs_override.map(SavedCrsOverride::from),
            style: style.map(SavedLayerStyle::from),
        })
    }

    pub(crate) fn from_persist(entry: OverlayPersistEntry) -> Self {
        match entry {
            OverlayPersistEntry::LocalRaster {
                path,
                visible,
                opacity,
                crs_override,
                style,
            } => Self::LocalRaster(SavedLocalRasterEntry {
                path,
                visible,
                opacity,
                crs_override: crs_override.as_ref().map(SavedCrsOverride::from),
                style: style.as_ref().map(SavedLayerStyle::from),
            }),
            OverlayPersistEntry::RemoteCog {
                canonical_url,
                title,
                auth_mode,
                external_bands,
                is_scl,
                stac_item_href,
                asset_key,
                visible,
                opacity,
                style,
            } => Self::RemoteCog(SavedRemoteCogEntry {
                url: canonical_url,
                title,
                auth_mode: Some(WireStacCogAuthMode::from(auth_mode)),
                external_bands: external_bands
                    .iter()
                    .map(SavedExternalBandMeta::from)
                    .collect(),
                is_scl,
                stac_item_href,
                asset_key,
                visible,
                opacity,
                style: style.as_ref().map(SavedLayerStyle::from),
            }),
            OverlayPersistEntry::BandMath {
                title,
                visible,
                opacity,
                bands,
                scl,
                style,
            } => Self::BandMath(SavedBandMathEntry {
                title,
                visible,
                opacity,
                bands: bands
                    .iter()
                    .map(SavedBandAssetSource::from_persist)
                    .collect(),
                scl: scl.as_ref().map(SavedBandAssetSource::from_persist),
                style: SavedBandMathStyle::from(&style),
            }),
        }
    }

    pub(crate) fn display_name(&self) -> String {
        match self {
            Self::LocalRaster(e) => crate::display_name_from_path(&e.path),
            Self::RemoteCog(e) => e.title.clone(),
            Self::BandMath(e) => e.title.clone(),
        }
    }

    pub(crate) fn display_detail(&self) -> String {
        match self {
            Self::LocalRaster(e) => e.path.display().to_string(),
            Self::RemoteCog(e) => e.url.clone(),
            Self::BandMath(e) => {
                let n = e.bands.len();
                let scl_part = if e.scl.is_some() { " + SCL" } else { "" };
                format!("Band Math ({n} bands{scl_part})")
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn crs_override(&self) -> Option<CrsOverride> {
        match self {
            Self::LocalRaster(e) => e.crs_override(),
            Self::RemoteCog(_) | Self::BandMath(_) => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn style(&self) -> Option<LayerStyleSpec> {
        match self {
            Self::LocalRaster(e) => e.style(),
            Self::RemoteCog(e) => e.style(),
            Self::BandMath(_) => None,
        }
    }

    pub(super) fn sanitize(&mut self) -> bool {
        match self {
            Self::LocalRaster(e) => {
                if e.path.as_os_str().is_empty() {
                    return false;
                }
                e.opacity = clamp_opacity(e.opacity);
                true
            }
            Self::RemoteCog(e) => e.sanitize(),
            Self::BandMath(e) => e.sanitize(),
        }
    }

    /// Transform persisted `auth_mode` on every remote asset inside this entry
    /// using caller-supplied policy `map_mode` (keeping STAC origin-fencing
    /// rules outside `crate::storage`).
    pub(crate) fn map_remote_auth_modes(
        &mut self,
        mut map_mode: impl FnMut(&str, StacCogAuthMode) -> StacCogAuthMode,
    ) {
        match self {
            Self::LocalRaster(_) => {}
            Self::RemoteCog(e) => {
                let next = map_mode(&e.url, e.auth_mode());
                e.auth_mode = Some(WireStacCogAuthMode::from(next));
            }
            Self::BandMath(e) => {
                for src in e.bands.iter_mut().chain(e.scl.iter_mut()) {
                    if let SavedBandAssetSource::Remote { url, auth_mode, .. } = src {
                        let current =
                            auth_mode.map_or(StacCogAuthMode::None, StacCogAuthMode::from);
                        let next = map_mode(url, current);
                        *auth_mode = Some(WireStacCogAuthMode::from(next));
                    }
                }
            }
        }
    }
}

/// Saved read-only GeoJSON vector layer entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SavedVectorEntry {
    pub(crate) path: PathBuf,
    #[serde(default = "default_true")]
    pub(crate) visible: bool,
    #[serde(default = "default_one_f32")]
    pub(crate) opacity: f32,
    #[serde(default)]
    pub(crate) outline_only: bool,
}

impl Default for SavedVectorEntry {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            visible: true,
            opacity: 1.0,
            outline_only: false,
        }
    }
}

/// Persisted split-screen swipe curtain configuration referencing raster layers
/// by index inside `open_layers`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SavedSwipeState {
    pub(crate) active: bool,
    #[serde(default = "default_half_f32")]
    pub(crate) split_t: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) left_layer_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) right_layer_index: Option<usize>,
}

impl Default for SavedSwipeState {
    fn default() -> Self {
        Self {
            active: false,
            split_t: 0.5,
            left_layer_index: None,
            right_layer_index: None,
        }
    }
}

impl SavedSwipeState {
    pub(super) fn normalize(&mut self, layer_count: usize) -> bool {
        if !self.split_t.is_finite() {
            self.split_t = 0.5;
        } else {
            self.split_t = self.split_t.clamp(0.0, 1.0);
        }
        if self.left_layer_index.is_some_and(|i| i >= layer_count) {
            self.left_layer_index = None;
        }
        if self.right_layer_index.is_some_and(|i| i >= layer_count) {
            self.right_layer_index = None;
        }
        self.active && layer_count > 0
    }
}

/// Persisted camera pose (shared by `last_camera` and [`CameraBookmark`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct SavedCameraPose {
    pub(crate) lat_deg: f64,
    pub(crate) lon_deg: f64,
    pub(crate) height_m: f64,
    #[serde(default)]
    pub(crate) heading_deg: f64,
}

impl SavedCameraPose {
    pub(crate) fn from_camera(camera: &Camera) -> Self {
        let (lat_deg, lon_deg) = camera.center();
        Self {
            lat_deg,
            lon_deg,
            height_m: camera.height_m(),
            heading_deg: camera.heading_deg(),
        }
    }

    pub(crate) fn to_camera(self) -> Camera {
        let mut camera = Camera::at(self.lat_deg, self.lon_deg, self.height_m);
        if self.heading_deg.abs() > 1e-6 {
            camera.rotate_heading(self.heading_deg.to_radians());
        }
        camera
    }

    pub(super) fn is_valid(&self) -> bool {
        self.lat_deg.is_finite()
            && (-90.0..=90.0).contains(&self.lat_deg)
            && self.lon_deg.is_finite()
            && (-360.0..=360.0).contains(&self.lon_deg)
            && self.height_m.is_finite()
            && self.height_m > 0.0
            && self.heading_deg.is_finite()
    }
}

/// Named camera bookmark.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct CameraBookmark {
    pub(crate) name: String,
    #[serde(flatten)]
    pub(crate) pose: SavedCameraPose,
}

/// Durable workstation session persisted in `session.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct AppSession {
    pub(crate) version: u32,
    pub(crate) open_layers: Vec<SavedLayerEntry>,
    pub(crate) open_vectors: Vec<SavedVectorEntry>,
    pub(crate) recent_files: Vec<PathBuf>,
    pub(crate) bookmarks: Vec<CameraBookmark>,
    pub(crate) last_camera: Option<SavedCameraPose>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) swipe: Option<SavedSwipeState>,
}

impl Default for AppSession {
    fn default() -> Self {
        Self {
            version: SESSION_SCHEMA_VERSION,
            open_layers: Vec::new(),
            open_vectors: Vec::new(),
            recent_files: Vec::new(),
            bookmarks: Vec::new(),
            last_camera: None,
            swipe: None,
        }
    }
}

/// Mutable borrow bundle of workspace fields shared between [`AppSession`] and
/// portable [`super::project::ProjectFile`] normalization.
pub(super) struct WorkspaceFieldsMut<'a> {
    pub(super) version: &'a mut u32,
    pub(super) open_layers: &'a mut Vec<SavedLayerEntry>,
    pub(super) open_vectors: &'a mut Vec<SavedVectorEntry>,
    pub(super) bookmarks: &'a mut Vec<CameraBookmark>,
    pub(super) last_camera: &'a mut Option<SavedCameraPose>,
    pub(super) swipe: &'a mut Option<SavedSwipeState>,
}

/// Shared workspace normalization applied identically to both [`AppSession`]
/// and portable [`super::project::ProjectFile`] instances.
pub(super) fn normalize_workspace_fields(fields: WorkspaceFieldsMut<'_>) {
    *fields.version = SESSION_SCHEMA_VERSION;
    fields.open_layers.retain_mut(SavedLayerEntry::sanitize);
    fields
        .open_vectors
        .retain(|v| !v.path.as_os_str().is_empty());
    for vec_layer in fields.open_vectors.iter_mut() {
        vec_layer.opacity = clamp_opacity(vec_layer.opacity);
    }
    fields
        .bookmarks
        .retain(|b| !b.name.trim().is_empty() && b.pose.is_valid());
    fields.bookmarks.truncate(MAX_BOOKMARKS);
    if let Some(cam) = *fields.last_camera
        && !cam.is_valid()
    {
        *fields.last_camera = None;
    }
    let layer_count = fields.open_layers.len();
    if let Some(mut sw) = *fields.swipe {
        if sw.normalize(layer_count) {
            *fields.swipe = Some(sw);
        } else {
            *fields.swipe = None;
        }
    }
}

impl AppSession {
    /// Normalize bounds, sanitize all remote URLs, deduplicate recent files,
    /// and drop invalid entries.
    pub(super) fn normalize(&mut self) {
        normalize_workspace_fields(WorkspaceFieldsMut {
            version: &mut self.version,
            open_layers: &mut self.open_layers,
            open_vectors: &mut self.open_vectors,
            bookmarks: &mut self.bookmarks,
            last_camera: &mut self.last_camera,
            swipe: &mut self.swipe,
        });
        let mut deduped = Vec::with_capacity(self.recent_files.len().min(MAX_RECENT_FILES));
        for p in self.recent_files.drain(..) {
            if !p.as_os_str().is_empty() && !deduped.contains(&p) {
                deduped.push(p);
                if deduped.len() >= MAX_RECENT_FILES {
                    break;
                }
            }
        }
        self.recent_files = deduped;
    }

    /// Push `path` to the front of `recent_files` (deduplicated, capped).
    pub(crate) fn record_recent_file(&mut self, path: &Path) {
        if path.as_os_str().is_empty() {
            return;
        }
        self.recent_files.retain(|p| p != path);
        self.recent_files.insert(0, path.to_path_buf());
        self.recent_files.truncate(MAX_RECENT_FILES);
    }

    /// Append a camera bookmark (bounded to `MAX_BOOKMARKS` during normalize).
    pub(crate) fn add_bookmark(&mut self, name: String, pose: SavedCameraPose) {
        self.bookmarks.push(CameraBookmark { name, pose });
    }

    /// Remove the bookmark at `index`, if in bounds.
    pub(crate) fn remove_bookmark(&mut self, index: usize) {
        if index < self.bookmarks.len() {
            self.bookmarks.remove(index);
        }
    }

    /// Load `session.json` from `paths` (backing up corrupt files to `.corrupt`).
    pub(super) fn load_or_default(paths: &StoragePaths) -> Self {
        let mut session: Self = read_json_or_default_with_backup(&paths.session_path());
        session.normalize();
        session
    }

    /// Persist a normalized copy to `session.json` via durable private atomic write.
    pub(super) fn save(&self, paths: &StoragePaths) -> Result<(), StorageError> {
        let mut copy = self.clone();
        copy.normalize();
        write_json_durable_private(&paths.session_path(), &copy)
    }
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
