//! Minimal durable local workstation session, recent-files, and camera
//! bookmarks persistence (`config_dir/session.json` via
//! `write_json_durable_private()`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer, Serialize};
use veldmap_core::geo::{Camera, CustomTmParams};
use veldmap_storage::{
    StorageError, StoragePaths, read_json_or_default_with_backup, write_json_durable_private,
};

use crate::clamp_opacity;
use crate::colormap::{CategoricalPaletteId, ColormapId};
use crate::layers::{CrsOverride, LayerStyleSpec, StretchPreset, TrackKind};

/// Current on-disk schema version for `session.json`.
const SESSION_SCHEMA_VERSION: u32 = 1;

/// Maximum number of recent files kept in `session.json`.
const MAX_RECENT_FILES: usize = 16;

/// Maximum number of camera bookmarks kept in `session.json`.
pub(crate) const MAX_BOOKMARKS: usize = 64;

/// Forward-compatible `Option<T>` deserializer for string-encoded style enums:
/// unknown variants written by newer versions fall back to `None` instead of
/// failing the entire `session.json` load into `.corrupt`.
fn deserialize_lossy_opt<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: for<'a> Deserialize<'a>,
{
    let raw = Option::<String>::deserialize(deserializer)?;
    Ok(raw.and_then(|s| {
        let de = serde::de::value::StrDeserializer::<serde::de::value::Error>::new(&s);
        T::deserialize(de).ok()
    }))
}

macro_rules! wire_enum {
    ($wire:ident <-> $domain:ident { $($variant:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        enum $wire {
            $($variant,)+
        }

        impl From<$domain> for $wire {
            fn from(value: $domain) -> Self {
                match value {
                    $($domain::$variant => Self::$variant,)+
                }
            }
        }

        impl From<$wire> for $domain {
            fn from(value: $wire) -> Self {
                match value {
                    $($wire::$variant => Self::$variant,)+
                }
            }
        }
    };
}

wire_enum!(WireTrackKind <-> TrackKind {
    Display,
    Scientific,
    Categorical,
});

wire_enum!(WireStretchPreset <-> StretchPreset {
    Percentile2_98,
    MinMax,
    Log10Db,
    Custom,
});

wire_enum!(WireColormapId <-> ColormapId {
    Grayscale,
    Viridis,
    Inferno,
    Plasma,
    Turbo,
    RdBu,
    Terrain,
});

wire_enum!(WireCategoricalPaletteId <-> CategoricalPaletteId {
    SourceColorMap,
    Sentinel2Scl,
    DistinctClasses,
});

/// Forward-compatible `session.json` wire proxy for [`LayerStyleSpec`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct SavedLayerStyle {
    band_index: Option<u16>,
    #[serde(deserialize_with = "deserialize_lossy_opt")]
    track: Option<WireTrackKind>,
    #[serde(deserialize_with = "deserialize_lossy_opt")]
    stretch_preset: Option<WireStretchPreset>,
    stretch_min: Option<f64>,
    stretch_max: Option<f64>,
    #[serde(default)]
    custom_stretch_range: bool,
    #[serde(deserialize_with = "deserialize_lossy_opt")]
    colormap: Option<WireColormapId>,
    invert_colormap: bool,
    #[serde(deserialize_with = "deserialize_lossy_opt")]
    categorical_palette: Option<WireCategoricalPaletteId>,
}

impl From<&LayerStyleSpec> for SavedLayerStyle {
    fn from(spec: &LayerStyleSpec) -> Self {
        Self {
            band_index: spec.band_index,
            track: spec.track.map(WireTrackKind::from),
            stretch_preset: spec.stretch_preset.map(WireStretchPreset::from),
            stretch_min: spec.stretch_min,
            stretch_max: spec.stretch_max,
            custom_stretch_range: spec.custom_stretch_range,
            colormap: spec.colormap.map(WireColormapId::from),
            invert_colormap: spec.invert_colormap,
            categorical_palette: spec.categorical_palette.map(WireCategoricalPaletteId::from),
        }
    }
}

impl From<&SavedLayerStyle> for LayerStyleSpec {
    fn from(saved: &SavedLayerStyle) -> Self {
        Self {
            band_index: saved.band_index,
            track: saved.track.map(TrackKind::from),
            stretch_preset: saved.stretch_preset.map(StretchPreset::from),
            stretch_min: saved.stretch_min,
            stretch_max: saved.stretch_max,
            custom_stretch_range: saved.custom_stretch_range,
            colormap: saved.colormap.map(ColormapId::from),
            invert_colormap: saved.invert_colormap,
            categorical_palette: saved.categorical_palette.map(CategoricalPaletteId::from),
        }
    }
}

/// Minimal serde proxy preserving the exact flat `session.json` wire format
/// (`{"kind": "...", ...}`) for [`CrsOverride`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SavedCrsOverride {
    Epsg { code: u32 },
    Wkt { wkt: String },
    Proj { proj: String },
    CustomTm(CustomTmParams),
}

impl From<&CrsOverride> for SavedCrsOverride {
    fn from(ov: &CrsOverride) -> Self {
        match ov {
            CrsOverride::Epsg(code) => Self::Epsg { code: *code },
            CrsOverride::Wkt(wkt) => Self::Wkt { wkt: wkt.clone() },
            CrsOverride::Proj(proj) => Self::Proj { proj: proj.clone() },
            CrsOverride::CustomTm(params) => Self::CustomTm(params.clone()),
        }
    }
}

impl From<&SavedCrsOverride> for CrsOverride {
    fn from(saved: &SavedCrsOverride) -> Self {
        match saved {
            SavedCrsOverride::Epsg { code } => Self::Epsg(*code),
            SavedCrsOverride::Wkt { wkt } => Self::Wkt(wkt.clone()),
            SavedCrsOverride::Proj { proj } => Self::Proj(proj.clone()),
            SavedCrsOverride::CustomTm(params) => Self::CustomTm(params.clone()),
        }
    }
}

/// Saved local raster layer entry in painter's order (bottom-to-top).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SavedLayerEntry {
    pub(crate) path: PathBuf,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    crs_override: Option<SavedCrsOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    style: Option<SavedLayerStyle>,
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

    pub(crate) fn with_style(
        path: PathBuf,
        visible: bool,
        opacity: f32,
        crs_override: Option<&CrsOverride>,
        style: Option<&LayerStyleSpec>,
    ) -> Self {
        Self {
            path,
            visible,
            opacity,
            crs_override: crs_override.map(SavedCrsOverride::from),
            style: style.map(SavedLayerStyle::from),
        }
    }

    pub(crate) fn crs_override(&self) -> Option<CrsOverride> {
        self.crs_override.as_ref().map(CrsOverride::from)
    }

    pub(crate) fn style(&self) -> Option<LayerStyleSpec> {
        self.style.as_ref().map(LayerStyleSpec::from)
    }
}

impl Default for SavedLayerEntry {
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

/// Saved read-only GeoJSON vector layer entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SavedVectorEntry {
    pub(crate) path: PathBuf,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
}

impl Default for SavedVectorEntry {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            visible: true,
            opacity: 1.0,
        }
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

    fn is_valid(&self) -> bool {
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

/// Durable minimal local workstation session persisted in `session.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct AppSession {
    pub(crate) version: u32,
    pub(crate) open_layers: Vec<SavedLayerEntry>,
    pub(crate) open_vectors: Vec<SavedVectorEntry>,
    pub(crate) recent_files: Vec<PathBuf>,
    pub(crate) bookmarks: Vec<CameraBookmark>,
    pub(crate) last_camera: Option<SavedCameraPose>,
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
        }
    }
}

impl AppSession {
    /// Normalize bounds, deduplicate recent files, and drop invalid entries.
    pub(super) fn normalize(&mut self) {
        self.version = SESSION_SCHEMA_VERSION;
        self.open_layers.retain(|l| !l.path.as_os_str().is_empty());
        for layer in &mut self.open_layers {
            layer.opacity = clamp_opacity(layer.opacity);
        }
        self.open_vectors.retain(|v| !v.path.as_os_str().is_empty());
        for vec_layer in &mut self.open_vectors {
            vec_layer.opacity = clamp_opacity(vec_layer.opacity);
        }
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
        self.bookmarks
            .retain(|b| !b.name.trim().is_empty() && b.pose.is_valid());
        self.bookmarks.truncate(MAX_BOOKMARKS);
        if let Some(cam) = self.last_camera
            && !cam.is_valid()
        {
            self.last_camera = None;
        }
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
