//! Minimal durable local workstation session, recent-files, and camera
//! bookmarks persistence (`config_dir/session.json` via
//! `write_json_durable_private()`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
#[cfg(test)]
use veldmap_core::geo::CrsSpec;
use veldmap_core::geo::{Camera, CustomTmParams, DatumPreset, LinearUnit};
use veldmap_storage::{
    StorageError, StoragePaths, read_json_or_default_with_backup, write_json_durable_private,
};

use crate::clamp_opacity;
use crate::crs_override::CrsOverride;

/// Current on-disk schema version for `session.json`.
const SESSION_SCHEMA_VERSION: u32 = 1;

/// Maximum number of recent files kept in `session.json`.
const MAX_RECENT_FILES: usize = 16;

/// Maximum number of camera bookmarks kept in `session.json`.
pub(crate) const MAX_BOOKMARKS: usize = 64;

/// Serializable datum choice for custom Transverse Mercator overrides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SavedCustomTmDatum {
    #[default]
    Wgs84,
    Pulkovo1942,
}

impl From<DatumPreset> for SavedCustomTmDatum {
    fn from(d: DatumPreset) -> Self {
        match d {
            DatumPreset::Wgs84 => Self::Wgs84,
            DatumPreset::Pulkovo1942 => Self::Pulkovo1942,
        }
    }
}

impl From<SavedCustomTmDatum> for DatumPreset {
    fn from(d: SavedCustomTmDatum) -> Self {
        match d {
            SavedCustomTmDatum::Wgs84 => Self::Wgs84,
            SavedCustomTmDatum::Pulkovo1942 => Self::Pulkovo1942,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SavedLinearUnit {
    #[default]
    Metre,
    InternationalFoot,
    UsSurveyFoot,
}

impl From<LinearUnit> for SavedLinearUnit {
    fn from(u: LinearUnit) -> Self {
        match u {
            LinearUnit::Metre => Self::Metre,
            LinearUnit::InternationalFoot => Self::InternationalFoot,
            LinearUnit::UsSurveyFoot => Self::UsSurveyFoot,
        }
    }
}

impl From<SavedLinearUnit> for LinearUnit {
    fn from(u: SavedLinearUnit) -> Self {
        match u {
            SavedLinearUnit::Metre => Self::Metre,
            SavedLinearUnit::InternationalFoot => Self::InternationalFoot,
            SavedLinearUnit::UsSurveyFoot => Self::UsSurveyFoot,
        }
    }
}

/// Serializable custom Transverse Mercator parameters matching [`CustomTmParams`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct SavedCustomTmParams {
    name: String,
    #[serde(default)]
    datum: SavedCustomTmDatum,
    central_meridian_deg: f64,
    origin_lat_deg: f64,
    scale: f64,
    false_easting_m: f64,
    false_northing_m: f64,
    unit: SavedLinearUnit,
}

impl From<&CustomTmParams> for SavedCustomTmParams {
    fn from(p: &CustomTmParams) -> Self {
        Self {
            name: p.name.clone(),
            datum: p.datum.into(),
            central_meridian_deg: p.central_meridian_deg,
            origin_lat_deg: p.origin_lat_deg,
            scale: p.scale,
            false_easting_m: p.false_easting_m,
            false_northing_m: p.false_northing_m,
            unit: p.unit.into(),
        }
    }
}

impl From<&SavedCustomTmParams> for CustomTmParams {
    fn from(p: &SavedCustomTmParams) -> Self {
        Self {
            name: p.name.clone(),
            datum: p.datum.into(),
            central_meridian_deg: p.central_meridian_deg,
            origin_lat_deg: p.origin_lat_deg,
            scale: p.scale,
            false_easting_m: p.false_easting_m,
            false_northing_m: p.false_northing_m,
            unit: p.unit.into(),
        }
    }
}

/// Serializable CRS override stored alongside a saved raster layer entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SavedCrsOverride {
    Epsg { code: u32 },
    Wkt { wkt: String },
    Proj { proj: String },
    CustomTm(SavedCustomTmParams),
}

impl From<&CrsOverride> for SavedCrsOverride {
    fn from(ov: &CrsOverride) -> Self {
        match ov {
            CrsOverride::Epsg(code) => Self::Epsg { code: *code },
            CrsOverride::Wkt(wkt) => Self::Wkt { wkt: wkt.clone() },
            CrsOverride::Proj(proj) => Self::Proj { proj: proj.clone() },
            CrsOverride::CustomTm(params) => Self::CustomTm(SavedCustomTmParams::from(params)),
        }
    }
}

impl From<SavedCrsOverride> for CrsOverride {
    fn from(saved: SavedCrsOverride) -> Self {
        match saved {
            SavedCrsOverride::Epsg { code } => Self::Epsg(code),
            SavedCrsOverride::Wkt { wkt } => Self::Wkt(wkt),
            SavedCrsOverride::Proj { proj } => Self::Proj(proj),
            SavedCrsOverride::CustomTm(params) => Self::CustomTm(CustomTmParams::from(&params)),
        }
    }
}

impl From<&SavedCrsOverride> for CrsOverride {
    fn from(saved: &SavedCrsOverride) -> Self {
        match saved {
            SavedCrsOverride::Epsg { code } => Self::Epsg(*code),
            SavedCrsOverride::Wkt { wkt } => Self::Wkt(wkt.clone()),
            SavedCrsOverride::Proj { proj } => Self::Proj(proj.clone()),
            SavedCrsOverride::CustomTm(params) => Self::CustomTm(CustomTmParams::from(params)),
        }
    }
}

#[cfg(test)]
impl SavedCrsOverride {
    /// Resolve back into a live [`CrsSpec`].
    fn to_crs_spec(&self) -> Result<CrsSpec, String> {
        CrsOverride::from(self).to_crs_spec()
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
}

impl SavedLayerEntry {
    pub(crate) fn new(
        path: PathBuf,
        visible: bool,
        opacity: f32,
        crs_override: Option<&CrsOverride>,
    ) -> Self {
        Self {
            path,
            visible,
            opacity,
            crs_override: crs_override.map(SavedCrsOverride::from),
        }
    }

    pub(crate) fn crs_override(&self) -> Option<CrsOverride> {
        self.crs_override.as_ref().map(CrsOverride::from)
    }
}

impl Default for SavedLayerEntry {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            visible: true,
            opacity: 1.0,
            crs_override: None,
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

    pub(crate) fn is_valid(&self) -> bool {
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
    pub(crate) fn normalize(&mut self) {
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

    /// Load `session.json` from `paths` (backing up corrupt files to `.corrupt`).
    pub(crate) fn load_or_default(paths: &StoragePaths) -> Self {
        let mut session: Self = read_json_or_default_with_backup(&paths.session_path());
        session.normalize();
        session
    }

    /// Persist a normalized copy to `session.json` via durable private atomic write.
    pub(crate) fn save(&self, paths: &StoragePaths) -> Result<(), StorageError> {
        let mut copy = self.clone();
        copy.normalize();
        write_json_durable_private(&paths.session_path(), &copy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use veldmap_storage::test_support::TestTempDir;

    #[test]
    fn session_persists_layers_crs_overrides_recent_files_and_bookmarks() {
        let tmp = TestTempDir::new("session-roundtrip");
        let paths = StoragePaths::with_root(tmp.path());

        let mut session = AppSession::default();
        session.open_layers.push(SavedLayerEntry {
            path: PathBuf::from("/tmp/ortho.tif"),
            visible: true,
            opacity: 0.75,
            crs_override: Some(SavedCrsOverride::CustomTm(SavedCustomTmParams {
                name: "MSK-61 Custom".to_string(),
                datum: SavedCustomTmDatum::Pulkovo1942,
                central_meridian_deg: 37.983_333_333_33,
                origin_lat_deg: 0.0,
                scale: 1.0,
                false_easting_m: 1_300_000.0,
                false_northing_m: -4_811_057.628,
                unit: SavedLinearUnit::Metre,
            })),
        });
        session.open_vectors.push(SavedVectorEntry {
            path: PathBuf::from("/tmp/parcels.geojson"),
            visible: true,
            opacity: 0.9,
        });
        session.record_recent_file(Path::new("/tmp/a.tif"));
        session.record_recent_file(Path::new("/tmp/b.tif"));
        session.record_recent_file(Path::new("/tmp/a.tif"));
        let pose = SavedCameraPose {
            lat_deg: 47.23,
            lon_deg: 39.72,
            height_m: 15_000.0,
            heading_deg: 12.5,
        };
        session.bookmarks.push(CameraBookmark {
            name: "Rostov".to_string(),
            pose,
        });
        session.last_camera = Some(pose);

        session.save(&paths).expect("save session");
        // Verify on-disk JSON stays flat for `custom_tm` and `CameraBookmark`.
        let raw = std::fs::read_to_string(paths.session_path()).expect("read raw json");
        assert!(raw.contains(r#""kind": "custom_tm""#));
        assert!(raw.contains(r#""central_meridian_deg""#));
        assert!(raw.contains(r#""name": "Rostov""#));
        assert!(!raw.contains(r#""pose""#));

        let loaded = AppSession::load_or_default(&paths);
        assert_eq!(loaded, session);
        assert_eq!(
            loaded.recent_files,
            vec![PathBuf::from("/tmp/a.tif"), PathBuf::from("/tmp/b.tif")]
        );
        let spec = loaded.open_layers[0]
            .crs_override
            .as_ref()
            .expect("crs override")
            .to_crs_spec()
            .expect("valid custom TM");
        assert_eq!(spec.label(), "MSK-61 Custom");
    }

    #[test]
    fn session_normalize_rejects_out_of_range_poses_and_nan_opacity() {
        let mut session = AppSession::default();
        session.open_layers.push(SavedLayerEntry {
            path: PathBuf::from("/tmp/a.tif"),
            visible: true,
            opacity: f32::NAN,
            crs_override: None,
        });
        session.bookmarks.push(CameraBookmark {
            name: "Invalid Lat".to_string(),
            pose: SavedCameraPose {
                lat_deg: 120.0,
                lon_deg: 0.0,
                height_m: 1000.0,
                heading_deg: 0.0,
            },
        });
        session.last_camera = Some(SavedCameraPose {
            lat_deg: -95.0,
            lon_deg: 0.0,
            height_m: 1000.0,
            heading_deg: 0.0,
        });
        session.normalize();
        assert_eq!(session.open_layers[0].opacity, 1.0);
        assert!(session.bookmarks.is_empty());
        assert!(session.last_camera.is_none());
    }
}
