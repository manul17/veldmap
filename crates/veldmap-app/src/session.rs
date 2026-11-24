//! Minimal durable local workstation session, recent-files, and camera
//! bookmarks persistence (`config_dir/session.json` via
//! `write_json_durable_private()`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use veldmap_core::geo::{CrsSpec, CustomTmParams, DatumPreset, LinearUnit};
use veldmap_storage::{
    StorageError, StoragePaths, read_json_or_default_with_backup, write_json_durable_private,
};

/// Current on-disk schema version for `session.json`.
pub const SESSION_SCHEMA_VERSION: u32 = 1;

/// Maximum number of recent files kept in `session.json`.
pub const MAX_RECENT_FILES: usize = 16;

/// Maximum number of camera bookmarks kept in `session.json`.
pub const MAX_BOOKMARKS: usize = 64;

/// Serializable datum choice for custom Transverse Mercator overrides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SavedCustomTmDatum {
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

/// Serializable CRS override stored alongside a saved raster layer entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SavedCrsOverride {
    Epsg {
        code: u32,
    },
    Wkt {
        wkt: String,
    },
    Proj {
        proj: String,
    },
    CustomTm {
        name: String,
        #[serde(default)]
        datum: SavedCustomTmDatum,
        central_meridian_deg: f64,
        origin_lat_deg: f64,
        scale: f64,
        false_easting_m: f64,
        false_northing_m: f64,
        unit: SavedLinearUnit,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SavedLinearUnit {
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

impl SavedCrsOverride {
    /// Resolve back into a live [`CrsSpec`].
    pub fn to_crs_spec(&self) -> Result<CrsSpec, String> {
        match self {
            Self::Epsg { code } => CrsSpec::from_epsg(*code).map_err(|e| e.to_string()),
            Self::Wkt { wkt } => CrsSpec::from_wkt(wkt).map_err(|e| e.to_string()),
            Self::Proj { proj } => CrsSpec::from_user_string(proj).map_err(|e| e.to_string()),
            Self::CustomTm {
                name,
                datum,
                central_meridian_deg,
                origin_lat_deg,
                scale,
                false_easting_m,
                false_northing_m,
                unit,
            } => {
                let params = CustomTmParams {
                    name: name.clone(),
                    datum: (*datum).into(),
                    central_meridian_deg: *central_meridian_deg,
                    origin_lat_deg: *origin_lat_deg,
                    scale: *scale,
                    false_easting_m: *false_easting_m,
                    false_northing_m: *false_northing_m,
                    unit: (*unit).into(),
                };
                let proj = params.to_projection().map_err(|e| e.to_string())?;
                Ok(CrsSpec::Projected(proj))
            }
        }
    }
}

/// Saved local raster layer entry in painter's order (bottom-to-top).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SavedLayerEntry {
    pub path: PathBuf,
    pub visible: bool,
    pub opacity: f32,
    pub crs_override: Option<SavedCrsOverride>,
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
pub struct SavedVectorEntry {
    pub path: PathBuf,
    pub visible: bool,
    pub opacity: f32,
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

/// Named camera bookmark.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraBookmark {
    pub name: String,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub height_m: f64,
    #[serde(default)]
    pub heading_deg: f64,
}

/// Persisted last-viewed camera pose.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedCameraPose {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub height_m: f64,
    #[serde(default)]
    pub heading_deg: f64,
}

/// Durable minimal local workstation session persisted in `session.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSession {
    pub version: u32,
    pub open_layers: Vec<SavedLayerEntry>,
    pub open_vectors: Vec<SavedVectorEntry>,
    pub recent_files: Vec<PathBuf>,
    pub bookmarks: Vec<CameraBookmark>,
    pub last_camera: Option<SavedCameraPose>,
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

fn is_valid_camera_pose(lat_deg: f64, lon_deg: f64, height_m: f64, heading_deg: f64) -> bool {
    lat_deg.is_finite()
        && (-90.0..=90.0).contains(&lat_deg)
        && lon_deg.is_finite()
        && (-360.0..=360.0).contains(&lon_deg)
        && height_m.is_finite()
        && height_m > 0.0
        && heading_deg.is_finite()
}

impl AppSession {
    /// Normalize bounds, deduplicate recent files, and drop invalid entries.
    pub fn normalize(&mut self) {
        self.version = SESSION_SCHEMA_VERSION;
        self.open_layers.retain(|l| !l.path.as_os_str().is_empty());
        for layer in &mut self.open_layers {
            layer.opacity = if layer.opacity.is_finite() {
                layer.opacity.clamp(0.0, 1.0)
            } else {
                1.0
            };
        }
        self.open_vectors.retain(|v| !v.path.as_os_str().is_empty());
        for vec_layer in &mut self.open_vectors {
            vec_layer.opacity = if vec_layer.opacity.is_finite() {
                vec_layer.opacity.clamp(0.0, 1.0)
            } else {
                1.0
            };
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
        self.bookmarks.retain(|b| {
            !b.name.trim().is_empty()
                && is_valid_camera_pose(b.lat_deg, b.lon_deg, b.height_m, b.heading_deg)
        });
        self.bookmarks.truncate(MAX_BOOKMARKS);
        if let Some(cam) = self.last_camera
            && !is_valid_camera_pose(cam.lat_deg, cam.lon_deg, cam.height_m, cam.heading_deg)
        {
            self.last_camera = None;
        }
    }

    /// Push `path` to the front of `recent_files` (deduplicated, capped).
    pub fn record_recent_file(&mut self, path: &Path) {
        if path.as_os_str().is_empty() {
            return;
        }
        self.recent_files.retain(|p| p != path);
        self.recent_files.insert(0, path.to_path_buf());
        self.recent_files.truncate(MAX_RECENT_FILES);
    }

    /// Load `session.json` from `paths` (backing up corrupt files to `.corrupt`).
    pub fn load_or_default(paths: &StoragePaths) -> Self {
        let mut session: Self = read_json_or_default_with_backup(&paths.session_path());
        session.normalize();
        session
    }

    /// Persist a normalized copy to `session.json` via durable private atomic write.
    pub fn save(&self, paths: &StoragePaths) -> Result<(), StorageError> {
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
            crs_override: Some(SavedCrsOverride::CustomTm {
                name: "MSK-61 Custom".to_string(),
                datum: SavedCustomTmDatum::Pulkovo1942,
                central_meridian_deg: 37.983_333_333_33,
                origin_lat_deg: 0.0,
                scale: 1.0,
                false_easting_m: 1_300_000.0,
                false_northing_m: -4_811_057.628,
                unit: SavedLinearUnit::Metre,
            }),
        });
        session.open_vectors.push(SavedVectorEntry {
            path: PathBuf::from("/tmp/parcels.geojson"),
            visible: true,
            opacity: 0.9,
        });
        session.record_recent_file(Path::new("/tmp/a.tif"));
        session.record_recent_file(Path::new("/tmp/b.tif"));
        session.record_recent_file(Path::new("/tmp/a.tif"));
        session.bookmarks.push(CameraBookmark {
            name: "Rostov".to_string(),
            lat_deg: 47.23,
            lon_deg: 39.72,
            height_m: 15_000.0,
            heading_deg: 12.5,
        });
        session.last_camera = Some(SavedCameraPose {
            lat_deg: 47.23,
            lon_deg: 39.72,
            height_m: 15_000.0,
            heading_deg: 12.5,
        });

        session.save(&paths).expect("save session");
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
            lat_deg: 120.0,
            lon_deg: 0.0,
            height_m: 1000.0,
            heading_deg: 0.0,
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
