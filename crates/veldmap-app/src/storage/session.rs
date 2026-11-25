//! Minimal durable local workstation session, recent-files, and camera
//! bookmarks persistence (`config_dir/session.json` via
//! `write_json_durable_private()`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use veldmap_core::geo::{Camera, CustomTmParams, DatumPreset, LinearUnit};
use veldmap_storage::{
    StorageError, StoragePaths, read_json_or_default_with_backup, write_json_durable_private,
};

use crate::clamp_opacity;
use crate::layers::CrsOverride;

/// Current on-disk schema version for `session.json`.
const SESSION_SCHEMA_VERSION: u32 = 1;

/// Maximum number of recent files kept in `session.json`.
const MAX_RECENT_FILES: usize = 16;

/// Maximum number of camera bookmarks kept in `session.json`.
pub(crate) const MAX_BOOKMARKS: usize = 64;

#[derive(Serialize, Deserialize)]
#[serde(remote = "DatumPreset", rename_all = "snake_case")]
enum DatumPresetDef {
    Wgs84,
    Pulkovo1942,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "LinearUnit", rename_all = "snake_case")]
enum LinearUnitDef {
    Metre,
    InternationalFoot,
    UsSurveyFoot,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "CustomTmParams")]
struct CustomTmParamsDef {
    name: String,
    #[serde(default, with = "DatumPresetDef")]
    datum: DatumPreset,
    central_meridian_deg: f64,
    origin_lat_deg: f64,
    scale: f64,
    false_easting_m: f64,
    false_northing_m: f64,
    #[serde(with = "LinearUnitDef")]
    unit: LinearUnit,
}

/// Minimal serde proxy preserving the exact flat `session.json` wire format
/// (`{"kind": "...", ...}`) for [`CrsOverride`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SavedCrsOverride {
    Epsg { code: u32 },
    Wkt { wkt: String },
    Proj { proj: String },
    CustomTm(#[serde(with = "CustomTmParamsDef")] CustomTmParams),
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
    fn session_persists_all_crs_override_variants_flat_json_recent_files_and_bookmarks() {
        let tmp = TestTempDir::new("session-roundtrip");
        let paths = StoragePaths::with_root(tmp.path());

        let epsg_ov = CrsOverride::Epsg(32637);
        let wkt_ov = CrsOverride::Wkt(
            concat!(
                "GEOGCS[\"WGS 84\",",
                "DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],",
                "PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]"
            )
            .to_owned(),
        );
        let proj_ov =
            CrsOverride::Proj("+proj=utm +zone=37 +datum=WGS84 +units=m +no_defs".to_owned());
        let custom_tm_ov = CrsOverride::CustomTm(CustomTmParams {
            name: "MSK-61 Custom".to_string(),
            datum: DatumPreset::Pulkovo1942,
            central_meridian_deg: 37.983_333_333_33,
            origin_lat_deg: 0.0,
            scale: 1.0,
            false_easting_m: 1_300_000.0,
            false_northing_m: -4_811_057.628,
            unit: LinearUnit::Metre,
        });

        let mut session = AppSession::default();
        session.open_layers.push(SavedLayerEntry::new(
            PathBuf::from("/tmp/epsg.tif"),
            true,
            1.0,
            Some(&epsg_ov),
        ));
        session.open_layers.push(SavedLayerEntry::new(
            PathBuf::from("/tmp/wkt.tif"),
            false,
            0.5,
            Some(&wkt_ov),
        ));
        session.open_layers.push(SavedLayerEntry::new(
            PathBuf::from("/tmp/proj.tif"),
            true,
            0.8,
            Some(&proj_ov),
        ));
        session.open_layers.push(SavedLayerEntry::new(
            PathBuf::from("/tmp/ortho.tif"),
            true,
            0.75,
            Some(&custom_tm_ov),
        ));
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
        // Verify on-disk JSON preserves the exact flat wire format for every
        // `CrsOverride` variant and `CameraBookmark`.
        let raw = std::fs::read_to_string(paths.session_path()).expect("read raw json");
        assert!(raw.contains(r#""kind": "epsg""#));
        assert!(raw.contains(r#""code": 32637"#));
        assert!(raw.contains(r#""kind": "wkt""#));
        assert!(raw.contains(r#""wkt": "GEOGCS"#));
        assert!(raw.contains(r#""kind": "proj""#));
        assert!(raw.contains(r#""proj": "+proj=utm"#));
        assert!(raw.contains(r#""kind": "custom_tm""#));
        assert!(raw.contains(r#""datum": "pulkovo1942""#));
        assert!(raw.contains(r#""central_meridian_deg""#));
        assert!(raw.contains(r#""unit": "metre""#));
        assert!(raw.contains(r#""name": "Rostov""#));
        assert!(!raw.contains(r#""pose""#));

        let loaded = AppSession::load_or_default(&paths);
        assert_eq!(loaded, session);
        assert_eq!(
            loaded.recent_files,
            vec![PathBuf::from("/tmp/a.tif"), PathBuf::from("/tmp/b.tif")]
        );

        let expected = [&epsg_ov, &wkt_ov, &proj_ov, &custom_tm_ov];
        for (entry, expected_ov) in loaded.open_layers.iter().zip(expected) {
            let roundtripped = entry.crs_override().expect("crs override");
            assert_eq!(&roundtripped, expected_ov);
            assert!(roundtripped.to_crs_spec().is_ok());
        }
        assert_eq!(
            loaded.open_layers[3]
                .crs_override()
                .expect("custom tm")
                .to_crs_spec()
                .expect("valid custom TM")
                .label(),
            "MSK-61 Custom"
        );
    }

    #[test]
    fn session_custom_tm_omitted_datum_defaults_to_wgs84() {
        let tmp = TestTempDir::new("session-default-datum");
        let paths = StoragePaths::with_root(tmp.path());
        std::fs::create_dir_all(paths.session_path().parent().expect("parent")).expect("mkdir");
        std::fs::write(
            paths.session_path(),
            r#"{
              "version": 1,
              "open_layers": [{
                "path": "/tmp/legacy.tif",
                "visible": true,
                "opacity": 1.0,
                "crs_override": {
                  "kind": "custom_tm",
                  "name": "Legacy TM",
                  "central_meridian_deg": 39.0,
                  "origin_lat_deg": 0.0,
                  "scale": 1.0,
                  "false_easting_m": 500000.0,
                  "false_northing_m": 0.0,
                  "unit": "us_survey_foot"
                }
              }]
            }"#,
        )
        .expect("write legacy session");

        let loaded = AppSession::load_or_default(&paths);
        assert_eq!(loaded.open_layers.len(), 1);
        match loaded.open_layers[0].crs_override() {
            Some(CrsOverride::CustomTm(params)) => {
                assert_eq!(params.datum, DatumPreset::Wgs84);
                assert_eq!(params.unit, LinearUnit::UsSurveyFoot);
            }
            other => panic!("expected CustomTm, got {other:?}"),
        }
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
