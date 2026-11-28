//! Explicit user-saved `.veldmap.json` project files with sanitized remote
//! COG / `BandMath` URLs (zero credentials/presigned query tokens) and
//! same-root relative local path serialization with absolute fallback.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use veldmap_storage::{StorageError, read_json_limited, write_json_durable};

use super::session::{
    CameraBookmark, SESSION_SCHEMA_VERSION, SavedBandAssetSource, SavedCameraPose, SavedLayerEntry,
    SavedSwipeState, SavedVectorEntry, normalize_workspace_fields,
};

/// File suffix identifying portable VeldMap project files.
pub(crate) const PROJECT_FILE_SUFFIX: &str = ".veldmap.json";

/// Upper size bound for loading a user-selected `.veldmap.json` project file (16 MiB).
const MAX_PROJECT_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Return whether `path` is a `.veldmap.json` project file (checked before
/// `is_geojson_path`, which matches generic `.json`).
pub(crate) fn is_project_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| name.to_ascii_lowercase().ends_with(PROJECT_FILE_SUFFIX))
}

/// Normalize a path lexically (collapsing `.` and resolving non-leading `..`)
/// without touching the filesystem so path relativization works deterministically
/// even for offline/unmounted paths.
fn lexical_clean(path: &Path) -> PathBuf {
    let mut out = Vec::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => match out.last() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                Some(Component::RootDir | Component::Prefix(_)) => {}
                _ => out.push(comp),
            },
            other => out.push(other),
        }
    }
    let mut result = PathBuf::new();
    for comp in out {
        result.push(comp.as_os_str());
    }
    result
}

fn components_equal(a: Component<'_>, b: Component<'_>) -> bool {
    #[cfg(windows)]
    {
        use std::path::Prefix;
        match (a, b) {
            (Component::Prefix(pa), Component::Prefix(pb)) => match (pa.kind(), pb.kind()) {
                (
                    Prefix::Disk(da) | Prefix::VerbatimDisk(da),
                    Prefix::Disk(db) | Prefix::VerbatimDisk(db),
                ) => da.eq_ignore_ascii_case(&db),
                (
                    Prefix::UNC(sa, sha) | Prefix::VerbatimUNC(sa, sha),
                    Prefix::UNC(sb, shb) | Prefix::VerbatimUNC(sb, shb),
                ) => {
                    sa.to_string_lossy().to_lowercase() == sb.to_string_lossy().to_lowercase()
                        && sha.to_string_lossy().to_lowercase()
                            == shb.to_string_lossy().to_lowercase()
                }
                _ => pa
                    .as_os_str()
                    .to_string_lossy()
                    .eq_ignore_ascii_case(&pb.as_os_str().to_string_lossy()),
            },
            (Component::Normal(na), Component::Normal(nb)) => {
                na.to_string_lossy().to_lowercase() == nb.to_string_lossy().to_lowercase()
            }
            _ => a == b,
        }
    }
    #[cfg(not(windows))]
    {
        a == b
    }
}

/// Express `target` relative to `base_dir` when both are absolute paths on the
/// exact same root/drive/UNC prefix, falling back to the cleaned absolute
/// `target` across different drives or when `base_dir` is relative.
pub(crate) fn make_path_relative_if_same_root(base_dir: &Path, target: &Path) -> PathBuf {
    if !base_dir.is_absolute() || !target.is_absolute() {
        return target.to_path_buf();
    }
    let base_clean = lexical_clean(base_dir);
    let target_clean = lexical_clean(target);
    let base_comps: Vec<Component<'_>> = base_clean.components().collect();
    let target_comps: Vec<Component<'_>> = target_clean.components().collect();

    // Verify identical root/prefix (e.g. same Windows drive letter or Unix `/`).
    let mut prefix_len = 0usize;
    while prefix_len < base_comps.len()
        && prefix_len < target_comps.len()
        && matches!(
            base_comps[prefix_len],
            Component::Prefix(_) | Component::RootDir
        )
    {
        if !components_equal(base_comps[prefix_len], target_comps[prefix_len]) {
            return target_clean;
        }
        prefix_len += 1;
    }
    if prefix_len == 0
        || target_comps
            .get(prefix_len)
            .is_some_and(|c| matches!(c, Component::Prefix(_) | Component::RootDir))
    {
        return target_clean;
    }

    let mut common = prefix_len;
    while common < base_comps.len()
        && common < target_comps.len()
        && components_equal(base_comps[common], target_comps[common])
    {
        common += 1;
    }

    let mut rel = PathBuf::new();
    for _ in common..base_comps.len() {
        rel.push("..");
    }
    for comp in &target_comps[common..] {
        rel.push(comp.as_os_str());
    }
    if rel.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        rel
    }
}

/// Resolve a `stored` path from a `.veldmap.json` file against `base_dir`
/// (returning absolute paths cleaned as-is, and joining relative paths onto
/// `base_dir`).
pub(crate) fn resolve_project_path(base_dir: &Path, stored: &Path) -> PathBuf {
    if stored.as_os_str().is_empty() || stored.is_absolute() {
        return stored.to_path_buf();
    }
    lexical_clean(&base_dir.join(stored))
}

fn map_layer_paths(layers: &mut [SavedLayerEntry], mut f: impl FnMut(&Path) -> PathBuf) {
    for layer in layers {
        match layer {
            SavedLayerEntry::LocalRaster(e) => {
                e.path = f(&e.path);
            }
            SavedLayerEntry::RemoteCog(_) => {}
            SavedLayerEntry::BandMath(e) => {
                for band in &mut e.bands {
                    if let SavedBandAssetSource::Local { path, .. } = band {
                        *path = f(path);
                    }
                }
                if let Some(SavedBandAssetSource::Local { path, .. }) = &mut e.scl {
                    *path = f(path);
                }
            }
        }
    }
}

/// Portable `.veldmap.json` workspace project file (sanitized URLs, relative
/// local paths on the same drive/root, zero credentials or recent-file history).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct ProjectFile {
    pub(crate) version: u32,
    pub(crate) open_layers: Vec<SavedLayerEntry>,
    pub(crate) open_vectors: Vec<SavedVectorEntry>,
    pub(crate) bookmarks: Vec<CameraBookmark>,
    pub(crate) last_camera: Option<SavedCameraPose>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) swipe: Option<SavedSwipeState>,
}

impl Default for ProjectFile {
    fn default() -> Self {
        Self {
            version: SESSION_SCHEMA_VERSION,
            open_layers: Vec::new(),
            open_vectors: Vec::new(),
            bookmarks: Vec::new(),
            last_camera: None,
            swipe: None,
        }
    }
}

impl ProjectFile {
    /// Normalize bounds, sanitize all remote URLs, and drop invalid entries.
    pub(crate) fn normalize(&mut self) {
        normalize_workspace_fields(
            &mut self.version,
            &mut self.open_layers,
            &mut self.open_vectors,
            &mut self.bookmarks,
            &mut self.last_camera,
            &mut self.swipe,
        );
    }

    /// Save a normalized, sanitized copy to `project_path`, relativizing local
    /// raster and vector paths against `project_path.parent()` (anchoring
    /// relative inputs against the current working directory first) when on the
    /// same root/drive.
    pub(crate) fn save_to(&self, project_path: &Path) -> Result<(), StorageError> {
        let mut copy = self.clone();
        copy.normalize();
        let cwd = std::env::current_dir().ok();
        let anchor = |p: &Path| -> PathBuf {
            if p.is_absolute() {
                p.to_path_buf()
            } else if let Some(ref dir) = cwd {
                dir.join(p)
            } else {
                p.to_path_buf()
            }
        };
        let anchored = anchor(project_path);
        if let Some(base_dir) = anchored.parent().filter(|p| p.is_absolute()) {
            map_layer_paths(&mut copy.open_layers, |p| {
                let abs = anchor(p);
                make_path_relative_if_same_root(base_dir, &abs)
            });
            for v in &mut copy.open_vectors {
                let abs = anchor(&v.path);
                v.path = make_path_relative_if_same_root(base_dir, &abs);
            }
        }
        write_json_durable(project_path, &copy)
    }

    /// Load and normalize a `.veldmap.json` project file, resolving relative
    /// local paths against `project_path.parent()` (anchored to the current
    /// working directory when `project_path` itself is relative).
    pub(crate) fn load_from(project_path: &Path) -> Result<Self, StorageError> {
        let mut project: Self = read_json_limited(project_path, MAX_PROJECT_FILE_BYTES)?;
        let anchored = if project_path.is_absolute() {
            project_path.to_path_buf()
        } else if let Ok(cwd) = std::env::current_dir() {
            cwd.join(project_path)
        } else {
            project_path.to_path_buf()
        };
        if let Some(base_dir) = anchored.parent().filter(|p| !p.as_os_str().is_empty()) {
            map_layer_paths(&mut project.open_layers, |p| {
                resolve_project_path(base_dir, p)
            });
            for v in &mut project.open_vectors {
                v.path = resolve_project_path(base_dir, &v.path);
            }
        }
        project.normalize();
        Ok(project)
    }
}

#[cfg(test)]
mod tests {
    use veldmap_storage::test_support::TestTempDir;

    use super::*;
    use crate::layers::{BandAssetPersistEntry, BandMathStyleSpec, OverlayPersistEntry};
    use crate::stac::StacCogAuthMode;

    #[test]
    fn project_path_detection_distinguishes_veldmap_project_from_generic_json() {
        assert!(is_project_path(Path::new("analysis.veldmap.json")));
        assert!(is_project_path(Path::new("/dir/MyProject.VELDMAP.JSON")));
        assert!(!is_project_path(Path::new("parcels.geojson")));
        assert!(!is_project_path(Path::new("parcels.json")));
    }

    #[test]
    fn relative_path_round_trips_on_same_root_and_falls_back_across_roots() {
        let tmp = TestTempDir::new("project-rel");
        let base = tmp.path().join("projects").join("run1");
        let sibling = tmp.path().join("data").join("ortho.tif");
        let nested = base.join("bands").join("b04.tif");

        let rel_sib = make_path_relative_if_same_root(&base, &sibling);
        assert!(!rel_sib.is_absolute(), "expected relative, got {rel_sib:?}");
        assert_eq!(
            resolve_project_path(&base, &rel_sib),
            lexical_clean(&sibling)
        );

        let rel_nested = make_path_relative_if_same_root(&base, &nested);
        assert_eq!(rel_nested, PathBuf::from("bands").join("b04.tif"));
        assert_eq!(
            resolve_project_path(&base, &rel_nested),
            lexical_clean(&nested)
        );

        #[cfg(windows)]
        {
            let other_drive = Path::new(r"Z:\external\scene.tif");
            let c_base = Path::new(r"C:\work\proj");
            assert_eq!(
                make_path_relative_if_same_root(c_base, other_drive),
                PathBuf::from(r"Z:\external\scene.tif")
            );
        }
    }

    #[test]
    fn project_file_strips_secrets_relativizes_paths_and_round_trips_swipe_and_band_math() {
        let tmp = TestTempDir::new("project-roundtrip");
        let proj_dir = tmp.path().join("workspace");
        let proj_path = proj_dir.join("study.veldmap.json");
        let local_tif = tmp.path().join("imagery").join("drone.tif");
        let local_vec = proj_dir.join("aoi.geojson");

        let mut proj = ProjectFile::default();
        proj.open_layers.push(SavedLayerEntry::with_style(
            local_tif.clone(),
            true,
            0.9,
            None,
            None,
        ));
        proj.open_layers
            .push(SavedLayerEntry::from_persist(OverlayPersistEntry::RemoteCog {
                canonical_url: "https://user:supersecret@sentinel-cogs.example.org/B04.tif?X-Amz-Signature=deadbeef&sig=secret#frag".to_owned(),
                title: "S2 Red".to_owned(),
                auth_mode: StacCogAuthMode::FromSearchResult,
                external_bands: vec![veldmap_raster::ExternalBandMeta {
                    scale: Some(0.0001),
                    offset: Some(-0.1),
                    nodata: Some(0.0),
                    unit: Some("reflectance".to_owned()),
                }],
                is_scl: false,
                stac_item_href: Some("https://user:pw@stac.example.org/items/s2?token=abc".to_owned()),
                asset_key: Some("red".to_owned()),
                visible: true,
                opacity: 0.85,
                style: None,
            }));
        proj.open_layers.push(SavedLayerEntry::from_persist(
            OverlayPersistEntry::BandMath {
                title: "Mixed NDVI".to_owned(),
                visible: true,
                opacity: 1.0,
                bands: vec![
                    BandAssetPersistEntry::Local {
                        label: "NIR".to_owned(),
                        path: local_tif.clone(),
                        crs_override: None,
                        band_index: 3,
                    },
                    BandAssetPersistEntry::Remote {
                        label: "Red".to_owned(),
                        canonical_url: "https://api:key@example.org/b04.tif?sig=xyz".to_owned(),
                        auth_mode: StacCogAuthMode::DirectCustom,
                        external_bands: Vec::new(),
                        stac_item_href: None,
                        asset_key: Some("B04".to_owned()),
                        band_index: 0,
                    },
                ],
                scl: None,
                style: BandMathStyleSpec {
                    preset: crate::layers::BandMathPreset::Ndvi,
                    formula: crate::layers::BandMathFormula::NormalizedDiff,
                    recipe_label: "NDVI".to_owned(),
                    colormap: crate::colormap::ColormapId::RdYlGn,
                    invert_colormap: false,
                    stretch_min: -0.2,
                    stretch_max: 0.85,
                    rgb_min: [0.0; 3],
                    rgb_max: [0.35; 3],
                    default_rgb_range: ([0.0; 3], [0.35; 3]),
                    rgb_gamma: 1.0,
                    scl_mask_enabled: false,
                    scl_mask_bits: 0,
                },
            },
        ));
        proj.open_vectors.push(SavedVectorEntry {
            path: local_vec.clone(),
            visible: true,
            opacity: 0.7,
            outline_only: true,
        });
        proj.swipe = Some(SavedSwipeState {
            active: true,
            split_t: 0.35,
            left_layer_index: Some(0),
            right_layer_index: Some(2),
        });

        proj.save_to(&proj_path).expect("save project");
        let raw = std::fs::read_to_string(&proj_path).expect("read raw project");

        // Secret hygiene: no userinfo, query tokens, or signatures on disk.
        for forbidden in ["supersecret", "deadbeef", "sig=", "token=abc", "api:key"] {
            assert!(
                !raw.contains(forbidden),
                "leaked secret '{forbidden}' into project file: {raw}"
            );
        }
        assert!(raw.contains(r#""kind": "local_raster""#));
        assert!(raw.contains(r#""kind": "remote_cog""#));
        assert!(raw.contains(r#""kind": "band_math""#));
        assert!(raw.contains(r#""outline_only": true"#));
        assert!(raw.contains(r#""aoi.geojson""#));

        let loaded = ProjectFile::load_from(&proj_path).expect("load project");
        assert_eq!(loaded.open_layers.len(), 3);
        assert_eq!(loaded.open_vectors.len(), 1);
        assert!(loaded.open_vectors[0].outline_only);
        assert_eq!(loaded.open_vectors[0].path, lexical_clean(&local_vec));
        assert_eq!(loaded.swipe, proj.swipe);

        match &loaded.open_layers[0] {
            SavedLayerEntry::LocalRaster(e) => {
                assert_eq!(e.path, lexical_clean(&local_tif));
            }
            other => panic!("expected LocalRaster, got {other:?}"),
        }
        match &loaded.open_layers[1] {
            SavedLayerEntry::RemoteCog(e) => {
                assert_eq!(e.url, "https://sentinel-cogs.example.org/B04.tif");
                assert_eq!(
                    e.stac_item_href.as_deref(),
                    Some("https://stac.example.org/items/s2")
                );
                assert_eq!(e.asset_key.as_deref(), Some("red"));
                assert_eq!(e.auth_mode(), StacCogAuthMode::FromSearchResult);
            }
            other => panic!("expected RemoteCog, got {other:?}"),
        }
    }
}
