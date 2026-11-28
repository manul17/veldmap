//! Tagged persisted raster layer entries (`local_raster`, `remote_cog`,
//! `band_math`, and upcoming `timeline`/`dem_mosaic`) shared between
//! `session.json` and portable `.veldmap.json` project files.

use std::path::PathBuf;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use veldmap_net::sanitize_url_for_persistence;
use veldmap_raster::ExternalBandMeta;

use crate::clamp_opacity;
use crate::colormap::ColormapId;
use crate::layers::{
    BandAssetPersistEntry, BandMathPreset, CrsOverride, LayerStyleSpec, OverlayPersistEntry,
    StacCogAuthMode, TimelineCompareMode, TimelineEpochPersistEntry, TimelineEpochPersistSource,
};

use super::wire::{
    SavedBandMathStyle, SavedCrsOverride, SavedExternalBandMeta, SavedLayerStyle,
    WireBandMathPreset, WireColormapId, WireStacCogAuthMode, deserialize_lossy_opt,
};

pub(super) fn default_true() -> bool {
    true
}

pub(super) fn default_one_f32() -> f32 {
    1.0
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
    pub(super) crs_override: Option<SavedCrsOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) style: Option<SavedLayerStyle>,
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

    pub(super) fn sanitize(&mut self) -> bool {
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

    pub(super) fn sanitize(&mut self) -> bool {
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
    pub(super) fn sanitize(&mut self) -> bool {
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

/// Saved epoch source inside [`SavedTimelineEpoch`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub(crate) enum SavedTimelineEpochSource {
    LocalSingle {
        path: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        crs_override: Option<SavedCrsOverride>,
    },
    RemoteSingle {
        url: String,
        #[serde(default, deserialize_with = "deserialize_lossy_opt")]
        auth_mode: Option<WireStacCogAuthMode>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        external_bands: Vec<SavedExternalBandMeta>,
        #[serde(default)]
        is_scl: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stac_item_href: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        asset_key: Option<String>,
    },
    BandMath {
        bands: Vec<SavedBandAssetSource>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scl: Option<SavedBandAssetSource>,
    },
}

impl SavedTimelineEpochSource {
    fn from_persist(src: &TimelineEpochPersistSource) -> Self {
        match src {
            TimelineEpochPersistSource::LocalSingle { path, crs_override } => Self::LocalSingle {
                path: path.clone(),
                crs_override: crs_override.as_ref().map(SavedCrsOverride::from),
            },
            TimelineEpochPersistSource::RemoteSingle {
                canonical_url,
                auth_mode,
                external_bands,
                is_scl,
                stac_item_href,
                asset_key,
            } => Self::RemoteSingle {
                url: canonical_url.clone(),
                auth_mode: Some(WireStacCogAuthMode::from(*auth_mode)),
                external_bands: external_bands
                    .iter()
                    .map(SavedExternalBandMeta::from)
                    .collect(),
                is_scl: *is_scl,
                stac_item_href: stac_item_href.clone(),
                asset_key: asset_key.clone(),
            },
            TimelineEpochPersistSource::BandMath { bands, scl } => Self::BandMath {
                bands: bands
                    .iter()
                    .map(SavedBandAssetSource::from_persist)
                    .collect(),
                scl: scl.as_ref().map(SavedBandAssetSource::from_persist),
            },
        }
    }

    pub(crate) fn remote_auth_mode(&self) -> StacCogAuthMode {
        match self {
            Self::RemoteSingle { auth_mode, .. } => {
                auth_mode.map_or(StacCogAuthMode::None, StacCogAuthMode::from)
            }
            _ => StacCogAuthMode::None,
        }
    }

    pub(crate) fn local_crs_override(&self) -> Option<CrsOverride> {
        match self {
            Self::LocalSingle { crs_override, .. } => crs_override.as_ref().map(CrsOverride::from),
            _ => None,
        }
    }

    fn sanitize(&mut self) -> bool {
        match self {
            Self::LocalSingle { path, .. } => !path.as_os_str().is_empty(),
            Self::RemoteSingle {
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
            Self::BandMath { bands, scl } => {
                if bands.is_empty() || !bands.iter_mut().all(SavedBandAssetSource::sanitize) {
                    return false;
                }
                if let Some(s) = scl
                    && !s.sanitize()
                {
                    *scl = None;
                }
                true
            }
        }
    }
}

/// Saved chronological epoch inside [`SavedTimelineEntry`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct SavedTimelineEpoch {
    pub(crate) label: String,
    pub(crate) date_label: String,
    #[serde(default)]
    pub(crate) sort_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) cloud_cover: Option<f32>,
    pub(crate) source: SavedTimelineEpochSource,
}

/// Wire representation of [`TimelineCompareMode`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub(crate) enum SavedTimelineCompareMode {
    #[default]
    Single,
    ScalarDiff {
        ref_idx: usize,
    },
    CrossFade {
        ref_idx: usize,
        blend_t: f32,
    },
}

impl From<TimelineCompareMode> for SavedTimelineCompareMode {
    fn from(m: TimelineCompareMode) -> Self {
        match m {
            TimelineCompareMode::Single => Self::Single,
            TimelineCompareMode::ScalarDiff { ref_idx } => Self::ScalarDiff { ref_idx },
            TimelineCompareMode::CrossFade { ref_idx, blend_t } => {
                Self::CrossFade { ref_idx, blend_t }
            }
        }
    }
}

impl From<SavedTimelineCompareMode> for TimelineCompareMode {
    fn from(m: SavedTimelineCompareMode) -> Self {
        match m {
            SavedTimelineCompareMode::Single => Self::Single,
            SavedTimelineCompareMode::ScalarDiff { ref_idx } => Self::ScalarDiff { ref_idx },
            SavedTimelineCompareMode::CrossFade { ref_idx, blend_t } => Self::CrossFade {
                ref_idx,
                blend_t: if blend_t.is_finite() {
                    blend_t.clamp(0.0, 1.0)
                } else {
                    0.5
                },
            },
        }
    }
}

/// Saved multi-temporal `Timeline` layer entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SavedTimelineEntry {
    pub(crate) title: String,
    #[serde(default = "default_true")]
    pub(crate) visible: bool,
    #[serde(default = "default_one_f32")]
    pub(crate) opacity: f32,
    pub(crate) epochs: Vec<SavedTimelineEpoch>,
    #[serde(default, deserialize_with = "deserialize_lossy_opt")]
    recipe_preset: Option<WireBandMathPreset>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) band_math_style: Option<SavedBandMathStyle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    single_style: Option<SavedLayerStyle>,
    #[serde(default)]
    pub(crate) active_idx: usize,
    #[serde(default)]
    pub(crate) compare: SavedTimelineCompareMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) max_cloud_cover: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_lossy_opt")]
    diff_colormap: Option<WireColormapId>,
    #[serde(default)]
    pub(crate) diff_invert_colormap: bool,
    #[serde(default = "default_diff_min")]
    pub(crate) diff_stretch_min: f64,
    #[serde(default = "default_diff_max")]
    pub(crate) diff_stretch_max: f64,
}

fn default_diff_min() -> f64 {
    -0.5
}

fn default_diff_max() -> f64 {
    0.5
}

impl Default for SavedTimelineEntry {
    fn default() -> Self {
        Self {
            title: String::new(),
            visible: true,
            opacity: 1.0,
            epochs: Vec::new(),
            recipe_preset: None,
            band_math_style: None,
            single_style: None,
            active_idx: 0,
            compare: SavedTimelineCompareMode::Single,
            max_cloud_cover: None,
            diff_colormap: Some(WireColormapId::RdBu),
            diff_invert_colormap: false,
            diff_stretch_min: -0.5,
            diff_stretch_max: 0.5,
        }
    }
}

impl SavedTimelineEntry {
    pub(crate) fn recipe_preset(&self) -> Option<BandMathPreset> {
        self.recipe_preset.map(BandMathPreset::from)
    }

    pub(crate) fn single_style(&self) -> Option<LayerStyleSpec> {
        self.single_style.as_ref().map(LayerStyleSpec::from)
    }

    pub(crate) fn diff_colormap(&self) -> ColormapId {
        self.diff_colormap
            .map_or(ColormapId::RdBu, ColormapId::from)
    }

    fn sanitize(&mut self) -> bool {
        self.opacity = clamp_opacity(self.opacity);
        self.epochs.retain_mut(|e| e.source.sanitize());
        if self.epochs.is_empty() {
            return false;
        }
        let max_idx = self.epochs.len() - 1;
        self.active_idx = self.active_idx.min(max_idx);
        self.compare = match self.compare {
            SavedTimelineCompareMode::Single => SavedTimelineCompareMode::Single,
            SavedTimelineCompareMode::ScalarDiff { ref_idx } => {
                SavedTimelineCompareMode::ScalarDiff {
                    ref_idx: ref_idx.min(max_idx),
                }
            }
            SavedTimelineCompareMode::CrossFade { ref_idx, blend_t } => {
                SavedTimelineCompareMode::CrossFade {
                    ref_idx: ref_idx.min(max_idx),
                    blend_t: if blend_t.is_finite() {
                        blend_t.clamp(0.0, 1.0)
                    } else {
                        0.5
                    },
                }
            }
        };
        if let Some(bm) = &mut self.band_math_style {
            bm.sanitize();
        }
        self.max_cloud_cover = self
            .max_cloud_cover
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 100.0));
        if !self.diff_stretch_min.is_finite() || !self.diff_stretch_max.is_finite() {
            self.diff_stretch_min = -0.5;
            self.diff_stretch_max = 0.5;
        }
        if self.title.trim().is_empty() {
            self.title = "Timeline".to_owned();
        }
        true
    }
}

/// Tagged persisted raster layer entry in painter's order (bottom-to-top).
///
/// Serializes strictly as tagged JSON (`"kind": "local_raster" | "remote_cog" | "band_math" | "timeline"`),
/// while deserializing both the tagged v2 schema and legacy v1 untagged local
/// raster entries seamlessly.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SavedLayerEntry {
    LocalRaster(SavedLocalRasterEntry),
    RemoteCog(SavedRemoteCogEntry),
    BandMath(SavedBandMathEntry),
    Timeline(Box<SavedTimelineEntry>),
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SavedLayerEntryRef<'a> {
    LocalRaster(&'a SavedLocalRasterEntry),
    RemoteCog(&'a SavedRemoteCogEntry),
    BandMath(&'a SavedBandMathEntry),
    Timeline(&'a SavedTimelineEntry),
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SavedLayerEntryTagged {
    LocalRaster(SavedLocalRasterEntry),
    RemoteCog(SavedRemoteCogEntry),
    BandMath(SavedBandMathEntry),
    Timeline(Box<SavedTimelineEntry>),
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
            Self::Timeline(e) => SavedLayerEntryRef::Timeline(e),
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
            SavedLayerEntryWire::Tagged(SavedLayerEntryTagged::Timeline(e)) => Self::Timeline(e),
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
            OverlayPersistEntry::Timeline {
                title,
                visible,
                opacity,
                epochs,
                recipe_preset,
                band_math_style,
                single_style,
                active_idx,
                compare,
                max_cloud_cover,
                diff_colormap,
                diff_invert_colormap,
                diff_stretch_min,
                diff_stretch_max,
            } => Self::Timeline(Box::new(SavedTimelineEntry {
                title,
                visible,
                opacity,
                epochs: epochs
                    .iter()
                    .map(|e: &TimelineEpochPersistEntry| SavedTimelineEpoch {
                        label: e.label.clone(),
                        date_label: e.date_label.clone(),
                        sort_key: e.sort_key.clone(),
                        cloud_cover: e.cloud_cover,
                        source: SavedTimelineEpochSource::from_persist(&e.source),
                    })
                    .collect(),
                recipe_preset: recipe_preset.map(WireBandMathPreset::from),
                band_math_style: band_math_style.as_ref().map(SavedBandMathStyle::from),
                single_style: single_style.as_ref().map(SavedLayerStyle::from),
                active_idx,
                compare: SavedTimelineCompareMode::from(compare),
                max_cloud_cover,
                diff_colormap: Some(WireColormapId::from(diff_colormap)),
                diff_invert_colormap,
                diff_stretch_min,
                diff_stretch_max,
            })),
        }
    }

    pub(crate) fn display_name(&self) -> String {
        match self {
            Self::LocalRaster(e) => crate::display_name_from_path(&e.path),
            Self::RemoteCog(e) => e.title.clone(),
            Self::BandMath(e) => e.title.clone(),
            Self::Timeline(e) => e.title.clone(),
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
            Self::Timeline(e) => format!("Timeline ({} epochs)", e.epochs.len()),
        }
    }

    #[cfg(test)]
    pub(crate) fn crs_override(&self) -> Option<CrsOverride> {
        match self {
            Self::LocalRaster(e) => e.crs_override(),
            Self::RemoteCog(_) | Self::BandMath(_) | Self::Timeline(_) => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn style(&self) -> Option<LayerStyleSpec> {
        match self {
            Self::LocalRaster(e) => e.style(),
            Self::RemoteCog(e) => e.style(),
            Self::BandMath(_) | Self::Timeline(_) => None,
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
            Self::Timeline(e) => e.sanitize(),
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
            Self::Timeline(e) => {
                for epoch in &mut e.epochs {
                    match &mut epoch.source {
                        SavedTimelineEpochSource::LocalSingle { .. } => {}
                        SavedTimelineEpochSource::RemoteSingle { url, auth_mode, .. } => {
                            let current =
                                auth_mode.map_or(StacCogAuthMode::None, StacCogAuthMode::from);
                            let next = map_mode(url, current);
                            *auth_mode = Some(WireStacCogAuthMode::from(next));
                        }
                        SavedTimelineEpochSource::BandMath { bands, scl } => {
                            for src in bands.iter_mut().chain(scl.iter_mut()) {
                                if let SavedBandAssetSource::Remote { url, auth_mode, .. } = src {
                                    let current = auth_mode
                                        .map_or(StacCogAuthMode::None, StacCogAuthMode::from);
                                    let next = map_mode(url, current);
                                    *auth_mode = Some(WireStacCogAuthMode::from(next));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
