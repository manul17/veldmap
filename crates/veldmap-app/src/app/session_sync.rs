//! Session & project restoration tracking (generalized beyond local `Path` keys
//! via unique restore tokens for local rasters, remote COGs, heterogeneous
//! `BandMath` recipes, vectors, and swipe state), unified `session.json` /
//! `.veldmap.json` snapshot persistence, and asynchronous scene-I/O event
//! draining for [`VeldmapApp`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use veldmap_core::geo::FootprintBounds;
use veldmap_raster::ExternalBandMeta;

use crate::display_name_from_path;
use crate::layers::{
    BandMathAssetOpenSpec, LayerId, OpenOverlayRequest, OverlayIoEvent, RemoteBandMathRequest,
    RemoteCogRequest,
};
use crate::measure::MeasureMode;
use crate::scene::SceneIoEvent;
use crate::stac::{AuthTrust, resolve_persisted_cog_auth};
use crate::storage::project::ProjectFile;
use crate::storage::session::{
    SavedBandAssetSource, SavedCameraPose, SavedLayerEntry, SavedSwipeState, SavedVectorEntry,
};
use crate::storage::settings::StacSettings;
use crate::ui::layers::UnavailableLayerNotice;
use crate::ui::workbench::WorkbenchTab;
use crate::vectors::VectorIoEvent;

use super::VeldmapApp;

#[derive(Clone, Copy, PartialEq, Eq)]
enum RestoredEntryKind {
    LocalRaster,
    RemoteCog,
    BandMath,
    Vector,
}

impl RestoredEntryKind {
    fn label(self) -> &'static str {
        match self {
            Self::LocalRaster => "local overlay",
            Self::RemoteCog => "remote COG",
            Self::BandMath => "band-math recipe",
            Self::Vector => "GeoJSON vector",
        }
    }

    fn badge(self) -> &'static str {
        match self {
            Self::LocalRaster => "Raster",
            Self::RemoteCog => "Cloud COG",
            Self::BandMath => "Band Math",
            Self::Vector => "Vector",
        }
    }
}

#[derive(Clone)]
enum SavedEntry {
    Raster(Box<SavedLayerEntry>),
    Vector(SavedVectorEntry),
}

impl SavedEntry {
    fn kind(&self) -> RestoredEntryKind {
        match self {
            Self::Raster(entry) => match entry.as_ref() {
                SavedLayerEntry::LocalRaster(_) => RestoredEntryKind::LocalRaster,
                SavedLayerEntry::RemoteCog(_) => RestoredEntryKind::RemoteCog,
                SavedLayerEntry::BandMath(_) => RestoredEntryKind::BandMath,
            },
            Self::Vector(_) => RestoredEntryKind::Vector,
        }
    }

    fn local_path(&self) -> Option<&Path> {
        match self {
            Self::Raster(entry) => match entry.as_ref() {
                SavedLayerEntry::LocalRaster(e) => Some(&e.path),
                SavedLayerEntry::RemoteCog(_) | SavedLayerEntry::BandMath(_) => None,
            },
            Self::Vector(e) => Some(&e.path),
        }
    }

    fn to_notice(&self, error: String) -> UnavailableLayerNotice {
        match self {
            Self::Raster(entry) => UnavailableLayerNotice {
                name: entry.display_name(),
                detail: entry.display_detail(),
                kind_tag: self.kind().badge(),
                error,
            },
            Self::Vector(entry) => UnavailableLayerNotice {
                name: display_name_from_path(&entry.path),
                detail: entry.path.display().to_string(),
                kind_tag: self.kind().badge(),
                error,
            },
        }
    }
}

struct PendingEntry {
    restore_id: u64,
    layer_index: Option<usize>,
    saved: SavedEntry,
}

struct UnavailableEntry {
    layer_index: Option<usize>,
    saved: SavedEntry,
    notice: UnavailableLayerNotice,
}

/// Tracks layers and vectors queued from `session.json` or `.veldmap.json`
/// whose background open events have not yet drained, plus any restored entries
/// that failed to open (e.g. unmounted drives, offline network, or 401/403
/// transient outages), so quitting or saving never silently wipes persisted
/// styles, remote COG metadata, `BandMath` recipes, painter's order, or swipe
/// configuration.
pub(super) struct SessionRestoreTracker {
    next_restore_id: u64,
    auth_trust: AuthTrust,
    pending: Vec<PendingEntry>,
    unavailable: Vec<UnavailableEntry>,
    notices_hidden: bool,
    pending_swipe: Option<SavedSwipeState>,
    restored_layer_ids: HashMap<usize, LayerId>,
    layer_slots: HashMap<LayerId, usize>,
}

impl Default for SessionRestoreTracker {
    fn default() -> Self {
        Self {
            next_restore_id: 1,
            auth_trust: AuthTrust::LocalSession,
            pending: Vec::new(),
            unavailable: Vec::new(),
            notices_hidden: false,
            pending_swipe: None,
            restored_layer_ids: HashMap::new(),
            layer_slots: HashMap::new(),
        }
    }
}

impl SessionRestoreTracker {
    fn alloc_restore_id(&mut self) -> u64 {
        let id = self.next_restore_id;
        self.next_restore_id = self.next_restore_id.saturating_add(1);
        id
    }

    pub(super) fn reset_for_project(
        &mut self,
        swipe: Option<SavedSwipeState>,
        auth_trust: AuthTrust,
    ) {
        self.auth_trust = auth_trust;
        self.pending.clear();
        self.unavailable.clear();
        self.notices_hidden = false;
        self.pending_swipe = swipe;
        self.restored_layer_ids.clear();
        self.layer_slots.clear();
    }

    pub(super) fn set_pending_swipe(
        &mut self,
        swipe: Option<SavedSwipeState>,
        auth_trust: AuthTrust,
    ) {
        self.auth_trust = auth_trust;
        self.pending_swipe = swipe;
        self.restored_layer_ids.clear();
        self.layer_slots.clear();
    }

    pub(super) fn clear_pending_swipe(&mut self) {
        self.pending_swipe = None;
    }

    fn push_layer(&mut self, restore_id: u64, layer_index: Option<usize>, entry: SavedLayerEntry) {
        self.pending.push(PendingEntry {
            restore_id,
            layer_index,
            saved: SavedEntry::Raster(Box::new(entry)),
        });
    }

    fn push_vector(&mut self, restore_id: u64, entry: SavedVectorEntry) {
        self.pending.push(PendingEntry {
            restore_id,
            layer_index: None,
            saved: SavedEntry::Vector(entry),
        });
    }

    fn record_unavailable(&mut self, layer_index: Option<usize>, saved: SavedEntry, error: String) {
        let notice = saved.to_notice(error);
        self.unavailable.push(UnavailableEntry {
            layer_index,
            saved,
            notice,
        });
        self.notices_hidden = false;
    }

    fn record_unavailable_layer(
        &mut self,
        layer_index: Option<usize>,
        entry: SavedLayerEntry,
        error: String,
    ) {
        self.record_unavailable(layer_index, SavedEntry::Raster(Box::new(entry)), error);
    }

    fn record_unavailable_vector(&mut self, entry: SavedVectorEntry, error: String) {
        self.record_unavailable(None, SavedEntry::Vector(entry), error);
    }

    /// Consume a matching entry when a raster or vector layer opens, returning
    /// its saved painter's-order slot index (if any) so the caller can place it
    /// in original z-order even when async opens finish out of order.
    fn on_opened(
        &mut self,
        restore_id: Option<u64>,
        opened_layer_id: Option<LayerId>,
        kind: RestoredEntryKind,
        local_path: Option<&Path>,
    ) -> Option<usize> {
        if let Some(rid) = restore_id
            && let Some(pos) = self.pending.iter().position(|e| e.restore_id == rid)
        {
            let entry = self.pending.remove(pos);
            if let (Some(idx), Some(lid)) = (entry.layer_index, opened_layer_id) {
                self.restored_layer_ids.insert(idx, lid);
                self.layer_slots.insert(lid, idx);
                return Some(idx);
            }
            return entry.layer_index;
        }
        if let Some(path) = local_path
            && let Some(pos) = self
                .unavailable
                .iter()
                .position(|e| e.saved.kind() == kind && e.saved.local_path() == Some(path))
        {
            self.unavailable.remove(pos);
        }
        None
    }

    /// Compute the painter's-order insertion index for a newly opened restored
    /// layer `new_id` with saved slot `target_slot` relative to already-live
    /// restored layers in `live_ids`.
    fn insertion_index_for_slot(
        &self,
        live_ids: &[LayerId],
        new_id: LayerId,
        target_slot: usize,
    ) -> usize {
        let mut insert_pos = 0usize;
        let mut idx = 0usize;
        for &id in live_ids {
            if id == new_id {
                continue;
            }
            if self
                .layer_slots
                .get(&id)
                .is_some_and(|&slot| slot < target_slot)
            {
                insert_pos = idx + 1;
            }
            idx += 1;
        }
        insert_pos
    }

    /// If `restore_id` belonged to a pending restore, move its full saved entry
    /// (preserving `layer_index`) into `unavailable` so offline/401/403 outages
    /// never wipe workspace state or swipe slot references.
    fn on_failed(&mut self, restore_id: Option<u64>, error: &str) -> bool {
        let Some(rid) = restore_id else {
            return false;
        };
        let Some(pos) = self.pending.iter().position(|e| e.restore_id == rid) else {
            return false;
        };
        let entry = self.pending.remove(pos);
        self.record_unavailable(entry.layer_index, entry.saved, error.to_owned());
        true
    }

    fn has_pending_slot(&self, slot: usize) -> bool {
        self.pending
            .iter()
            .any(|p| matches!(p.saved, SavedEntry::Raster(_)) && p.layer_index == Some(slot))
    }

    fn has_unavailable_slot(&self, slot: usize) -> bool {
        self.unavailable
            .iter()
            .any(|u| matches!(u.saved, SavedEntry::Raster(_)) && u.layer_index == Some(slot))
    }

    /// Resolve saved swipe indices into live [`LayerId`]s once all referenced
    /// slots have opened. If a referenced slot failed into `unavailable` (e.g.
    /// offline launch), keeps `pending_swipe` preserved so quitting never wipes
    /// the user's swipe setup and retrying unavailable layers restores it.
    fn try_resolve_pending_swipe(&mut self) -> Option<(f32, Option<LayerId>, Option<LayerId>)> {
        let sw = self.pending_swipe?;
        if !sw.active {
            self.pending_swipe = None;
            return None;
        }
        let slots = [sw.left_layer_index, sw.right_layer_index];
        // Wait while any referenced slot is still pending or preserved as unavailable.
        if slots
            .into_iter()
            .flatten()
            .any(|idx| self.has_pending_slot(idx) || self.has_unavailable_slot(idx))
        {
            return None;
        }
        self.pending_swipe = None;
        let left = sw
            .left_layer_index
            .and_then(|i| self.restored_layer_ids.get(&i).copied());
        let right = sw
            .right_layer_index
            .and_then(|i| self.restored_layer_ids.get(&i).copied());
        if left.is_none() && right.is_none() {
            return None;
        }
        Some((sw.split_t, left, right))
    }

    fn take_unavailable_for_retry(
        &mut self,
    ) -> (Vec<(Option<usize>, SavedLayerEntry)>, Vec<SavedVectorEntry>) {
        self.notices_hidden = false;
        let mut layers = Vec::new();
        let mut vectors = Vec::new();
        for item in self.unavailable.drain(..) {
            match item.saved {
                SavedEntry::Raster(e) => layers.push((item.layer_index, *e)),
                SavedEntry::Vector(e) => vectors.push(e),
            }
        }
        (layers, vectors)
    }

    pub(super) fn hide_notices(&mut self) {
        self.notices_hidden = true;
    }

    pub(super) fn unhide_notices(&mut self) {
        self.notices_hidden = false;
    }

    pub(super) fn forget_unavailable(&mut self) -> bool {
        let had_any = !self.unavailable.is_empty();
        self.unavailable.clear();
        self.notices_hidden = false;
        had_any
    }

    pub(super) fn unavailable_count(&self) -> usize {
        self.unavailable.len()
    }

    pub(super) fn visible_notices(&self) -> Vec<UnavailableLayerNotice> {
        if self.notices_hidden || self.unavailable.is_empty() {
            return Vec::new();
        }
        self.unavailable
            .iter()
            .map(|item| item.notice.clone())
            .collect()
    }

    /// Assemble live + preserved raster entries in original saved painter's
    /// order and compute [`SavedSwipeState`] indices directly against the final
    /// serialized `open_layers` list (preventing index skew across out-of-order
    /// opens, unavailable offline layers, or user deletions).
    fn assemble_layers_and_swipe(
        &self,
        live_entries: Vec<(LayerId, SavedLayerEntry)>,
        live_swipe: Option<(f32, Option<LayerId>, Option<LayerId>)>,
    ) -> (Vec<SavedLayerEntry>, Option<SavedSwipeState>) {
        let mut slotted_preserved: Vec<(usize, SavedLayerEntry)> = Vec::new();
        let mut unslotted_preserved: Vec<SavedLayerEntry> = Vec::new();
        for (layer_index, saved) in self
            .pending
            .iter()
            .map(|p| (p.layer_index, &p.saved))
            .chain(self.unavailable.iter().map(|u| (u.layer_index, &u.saved)))
        {
            if let SavedEntry::Raster(entry) = saved {
                match layer_index {
                    Some(slot) => slotted_preserved.push((slot, entry.as_ref().clone())),
                    None => unslotted_preserved.push(entry.as_ref().clone()),
                }
            }
        }
        slotted_preserved.sort_by_key(|(slot, _)| *slot);

        let mut merged: Vec<(Option<LayerId>, Option<usize>, SavedLayerEntry)> = Vec::with_capacity(
            live_entries.len() + slotted_preserved.len() + unslotted_preserved.len(),
        );
        let mut preserved_iter = slotted_preserved.into_iter().peekable();
        for (lid, entry) in live_entries {
            let live_slot = self.layer_slots.get(&lid).copied();
            if let Some(slot) = live_slot {
                while preserved_iter
                    .peek()
                    .is_some_and(|&(p_slot, _)| p_slot < slot)
                    && let Some((ps, pe)) = preserved_iter.next()
                {
                    merged.push((None, Some(ps), pe));
                }
                merged.push((Some(lid), Some(slot), entry));
            } else {
                // Flush remaining slotted restored entries ahead of trailing
                // manually opened layers so restored z-order stays contiguous.
                for (ps, pe) in preserved_iter.by_ref() {
                    merged.push((None, Some(ps), pe));
                }
                merged.push((Some(lid), None, entry));
            }
        }
        for (ps, pe) in preserved_iter {
            merged.push((None, Some(ps), pe));
        }
        for pe in unslotted_preserved {
            merged.push((None, None, pe));
        }

        let swipe = if let Some((split_t, left_id, right_id)) = live_swipe {
            if merged.is_empty() {
                None
            } else {
                let find_live = |target: Option<LayerId>| {
                    target.and_then(|tid| merged.iter().position(|(lid, _, _)| *lid == Some(tid)))
                };
                Some(SavedSwipeState {
                    active: true,
                    split_t,
                    left_layer_index: find_live(left_id),
                    right_layer_index: find_live(right_id),
                })
            }
        } else if let Some(sw) = self.pending_swipe
            && sw.active
            && !merged.is_empty()
        {
            let remap_slot = |old_slot: Option<usize>| {
                old_slot.and_then(|os| merged.iter().position(|(_, slot, _)| *slot == Some(os)))
            };
            let left = remap_slot(sw.left_layer_index);
            let right = remap_slot(sw.right_layer_index);
            if left.is_none() && right.is_none() {
                None
            } else {
                Some(SavedSwipeState {
                    active: true,
                    split_t: sw.split_t,
                    left_layer_index: left,
                    right_layer_index: right,
                })
            }
        } else {
            None
        };

        let open_layers = merged.into_iter().map(|(_, _, entry)| entry).collect();
        (open_layers, swipe)
    }

    fn all_preserved(&self) -> impl Iterator<Item = &SavedEntry> {
        self.pending
            .iter()
            .map(|p| &p.saved)
            .chain(self.unavailable.iter().map(|e| &e.saved))
    }

    #[cfg(test)]
    fn preserved_layers(&self) -> impl Iterator<Item = &SavedLayerEntry> {
        self.all_preserved().filter_map(|e| match e {
            SavedEntry::Raster(layer) => Some(layer.as_ref()),
            SavedEntry::Vector(_) => None,
        })
    }

    fn preserved_vectors(&self) -> impl Iterator<Item = &SavedVectorEntry> {
        self.all_preserved().filter_map(|e| match e {
            SavedEntry::Vector(vec_layer) => Some(vec_layer),
            SavedEntry::Raster(_) => None,
        })
    }
}

fn build_band_math_asset_spec(
    src: &SavedBandAssetSource,
    trust: AuthTrust,
    stac_settings: &StacSettings,
) -> BandMathAssetOpenSpec {
    match src {
        SavedBandAssetSource::Local {
            label,
            path,
            band_index,
            ..
        } => BandMathAssetOpenSpec::Local {
            label: label.clone(),
            path: path.clone(),
            crs_override: src.local_crs_override(),
            band_index: *band_index,
        },
        SavedBandAssetSource::Remote {
            label,
            url,
            external_bands,
            stac_item_href,
            asset_key,
            band_index,
            ..
        } => {
            let auth_mode = src.remote_auth_mode();
            let (final_url, extra_headers) =
                resolve_persisted_cog_auth(url, auth_mode, trust, stac_settings);
            BandMathAssetOpenSpec::Remote {
                label: label.clone(),
                url: final_url,
                auth_mode,
                extra_headers,
                external_bands: external_bands.iter().map(ExternalBandMeta::from).collect(),
                stac_item_href: stac_item_href.clone(),
                asset_key: asset_key.clone(),
                band_index: *band_index,
            }
        }
    }
}

impl VeldmapApp {
    /// Queue persisted open layers (local rasters, sanitized remote COGs, and
    /// heterogeneous `BandMath` recipes) and vectors strictly off the UI thread.
    pub(super) fn queue_restored_session(
        &mut self,
        open_layers: Vec<SavedLayerEntry>,
        open_vectors: Vec<SavedVectorEntry>,
        swipe: Option<SavedSwipeState>,
        trust: AuthTrust,
    ) {
        self.restore_tracker.set_pending_swipe(swipe, trust);
        let indexed_layers: Vec<(Option<usize>, SavedLayerEntry)> = open_layers
            .into_iter()
            .enumerate()
            .map(|(i, e)| (Some(i), e))
            .collect();
        self.queue_restored_entries(indexed_layers, open_vectors, trust);
    }

    fn queue_restored_entries(
        &mut self,
        open_layers: Vec<(Option<usize>, SavedLayerEntry)>,
        open_vectors: Vec<SavedVectorEntry>,
        trust: AuthTrust,
    ) {
        let stac_settings = self.storage.settings().stac.clone();

        for (layer_idx, saved) in open_layers {
            let restore_id = self.restore_tracker.alloc_restore_id();
            let res = match &saved {
                SavedLayerEntry::LocalRaster(entry) => {
                    self.scene.begin_open_overlay(OpenOverlayRequest {
                        path: &entry.path,
                        fly_to: false,
                        visible: entry.visible,
                        opacity: entry.opacity,
                        crs_override: entry.crs_override(),
                        style: entry.style(),
                        restore_id: Some(restore_id),
                    })
                }
                SavedLayerEntry::RemoteCog(entry) => {
                    let auth_mode = entry.auth_mode();
                    let (final_url, extra_headers) =
                        resolve_persisted_cog_auth(&entry.url, auth_mode, trust, &stac_settings);
                    self.scene.begin_open_remote_cog(RemoteCogRequest {
                        url: final_url,
                        title: entry.title.clone(),
                        auth_mode,
                        extra_headers,
                        external_bands: entry.external_band_metas(),
                        is_scl: entry.is_scl,
                        stac_item_href: entry.stac_item_href.clone(),
                        asset_key: entry.asset_key.clone(),
                        visible: entry.visible,
                        opacity: entry.opacity,
                        style: entry.style(),
                        fly_to: false,
                        restore_id: Some(restore_id),
                    })
                }
                SavedLayerEntry::BandMath(entry) => {
                    let style_spec = entry.style.to_spec();
                    let bands = entry
                        .bands
                        .iter()
                        .map(|b| build_band_math_asset_spec(b, trust, &stac_settings))
                        .collect();
                    let scl = entry
                        .scl
                        .as_ref()
                        .map(|s| build_band_math_asset_spec(s, trust, &stac_settings));
                    self.scene
                        .begin_open_remote_band_math(RemoteBandMathRequest {
                            title: entry.title.clone(),
                            preset: style_spec.preset,
                            bands,
                            scl,
                            visible: entry.visible,
                            opacity: entry.opacity,
                            style: Some(style_spec),
                            fly_to: false,
                            restore_id: Some(restore_id),
                        })
                }
            };
            match res {
                Ok(()) => self
                    .restore_tracker
                    .push_layer(restore_id, layer_idx, saved),
                Err(err) => self
                    .restore_tracker
                    .record_unavailable_layer(layer_idx, saved, err),
            }
        }

        for saved_vec in open_vectors {
            let restore_id = self.restore_tracker.alloc_restore_id();
            match self.scene.begin_open_vector(
                &saved_vec.path,
                false,
                saved_vec.visible,
                saved_vec.opacity,
                saved_vec.outline_only,
                Some(restore_id),
            ) {
                Ok(()) => self.restore_tracker.push_vector(restore_id, saved_vec),
                Err(err) => self
                    .restore_tracker
                    .record_unavailable_vector(saved_vec, err),
            }
        }
    }

    pub(super) fn retry_unavailable_session_layers(&mut self) {
        let trust = self.restore_tracker.auth_trust;
        let (layers, vectors) = self.restore_tracker.take_unavailable_for_retry();
        self.queue_restored_entries(layers, vectors, trust);
    }

    /// Snapshot live + preserved layers, vectors, bookmarks, camera pose, and
    /// swipe layout into a portable [`ProjectFile`] (ready for `.veldmap.json`
    /// export).
    pub(super) fn build_project_file(&self) -> ProjectFile {
        let (open_layers, open_vectors, swipe) = self.collect_workspace_state();
        ProjectFile {
            version: 0,
            open_layers,
            open_vectors,
            bookmarks: self.storage.session().bookmarks.clone(),
            last_camera: Some(SavedCameraPose::from_camera(&self.camera)),
            swipe,
        }
    }

    /// Replace the active workspace with a loaded `.veldmap.json` project file,
    /// enforcing imported-project credential trust boundaries and emitting a
    /// single clean `session.json` snapshot.
    pub(super) fn apply_loaded_project(&mut self, mut project: ProjectFile, source_path: &Path) {
        // Downgrade untrusted cross-origin auth modes before queuing or
        // persisting into `session.json`.
        let stac_settings = self.storage.settings().stac.clone();
        for entry in &mut project.open_layers {
            entry.sanitize_imported_auth(&stac_settings);
        }

        // Reset transient tools, UI error/picker state, and in-flight opens
        // cleanly without writing intermediate partial session snapshots.
        self.deactivate_draw_aoi();
        self.scene.mutate_measure(|m| {
            m.clear();
            m.set_mode(MeasureMode::Off);
        });
        self.panels.measure.clear_notice();
        self.panels.layers.clear_error();
        self.panels.crs_picker.close();
        self.swipe.reset();
        self.viewport.cancel_swipe_drag();
        self.viewport.set_swipe_bar_rect(None);

        let _ = self.scene.clear_workspace_layers();
        self.restore_tracker
            .reset_for_project(project.swipe, AuthTrust::ImportedProject);
        if let Some(pose) = project.last_camera {
            self.camera = pose.to_camera();
        }
        let indexed_layers: Vec<(Option<usize>, SavedLayerEntry)> = project
            .open_layers
            .into_iter()
            .enumerate()
            .map(|(i, e)| (Some(i), e))
            .collect();
        self.queue_restored_entries(
            indexed_layers,
            project.open_vectors,
            AuthTrust::ImportedProject,
        );

        let (open_layers, open_vectors, swipe) = self.collect_workspace_state();
        let cam_pose = Some(SavedCameraPose::from_camera(&self.camera));
        let bookmarks = project.bookmarks;
        let _ = self.storage.mutate_session(|s| {
            s.open_layers = open_layers;
            s.open_vectors = open_vectors;
            s.bookmarks = bookmarks;
            s.last_camera = cam_pose;
            s.swipe = swipe;
            s.record_recent_file(source_path);
        });
        self.set_status_notice(format!(
            "Loaded project {}",
            display_name_from_path(source_path)
        ));
    }

    fn collect_workspace_state(
        &self,
    ) -> (
        Vec<SavedLayerEntry>,
        Vec<SavedVectorEntry>,
        Option<SavedSwipeState>,
    ) {
        let live_entries: Vec<(LayerId, SavedLayerEntry)> = self
            .scene
            .overlays()
            .saved_entries_iter()
            .map(|(id, entry)| (id, SavedLayerEntry::from_persist(entry)))
            .collect();
        let live_swipe = self.swipe.is_active().then(|| {
            (
                self.swipe.split_t(),
                self.swipe.left_layer(),
                self.swipe.right_layer(),
            )
        });
        let (open_layers, swipe) = self
            .restore_tracker
            .assemble_layers_and_swipe(live_entries, live_swipe);

        let mut open_vectors: Vec<SavedVectorEntry> = self
            .scene
            .vectors()
            .saved_entries_iter()
            .map(|entry| SavedVectorEntry {
                path: entry.path.to_path_buf(),
                visible: entry.visible,
                opacity: entry.opacity,
                outline_only: entry.outline_only,
            })
            .collect();
        open_vectors.extend(self.restore_tracker.preserved_vectors().cloned());

        (open_layers, open_vectors, swipe)
    }

    pub(super) fn persist_session_snapshot(
        &mut self,
        include_camera: bool,
        newly_opened: &[PathBuf],
    ) {
        // While a startup CLI `.veldmap.json` load is still in flight, do not
        // overwrite `session.json`'s saved layers/vectors/swipe with an empty
        // pre-load workspace.
        if self.deferred_startup_session.is_some() {
            if !newly_opened.is_empty() {
                let _ = self.storage.mutate_session(|s| {
                    for path in newly_opened {
                        s.record_recent_file(path);
                    }
                });
            }
            return;
        }
        let (open_layers, open_vectors, swipe) = self.collect_workspace_state();
        let cam_pose = if include_camera {
            Some(SavedCameraPose::from_camera(&self.camera))
        } else {
            self.storage.session().last_camera
        };
        let _ = self.storage.mutate_session(|s| {
            s.open_layers = open_layers;
            s.open_vectors = open_vectors;
            s.last_camera = cam_pose;
            s.swipe = swipe;
            for path in newly_opened {
                s.record_recent_file(path);
            }
        });
    }

    fn place_restored_layer_in_order(&mut self, id: LayerId, saved_slot: Option<usize>) {
        if let Some(slot) = saved_slot {
            let live_ids = self.scene.overlays().layer_ids();
            let target_idx = self
                .restore_tracker
                .insertion_index_for_slot(&live_ids, id, slot);
            let _ = self.scene.move_overlay_to_index(id, target_idx);
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Private event handler consolidating local raster and vector open bookkeeping."
    )]
    fn handle_layer_opened(
        &mut self,
        kind: RestoredEntryKind,
        opened_layer_id: Option<LayerId>,
        path: PathBuf,
        bounds: Option<FootprintBounds>,
        fly_to: bool,
        restore_id: Option<u64>,
        newly_opened_paths: &mut Vec<PathBuf>,
        session_changed: &mut bool,
    ) {
        self.panels.layers.on_layer_opened();
        if restore_id.is_none() {
            self.open_workbench_tab(WorkbenchTab::Layers);
        }
        let saved_slot =
            self.restore_tracker
                .on_opened(restore_id, opened_layer_id, kind, Some(&path));
        if let Some(lid) = opened_layer_id {
            self.place_restored_layer_in_order(lid, saved_slot);
        }
        if restore_id.is_none() {
            newly_opened_paths.push(path);
        }
        *session_changed = true;
        if fly_to && let Some(b) = bounds {
            self.focus_footprint(b);
        }
    }

    fn handle_layer_failed(
        &mut self,
        kind: RestoredEntryKind,
        path: PathBuf,
        error: String,
        can_assign_crs: bool,
        restore_id: Option<u64>,
    ) {
        if self.restore_tracker.on_failed(restore_id, &error) {
            tracing::warn!(
                kind = kind.label(),
                path = %path.display(),
                error = %error,
                "Saved workspace layer unavailable; preserving entry in session.json"
            );
        } else {
            tracing::warn!(
                kind = kind.label(),
                path = %path.display(),
                error = %error,
                "Failed to open layer"
            );
            self.panels
                .layers
                .note_failed_path(path, error, can_assign_crs);
            self.open_workbench_tab(WorkbenchTab::Layers);
        }
    }

    pub(super) fn drain_scene_io_events(&mut self) -> bool {
        let (events, stac_changed) = self.scene.poll_io_events();
        if events.is_empty() {
            return stac_changed;
        }
        let mut session_changed = false;
        let mut newly_opened_paths: Vec<PathBuf> = Vec::new();
        for event in events {
            match event {
                SceneIoEvent::Raster(OverlayIoEvent::Opened {
                    id,
                    path,
                    bounds,
                    fly_to,
                    restore_id,
                }) => {
                    tracing::info!(?id, path = %path.display(), "Opened local overlay layer");
                    self.handle_layer_opened(
                        RestoredEntryKind::LocalRaster,
                        Some(id),
                        path,
                        bounds,
                        fly_to,
                        restore_id,
                        &mut newly_opened_paths,
                        &mut session_changed,
                    );
                }
                SceneIoEvent::Raster(OverlayIoEvent::Reloaded { id }) => {
                    tracing::info!(?id, "Reloaded local overlay layer");
                    self.panels.layers.clear_error();
                }
                SceneIoEvent::Raster(OverlayIoEvent::ReloadFailed {
                    path,
                    error,
                    can_assign_crs,
                }) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %error,
                        "Failed to reload local overlay layer"
                    );
                    self.panels
                        .layers
                        .note_failed_path(path, error, can_assign_crs);
                    self.open_workbench_tab(WorkbenchTab::Layers);
                }
                SceneIoEvent::Raster(OverlayIoEvent::Failed {
                    path,
                    error,
                    can_assign_crs,
                    restore_id,
                }) => {
                    self.handle_layer_failed(
                        RestoredEntryKind::LocalRaster,
                        path,
                        error,
                        can_assign_crs,
                        restore_id,
                    );
                }
                SceneIoEvent::Raster(OverlayIoEvent::RemoteOpened {
                    id,
                    title,
                    display_url,
                    is_band_math,
                    bounds,
                    fly_to,
                    restore_id,
                }) => {
                    tracing::info!(?id, title = %title, url = %display_url, "Opened remote COG / BandMath overlay layer");
                    let kind = if is_band_math {
                        RestoredEntryKind::BandMath
                    } else {
                        RestoredEntryKind::RemoteCog
                    };
                    let saved_slot =
                        self.restore_tracker
                            .on_opened(restore_id, Some(id), kind, None);
                    self.place_restored_layer_in_order(id, saved_slot);
                    session_changed = true;
                    if restore_id.is_none() && !self.panels.workbench_open {
                        self.open_workbench_tab(WorkbenchTab::Layers);
                    }
                    if fly_to && let Some(b) = bounds {
                        self.focus_footprint(b);
                    }
                }
                SceneIoEvent::Raster(OverlayIoEvent::RemoteFailed {
                    title,
                    display_url,
                    is_band_math,
                    error,
                    restore_id,
                }) => {
                    if self.restore_tracker.on_failed(restore_id, &error) {
                        tracing::warn!(
                            title = %title,
                            url = %display_url,
                            error = %error,
                            "Saved remote COG / BandMath layer unavailable; preserving entry in session.json"
                        );
                    } else {
                        tracing::warn!(
                            title = %title,
                            url = %display_url,
                            error = %error,
                            "Failed to open remote COG / BandMath overlay layer"
                        );
                        let prefix = if is_band_math {
                            "Band Math"
                        } else {
                            "Remote COG"
                        };
                        let msg = format!("{prefix} '{title}': {error}");
                        if self.panels.workbench_open
                            && self.panels.workbench_tab == WorkbenchTab::CloudStac
                        {
                            self.scene.mutate_stac(|s| s.set_error(msg));
                        } else {
                            self.panels.layers.push_error(msg);
                            self.open_workbench_tab(WorkbenchTab::Layers);
                        }
                    }
                }
                SceneIoEvent::Vector(VectorIoEvent::Opened {
                    id,
                    path,
                    bounds,
                    fly_to,
                    restore_id,
                }) => {
                    tracing::info!(?id, path = %path.display(), "Opened GeoJSON vector layer");
                    self.handle_layer_opened(
                        RestoredEntryKind::Vector,
                        None,
                        path,
                        bounds,
                        fly_to,
                        restore_id,
                        &mut newly_opened_paths,
                        &mut session_changed,
                    );
                }
                SceneIoEvent::Vector(VectorIoEvent::Failed {
                    path,
                    error,
                    restore_id,
                }) => {
                    self.handle_layer_failed(
                        RestoredEntryKind::Vector,
                        path,
                        error,
                        false,
                        restore_id,
                    );
                }
            }
        }
        if let Some((split_t, left, right)) = self.restore_tracker.try_resolve_pending_swipe()
            && matches!(
                self.active_viewport_tool(),
                super::viewport_input::ViewportToolMode::Navigate
                    | super::viewport_input::ViewportToolMode::Swipe
            )
        {
            self.swipe.restore(split_t, left, right);
        }
        if session_changed || !newly_opened_paths.is_empty() {
            self.persist_session_snapshot(false, &newly_opened_paths);
        }
        true
    }
}

#[cfg(test)]
#[path = "session_sync_tests.rs"]
mod tests;
