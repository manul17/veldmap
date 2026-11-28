//! Pure state machine tracking layers and vectors queued from `session.json`
//! or `.veldmap.json` whose background open events have not yet drained, plus
//! any restored entries that failed to open (e.g. unmounted drives, offline
//! network, or 401/403 transient outages), preserving painter's order and swipe
//! slot indices across out-of-order async opens and retries.

use std::collections::HashMap;
use std::path::Path;

use crate::display_name_from_path;
use crate::layers::LayerId;
use crate::stac::AuthTrust;
use crate::storage::session::{SavedLayerEntry, SavedSwipeState, SavedVectorEntry};
use crate::ui::layers::UnavailableLayerNotice;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum RestoredEntryKind {
    LocalRaster,
    RemoteCog,
    BandMath,
    Vector,
}

impl RestoredEntryKind {
    pub(super) fn label(self) -> &'static str {
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
    pub(super) fn auth_trust(&self) -> AuthTrust {
        self.auth_trust
    }

    pub(super) fn alloc_restore_id(&mut self) -> u64 {
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

    pub(super) fn push_layer(
        &mut self,
        restore_id: u64,
        layer_index: Option<usize>,
        entry: SavedLayerEntry,
    ) {
        self.pending.push(PendingEntry {
            restore_id,
            layer_index,
            saved: SavedEntry::Raster(Box::new(entry)),
        });
    }

    pub(super) fn push_vector(&mut self, restore_id: u64, entry: SavedVectorEntry) {
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

    pub(super) fn record_unavailable_layer(
        &mut self,
        layer_index: Option<usize>,
        entry: SavedLayerEntry,
        error: String,
    ) {
        self.record_unavailable(layer_index, SavedEntry::Raster(Box::new(entry)), error);
    }

    pub(super) fn record_unavailable_vector(&mut self, entry: SavedVectorEntry, error: String) {
        self.record_unavailable(None, SavedEntry::Vector(entry), error);
    }

    /// Consume a matching entry when a raster or vector layer opens, returning
    /// its saved painter's-order slot index (if any) so the caller can place it
    /// in original z-order even when async opens finish out of order.
    pub(super) fn on_opened(
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
    pub(super) fn insertion_index_for_slot(
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
    pub(super) fn on_failed(&mut self, restore_id: Option<u64>, error: &str) -> bool {
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
    pub(super) fn try_resolve_pending_swipe(
        &mut self,
    ) -> Option<(f32, Option<LayerId>, Option<LayerId>)> {
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

    pub(super) fn take_unavailable_for_retry(
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
    pub(super) fn assemble_layers_and_swipe(
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

    pub(super) fn preserved_vectors(&self) -> impl Iterator<Item = &SavedVectorEntry> {
        self.all_preserved().filter_map(|e| match e {
            SavedEntry::Vector(vec_layer) => Some(vec_layer),
            SavedEntry::Raster(_) => None,
        })
    }
}

#[cfg(test)]
#[path = "session_sync_tests.rs"]
mod tests;
