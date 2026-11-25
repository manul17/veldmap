//! Generation-checked table of active disk/HTTP/decode tasks for
//! [`super::PipelineState`].

use std::collections::HashMap;

use veldmap_core::sync::CancelToken;
use veldmap_core::tile::TileKey;

#[derive(Default)]
pub(super) struct InFlightTable {
    entries: HashMap<TileKey, (u64, CancelToken)>,
    next_gen: u64,
}

impl InFlightTable {
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn contains(&self, key: &TileKey) -> bool {
        self.entries.contains_key(key)
    }

    pub(super) fn register(&mut self, key: TileKey, shutdown: &CancelToken) -> (u64, CancelToken) {
        self.next_gen = self.next_gen.wrapping_add(1);
        let gen_id = self.next_gen;
        let tile_cancel = shutdown.child_token();
        self.entries.insert(key, (gen_id, tile_cancel.clone()));
        (gen_id, tile_cancel)
    }

    pub(super) fn remove_if_gen(&mut self, key: &TileKey, generation: u64) -> bool {
        if self
            .entries
            .get(key)
            .is_some_and(|(gen_id, _)| *gen_id == generation)
        {
            self.entries.remove(key);
            true
        } else {
            false
        }
    }

    pub(super) fn drain_tokens(&mut self) -> Vec<CancelToken> {
        self.entries.drain().map(|(_, (_, tok))| tok).collect()
    }

    pub(super) fn retain_and_collect_cancelled(
        &mut self,
        mut keep: impl FnMut(&TileKey) -> bool,
    ) -> Vec<CancelToken> {
        let mut cancelled = Vec::new();
        self.entries.retain(|key, (_, tok)| {
            let ok = keep(key);
            if !ok {
                cancelled.push(tok.clone());
            }
            ok
        });
        cancelled
    }
}
