//! Bounded pool of `SQLITE_OPEN_READ_ONLY` WAL connections for concurrent
//! [`super::TileDiskCache`] reads and O(1) stats queries.

use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex};

use rusqlite::Connection;
use veldmap_core::sync::{lock_unpoisoned, wait_unpoisoned};

use crate::StorageError;

use super::schema::open_read_only_connection;

/// Maximum concurrent and retained read-only WAL connections for lookups.
const MAX_READ_POOL_CONNS: usize = 4;

struct ReaderPoolState {
    idle: Vec<Connection>,
    active: usize,
}

struct ActiveReaderGuard<'a> {
    pool: &'a ReaderPool,
    conn: Option<Connection>,
    keep: bool,
}

impl Drop for ActiveReaderGuard<'_> {
    fn drop(&mut self) {
        let mut state = lock_unpoisoned(&self.pool.state);
        state.active = state.active.saturating_sub(1);
        if self.keep
            && state.idle.len() < MAX_READ_POOL_CONNS
            && let Some(conn) = self.conn.take()
        {
            state.idle.push(conn);
        }
        self.pool.cvar.notify_one();
    }
}

pub(super) struct ReaderPool {
    db_path: PathBuf,
    state: Mutex<ReaderPoolState>,
    cvar: Condvar,
}

impl ReaderPool {
    pub(super) fn new(db_path: PathBuf) -> Self {
        Self {
            db_path,
            state: Mutex::new(ReaderPoolState {
                idle: Vec::with_capacity(MAX_READ_POOL_CONNS),
                active: 0,
            }),
            cvar: Condvar::new(),
        }
    }

    pub(super) fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub(super) fn clear(&self) {
        lock_unpoisoned(&self.state).idle.clear();
    }

    pub(super) fn with_reader<T>(
        &self,
        f: impl FnOnce(&Connection) -> Result<T, rusqlite::Error>,
    ) -> Result<T, StorageError> {
        let pooled = {
            let mut state = lock_unpoisoned(&self.state);
            while state.active >= MAX_READ_POOL_CONNS {
                state = wait_unpoisoned(&self.cvar, state);
            }
            state.active += 1;
            state.idle.pop()
        };

        // Guard ensures `active` is always decremented and a waiter is woken
        // even if opening a connection or executing `f` returns an error or panics.
        let mut guard = ActiveReaderGuard {
            pool: self,
            conn: pooled,
            keep: false,
        };

        let conn: &Connection = match &mut guard.conn {
            Some(conn) => conn,
            slot @ None => slot.insert(open_read_only_connection(&self.db_path)?),
        };
        let result = f(conn);
        if result.is_ok() {
            guard.keep = true;
        }
        Ok(result?)
    }
}
