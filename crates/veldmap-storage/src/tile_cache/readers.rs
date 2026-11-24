//! Bounded pool of `SQLITE_OPEN_READ_ONLY` WAL connections for concurrent
//! [`super::TileDiskCache`] reads and O(1) stats queries.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::Connection;
use veldmap_core::sync::lock_unpoisoned;

use crate::StorageError;

use super::schema::open_read_only_connection;

/// Maximum pooled read-only WAL connections retained for concurrent lookups.
const MAX_READ_POOL_CONNS: usize = 4;

pub(super) struct ReaderPool {
    db_path: PathBuf,
    conns: Mutex<Vec<Connection>>,
}

impl ReaderPool {
    pub(super) fn new(db_path: PathBuf) -> Self {
        Self {
            db_path,
            conns: Mutex::new(Vec::with_capacity(MAX_READ_POOL_CONNS)),
        }
    }

    pub(super) fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub(super) fn clear(&self) {
        lock_unpoisoned(&self.conns).clear();
    }

    pub(super) fn with_reader<T>(
        &self,
        f: impl FnOnce(&Connection) -> Result<T, rusqlite::Error>,
    ) -> Result<T, StorageError> {
        let conn = {
            let mut guard = lock_unpoisoned(&self.conns);
            guard.pop()
        };
        let conn = match conn {
            Some(c) => c,
            None => open_read_only_connection(&self.db_path)?,
        };
        let result = f(&conn);
        if result.is_ok() {
            let mut guard = lock_unpoisoned(&self.conns);
            if guard.len() < MAX_READ_POOL_CONNS {
                guard.push(conn);
            }
        }
        Ok(result?)
    }
}
