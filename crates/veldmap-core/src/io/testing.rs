//! In-memory and read-amplification-instrumented [`ByteSource`] test doubles.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::sync::{CancelToken, lock_unpoisoned};

use super::byte_source::{ByteSource, IoError};

/// In-memory [`ByteSource`] backed by a shared byte slice.
#[derive(Clone, Debug)]
pub struct MemoryByteSource {
    bytes: Arc<[u8]>,
}

impl MemoryByteSource {
    /// Wrap an owned `Vec<u8>` as a [`MemoryByteSource`].
    pub fn from_vec(bytes: Vec<u8>) -> Self {
        Self {
            bytes: Arc::from(bytes),
        }
    }

    /// Wrap a shared slice as a [`MemoryByteSource`].
    pub fn new(bytes: Arc<[u8]>) -> Self {
        Self { bytes }
    }
}

impl ByteSource for MemoryByteSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8], cancel: &CancelToken) -> Result<usize, IoError> {
        if cancel.is_cancelled() {
            return Err(IoError::Cancelled);
        }
        if buf.is_empty() || offset >= self.bytes.len() as u64 {
            return Ok(0);
        }
        let start = offset as usize;
        let avail = self.bytes.len() - start;
        let count = avail.min(buf.len());
        buf[..count].copy_from_slice(&self.bytes[start..start + count]);
        Ok(count)
    }
}

/// Recorded positioned read span captured by [`CountingByteSource`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteReadSpan {
    pub offset: u64,
    pub len: usize,
}

/// Snapshot of I/O activity observed by [`CountingByteSource`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReadStats {
    pub read_calls: u64,
    pub bytes_read: u64,
    pub spans: Vec<ByteReadSpan>,
}

/// Instrumentation wrapper around any [`ByteSource`] recording read calls,
/// byte counts, and exact `(offset, len)` spans for read-amplification tests.
pub struct CountingByteSource {
    inner: Arc<dyn ByteSource>,
    read_calls: AtomicU64,
    bytes_read: AtomicU64,
    spans: std::sync::Mutex<Vec<ByteReadSpan>>,
}

impl CountingByteSource {
    /// Wrap `inner` with positioned read accounting.
    pub fn new(inner: Arc<dyn ByteSource>) -> Self {
        Self {
            inner,
            read_calls: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            spans: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Total `read_at` calls issued so far.
    pub fn read_calls(&self) -> u64 {
        self.read_calls.load(Ordering::Acquire)
    }

    /// Total bytes returned by `read_at` calls so far.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Acquire)
    }

    /// Snapshot read counts and exact `(offset, len)` spans.
    pub fn snapshot(&self) -> ReadStats {
        let spans = lock_unpoisoned(&self.spans).clone();
        ReadStats {
            read_calls: self.read_calls(),
            bytes_read: self.bytes_read(),
            spans,
        }
    }

    /// Clear recorded counters and spans.
    pub fn reset(&self) {
        self.read_calls.store(0, Ordering::Release);
        self.bytes_read.store(0, Ordering::Release);
        lock_unpoisoned(&self.spans).clear();
    }
}

impl ByteSource for CountingByteSource {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8], cancel: &CancelToken) -> Result<usize, IoError> {
        let n = self.inner.read_at(offset, buf, cancel)?;
        self.read_calls.fetch_add(1, Ordering::AcqRel);
        self.bytes_read.fetch_add(n as u64, Ordering::AcqRel);
        if n > 0 {
            lock_unpoisoned(&self.spans).push(ByteReadSpan { offset, len: n });
        }
        Ok(n)
    }
}
