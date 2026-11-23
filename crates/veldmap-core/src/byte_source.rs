//! Synchronous caller-buffered positioned I/O (`ByteSource`), read
//! amplification instrumentation (`CountingByteSource`), buffered `Read + Seek`
//! adapter (`BufferedByteReader`), and opaque [`SourceFingerprint`].
//!
//! Avoids memory-mapped files (`memmap2`) to prevent `SIGBUS` UB on truncated,
//! removable, or network-backed inputs.

use std::fmt;
use std::io::{self, Read, Seek, SeekFrom};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

use crate::io::{CancelToken, lock_unpoisoned};
use crate::tile::SourceId;

/// Default internal buffer size for [`BufferedByteReader`] (`64 KiB`).
pub const DEFAULT_READER_BUFFER_BYTES: usize = 64 * 1024;

/// Errors returned by synchronous positioned [`ByteSource`] operations.
#[derive(Debug, Error)]
pub enum IoError {
    #[error("I/O operation cancelled")]
    Cancelled,
    #[error("unexpected EOF reading {requested} bytes at offset {offset} (read {read} bytes)")]
    UnexpectedEof {
        offset: u64,
        requested: usize,
        read: usize,
    },
    #[error("read offset {offset} exceeds source length {len}")]
    OutOfBounds { offset: u64, len: u64 },
    #[error("invalid I/O argument: {0}")]
    InvalidInput(String),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

impl IoError {
    /// Return whether this error represents task cancellation.
    pub const fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }

    /// Convert into `std::io::Error` for `std::io::Read`/`Seek` adapters.
    pub fn into_std_io(self) -> io::Error {
        match self {
            Self::Cancelled => io::Error::new(io::ErrorKind::Interrupted, self),
            Self::UnexpectedEof { .. } => io::Error::new(io::ErrorKind::UnexpectedEof, self),
            Self::OutOfBounds { .. } | Self::InvalidInput(_) => {
                io::Error::new(io::ErrorKind::InvalidInput, self)
            }
            Self::Io(err) => err,
        }
    }
}

/// Strictly synchronous, caller-buffered, thread-safe positioned byte source.
pub trait ByteSource: Send + Sync {
    /// Total byte length of the source.
    fn len(&self) -> u64;

    /// Return `true` when the source has zero bytes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read up to `buf.len()` bytes starting at `offset`, returning the number
    /// of bytes read (`0` only when `buf` is empty or `offset >= len()`).
    fn read_at(&self, offset: u64, buf: &mut [u8], cancel: &CancelToken) -> Result<usize, IoError>;

    /// Fill `buf` completely starting at `offset`, looping across short reads
    /// and returning [`IoError::UnexpectedEof`] if EOF is reached early.
    fn read_exact_at(
        &self,
        offset: u64,
        mut buf: &mut [u8],
        cancel: &CancelToken,
    ) -> Result<(), IoError> {
        let mut cursor = offset;
        let requested = buf.len();
        let mut total_read = 0usize;
        while !buf.is_empty() {
            if cancel.is_cancelled() {
                return Err(IoError::Cancelled);
            }
            let n = self.read_at(cursor, buf, cancel)?;
            if n == 0 {
                return Err(IoError::UnexpectedEof {
                    offset,
                    requested,
                    read: total_read,
                });
            }
            total_read += n;
            cursor = cursor.checked_add(n as u64).ok_or_else(|| {
                IoError::InvalidInput("byte offset overflow during read_exact_at".to_string())
            })?;
            buf = &mut buf[n..];
        }
        Ok(())
    }
}

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

/// Synchronous buffered `std::io::Read + std::io::Seek` adapter over a
/// positioned [`ByteSource`].
///
/// Satisfies small header/IFD tag reads and backward seeks within the currently
/// buffered window without issuing extra `ByteSource::read_at` calls.
pub struct BufferedByteReader {
    source: Arc<dyn ByteSource>,
    cancel: CancelToken,
    len: u64,
    pos: u64,
    buf: Vec<u8>,
    buf_offset: u64,
    buf_len: usize,
}

impl BufferedByteReader {
    /// Create a buffered reader with the default `64 KiB` window.
    pub fn new(source: Arc<dyn ByteSource>, cancel: CancelToken) -> Self {
        Self::with_capacity(source, cancel, DEFAULT_READER_BUFFER_BYTES)
    }

    /// Create a buffered reader with an explicit window size in bytes.
    pub fn with_capacity(
        source: Arc<dyn ByteSource>,
        cancel: CancelToken,
        capacity: usize,
    ) -> Self {
        let len = source.len();
        Self {
            source,
            cancel,
            len,
            pos: 0,
            buf: vec![0u8; capacity.max(1)],
            buf_offset: 0,
            buf_len: 0,
        }
    }

    /// Return the current logical stream position.
    pub const fn position(&self) -> u64 {
        self.pos
    }

    /// Return the underlying [`ByteSource`].
    pub fn source(&self) -> &Arc<dyn ByteSource> {
        &self.source
    }

    fn buffered_end(&self) -> u64 {
        self.buf_offset.saturating_add(self.buf_len as u64)
    }

    fn refill_at(&mut self, offset: u64) -> io::Result<usize> {
        if self.cancel.is_cancelled() {
            return Err(IoError::Cancelled.into_std_io());
        }
        let n = self
            .source
            .read_at(offset, &mut self.buf, &self.cancel)
            .map_err(IoError::into_std_io)?;
        self.buf_offset = offset;
        self.buf_len = n;
        Ok(n)
    }
}

impl Read for BufferedByteReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.cancel.is_cancelled() {
            return Err(IoError::Cancelled.into_std_io());
        }
        if out.is_empty() || self.pos >= self.len {
            return Ok(0);
        }

        // Bypass the internal buffer for very large contiguous reads when the
        // current position is not already buffered, avoiding an extra copy.
        let end = self.buffered_end();
        let in_window = self.pos >= self.buf_offset && self.pos < end;
        if !in_window && out.len() >= self.buf.len() {
            let n = self
                .source
                .read_at(self.pos, out, &self.cancel)
                .map_err(IoError::into_std_io)?;
            self.pos = self.pos.saturating_add(n as u64);
            return Ok(n);
        }

        if !in_window && self.refill_at(self.pos)? == 0 {
            return Ok(0);
        }

        let start = (self.pos - self.buf_offset) as usize;
        let avail = self.buf_len - start;
        let copy_len = avail.min(out.len());
        out[..copy_len].copy_from_slice(&self.buf[start..start + copy_len]);
        self.pos = self.pos.saturating_add(copy_len as u64);
        Ok(copy_len)
    }
}

impl Seek for BufferedByteReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let base: i128 = match pos {
            SeekFrom::Start(off) => i128::from(off),
            SeekFrom::Current(delta) => i128::from(self.pos) + i128::from(delta),
            SeekFrom::End(delta) => i128::from(self.len) + i128::from(delta),
        };
        if base < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot seek before start of ByteSource",
            ));
        }
        let target = u64::try_from(base).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "seek position overflows u64")
        })?;
        self.pos = target;
        Ok(target)
    }
}

/// Opaque 256-bit content fingerprint uniquely identifying a raster byte stream
/// (sampled BLAKE3 digest for local files, or metadata digest for remote COGs).
///
/// Formats as 64 lowercase ASCII hex characters, which always satisfies strict
/// disk-cache `source_key` validation (`<= 128` safe ASCII bytes).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceFingerprint([u8; 32]);

impl SourceFingerprint {
    /// Wrap a 32-byte digest into an opaque [`SourceFingerprint`].
    pub const fn from_digest(digest: [u8; 32]) -> Self {
        Self(digest)
    }

    /// Borrow the raw 32-byte digest.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Encode as 64 lowercase hexadecimal ASCII characters.
    pub fn to_hex(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(64);
        for &byte in &self.0 {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        out
    }

    /// Derive a deterministic [`SourceId`] from this fingerprint.
    pub fn to_source_id(&self) -> SourceId {
        SourceId::from_stable_key(&self.to_hex())
    }
}

impl fmt::Display for SourceFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for SourceFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hex = self.to_hex();
        write!(f, "SourceFingerprint({}..)", &hex[..12])
    }
}
