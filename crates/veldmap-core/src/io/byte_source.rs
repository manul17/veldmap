//! Synchronous caller-buffered positioned I/O (`ByteSource`) and buffered
//! `Read + Seek` adapter (`BufferedByteReader`).
//!
//! Avoids memory-mapped files (`memmap2`) to prevent `SIGBUS` UB on truncated,
//! removable, or network-backed inputs.

use std::io::{self, BufRead, Read, Seek, SeekFrom};
use std::sync::Arc;

use thiserror::Error;

use crate::sync::CancelToken;

/// Default internal buffer size for [`BufferedByteReader`] (`64 KiB`).
const DEFAULT_READER_BUFFER_BYTES: usize = 64 * 1024;

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
    ///
    /// Uses `ErrorKind::Other` (never `ErrorKind::Interrupted`) for
    /// [`Self::Cancelled`] so `std::io::Read::read_exact` / `read_to_end` abort
    /// immediately instead of retrying in a loop.
    pub fn into_std_io(self) -> io::Error {
        match self {
            Self::Cancelled => io::Error::other(self),
            Self::UnexpectedEof { .. } => io::Error::new(io::ErrorKind::UnexpectedEof, self),
            Self::InvalidInput(_) => io::Error::new(io::ErrorKind::InvalidInput, self),
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

    /// Return `true` when reads cross a high-latency remote transport (such as
    /// HTTP Range streaming), allowing decoders and schedulers to reject
    /// whole-file sequential passes and avoid un-prefetched hover I/O.
    fn is_remote(&self) -> bool {
        false
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
            if n > buf.len() {
                return Err(IoError::InvalidInput(format!(
                    "ByteSource::read_at returned {n} bytes for a {}-byte buffer",
                    buf.len()
                )));
            }
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
    fn with_capacity(source: Arc<dyn ByteSource>, cancel: CancelToken, capacity: usize) -> Self {
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

    fn buffered_end(&self) -> u64 {
        self.buf_offset.saturating_add(self.buf_len as u64)
    }

    fn refill_at(&mut self, offset: u64) -> io::Result<usize> {
        if self.cancel.is_cancelled() {
            return Err(IoError::Cancelled.into_std_io());
        }
        // Invalidate the cached window before calling `read_at` so a failed or
        // short-written refill cannot leave stale metadata pointing at overwritten bytes.
        self.buf_len = 0;
        let n = self
            .source
            .read_at(offset, &mut self.buf, &self.cancel)
            .map_err(IoError::into_std_io)?;
        if n > self.buf.len() {
            return Err(IoError::InvalidInput(format!(
                "ByteSource::read_at returned {n} bytes for a {}-byte window",
                self.buf.len()
            ))
            .into_std_io());
        }
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
            if n > out.len() {
                return Err(IoError::InvalidInput(format!(
                    "ByteSource::read_at returned {n} bytes for a {}-byte output buffer",
                    out.len()
                ))
                .into_std_io());
            }
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

impl BufRead for BufferedByteReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.cancel.is_cancelled() {
            return Err(IoError::Cancelled.into_std_io());
        }
        if self.pos >= self.len {
            return Ok(&[]);
        }
        let end = self.buffered_end();
        let in_window = self.pos >= self.buf_offset && self.pos < end;
        if !in_window && self.refill_at(self.pos)? == 0 {
            return Ok(&[]);
        }
        let start = (self.pos - self.buf_offset) as usize;
        Ok(&self.buf[start..self.buf_len])
    }

    fn consume(&mut self, amt: usize) {
        let end = self.buffered_end();
        if self.pos >= self.buf_offset && self.pos < end {
            let avail = (end - self.pos) as usize;
            let take = amt.min(avail);
            self.pos = self.pos.saturating_add(take as u64);
        }
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

#[cfg(test)]
#[path = "byte_source_tests.rs"]
mod tests;
