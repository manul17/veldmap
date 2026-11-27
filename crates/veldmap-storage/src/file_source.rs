//! Safe regular-file helpers and positioned local-file [`ByteSource`]
//! implementation ([`LocalFileSource`] and [`read_regular_file_utf8_limited`]):
//! lock-free POSIX `pread` on a single immutable file descriptor on Unix,
//! pooled `std::fs::File` handles with `seek_read` on Windows (avoiding
//! `memmap2`), and size-bounded UTF-8 file reading without TOCTOU races.

use std::fs::{File, Metadata};
use std::io;
use std::path::Path;
#[cfg(not(unix))]
use std::path::PathBuf;
#[cfg(not(unix))]
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

use veldmap_core::io::{ByteSource, IoError};
use veldmap_core::sync::CancelToken;
#[cfg(not(unix))]
use veldmap_core::sync::lock_unpoisoned;

use crate::StorageError;

/// Verify that `meta` describes a regular file, rejecting directories, FIFOs,
/// and device nodes with a consistent [`StorageError::InvalidPath`].
fn ensure_regular_file(meta: &Metadata, path: &Path) -> Result<(), StorageError> {
    if !meta.is_file() {
        return Err(StorageError::InvalidPath(format!(
            "'{}' is not a regular file",
            path.display()
        )));
    }
    Ok(())
}

/// Open `path` after verifying regular-file status both before and after
/// `File::open`, returning the open handle and its handle-level [`Metadata`].
///
/// Checking before `File::open` ensures Windows directories report
/// [`StorageError::InvalidPath`] (rather than `PermissionDenied`) and Unix
/// FIFOs do not block; re-checking on the opened handle closes the TOCTOU
/// window against path replacement.
fn open_regular_file(path: &Path) -> Result<(File, Metadata), StorageError> {
    let path_meta = std::fs::metadata(path)?;
    ensure_regular_file(&path_meta, path)?;
    let file = File::open(path)?;
    let meta = file.metadata()?;
    ensure_regular_file(&meta, path)?;
    Ok((file, meta))
}

/// Safely read a regular UTF-8 text file up to `max_bytes` without a TOCTOU
/// window between metadata size checking and reading.
///
/// Rejects directories, FIFOs, and device nodes before and after `File::open`,
/// checks `file.metadata().len() <= max_bytes`, and caps the actual read via
/// [`std::io::Read::take`] so a file growing concurrently cannot exhaust RAM.
pub fn read_regular_file_utf8_limited(path: &Path, max_bytes: u64) -> Result<String, StorageError> {
    use std::io::Read;

    let (file, meta) = open_regular_file(path)?;
    if meta.len() > max_bytes {
        return Err(StorageError::FileTooLarge {
            path: path.display().to_string(),
            limit_bytes: max_bytes,
        });
    }
    let initial_cap = usize::try_from(meta.len().min(max_bytes))
        .unwrap_or(0)
        .min(4 * 1024 * 1024);
    let mut bytes = Vec::with_capacity(initial_cap);
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes {
        return Err(StorageError::FileTooLarge {
            path: path.display().to_string(),
            limit_bytes: max_bytes,
        });
    }
    String::from_utf8(bytes)
        .map_err(|err| StorageError::Io(io::Error::new(io::ErrorKind::InvalidData, err)))
}

/// Maximum idle read-only `File` handles cached per [`LocalFileSource`] on
/// platforms where positioned reads mutate the kernel file pointer.
#[cfg(not(unix))]
const MAX_POOLED_HANDLES: usize = 8;

/// Thread-safe, caller-buffered positioned reader over a local file.
///
/// On Unix, retains a single immutable open file descriptor and calls lock-free
/// POSIX `pread(2)` (`FileExt::read_at`), eliminating pool contention and
/// path-reopen inode skew. On Windows/other platforms, pools read-only `File`
/// handles without holding the pool lock across disk I/O.
pub struct LocalFileSource {
    #[cfg(not(unix))]
    path: PathBuf,
    len: u64,
    mtime_ns: Option<i128>,
    #[cfg(unix)]
    file: File,
    #[cfg(not(unix))]
    pool: Mutex<Vec<File>>,
}

impl LocalFileSource {
    /// Open `path` for safe positioned reading and capture its `(len, mtime_ns)`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let path_ref = path.as_ref();
        let (file, meta) = open_regular_file(path_ref)?;
        let len = meta.len();
        let mtime_ns = metadata_mtime_ns(&meta);
        Ok(Self {
            #[cfg(not(unix))]
            path: path_ref.to_path_buf(),
            len,
            mtime_ns,
            #[cfg(unix)]
            file,
            #[cfg(not(unix))]
            pool: Mutex::new(vec![file]),
        })
    }

    /// Last-modified timestamp in nanoseconds relative to `UNIX_EPOCH`, if
    /// available on this platform/filesystem.
    pub(crate) const fn mtime_ns(&self) -> Option<i128> {
        self.mtime_ns
    }

    #[cfg(not(unix))]
    fn acquire_handle(&self) -> Result<PooledFile<'_>, IoError> {
        let maybe = lock_unpoisoned(&self.pool).pop();
        let file = match maybe {
            Some(file) => file,
            None => open_regular_file(&self.path)
                .map(|(file, _)| file)
                .map_err(|err| match err {
                    StorageError::Io(io_err) => IoError::Io(io_err),
                    StorageError::InvalidPath(msg) => IoError::InvalidInput(msg),
                    other => IoError::InvalidInput(other.to_string()),
                })?,
        };
        Ok(PooledFile {
            pool: &self.pool,
            file: Some(file),
            reusable: false,
        })
    }
}

#[cfg(not(unix))]
struct PooledFile<'a> {
    pool: &'a Mutex<Vec<File>>,
    file: Option<File>,
    reusable: bool,
}

#[cfg(not(unix))]
impl Drop for PooledFile<'_> {
    fn drop(&mut self) {
        if !self.reusable {
            return;
        }
        if let Some(file) = self.file.take() {
            let mut pool = lock_unpoisoned(self.pool);
            if pool.len() < MAX_POOLED_HANDLES {
                pool.push(file);
            }
        }
    }
}

impl ByteSource for LocalFileSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8], cancel: &CancelToken) -> Result<usize, IoError> {
        if cancel.is_cancelled() {
            return Err(IoError::Cancelled);
        }
        if buf.is_empty() || offset >= self.len {
            return Ok(0);
        }

        #[cfg(unix)]
        {
            read_file_at(&self.file, offset, buf, cancel)
        }
        #[cfg(not(unix))]
        {
            let mut handle = self.acquire_handle()?;
            let file = handle
                .file
                .as_mut()
                .ok_or_else(|| IoError::InvalidInput("missing pooled file handle".to_string()))?;
            let read = read_file_at(file, offset, buf, cancel)?;
            handle.reusable = true;
            Ok(read)
        }
    }
}

pub(crate) fn metadata_mtime_ns(meta: &Metadata) -> Option<i128> {
    let modified = meta.modified().ok()?;
    match modified.duration_since(UNIX_EPOCH) {
        Ok(dur) => i128::try_from(dur.as_nanos()).ok(),
        Err(err) => i128::try_from(err.duration().as_nanos()).ok().map(|ns| -ns),
    }
}

#[cfg(unix)]
fn read_file_at(
    file: &File,
    offset: u64,
    buf: &mut [u8],
    cancel: &CancelToken,
) -> Result<usize, IoError> {
    use std::os::unix::fs::FileExt;
    loop {
        if cancel.is_cancelled() {
            return Err(IoError::Cancelled);
        }
        match file.read_at(buf, offset) {
            Ok(n) => return Ok(n),
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(IoError::Io(err)),
        }
    }
}

#[cfg(windows)]
fn read_file_at(
    file: &mut File,
    offset: u64,
    buf: &mut [u8],
    cancel: &CancelToken,
) -> Result<usize, IoError> {
    use std::os::windows::fs::FileExt;
    let mut total = 0usize;
    while total < buf.len() {
        if cancel.is_cancelled() {
            return Err(IoError::Cancelled);
        }
        let cursor = offset
            .checked_add(total as u64)
            .ok_or_else(|| IoError::InvalidInput("file read offset overflow".to_string()))?;
        match file.seek_read(&mut buf[total..], cursor) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(IoError::Io(err)),
        }
    }
    Ok(total)
}

#[cfg(not(any(unix, windows)))]
fn read_file_at(
    file: &mut File,
    offset: u64,
    buf: &mut [u8],
    cancel: &CancelToken,
) -> Result<usize, IoError> {
    use std::io::{Read, Seek, SeekFrom};
    if cancel.is_cancelled() {
        return Err(IoError::Cancelled);
    }
    file.seek(SeekFrom::Start(offset))?;
    loop {
        if cancel.is_cancelled() {
            return Err(IoError::Cancelled);
        }
        match file.read(buf) {
            Ok(n) => return Ok(n),
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(IoError::Io(err)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::Arc;
    use std::thread;

    use super::*;
    use crate::test_support::TestTempDir;

    #[test]
    fn local_file_source_reads_concurrently_and_checks_cancellation() {
        let dir = TestTempDir::new("file-source");
        let path = dir.path().join("sample.bin");
        let payload: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        {
            let mut f = File::create(&path).expect("create sample");
            f.write_all(&payload).expect("write sample");
        }

        let source = Arc::new(LocalFileSource::open(&path).expect("open source"));
        assert_eq!(source.len(), 4096);
        assert!(source.mtime_ns().is_some());

        let handles: Vec<_> = (0..8)
            .map(|tid| {
                let src = Arc::clone(&source);
                let expected = payload.clone();
                thread::spawn(move || {
                    let cancel = CancelToken::new();
                    let offset = (tid * 331) as u64;
                    let mut buf = [0u8; 500];
                    src.read_exact_at(offset, &mut buf, &cancel)
                        .expect("concurrent read_exact_at");
                    assert_eq!(&buf, &expected[offset as usize..offset as usize + 500]);
                })
            })
            .collect();
        for h in handles {
            h.join().expect("join reader");
        }

        let cancel = CancelToken::new();
        cancel.cancel();
        let mut buf = [0u8; 16];
        assert!(
            source
                .read_at(0, &mut buf, &cancel)
                .is_err_and(|e| e.is_cancelled())
        );

        assert!(matches!(
            LocalFileSource::open(dir.path()),
            Err(StorageError::InvalidPath(_))
        ));
    }

    #[test]
    fn read_regular_file_utf8_limited_enforces_cap_utf8_and_regular_file() {
        let dir = TestTempDir::new("utf8_limited");
        let path = dir.path().join("sample.txt");

        // Multi-byte UTF-8: "abé" is 4 bytes (`[0x61, 0x62, 0xC3, 0xA9]`).
        std::fs::write(&path, "abé").expect("write utf8");
        assert_eq!(
            read_regular_file_utf8_limited(&path, 4).expect("exact limit"),
            "abé"
        );
        assert_eq!(
            read_regular_file_utf8_limited(&path, u64::MAX).expect("u64::MAX limit"),
            "abé"
        );
        // Limit smaller than file size (even splitting a multi-byte char) must
        // report `FileTooLarge`, never `Io(InvalidData)`.
        assert!(matches!(
            read_regular_file_utf8_limited(&path, 2),
            Err(StorageError::FileTooLarge { limit_bytes: 2, .. })
        ));

        // Invalid UTF-8 within limit reports `Io(InvalidData)`.
        let bad_utf8 = dir.path().join("bad.bin");
        std::fs::write(&bad_utf8, [0xFF, 0xFE, 0xFD]).expect("write invalid utf8");
        assert!(matches!(
            read_regular_file_utf8_limited(&bad_utf8, 16),
            Err(StorageError::Io(ref err)) if err.kind() == io::ErrorKind::InvalidData
        ));

        // Directories are rejected as non-regular files.
        assert!(matches!(
            read_regular_file_utf8_limited(dir.path(), 64),
            Err(StorageError::InvalidPath(_))
        ));
    }
}
