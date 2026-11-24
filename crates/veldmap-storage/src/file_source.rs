//! Safe positioned local-file [`ByteSource`] implementation using lock-free
//! POSIX `pread` on a single immutable file descriptor on Unix, and pooled
//! `std::fs::File` handles with `seek_read` on Windows (avoiding `memmap2`).

use std::fs::{File, Metadata};
use std::io;
use std::path::{Path, PathBuf};
#[cfg(not(unix))]
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

use veldmap_core::io::{ByteSource, IoError};
use veldmap_core::sync::CancelToken;
#[cfg(not(unix))]
use veldmap_core::sync::lock_unpoisoned;

use crate::StorageError;

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
        // Check regular-file status before `File::open` so Windows directories
        // report `InvalidPath` (not `PermissionDenied`) and Unix FIFOs do not block.
        let path_meta = std::fs::metadata(path_ref)?;
        if !path_meta.is_file() {
            return Err(StorageError::InvalidPath(format!(
                "'{}' is not a regular file",
                path_ref.display()
            )));
        }
        let file = File::open(path_ref)?;
        let meta = file.metadata()?;
        if !meta.is_file() {
            return Err(StorageError::InvalidPath(format!(
                "'{}' is not a regular file",
                path_ref.display()
            )));
        }
        let len = meta.len();
        let mtime_ns = metadata_mtime_ns(&meta);
        Ok(Self {
            path: path_ref.to_path_buf(),
            len,
            mtime_ns,
            #[cfg(unix)]
            file,
            #[cfg(not(unix))]
            pool: Mutex::new(vec![file]),
        })
    }

    /// Underlying file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Last-modified timestamp in nanoseconds relative to `UNIX_EPOCH`, if
    /// available on this platform/filesystem.
    pub const fn mtime_ns(&self) -> Option<i128> {
        self.mtime_ns
    }

    #[cfg(not(unix))]
    fn acquire_handle(&self) -> Result<PooledFile<'_>, IoError> {
        let maybe = lock_unpoisoned(&self.pool).pop();
        let file = match maybe {
            Some(file) => file,
            None => File::open(&self.path)?,
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
}
