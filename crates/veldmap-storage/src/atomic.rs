//! Durable atomic file writes (`sync_all()` + synchronous handle close before
//! rename on Windows) for user settings and catalog state.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::StorageError;
use crate::paths::append_path_suffix;

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);
const MAX_TEMP_CREATE_ATTEMPTS: usize = 16;

fn parent_and_file_name(target: &Path) -> Result<(PathBuf, &std::ffi::OsStr), StorageError> {
    let parent = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let file_name = target
        .file_name()
        .filter(|n| !n.is_empty())
        .ok_or_else(|| StorageError::InvalidPath(target.display().to_string()))?;
    Ok((parent, file_name))
}

fn sibling_temp_path(parent: &Path, file_name: &std::ffi::OsStr) -> PathBuf {
    let pid = std::process::id();
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let mut name = OsString::from(".");
    name.push(file_name);
    name.push(format!(".tmp.{pid}.{seq}"));
    parent.join(name)
}

#[cfg(unix)]
fn sync_parent_dir(parent: &Path) {
    if let Ok(dir) = File::open(parent) {
        let _ = dir.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_parent_dir(_parent: &Path) {}

#[cfg(windows)]
fn is_transient_windows_rename_error(err: &std::io::Error) -> bool {
    const ERROR_SHARING_VIOLATION: i32 = 32;
    const ERROR_LOCK_VIOLATION: i32 = 33;
    err.kind() == std::io::ErrorKind::PermissionDenied
        || matches!(
            err.raw_os_error(),
            Some(ERROR_SHARING_VIOLATION | ERROR_LOCK_VIOLATION)
        )
}

#[cfg(windows)]
fn rename_replace(tmp: &Path, target: &Path) -> std::io::Result<()> {
    // On Windows, a transient antivirus/indexer scan or concurrent reader can
    // momentarily hold the destination path; retry briefly before failing.
    let mut last_err = None;
    for attempt in 0..5u64 {
        match fs::rename(tmp, target) {
            Ok(()) => return Ok(()),
            Err(err) if is_transient_windows_rename_error(&err) && attempt < 4 => {
                last_err = Some(err);
                std::thread::sleep(std::time::Duration::from_millis(10 * (1 << attempt)));
            }
            Err(err) => return Err(err),
        }
    }
    Err(last_err.unwrap_or_else(|| std::io::Error::other("atomic rename failed")))
}

#[cfg(not(windows))]
fn rename_replace(tmp: &Path, target: &Path) -> std::io::Result<()> {
    fs::rename(tmp, target)
}

/// Atomically and durably write `bytes` to `path`.
///
/// Writes to a unique sibling temporary file in the same directory, flushes
/// data and metadata via `sync_all()`, **closes the `File` handle synchronously
/// before renaming** (required on Windows), atomically renames over `path`,
/// and best-effort fsyncs the parent directory on Unix.
pub fn write_atomic_durable(path: &Path, bytes: &[u8]) -> Result<(), StorageError> {
    write_atomic_impl(path, bytes, false)
}

/// Atomically and durably write `bytes` to `path` with owner-only permissions
/// (`0600` on Unix; on Windows, relies on parent user-profile directory ACLs)
/// for configuration files that may store API keys.
fn write_atomic_durable_private(path: &Path, bytes: &[u8]) -> Result<(), StorageError> {
    write_atomic_impl(path, bytes, true)
}

fn create_unique_temp(
    path: &Path,
    private: bool,
) -> Result<(PathBuf, PathBuf, File), StorageError> {
    let (parent, file_name) = parent_and_file_name(path)?;
    fs::create_dir_all(&parent)?;
    let mut last_err = None;
    for _ in 0..MAX_TEMP_CREATE_ATTEMPTS {
        let tmp_path = sibling_temp_path(&parent, file_name);
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        if private {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        #[cfg(not(unix))]
        let _ = private;
        match opts.open(&tmp_path) {
            Ok(file) => return Ok((parent, tmp_path, file)),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                // Retry with the next sequence number without touching the
                // pre-existing colliding file.
                last_err = Some(err);
            }
            Err(err) => return Err(StorageError::Io(err)),
        }
    }
    Err(StorageError::Io(last_err.unwrap_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "failed to allocate unique temporary file",
        )
    })))
}

fn write_atomic_impl(path: &Path, bytes: &[u8], private: bool) -> Result<(), StorageError> {
    let (parent, tmp_path, mut file) = create_unique_temp(path, private)?;

    let write_res = (|| -> std::io::Result<()> {
        file.write_all(bytes)?;
        file.sync_all()?;
        // Explicitly close the file handle before rename (Windows requirement).
        drop(file);
        Ok(())
    })();

    if let Err(err) = write_res {
        let _ = fs::remove_file(&tmp_path);
        return Err(StorageError::Io(err));
    }

    if let Err(err) = rename_replace(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(StorageError::Io(err));
    }

    sync_parent_dir(&parent);
    Ok(())
}

/// Serialize `value` as pretty JSON and persist it via atomic durable write
/// with standard user permissions (suitable for sanitized user-exported project
/// files such as `.veldmap.json`).
pub fn write_json_durable<T: Serialize>(path: &Path, value: &T) -> Result<(), StorageError> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_atomic_durable(path, &bytes)
}

/// Serialize `value` as pretty JSON and persist it via private atomic durable
/// write (`0600` permissions on Unix).
pub fn write_json_durable_private<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), StorageError> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_atomic_durable_private(path, &bytes)
}

/// Read and deserialize a bounded regular UTF-8 JSON file without renaming it
/// on parse failure (suitable for explicitly opened user project files).
pub fn read_json_limited<T: DeserializeOwned>(
    path: &Path,
    max_bytes: u64,
) -> Result<T, StorageError> {
    let text = crate::read_regular_file_utf8_limited(path, max_bytes)?;
    Ok(serde_json::from_str(&text)?)
}

/// Read JSON from `path`, returning `T::default()` if the file does not exist.
fn read_json_or_default<T: DeserializeOwned + Default>(path: &Path) -> Result<T, StorageError> {
    match fs::read(path) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(T::default()),
        Err(err) => Err(StorageError::Io(err)),
    }
}

/// Read JSON from `path`, falling back to `T::default()` when missing or
/// unreadable. On JSON parse failure, preserves the corrupt file as
/// `<path>.corrupt` via atomic rename so a subsequent save never silently
/// destroys a hand-edited configuration file.
pub fn read_json_or_default_with_backup<T: DeserializeOwned + Default>(path: &Path) -> T {
    match read_json_or_default::<T>(path) {
        Ok(value) => value,
        Err(StorageError::Json(err)) => {
            let backup = append_path_suffix(path, ".corrupt");
            match rename_replace(path, &backup) {
                Ok(()) => {
                    tracing::warn!(
                        path = %path.display(),
                        backup = %backup.display(),
                        error = %err,
                        "Failed to parse persisted JSON; backed up corrupt file and using defaults"
                    );
                }
                Err(rename_err) => {
                    tracing::warn!(
                        path = %path.display(),
                        backup = %backup.display(),
                        parse_error = %err,
                        backup_error = %rename_err,
                        "Failed to parse persisted JSON and failed to back up corrupt file; using defaults"
                    );
                }
            }
            T::default()
        }
        Err(err) => {
            tracing::warn!(
                path = %path.display(),
                error = %err,
                "Failed to read persisted JSON; using defaults"
            );
            T::default()
        }
    }
}

#[cfg(test)]
#[path = "atomic_tests.rs"]
mod tests;
