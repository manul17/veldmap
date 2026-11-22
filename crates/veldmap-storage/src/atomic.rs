//! Durable atomic file writes (`sync_all()` + synchronous handle close before
//! rename on Windows) for user settings and catalog state.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::StorageError;

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

fn sibling_temp_path(target: &Path) -> Result<(PathBuf, PathBuf), StorageError> {
    let parent = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let file_name = target
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| StorageError::InvalidPath(target.display().to_string()))?;
    let pid = std::process::id();
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = parent.join(format!(".{file_name}.tmp.{pid}.{seq}"));
    Ok((parent, tmp))
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
fn rename_replace(tmp: &Path, target: &Path) -> std::io::Result<()> {
    // On Windows, a transient antivirus/indexer scan can momentarily hold the
    // destination path; retry briefly before failing.
    let mut last_err = None;
    for attempt in 0..4u64 {
        match fs::rename(tmp, target) {
            Ok(()) => return Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied && attempt < 3 => {
                last_err = Some(err);
                std::thread::sleep(std::time::Duration::from_millis(5 * (attempt + 1)));
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

/// Atomically and durably write `bytes` to `path` with owner-only (`0600` on
/// Unix) file permissions for configuration files that may store API keys.
pub fn write_atomic_durable_private(path: &Path, bytes: &[u8]) -> Result<(), StorageError> {
    write_atomic_impl(path, bytes, true)
}

fn write_atomic_impl(path: &Path, bytes: &[u8], private: bool) -> Result<(), StorageError> {
    let (parent, tmp_path) = sibling_temp_path(path)?;
    fs::create_dir_all(&parent)?;

    let write_res = (|| -> std::io::Result<()> {
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        if private {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        #[cfg(not(unix))]
        let _ = private;
        let mut file = opts.open(&tmp_path)?;
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

/// Serialize `value` as pretty JSON and persist it via [`write_atomic_durable`].
pub fn write_json_durable<T: Serialize>(path: &Path, value: &T) -> Result<(), StorageError> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_atomic_durable(path, &bytes)
}

/// Serialize `value` as pretty JSON and persist it via
/// [`write_atomic_durable_private`] (`0600` permissions on Unix).
pub fn write_json_durable_private<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), StorageError> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_atomic_durable_private(path, &bytes)
}

/// Read JSON from `path`, returning `T::default()` if the file does not exist.
pub fn read_json_or_default<T: DeserializeOwned + Default>(path: &Path) -> Result<T, StorageError> {
    match fs::read(path) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(T::default()),
        Err(err) => Err(StorageError::Io(err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = std::env::temp_dir().join(format!(
            "veldmap-storage-{label}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp test dir");
        dir
    }

    #[test]
    fn write_atomic_durable_creates_overwrites_and_leaves_no_temp_files() {
        let dir = unique_temp_dir("atomic");
        let target = dir.join("nested").join("settings.json");

        write_atomic_durable(&target, b"{\"v\":1}\n").expect("first write");
        assert_eq!(fs::read(&target).expect("read v1"), b"{\"v\":1}\n");

        // Overwrite existing file cleanly (including Windows replace semantics).
        write_atomic_durable(&target, b"{\"v\":2,\"ok\":true}\n").expect("second write");
        assert_eq!(
            fs::read(&target).expect("read v2"),
            b"{\"v\":2,\"ok\":true}\n"
        );

        let entries: Vec<_> = fs::read_dir(target.parent().expect("parent"))
            .expect("list dir")
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            entries,
            vec!["settings.json"],
            "stray temp files remained: {entries:?}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_json_or_default_handles_missing_and_valid_files() {
        #[derive(Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        struct Sample {
            count: u32,
        }

        let dir = unique_temp_dir("json");
        let path = dir.join("sample.json");

        let missing: Sample = read_json_or_default(&path).expect("missing returns default");
        assert_eq!(missing, Sample { count: 0 });

        write_json_durable(&path, &Sample { count: 42 }).expect("write json");
        let loaded: Sample = read_json_or_default(&path).expect("read json");
        assert_eq!(loaded, Sample { count: 42 });

        let _ = fs::remove_dir_all(&dir);
    }
}
