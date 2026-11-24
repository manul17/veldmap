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

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);
const MAX_TEMP_CREATE_ATTEMPTS: usize = 16;

fn sibling_temp_path(target: &Path) -> Result<(PathBuf, PathBuf), StorageError> {
    let parent = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let file_name = target
        .file_name()
        .filter(|n| !n.is_empty())
        .ok_or_else(|| StorageError::InvalidPath(target.display().to_string()))?;
    let pid = std::process::id();
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let mut name = OsString::from(".");
    name.push(file_name);
    name.push(format!(".tmp.{pid}.{seq}"));
    let tmp = parent.join(name);
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
#[cfg(test)]
fn write_atomic_durable(path: &Path, bytes: &[u8]) -> Result<(), StorageError> {
    write_atomic_impl(path, bytes, false)
}

/// Atomically and durably write `bytes` to `path` with owner-only permissions
/// (`0600` on Unix; on Windows, relies on parent user-profile directory ACLs)
/// for configuration files that may store API keys.
pub fn write_atomic_durable_private(path: &Path, bytes: &[u8]) -> Result<(), StorageError> {
    write_atomic_impl(path, bytes, true)
}

fn create_unique_temp(
    path: &Path,
    private: bool,
) -> Result<(PathBuf, PathBuf, File), StorageError> {
    let mut last_err = None;
    for _ in 0..MAX_TEMP_CREATE_ATTEMPTS {
        let (parent, tmp_path) = sibling_temp_path(path)?;
        fs::create_dir_all(&parent)?;
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

fn write_json_impl<T: Serialize>(
    path: &Path,
    value: &T,
    private: bool,
) -> Result<(), StorageError> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_atomic_impl(path, &bytes, private)
}

/// Serialize `value` as pretty JSON and persist it via [`write_atomic_durable`].
#[cfg(test)]
fn write_json_durable<T: Serialize>(path: &Path, value: &T) -> Result<(), StorageError> {
    write_json_impl(path, value, false)
}

/// Serialize `value` as pretty JSON and persist it via
/// [`write_atomic_durable_private`] (`0600` permissions on Unix).
pub fn write_json_durable_private<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), StorageError> {
    write_json_impl(path, value, true)
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
            let mut backup_os = path.as_os_str().to_os_string();
            backup_os.push(".corrupt");
            let backup = PathBuf::from(backup_os);
            let _ = fs::rename(path, &backup);
            tracing::warn!(
                path = %path.display(),
                backup = %backup.display(),
                error = %err,
                "Failed to parse persisted JSON; backed up corrupt file and using defaults"
            );
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
mod tests {
    use super::*;
    use crate::test_support::TestTempDir;

    #[test]
    fn write_atomic_durable_creates_overwrites_and_leaves_no_temp_files() {
        let dir = TestTempDir::new("atomic");
        let target = dir.path().join("nested").join("settings.json");

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
    }

    #[test]
    fn private_write_sets_owner_only_mode_and_cleans_up_on_rename_failure() {
        let dir = TestTempDir::new("atomic-private");
        let secret_path = dir.path().join("secret.json");
        write_atomic_durable_private(&secret_path, b"{\"k\":1}\n").expect("private write");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&secret_path)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600, "private file must be 0600, got {mode:o}");
        }

        // Renaming onto an existing directory must fail and clean up its temp file.
        let dir_target = dir.path().join("target_is_dir");
        fs::create_dir_all(&dir_target).expect("create blocking dir");
        assert!(write_atomic_durable(&dir_target, b"payload").is_err());

        let leftovers: Vec<_> = fs::read_dir(dir.path())
            .expect("list dir")
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp."))
            .collect();
        assert!(leftovers.is_empty(), "temp file leaked: {leftovers:?}");

        // Invalid path with no filename fails cleanly.
        assert!(matches!(
            write_atomic_durable(Path::new(""), b"x"),
            Err(StorageError::InvalidPath(_))
        ));
    }

    #[test]
    fn read_json_or_default_handles_missing_and_valid_files() {
        #[derive(Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        struct Sample {
            count: u32,
        }

        let dir = TestTempDir::new("json");
        let path = dir.path().join("sample.json");

        let missing: Sample = read_json_or_default(&path).expect("missing returns default");
        assert_eq!(missing, Sample { count: 0 });

        write_json_durable(&path, &Sample { count: 42 }).expect("write json");
        let loaded: Sample = read_json_or_default(&path).expect("read json");
        assert_eq!(loaded, Sample { count: 42 });

        // Corrupt JSON is backed up to `<path>.corrupt` and falls back to Default.
        fs::write(&path, b"{not valid json").expect("write corrupt");
        let fallback: Sample = read_json_or_default_with_backup(&path);
        assert_eq!(fallback, Sample { count: 0 });
        let backup = path.with_extension("json.corrupt");
        assert_eq!(
            fs::read(&backup).expect("backup preserved"),
            b"{not valid json"
        );
    }
}
