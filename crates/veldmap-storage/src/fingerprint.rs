//! Sampled BLAKE3 content fingerprinting (`len || head_64k || mid_64k || tail_64k`)
//! and in-memory `(path, len, mtime_ns)` stat cache with explicit reload/bypass
//! support.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use veldmap_core::io::{ByteSource, CancelToken, IoError, SourceFingerprint, lock_unpoisoned};

use crate::StorageError;
use crate::file_source::{LocalFileSource, metadata_mtime_ns};

/// Edge/midpoint sample window size (`64 KiB`).
pub const FINGERPRINT_SAMPLE_BYTES: u64 = 64 * 1024;

/// Compute non-overlapping `(offset, len)` sample windows covering
/// `head_64k || mid_64k || tail_64k` for a stream of `total_len` bytes.
pub fn sample_ranges(total_len: u64) -> Vec<(u64, usize)> {
    let s = FINGERPRINT_SAMPLE_BYTES;
    if total_len == 0 {
        return Vec::new();
    }
    if total_len <= s {
        return vec![(0, total_len as usize)];
    }
    if total_len <= 2 * s {
        return vec![(0, s as usize), (s, (total_len - s) as usize)];
    }
    if total_len <= 3 * s {
        let tail_start = total_len - s;
        let mid_len = tail_start - s;
        return vec![
            (0, s as usize),
            (s, mid_len as usize),
            (tail_start, s as usize),
        ];
    }
    let tail_start = total_len - s;
    let mid_start = (total_len / 2) - (s / 2);
    vec![
        (0, s as usize),
        (mid_start, s as usize),
        (tail_start, s as usize),
    ]
}

/// Compute an opaque [`SourceFingerprint`] by hashing
/// `len || head_64k || mid_64k || tail_64k` via BLAKE3 over `source`.
pub fn fingerprint_byte_source(
    source: &dyn ByteSource,
    cancel: &CancelToken,
) -> Result<SourceFingerprint, IoError> {
    let total_len = source.len();
    let mut hasher = blake3::Hasher::new();
    hasher.update(&total_len.to_le_bytes());

    let mut scratch = vec![0u8; FINGERPRINT_SAMPLE_BYTES as usize];
    for (offset, size) in sample_ranges(total_len) {
        source.read_exact_at(offset, &mut scratch[..size], cancel)?;
        hasher.update(&scratch[..size]);
    }
    Ok(SourceFingerprint::from_digest(
        *hasher.finalize().as_bytes(),
    ))
}

#[derive(Clone, Copy)]
struct CachedStat {
    len: u64,
    mtime_ns: Option<i128>,
    fingerprint: SourceFingerprint,
}

/// In-memory `(path, len, mtime_ns)` stat cache avoiding repeated BLAKE3 reads
/// when re-opening unchanged local files, with explicit bypass/invalidation for
/// UI "Reload / bypass cache" actions.
#[derive(Default)]
pub struct LocalFingerprintCache {
    entries: Mutex<HashMap<PathBuf, CachedStat>>,
}

impl LocalFingerprintCache {
    /// Construct an empty stat cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the cached [`SourceFingerprint`] for `path` if `(len, mtime_ns)`
    /// still match, or recompute from disk and cache the result.
    pub fn get_or_compute(
        &self,
        path: &Path,
        cancel: &CancelToken,
    ) -> Result<SourceFingerprint, StorageError> {
        self.get_or_compute_with_bypass(path, false, cancel)
    }

    /// Return the [`SourceFingerprint`] for `path`, bypassing the in-memory stat
    /// cache when `bypass_cache` is `true` ("Reload / bypass cache").
    pub fn get_or_compute_with_bypass(
        &self,
        path: &Path,
        bypass_cache: bool,
        cancel: &CancelToken,
    ) -> Result<SourceFingerprint, StorageError> {
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let meta = std::fs::metadata(&key)?;
        if !meta.is_file() {
            return Err(StorageError::InvalidPath(format!(
                "'{}' is not a regular file",
                path.display()
            )));
        }
        let len = meta.len();
        let mtime_ns = metadata_mtime_ns(&meta);

        if !bypass_cache {
            let guard = lock_unpoisoned(&self.entries);
            if let Some(entry) = guard.get(&key)
                && entry.len == len
                && entry.mtime_ns.is_some()
                && entry.mtime_ns == mtime_ns
            {
                return Ok(entry.fingerprint);
            }
        }

        let source = LocalFileSource::open(&key)?;
        let fingerprint = fingerprint_byte_source(&source, cancel).map_err(|err| match err {
            IoError::Io(io_err) => StorageError::Io(io_err),
            other => StorageError::Io(other.into_std_io()),
        })?;

        // Re-stat after hashing so concurrent in-place edits or replacements
        // during the read never cache a torn `(mtime_ns, fingerprint)` pair.
        let post_stable = std::fs::metadata(&key).ok().is_some_and(|post| {
            source.mtime_ns().is_some()
                && post.len() == source.len()
                && metadata_mtime_ns(&post) == source.mtime_ns()
        });

        let mut guard = lock_unpoisoned(&self.entries);
        if post_stable {
            guard.insert(
                key,
                CachedStat {
                    len: source.len(),
                    mtime_ns: source.mtime_ns(),
                    fingerprint,
                },
            );
        } else {
            guard.remove(&key);
        }
        Ok(fingerprint)
    }

    /// Invalidate any cached entry for `path`.
    pub fn invalidate(&self, path: &Path) {
        let canonical = std::fs::canonicalize(path).ok().or_else(|| {
            let name = path.file_name()?;
            let parent = match path.parent() {
                Some(p) if !p.as_os_str().is_empty() => p,
                _ => Path::new("."),
            };
            std::fs::canonicalize(parent).ok().map(|p| p.join(name))
        });
        let mut guard = lock_unpoisoned(&self.entries);
        if let Some(key) = canonical {
            guard.remove(&key);
        }
        guard.remove(path);
    }

    /// Clear all cached stat entries.
    pub fn clear(&self) {
        lock_unpoisoned(&self.entries).clear();
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use std::sync::Arc;

    use veldmap_core::io::{CountingByteSource, MemoryByteSource};

    use super::*;
    use crate::test_support::TestTempDir;
    use crate::valid_source_key;

    #[test]
    fn sample_ranges_are_non_overlapping_and_cover_small_and_large_files() {
        let s = FINGERPRINT_SAMPLE_BYTES;
        assert!(sample_ranges(0).is_empty());
        assert_eq!(sample_ranges(100), vec![(0, 100)]);
        assert_eq!(sample_ranges(s + 10), vec![(0, s as usize), (s, 10)]);
        assert_eq!(
            sample_ranges(2 * s + 100),
            vec![(0, s as usize), (s, 100), (s + 100, s as usize)]
        );
        let large = sample_ranges(10 * s + 7);
        assert_eq!(large.len(), 3);
        for pair in large.windows(2) {
            assert!(pair[0].0 + pair[0].1 as u64 <= pair[1].0);
        }
    }

    #[test]
    fn fingerprint_detects_mid_changes_and_stat_cache_honors_bypass() {
        let cancel = CancelToken::new();
        let mut data = vec![0x11u8; (4 * FINGERPRINT_SAMPLE_BYTES) as usize];
        let fp1 = fingerprint_byte_source(&MemoryByteSource::from_vec(data.clone()), &cancel)
            .expect("fp1");
        assert!(valid_source_key(&fp1.to_hex()));

        // Flip a byte inside the midpoint window.
        let mid_idx = data.len() / 2;
        data[mid_idx] ^= 0xFF;
        let fp2 = fingerprint_byte_source(&MemoryByteSource::from_vec(data.clone()), &cancel)
            .expect("fp2");
        assert_ne!(fp1, fp2);

        // CountingByteSource verifies only <= 3 * 64 KiB read for a 1 MiB file.
        let large = vec![0x22u8; 1024 * 1024];
        let counting = CountingByteSource::new(Arc::new(MemoryByteSource::from_vec(large)));
        let _ = fingerprint_byte_source(&counting, &cancel).expect("fp large");
        assert_eq!(counting.bytes_read(), 3 * FINGERPRINT_SAMPLE_BYTES);

        // Stat cache caches unchanged files and recomputes when bypassed or modified.
        let dir = TestTempDir::new("fp-cache");
        let path = dir.path().join("raster.tif");
        {
            let mut f = File::create(&path).expect("create");
            f.write_all(&data).expect("write");
        }
        let cache = LocalFingerprintCache::new();
        let cached1 = cache.get_or_compute(&path, &cancel).expect("initial");
        assert_eq!(cached1, fp2);
        let cached2 = cache
            .get_or_compute_with_bypass(&path, true, &cancel)
            .expect("bypass");
        assert_eq!(cached2, fp2);

        // Same length rewrite with a different byte in head; bypass catches it even if coarse mtime ties.
        data[0] ^= 0x55;
        {
            let mut f = File::create(&path).expect("overwrite");
            f.write_all(&data).expect("write2");
        }
        let recomputed = cache
            .get_or_compute_with_bypass(&path, true, &cancel)
            .expect("recomputed");
        assert_ne!(recomputed, fp2);

        // Deleting the file before `invalidate` still removes the canonicalized cache entry.
        std::fs::remove_file(&path).expect("remove");
        cache.invalidate(&path);
        assert!(lock_unpoisoned(&cache.entries).is_empty());
        cache.clear();
    }
}
