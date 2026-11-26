//! Sampled BLAKE3 content fingerprinting (`len || head_64k || mid_64k || tail_64k`)
//! and bounded in-memory `(path, len, mtime_ns)` stat cache with explicit
//! reload/bypass support.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use veldmap_core::io::{ByteSource, IoError};
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::SourceFingerprint;

use crate::StorageError;
use crate::file_source::{LocalFileSource, metadata_mtime_ns};

/// Edge/midpoint sample window size (`64 KiB`).
const FINGERPRINT_SAMPLE_BYTES: u64 = 64 * 1024;

/// Maximum stat entries retained in [`LocalFingerprintCache`] so long-running
/// sessions cannot grow the in-memory cache without bound.
const MAX_FINGERPRINT_CACHE_ENTRIES: usize = 1024;

/// Compute non-overlapping `(offset, len)` sample windows covering
/// `head_64k || mid_64k || tail_64k` for a stream of `total_len` bytes.
fn sample_ranges(total_len: u64) -> Vec<(u64, usize)> {
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
    if cancel.is_cancelled() {
        return Err(IoError::Cancelled);
    }
    let total_len = source.len();
    let mut hasher = blake3::Hasher::new();
    hasher.update(&total_len.to_le_bytes());

    let ranges = sample_ranges(total_len);
    if !ranges.is_empty() {
        let mut scratch = vec![0u8; FINGERPRINT_SAMPLE_BYTES as usize];
        for (offset, size) in ranges {
            source.read_exact_at(offset, &mut scratch[..size], cancel)?;
            hasher.update(&scratch[..size]);
        }
    }
    Ok(SourceFingerprint::from_digest(
        *hasher.finalize().as_bytes(),
    ))
}

/// Domain separation prefix for remote HTTP source fingerprints so they can
/// never collide with local `len || head || mid || tail` digests.
const REMOTE_FINGERPRINT_DOMAIN: &[u8] = b"veldmap-remote-v1\0";

/// Derive a deterministic [`SourceFingerprint`] for a remote HTTP object
/// without issuing extra `mid_64k`/`tail_64k` range requests.
///
/// - When `strong_etag` is `Some` (a validated strong RFC 9110 ETag), hashes
///   `(canonical_url_without_query, strong_etag, content_length)`.
/// - Otherwise hashes `(canonical_url_without_query, content_length, blake3(head_64k))`
///   using at most the first `64 KiB` already fetched during the open probe.
///
/// The resulting 64-character lowercase hex string always satisfies
/// [`crate::TileDiskCache`]'s `<= 128`-byte ASCII source-key allowlist.
pub fn fingerprint_remote_source(
    canonical_url_without_query: &str,
    content_length: u64,
    strong_etag: Option<&str>,
    head_bytes: &[u8],
) -> SourceFingerprint {
    let mut hasher = blake3::Hasher::new();
    hasher.update(REMOTE_FINGERPRINT_DOMAIN);
    hasher.update(&(canonical_url_without_query.len() as u64).to_le_bytes());
    hasher.update(canonical_url_without_query.as_bytes());
    hasher.update(&content_length.to_le_bytes());
    if let Some(etag) = strong_etag {
        hasher.update(&[1u8]);
        hasher.update(&(etag.len() as u64).to_le_bytes());
        hasher.update(etag.as_bytes());
    } else {
        hasher.update(&[0u8]);
        let prefix = &head_bytes[..head_bytes.len().min(FINGERPRINT_SAMPLE_BYTES as usize)];
        let head_hash = blake3::hash(prefix);
        hasher.update(head_hash.as_bytes());
    }
    SourceFingerprint::from_digest(*hasher.finalize().as_bytes())
}

#[derive(Clone, Copy)]
struct CachedStat {
    len: u64,
    mtime_ns: Option<i128>,
    fingerprint: SourceFingerprint,
}

/// Bounded in-memory `(path, len, mtime_ns)` stat cache avoiding repeated
/// BLAKE3 reads when re-opening unchanged local files, with explicit
/// bypass/invalidation for UI "Reload / bypass cache" actions.
#[derive(Default)]
pub struct LocalFingerprintCache {
    entries: Mutex<HashMap<PathBuf, CachedStat>>,
}

impl LocalFingerprintCache {
    /// Construct an empty stat cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Open `path` once as a [`LocalFileSource`] and return both the opened
    /// handle and its [`SourceFingerprint`], avoiding a second `File::open` and
    /// closing the TOCTOU window between fingerprinting and decoding.
    ///
    /// On cache hit (matching canonical `(len, mtime_ns)` on the opened file
    /// handle), skips BLAKE3 I/O; when `bypass_cache` is `true` or on cache
    /// miss, hashes the exact returned [`LocalFileSource`].
    pub fn open_and_fingerprint(
        &self,
        path: &Path,
        bypass_cache: bool,
        cancel: &CancelToken,
    ) -> Result<(LocalFileSource, SourceFingerprint), StorageError> {
        if cancel.is_cancelled() {
            return Err(StorageError::Cancelled);
        }
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let source = LocalFileSource::open(&key)?;

        if !bypass_cache
            && let Some(fingerprint) = self.lookup_cached(&key, source.len(), source.mtime_ns())
        {
            return Ok((source, fingerprint));
        }

        if cancel.is_cancelled() {
            return Err(StorageError::Cancelled);
        }
        let fingerprint = self.compute_and_cache(&key, &source, cancel)?;
        Ok((source, fingerprint))
    }

    fn lookup_cached(
        &self,
        key: &Path,
        len: u64,
        mtime_ns: Option<i128>,
    ) -> Option<SourceFingerprint> {
        let mtime_ns = mtime_ns?;
        let guard = lock_unpoisoned(&self.entries);
        guard
            .get(key)
            .filter(|entry| entry.len == len && entry.mtime_ns == Some(mtime_ns))
            .map(|entry| entry.fingerprint)
    }

    fn compute_and_cache(
        &self,
        key: &Path,
        source: &LocalFileSource,
        cancel: &CancelToken,
    ) -> Result<SourceFingerprint, StorageError> {
        let fingerprint = fingerprint_byte_source(source, cancel)?;

        // Re-stat after hashing so concurrent in-place edits or replacements
        // during the read never cache a torn `(mtime_ns, fingerprint)` pair.
        let post_stable = std::fs::metadata(key).ok().is_some_and(|post| {
            source.mtime_ns().is_some()
                && post.len() == source.len()
                && metadata_mtime_ns(&post) == source.mtime_ns()
        });

        let mut guard = lock_unpoisoned(&self.entries);
        if post_stable {
            if !guard.contains_key(key) && guard.len() >= MAX_FINGERPRINT_CACHE_ENTRIES {
                guard.clear();
            }
            guard.insert(
                key.to_path_buf(),
                CachedStat {
                    len: source.len(),
                    mtime_ns: source.mtime_ns(),
                    fingerprint,
                },
            );
        } else {
            guard.remove(key);
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
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use std::sync::Arc;

    use veldmap_core::io::testing::{CountingByteSource, MemoryByteSource};

    use super::*;
    use crate::test_support::TestTempDir;
    use crate::tile_cache::valid_source_key;

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
    fn fingerprint_detects_mid_changes_and_stat_cache_honors_bypass_and_cancel() {
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
        let (opened_src, opened_fp) = cache
            .open_and_fingerprint(&path, false, &cancel)
            .expect("open_and_fingerprint miss");
        assert_eq!(opened_fp, fp2);
        assert_eq!(opened_src.len(), data.len() as u64);
        let (hit_src, hit_fp) = cache
            .open_and_fingerprint(&path, false, &cancel)
            .expect("open_and_fingerprint hit");
        assert_eq!(hit_fp, fp2);
        assert_eq!(hit_src.len(), data.len() as u64);
        let (_, bypassed_fp) = cache
            .open_and_fingerprint(&path, true, &cancel)
            .expect("bypass");
        assert_eq!(bypassed_fp, fp2);

        // Pre-cancelled token returns typed `StorageError::Cancelled` without I/O.
        let cancelled = CancelToken::new();
        cancelled.cancel();
        assert!(
            cache
                .open_and_fingerprint(&path, false, &cancelled)
                .is_err_and(|e| e.is_cancelled())
        );

        // Same length rewrite with a different byte in head; bypass catches it even if coarse mtime ties.
        data[0] ^= 0x55;
        {
            let mut f = File::create(&path).expect("overwrite");
            f.write_all(&data).expect("write2");
        }
        let (recomputed_src, recomputed) = cache
            .open_and_fingerprint(&path, true, &cancel)
            .expect("recomputed");
        assert_ne!(recomputed, fp2);
        assert_eq!(recomputed_src.len(), data.len() as u64);

        // Deleting the file before `invalidate` still removes the canonicalized cache entry.
        std::fs::remove_file(&path).expect("remove");
        cache.invalidate(&path);
        assert!(lock_unpoisoned(&cache.entries).is_empty());
    }

    #[test]
    fn remote_fingerprint_uses_strong_etag_or_head_fallback_and_stays_ascii_valid() {
        let head_a = vec![0xAAu8; 4096];
        let head_b = vec![0xBBu8; 4096];
        let url = "https://example.org/sentinel2/B04.tif";

        let fp_etag_a = fingerprint_remote_source(url, 1_000_000, Some("\"abc123\""), &head_a);
        let fp_etag_b = fingerprint_remote_source(url, 1_000_000, Some("\"abc123\""), &head_b);
        assert_eq!(fp_etag_a, fp_etag_b);
        assert!(valid_source_key(&fp_etag_a.to_hex()));

        let fp_etag_diff = fingerprint_remote_source(url, 1_000_000, Some("\"def456\""), &head_a);
        assert_ne!(fp_etag_a, fp_etag_diff);

        let fp_head_a = fingerprint_remote_source(url, 1_000_000, None, &head_a);
        let fp_head_b = fingerprint_remote_source(url, 1_000_000, None, &head_b);
        assert_ne!(fp_head_a, fp_head_b);
        assert_ne!(fp_head_a, fp_etag_a);
        assert!(valid_source_key(&fp_head_a.to_hex()));
    }
}
