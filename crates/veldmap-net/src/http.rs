//! HTTP client construction (`rustls` + `ring`), per-host concurrency &
//! rate-limit cooldown management, and size-capped tile download helpers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use veldmap_core::sync::lock_unpoisoned;

use crate::NetError;

/// Parameters for constructing a `rustls` + `ring` HTTP tile client without
/// coupling `http` back to [`crate::fetcher::FetcherConfig`].
pub(crate) struct HttpClientConfig<'a> {
    pub user_agent: &'a str,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub max_concurrent_per_host: usize,
}

/// Cap initial buffer pre-allocation from untrusted `Content-Length` headers so
/// a malicious or misconfigured server cannot force 16 MiB upfront allocations
/// before sending any payload bytes.
const MAX_INITIAL_CAPACITY_BYTES: usize = 256 * 1024;

/// Minimum and maximum clamped `Retry-After` cooldown window.
const MIN_RETRY_AFTER: Duration = Duration::from_secs(1);
const MAX_RETRY_AFTER: Duration = Duration::from_secs(120);

/// Compliant HTTP `User-Agent` identifying VeldMap.
pub(crate) fn default_user_agent() -> String {
    format!(
        "{}/{} (+https://github.com/veldmap/veldmap)",
        veldmap_core::APP_NAME,
        env!("CARGO_PKG_VERSION")
    )
}

fn ensure_rustls_ring_provider() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        // Explicit `ring` crypto provider for `rustls` (installed once per
        // process; ignores Err if another subsystem already installed one).
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

pub(crate) fn build_http_client(cfg: &HttpClientConfig<'_>) -> Result<reqwest::Client, NetError> {
    ensure_rustls_ring_provider();
    reqwest::Client::builder()
        .user_agent(cfg.user_agent)
        .connect_timeout(cfg.connect_timeout)
        .timeout(cfg.request_timeout)
        .pool_max_idle_per_host(cfg.max_concurrent_per_host.max(1))
        .build()
        .map_err(|err| NetError::HttpClient(err.to_string()))
}

struct HostEntry {
    semaphore: Arc<Semaphore>,
    cooldown_until: Option<Instant>,
}

/// Reason a per-host HTTP permit could not be acquired immediately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PermitDenied {
    Cooldown(Instant),
    Saturated,
}

/// Per-host concurrency semaphores and rate-limit cooldown tracking (`429` /
/// `503` `Retry-After`).
///
/// Lock ordering: `HostLimiters` is a strict leaf lock — it never acquires
/// external mutexes or signals notifiers while locked, so callers may safely
/// invoke it while holding higher-level scheduler locks.
pub(crate) struct HostLimiters {
    max_per_host: usize,
    map: Mutex<HashMap<String, HostEntry>>,
}

impl HostLimiters {
    pub(crate) fn new(max_per_host: usize) -> Self {
        Self {
            max_per_host: max_per_host.max(1),
            map: Mutex::new(HashMap::new()),
        }
    }

    fn with_entry<R>(&self, host: &str, f: impl FnOnce(&mut HostEntry) -> R) -> R {
        let mut map = lock_unpoisoned(&self.map);
        if let Some(existing) = map.get_mut(host) {
            return f(existing);
        }
        let entry = map.entry(host.to_owned()).or_insert_with(|| HostEntry {
            semaphore: Arc::new(Semaphore::new(self.max_per_host)),
            cooldown_until: None,
        });
        f(entry)
    }

    /// Atomically check host cooldown and attempt to acquire a per-host HTTP
    /// permit under a single lock, avoiding string allocation on existing host
    /// hits.
    pub(crate) fn try_acquire_permit(
        &self,
        host: &str,
        now: Instant,
    ) -> Result<OwnedSemaphorePermit, PermitDenied> {
        self.with_entry(host, |entry| {
            if let Some(until) = entry.cooldown_until.filter(|&u| u > now) {
                return Err(PermitDenied::Cooldown(until));
            }
            Arc::clone(&entry.semaphore)
                .try_acquire_owned()
                .map_err(|_| PermitDenied::Saturated)
        })
    }

    /// Record or extend a per-host cooldown window (e.g. on HTTP `429`/`503`),
    /// avoiding string allocation when `host` is already tracked.
    pub(crate) fn record_cooldown(&self, host: &str, until: Instant) {
        self.with_entry(host, |entry| {
            entry.cooldown_until = Some(match entry.cooldown_until {
                Some(existing) if existing > until => existing,
                _ => until,
            });
        });
    }

    /// Clear all host cooldowns (e.g. on explicit user Retry or basemap switch)
    /// and prune idle host entries with no outstanding permits.
    pub(crate) fn clear_cooldowns(&self) {
        let mut map = lock_unpoisoned(&self.map);
        let max_per_host = self.max_per_host;
        map.retain(|_, entry| {
            entry.cooldown_until = None;
            Arc::strong_count(&entry.semaphore) > 1
                || entry.semaphore.available_permits() < max_per_host
        });
    }
}

pub(crate) fn parse_retry_after_seconds(header: Option<&str>) -> Option<Duration> {
    let raw = header?.trim();
    let secs = raw.parse::<u64>().ok()?;
    Some(Duration::from_secs(secs).clamp(MIN_RETRY_AFTER, MAX_RETRY_AFTER))
}

pub(crate) enum FetchOutcome {
    Payload(Vec<u8>),
    /// HTTP `204 No Content` (valid empty/nodata tile, not an error).
    Empty204,
}

/// Classification of a transient HTTP/network error so impossible combinations
/// (such as non-rate-limited errors carrying `Retry-After`) are unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TransientKind {
    Network { offline: bool },
    ServerError,
    RateLimited { retry_after: Option<Duration> },
}

pub(crate) struct TransientFetchError {
    pub msg: String,
    pub kind: TransientKind,
}

pub(crate) enum FetchFailure {
    Terminal(String),
    Transient(TransientFetchError),
}

fn map_reqwest_err(err: reqwest::Error) -> FetchFailure {
    FetchFailure::Transient(TransientFetchError {
        msg: err.to_string(),
        kind: TransientKind::Network {
            offline: err.is_connect() || err.is_timeout(),
        },
    })
}

pub(crate) async fn download_capped(
    client: &reqwest::Client,
    url: &str,
    max_bytes: usize,
) -> Result<FetchOutcome, FetchFailure> {
    let mut resp = client.get(url).send().await.map_err(map_reqwest_err)?;

    let status = resp.status();
    if status.as_u16() == 204 {
        return Ok(FetchOutcome::Empty204);
    }
    if !status.is_success() {
        let code = status.as_u16();
        let msg = format!("HTTP {code}");
        if matches!(code, 400 | 401 | 403 | 404 | 410 | 451) {
            return Err(FetchFailure::Terminal(msg));
        }
        let kind = if matches!(code, 429 | 503) {
            let hdr = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok());
            TransientKind::RateLimited {
                retry_after: parse_retry_after_seconds(hdr),
            }
        } else {
            TransientKind::ServerError
        };
        return Err(FetchFailure::Transient(TransientFetchError { msg, kind }));
    }

    if let Some(len) = resp.content_length()
        && len > max_bytes as u64
    {
        return Err(FetchFailure::Terminal(format!(
            "tile payload {len} B exceeds limit {max_bytes} B"
        )));
    }

    let initial_cap = resp.content_length().map_or(16 * 1024, |n| {
        (n as usize).min(max_bytes).min(MAX_INITIAL_CAPACITY_BYTES)
    });
    let mut buf = Vec::with_capacity(initial_cap);
    while let Some(chunk) = resp.chunk().await.map_err(map_reqwest_err)? {
        if buf.len().saturating_add(chunk.len()) > max_bytes {
            return Err(FetchFailure::Terminal(format!(
                "tile payload exceeded limit {max_bytes} B"
            )));
        }
        buf.extend_from_slice(&chunk);
    }

    if buf.is_empty() {
        return Err(FetchFailure::Terminal("empty tile response".to_owned()));
    }
    Ok(FetchOutcome::Payload(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_after_and_host_limiters_enforce_cooldowns_and_permits() {
        assert_eq!(
            parse_retry_after_seconds(Some("0")),
            Some(Duration::from_secs(1))
        );
        assert_eq!(
            parse_retry_after_seconds(Some("30")),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            parse_retry_after_seconds(Some("86400")),
            Some(Duration::from_secs(120))
        );
        assert_eq!(
            parse_retry_after_seconds(Some("Wed, 21 Oct 2026 07:28:00 GMT")),
            None
        );
        assert_eq!(parse_retry_after_seconds(None), None);

        let limiters = HostLimiters::new(1);
        let now = Instant::now();
        let permit = limiters
            .try_acquire_permit("example.org", now)
            .expect("first permit");
        assert_eq!(
            limiters.try_acquire_permit("example.org", now).err(),
            Some(PermitDenied::Saturated)
        );
        drop(permit);

        let later = now + Duration::from_secs(30);
        limiters.record_cooldown("example.org", later);
        assert_eq!(
            limiters.try_acquire_permit("example.org", now).err(),
            Some(PermitDenied::Cooldown(later))
        );
        // Shorter cooldown must not shorten an existing longer cooldown.
        limiters.record_cooldown("example.org", now + Duration::from_secs(5));
        assert_eq!(
            limiters.try_acquire_permit("example.org", now).err(),
            Some(PermitDenied::Cooldown(later))
        );

        limiters.clear_cooldowns();
        assert!(limiters.try_acquire_permit("example.org", now).is_ok());
    }
}
