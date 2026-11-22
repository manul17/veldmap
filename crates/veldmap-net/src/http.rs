//! HTTP client construction (`rustls` + `ring`), per-host concurrency &
//! rate-limit cooldown management, and size-capped tile download helpers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use tokio::sync::Semaphore;
use veldmap_core::io::lock_unpoisoned;

use crate::NetError;

/// Parameters for constructing a `rustls` + `ring` HTTP tile client without
/// coupling `http` back to [`crate::fetcher::FetcherConfig`].
pub(crate) struct HttpClientConfig<'a> {
    pub user_agent: &'a str,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub max_concurrent_per_host: usize,
}

/// Default maximum concurrent HTTP requests per remote host.
pub const DEFAULT_MAX_CONCURRENT_PER_HOST: usize = 6;

/// Default maximum encoded response body accepted for one live tile (`16 MiB`).
pub const DEFAULT_MAX_TILE_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Cap initial buffer pre-allocation from untrusted `Content-Length` headers so
/// a malicious or misconfigured server cannot force 16 MiB upfront allocations
/// before sending any payload bytes.
const MAX_INITIAL_CAPACITY_BYTES: usize = 256 * 1024;

/// Minimum and maximum clamped `Retry-After` cooldown window.
const MIN_RETRY_AFTER: Duration = Duration::from_secs(1);
const MAX_RETRY_AFTER: Duration = Duration::from_secs(120);

/// Compliant HTTP `User-Agent` identifying VeldMap.
pub fn default_user_agent() -> String {
    format!(
        "VeldMap/{} (+https://github.com/veldmap/veldmap)",
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

/// Per-host concurrency semaphores and rate-limit cooldown tracking (`429` /
/// `503` `Retry-After`).
///
/// Lock ordering: always acquire `PipelineState` BEFORE `HostLimiters` (never
/// reverse; `HostLimiters` never locks `PipelineState` or calls `Notify`).
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

    pub(crate) fn semaphore_for_host(&self, host: &str) -> Arc<Semaphore> {
        let mut map = lock_unpoisoned(&self.map);
        Arc::clone(
            &map.entry(host.to_owned())
                .or_insert_with(|| HostEntry {
                    semaphore: Arc::new(Semaphore::new(self.max_per_host)),
                    cooldown_until: None,
                })
                .semaphore,
        )
    }

    /// Return `Some(until)` if `host` is currently in a rate-limit cooldown.
    pub(crate) fn cooldown_until(&self, host: &str, now: Instant) -> Option<Instant> {
        let map = lock_unpoisoned(&self.map);
        map.get(host)
            .and_then(|entry| entry.cooldown_until)
            .filter(|&until| until > now)
    }

    /// Record or extend a per-host cooldown window (e.g. on HTTP `429`/`503`).
    pub(crate) fn record_cooldown(&self, host: &str, until: Instant) {
        let mut map = lock_unpoisoned(&self.map);
        let entry = map.entry(host.to_owned()).or_insert_with(|| HostEntry {
            semaphore: Arc::new(Semaphore::new(self.max_per_host)),
            cooldown_until: None,
        });
        entry.cooldown_until = Some(match entry.cooldown_until {
            Some(existing) if existing > until => existing,
            _ => until,
        });
    }

    /// Clear all host cooldowns (e.g. on explicit user Retry or basemap switch).
    pub(crate) fn clear_cooldowns(&self) {
        let mut map = lock_unpoisoned(&self.map);
        for entry in map.values_mut() {
            entry.cooldown_until = None;
        }
    }
}

/// Extract the lowercased `host[:port]` authority key from an HTTP(S) URL.
pub(crate) fn extract_host(url: &str) -> String {
    let without_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = without_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("unknown");
    let host_port = authority.rsplit_once('@').map_or(authority, |(_, hp)| hp);
    host_port.to_ascii_lowercase()
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

pub(crate) enum FetchFailure {
    Terminal(String),
    Transient {
        msg: String,
        offline: bool,
        retry_after: Option<Duration>,
    },
}

pub(crate) async fn download_capped(
    client: &reqwest::Client,
    url: &str,
    max_bytes: usize,
) -> Result<FetchOutcome, FetchFailure> {
    let mut resp = client.get(url).send().await.map_err(|err| {
        let offline = err.is_connect() || err.is_timeout();
        FetchFailure::Transient {
            msg: err.to_string(),
            offline,
            retry_after: None,
        }
    })?;

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
        let retry_after = if matches!(code, 429 | 503) {
            let hdr = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok());
            Some(parse_retry_after_seconds(hdr).unwrap_or(Duration::from_secs(15)))
        } else {
            None
        };
        return Err(FetchFailure::Transient {
            msg,
            offline: false,
            retry_after,
        });
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
    while let Some(chunk) = resp.chunk().await.map_err(|err| FetchFailure::Transient {
        msg: err.to_string(),
        offline: err.is_connect() || err.is_timeout(),
        retry_after: None,
    })? {
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
    fn extract_host_and_retry_after_handle_edge_cases() {
        assert_eq!(
            extract_host("https://GIBS.Earthdata.NASA.gov/wmts/1/2/3.jpg"),
            "gibs.earthdata.nasa.gov"
        );
        assert_eq!(
            extract_host("http://user:pass@127.0.0.1:8080/0/0/0.png?key=1"),
            "127.0.0.1:8080"
        );

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
    }

    #[test]
    fn host_limiters_record_and_clear_cooldowns() {
        let limiters = HostLimiters::new(4);
        let now = Instant::now();
        assert!(limiters.cooldown_until("example.org", now).is_none());

        let later = now + Duration::from_secs(30);
        limiters.record_cooldown("example.org", later);
        assert_eq!(limiters.cooldown_until("example.org", now), Some(later));
        // Shorter cooldown must not shorten an existing longer cooldown.
        limiters.record_cooldown("example.org", now + Duration::from_secs(5));
        assert_eq!(limiters.cooldown_until("example.org", now), Some(later));

        limiters.clear_cooldowns();
        assert!(limiters.cooldown_until("example.org", now).is_none());
    }
}
