//! HTTP client construction (`rustls` + `ring`), per-host concurrency &
//! rate-limit cooldown management, and size-capped response body reading
//! helpers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use reqwest::header::{HeaderMap, HeaderName, HeaderValue, RETRY_AFTER};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use veldmap_core::sync::{CancelGuard, CancelToken, lock_unpoisoned};

use crate::NetError;
use crate::redact::{EncodedSecret, extract_header_secret};

/// Parameters for constructing a `rustls` + `ring` HTTP client with split
/// connect/read timeouts, explicit compression control, and redirect security.
pub(crate) struct HttpClientConfig<'a> {
    pub user_agent: &'a str,
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
    pub max_concurrent_per_host: usize,
    pub enable_gzip: bool,
    /// When `true`, reject cross-origin redirects so custom auth headers
    /// (`x-api-key`, `extra_headers`) can never leak to third-party origins.
    pub same_origin_redirects: bool,
}

/// Protocol-owned headers that user-configured custom headers must never
/// overwrite (breaking HTTP framing, byte-range offsets, or ETag guards).
const FORBIDDEN_CUSTOM_HEADERS: &[&str] = &[
    "host",
    "range",
    "if-match",
    "accept-encoding",
    "content-length",
    "transfer-encoding",
    "connection",
    "te",
    "upgrade",
];

/// Return whether `name` is a valid HTTP header name permitted for user-supplied
/// custom authentication headers (rejecting invalid syntax and protocol-owned
/// headers such as `Host`, `Range`, `If-Match`, `Accept-Encoding`, etc.).
pub fn is_valid_custom_header_name(name: &str) -> bool {
    validate_custom_header_name(name, &[]).is_ok()
}

fn validate_custom_header_name(name: &str, extra_forbidden: &[&str]) -> Result<HeaderName, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("header name must not be empty".to_owned());
    }
    let parsed = HeaderName::from_bytes(trimmed.as_bytes()).map_err(|e| e.to_string())?;
    let lower = parsed.as_str();
    if FORBIDDEN_CUSTOM_HEADERS.contains(&lower)
        || extra_forbidden
            .iter()
            .any(|f| f.eq_ignore_ascii_case(lower))
    {
        return Err(format!(
            "protocol-owned header '{trimmed}' cannot be overridden"
        ));
    }
    Ok(parsed)
}

/// Validated custom HTTP headers (marked sensitive) paired with extracted
/// error-redaction secrets.
pub(crate) struct ParsedCustomHeaders {
    pub(crate) headers: Vec<(HeaderName, HeaderValue)>,
    pub(crate) secrets: Vec<EncodedSecret>,
}

/// Validate and parse custom HTTP headers, marking every value sensitive (so
/// transport debug logging cannot leak credentials) and extracting redaction
/// secrets for error messages.
pub(crate) fn parse_custom_headers(
    headers: &[(String, String)],
    extra_forbidden: &[&str],
) -> Result<ParsedCustomHeaders, (String, String)> {
    let mut parsed = Vec::with_capacity(headers.len());
    let mut secrets = Vec::with_capacity(headers.len());
    for (k, v) in headers {
        let name = validate_custom_header_name(k, extra_forbidden)
            .map_err(|reason| (k.clone(), reason))?;
        let trimmed_val = v.trim();
        let mut value = HeaderValue::from_str(trimmed_val)
            .map_err(|_| (k.clone(), "invalid header value".to_owned()))?;
        value.set_sensitive(true);
        if let Some(secret) = extract_header_secret(name.as_str(), trimmed_val) {
            secrets.push(secret);
        }
        parsed.push((name, value));
    }
    Ok(ParsedCustomHeaders {
        headers: parsed,
        secrets,
    })
}

/// Link `caller` and `shutdown` into a single child [`CancelToken`] that fires
/// as soon as either parent cancels, with automatic callback cleanup on drop.
pub(crate) fn link_cancel_token(
    shutdown: &CancelToken,
    caller: &CancelToken,
) -> (CancelToken, CancelGuard) {
    let combined = shutdown.child_token();
    let combined_clone = combined.clone();
    let guard = caller.on_cancel_guarded(move || combined_clone.cancel());
    (combined, guard)
}

/// Record a host cooldown from an HTTP `429`/`503` response's `Retry-After`
/// header (or `default_backoff` if missing/unparseable).
pub(crate) fn record_rate_limit_cooldown(
    limiters: &HostLimiters,
    host: &str,
    headers: &HeaderMap,
    default_backoff: Duration,
) {
    let retry_after = headers
        .get(RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| parse_retry_after_seconds(Some(s)))
        .unwrap_or(default_backoff);
    limiters.record_cooldown(host, Instant::now() + retry_after);
}

/// Failure modes for [`read_body_capped`].
pub(crate) enum BodyReadError {
    Cancelled,
    TooLarge,
    Network(reqwest::Error),
}

/// Stream a `reqwest::Response` body into a `Vec<u8>` up to `max_bytes`,
/// cancelling promptly via `cancel`.
pub(crate) async fn read_body_capped(
    resp: &mut reqwest::Response,
    max_bytes: usize,
    initial_cap: usize,
    cancel: &CancelToken,
) -> Result<Vec<u8>, BodyReadError> {
    let cap = initial_cap.min(max_bytes).min(MAX_INITIAL_CAPACITY_BYTES);
    let mut body = Vec::with_capacity(cap);
    loop {
        let chunk_res = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(BodyReadError::Cancelled),
            chunk = resp.chunk() => chunk,
        };
        match chunk_res {
            Ok(Some(chunk)) => {
                if body.len().saturating_add(chunk.len()) > max_bytes {
                    return Err(BodyReadError::TooLarge);
                }
                body.extend_from_slice(&chunk);
            }
            Ok(None) => return Ok(body),
            Err(err) => return Err(BodyReadError::Network(err)),
        }
    }
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

fn build_redirect_policy(same_origin_only: bool) -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(move |attempt| {
        if attempt.previous().len() >= 10 {
            return attempt.error("too many HTTP redirects");
        }
        if attempt.previous().iter().any(|u| u.scheme() == "https")
            && attempt.url().scheme() != "https"
        {
            return attempt.error("refused insecure HTTPS-to-HTTP redirect");
        }
        if same_origin_only && let Some(first) = attempt.previous().first() {
            let origin_matches = first.scheme() == attempt.url().scheme()
                && first
                    .host_str()
                    .zip(attempt.url().host_str())
                    .is_some_and(|(a, b)| a.eq_ignore_ascii_case(b))
                && first.port_or_known_default() == attempt.url().port_or_known_default();
            if !origin_matches {
                return attempt.error("refused cross-origin redirect with custom auth headers");
            }
        }
        attempt.follow()
    })
}

pub(crate) fn build_http_client(cfg: &HttpClientConfig<'_>) -> Result<reqwest::Client, NetError> {
    ensure_rustls_ring_provider();
    reqwest::Client::builder()
        .user_agent(cfg.user_agent)
        .connect_timeout(cfg.connect_timeout)
        .read_timeout(cfg.read_timeout)
        .gzip(cfg.enable_gzip)
        .redirect(build_redirect_policy(cfg.same_origin_redirects))
        .pool_max_idle_per_host(cfg.max_concurrent_per_host.max(1))
        .build()
        .map_err(|err| NetError::HttpClient(err.to_string()))
}

struct HostEntry {
    semaphore: Arc<Semaphore>,
    cooldown_until: Option<Instant>,
    cooldown_notify: Arc<Notify>,
}

/// Reason a per-host HTTP permit could not be acquired immediately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PermitDenied {
    Cooldown(Instant),
    Saturated,
}

/// Async cancellation marker returned by [`HostLimiters::acquire_permit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PermitCancelled;

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
            cooldown_notify: Arc::new(Notify::new()),
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

    /// Asynchronously wait out any active host cooldown and acquire a per-host
    /// HTTP concurrency permit, aborting immediately if `cancel` triggers.
    pub(crate) async fn acquire_permit(
        &self,
        host: &str,
        cancel: &CancelToken,
    ) -> Result<OwnedSemaphorePermit, PermitCancelled> {
        loop {
            if cancel.is_cancelled() {
                return Err(PermitCancelled);
            }
            let cooldown_notify = self.with_entry(host, |entry| Arc::clone(&entry.cooldown_notify));
            let notified = cooldown_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            let (semaphore, cooldown_until) = self.with_entry(host, |entry| {
                (Arc::clone(&entry.semaphore), entry.cooldown_until)
            });

            if let Some(until) = cooldown_until.filter(|&u| u > Instant::now()) {
                let deadline = tokio::time::Instant::from_std(until);
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return Err(PermitCancelled),
                    _ = notified => continue,
                    _ = tokio::time::sleep_until(deadline) => continue,
                }
            }

            let permit = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(PermitCancelled),
                res = semaphore.acquire_owned() => match res {
                    Ok(permit) => permit,
                    Err(_) => return Err(PermitCancelled),
                },
            };

            // Re-check whether a `429`/`503` cooldown started while we waited
            // on the semaphore; if so, release the permit and honor cooldown.
            let active_cooldown = self.with_entry(host, |entry| {
                entry.cooldown_until.filter(|&u| u > Instant::now())
            });
            if active_cooldown.is_some() {
                drop(permit);
                continue;
            }
            return Ok(permit);
        }
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

    /// Clear cooldowns strictly for `hosts` (leaving unrelated hosts untouched)
    /// and prune idle entries with no active permits or active cooldowns.
    pub(crate) fn clear_cooldowns_for<'a>(&self, hosts: impl IntoIterator<Item = &'a str>) {
        let mut to_notify = Vec::new();
        {
            let mut map = lock_unpoisoned(&self.map);
            let now = Instant::now();
            for host in hosts {
                if let Some(entry) = map.get_mut(host)
                    && entry.cooldown_until.take().is_some()
                {
                    to_notify.push(Arc::clone(&entry.cooldown_notify));
                }
            }
            let max_per_host = self.max_per_host;
            map.retain(|_, entry| {
                let has_cooldown = entry.cooldown_until.is_some_and(|u| u > now);
                has_cooldown
                    || Arc::strong_count(&entry.semaphore) > 1
                    || entry.semaphore.available_permits() < max_per_host
                    || Arc::strong_count(&entry.cooldown_notify) > 1
            });
        }
        // Notify outside `self.map` so `HostLimiters` remains a strict leaf lock.
        for notify in to_notify {
            notify.notify_waiters();
        }
    }
}

pub(crate) fn parse_retry_after_seconds(header: Option<&str>) -> Option<Duration> {
    let raw = header?.trim();
    let secs = raw.parse::<u64>().ok()?;
    Some(Duration::from_secs(secs).clamp(MIN_RETRY_AFTER, MAX_RETRY_AFTER))
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

        limiters.record_cooldown("other.org", later);
        limiters.clear_cooldowns_for(["example.org"]);
        assert!(limiters.try_acquire_permit("example.org", now).is_ok());
        assert_eq!(
            limiters.try_acquire_permit("other.org", now).err(),
            Some(PermitDenied::Cooldown(later)),
            "clearing example.org must leave unrelated host cooldowns intact"
        );
    }
}
