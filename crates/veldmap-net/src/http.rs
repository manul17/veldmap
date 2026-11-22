//! HTTP client construction (`rustls` + `ring`), per-host concurrency &
//! rate-limit cooldown management, canonical URL authority/redaction helpers,
//! and size-capped tile download helpers.

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
pub(crate) const DEFAULT_MAX_CONCURRENT_PER_HOST: usize = 6;

/// Default maximum encoded response body accepted for one live tile (`16 MiB`).
pub(crate) const DEFAULT_MAX_TILE_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

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

/// Parsed HTTP(S) authority components (`host_port` without `user:pass@`, and
/// bare `host` preserving bracketed IPv6 literals like `[::1]`).
pub(crate) struct ParsedAuthority<'a> {
    pub host_port: &'a str,
    pub host: &'a str,
}

/// Parse the authority section of an HTTP(S) URL or template, handling
/// `user:pass@` userinfo and bracketed IPv6 literals (`http://[::1]:8080/...`).
pub(crate) fn parse_url_authority(url: &str) -> Option<ParsedAuthority<'_>> {
    let without_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let authority = without_scheme.split(['/', '?', '#']).next().unwrap_or("");
    let host_port = authority.rsplit_once('@').map_or(authority, |(_, hp)| hp);
    if host_port.is_empty() {
        return None;
    }
    let host = if let Some(rest) = host_port.strip_prefix('[') {
        let end = rest.find(']')?;
        let inner = &rest[..end];
        if inner.is_empty() {
            return None;
        }
        let after = &rest[end + 1..];
        if !after.is_empty() && !after.starts_with(':') {
            return None;
        }
        &host_port[..=end + 1]
    } else {
        let h = host_port.split(':').next().unwrap_or("");
        if h.is_empty() {
            return None;
        }
        h
    };
    Some(ParsedAuthority { host_port, host })
}

/// Extract the lowercased `host[:port]` authority key from an HTTP(S) URL.
pub(crate) fn extract_host(url: &str) -> String {
    parse_url_authority(url)
        .map(|auth| auth.host_port.to_ascii_lowercase())
        .unwrap_or_else(|| "unknown".to_owned())
}

/// Redact `user:pass@` authority credentials and `?query` strings from a URL
/// or URL template so inline secrets never appear in `Debug` output, logs, or
/// UI tooltips.
pub fn sanitize_url_for_display(url: &str) -> String {
    let trimmed = url.trim();
    let scheme_end = trimmed.find("://").filter(|&idx| {
        let scheme = &trimmed[..idx];
        scheme
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphabetic())
            && scheme
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
    });
    let Some(scheme_idx) = scheme_end else {
        return match trimmed.split_once('?') {
            Some((base, _)) => format!("{base}?<redacted>"),
            None => trimmed.to_owned(),
        };
    };

    let scheme_prefix = &trimmed[..scheme_idx + 3];
    let rest = &trimmed[scheme_idx + 3..];
    let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..auth_end];
    let tail = &rest[auth_end..];

    let sanitized_authority = match authority.rsplit_once('@') {
        Some((_, host_port)) => format!("<redacted>@{host_port}"),
        None => authority.to_owned(),
    };

    let sanitized_tail = match tail.split_once('?') {
        Some((path, query_and_frag)) => match query_and_frag.split_once('#') {
            Some((_, frag)) => format!("{path}?<redacted>#{frag}"),
            None => format!("{path}?<redacted>"),
        },
        None => tail.to_owned(),
    };

    format!("{scheme_prefix}{sanitized_authority}{sanitized_tail}")
}

/// Redact `user:pass@` credentials and `?query` tokens inside any URLs
/// embedded within an arbitrary error or diagnostic string (e.g.
/// `reqwest::Error::to_string()`).
pub(crate) fn sanitize_urls_in_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = find_url_scheme_start(rest) {
        out.push_str(&rest[..pos]);
        let candidate = &rest[pos..];
        let raw_end = candidate
            .find(|c: char| c.is_whitespace() || matches!(c, '"' | '<' | '>' | '`'))
            .unwrap_or(candidate.len());
        let end = trim_trailing_prose_punct(&candidate[..raw_end]);
        out.push_str(&sanitize_url_for_display(&candidate[..end]));
        rest = &candidate[end..];
    }
    out.push_str(rest);
    out
}

fn find_url_scheme_start(s: &str) -> Option<usize> {
    let mut offset = 0usize;
    while let Some(rel) = s[offset..].find("://") {
        let sep = offset + rel;
        let prefix = &s[..sep];
        let start = prefix
            .char_indices()
            .rfind(|&(_, c)| !(c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')))
            .map_or(0, |(idx, c)| idx + c.len_utf8());
        let scheme = &s[start..sep];
        if scheme
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphabetic())
        {
            return Some(start);
        }
        offset = sep + 3;
    }
    None
}

/// Strip trailing sentence/wrapper punctuation from a scanned URL token while
/// preserving balanced `[...]` IPv6 literals, balanced `(...)`, and internal
/// RFC 3986 sub-delimiters (`,`, `;`, `'`).
fn trim_trailing_prose_punct(token: &str) -> usize {
    let mut end = token.len();
    while end > 0 {
        let slice = &token[..end];
        let Some(last) = slice.chars().next_back() else {
            break;
        };
        let strip = match last {
            '.' | ',' | ';' | ':' | '\'' => true,
            ')' => slice.matches(')').count() > slice.matches('(').count(),
            ']' => slice.matches(']').count() > slice.matches('[').count(),
            _ => false,
        };
        if strip {
            end -= last.len_utf8();
        } else {
            break;
        }
    }
    end
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
    fn extract_host_ipv6_sanitization_and_retry_after_handle_edge_cases() {
        assert_eq!(
            extract_host("https://GIBS.Earthdata.NASA.gov/wmts/1/2/3.jpg"),
            "gibs.earthdata.nasa.gov"
        );
        assert_eq!(
            extract_host("http://user:pass@127.0.0.1:8080/0/0/0.png?key=1"),
            "127.0.0.1:8080"
        );
        assert_eq!(extract_host("http://[::1]:8080/0/0/0.png"), "[::1]:8080");
        let ipv6 = parse_url_authority("http://user:pw@[::1]:8080/0/0/0.png").expect("ipv6");
        assert_eq!(ipv6.host, "[::1]");
        assert_eq!(ipv6.host_port, "[::1]:8080");

        assert_eq!(
            sanitize_url_for_display("https://alice:secret@tiles.example.com/1/2/3.png?token=xyz"),
            "https://<redacted>@tiles.example.com/1/2/3.png?<redacted>"
        );
        assert_eq!(
            sanitize_url_for_display(
                "  HTTPS://alice:p,a;s]s@tiles.example.com/1/2/3.png?token=xyz  "
            ),
            "HTTPS://<redacted>@tiles.example.com/1/2/3.png?<redacted>"
        );
        assert_eq!(
            sanitize_urls_in_text(
                "error sending request for url (https://u:p@host.test/1/2/3.png?token=abc): timed out"
            ),
            "error sending request for url (https://<redacted>@host.test/1/2/3.png?<redacted>): timed out"
        );
        assert_eq!(
            sanitize_urls_in_text(
                "error sending request for url (http://[::1]:8080/1/2/3.png?token=secret): timed out"
            ),
            "error sending request for url (http://[::1]:8080/1/2/3.png?<redacted>): timed out"
        );
        assert_eq!(
            sanitize_urls_in_text(
                "request failed: https://tiles.example/wms?bbox=0,0,1,1&layers=a;b&token=secret, retrying"
            ),
            "request failed: https://tiles.example/wms?<redacted>, retrying"
        );
        assert_eq!(
            sanitize_urls_in_text(
                "failed [HTTPS://alice:p,a;s]s@tiles.example/0/0/0.png?key=secret]"
            ),
            "failed [HTTPS://<redacted>@tiles.example/0/0/0.png?<redacted>]"
        );
        // Multi-byte UTF-8 punctuation immediately before scheme must not panic.
        assert_eq!(
            sanitize_urls_in_text("failed—https://u:p@host.test/0/0/0.png?token=secret"),
            "failed—https://<redacted>@host.test/0/0/0.png?<redacted>"
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
