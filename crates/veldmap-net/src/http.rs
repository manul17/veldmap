//! HTTP client construction (`rustls` + `ring`), custom auth header validation,
//! cancel-token linking, and size-capped response body reading helpers.

use std::sync::OnceLock;
use std::time::Duration;

use reqwest::header::{HeaderName, HeaderValue};
use veldmap_core::sync::{CancelGuard, CancelToken};

use crate::NetError;
use crate::redact::{EncodedSecret, extract_header_secret};
use crate::url::same_url_origin;

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
/// overwrite (breaking HTTP framing, content negotiation, byte-range offsets,
/// or ETag guards).
const FORBIDDEN_CUSTOM_HEADERS: &[&str] = &[
    "host",
    "range",
    "if-match",
    "accept",
    "accept-encoding",
    "content-type",
    "content-length",
    "transfer-encoding",
    "connection",
    "te",
    "upgrade",
];

/// Return whether `name` is a valid HTTP header name permitted for user-supplied
/// custom authentication headers (rejecting invalid syntax and protocol-owned
/// headers such as `Host`, `Range`, `If-Match`, `Accept`, `Content-Type`, etc.).
pub fn is_valid_custom_header_name(name: &str) -> bool {
    validate_custom_header_name(name).is_ok()
}

fn validate_custom_header_name(name: &str) -> Result<HeaderName, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("header name must not be empty".to_owned());
    }
    let parsed = HeaderName::from_bytes(trimmed.as_bytes()).map_err(|e| e.to_string())?;
    let lower = parsed.as_str();
    if FORBIDDEN_CUSTOM_HEADERS.contains(&lower) {
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
) -> Result<ParsedCustomHeaders, (String, String)> {
    let mut parsed = Vec::with_capacity(headers.len());
    let mut secrets = Vec::with_capacity(headers.len());
    for (k, v) in headers {
        let name = validate_custom_header_name(k).map_err(|reason| (k.clone(), reason))?;
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
        if same_origin_only
            && let Some(first) = attempt.previous().first()
            && !same_url_origin(first, attempt.url())
        {
            return attempt.error("refused cross-origin redirect with custom auth headers");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_header_validation_rejects_protocol_owned_headers() {
        assert!(is_valid_custom_header_name("x-api-key"));
        assert!(is_valid_custom_header_name("Authorization"));
        for forbidden in [
            "",
            "   ",
            "Host",
            "Range",
            "If-Match",
            "Accept",
            "Accept-Encoding",
            "Content-Type",
            "Content-Length",
        ] {
            assert!(
                !is_valid_custom_header_name(forbidden),
                "expected '{forbidden}' to be rejected"
            );
        }
    }
}
