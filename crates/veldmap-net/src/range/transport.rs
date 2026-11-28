//! Unified RFC 9110 HTTP Range GET transport shared by `HttpRangeSource::open_async`
//! (1-RTT adaptive `0..probe_len` probe) and block-span fetches (`If-Match`,
//! response `ETag` verification, `Content-Range` validation, `429`/`503`
//! cooldowns, and cancel-aware retries).

#[cfg(any(test, feature = "test-support"))]
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reqwest::header::{
    ACCEPT_ENCODING, CONTENT_RANGE, ETAG, HeaderMap, HeaderName, HeaderValue, IF_MATCH, RANGE,
};
use veldmap_core::sync::CancelToken;

use crate::http::{BodyReadError, read_body_capped};
use crate::limiter::{HostLimiters, record_rate_limit_cooldown};
use crate::redact::{EncodedSecret, redact_text_many};

use super::RangeOpenError;
use super::protocol::{
    ContentRange, parse_content_range, parse_strong_etag, validate_identity_content_encoding,
};

/// Default fallback retry delay when a server returns `429`/`503` without
/// `Retry-After`.
const DEFAULT_RATE_LIMIT_BACKOFF: Duration = Duration::from_secs(2);

pub(super) enum RangeFetchMode<'a> {
    OpenProbe {
        probe_len: usize,
    },
    BlockSpan {
        byte_start: u64,
        byte_end_incl: u64,
        total_len: u64,
        strong_etag: Option<&'a str>,
    },
}

pub(super) struct RangeFetchRequest<'a> {
    pub(super) client: &'a reqwest::Client,
    pub(super) host_limiters: &'a HostLimiters,
    pub(super) url: &'a str,
    pub(super) host: &'a str,
    pub(super) extra_headers: &'a [(HeaderName, HeaderValue)],
    pub(super) secrets: &'a [EncodedSecret],
    pub(super) max_retries: usize,
    pub(super) mode: RangeFetchMode<'a>,
    #[cfg(any(test, feature = "test-support"))]
    pub(super) request_counter: &'a AtomicU64,
}

pub(super) struct RangeFetchResponse {
    pub(super) total_len: u64,
    pub(super) strong_etag: Option<String>,
    pub(super) body: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(super) enum RangeTransportError {
    Cancelled,
    RangeNotSupported(String),
    PreconditionFailed(String),
    HttpStatus { code: u16, detail: String },
    InvalidResponse(String),
    Network(String),
}

impl From<RangeTransportError> for RangeOpenError {
    fn from(err: RangeTransportError) -> Self {
        match err {
            RangeTransportError::Cancelled => Self::Cancelled,
            RangeTransportError::RangeNotSupported(msg) => Self::RangeNotSupported(msg),
            RangeTransportError::PreconditionFailed(msg) => Self::HttpStatus {
                code: 412,
                detail: msg,
            },
            RangeTransportError::HttpStatus { code, detail } => Self::HttpStatus { code, detail },
            RangeTransportError::InvalidResponse(msg) => Self::InvalidResponse(msg),
            RangeTransportError::Network(msg) => Self::Network(msg),
        }
    }
}

enum AttemptOutcome {
    Done(RangeFetchResponse),
    Retry {
        sleep_backoff: bool,
        err: RangeTransportError,
    },
    Fatal(RangeTransportError),
}

async fn sleep_cancel(dur: Duration, cancel: &CancelToken) -> Result<(), RangeTransportError> {
    tokio::select! {
        biased;
        _ = cancel.cancelled() => Err(RangeTransportError::Cancelled),
        _ = tokio::time::sleep(dur) => Ok(()),
    }
}

fn handle_416_open_probe(headers: &HeaderMap) -> Result<RangeFetchResponse, RangeTransportError> {
    let cr = headers
        .get(CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .map(parse_content_range)
        .transpose()
        .map_err(RangeTransportError::InvalidResponse)?;
    let etag = parse_strong_etag(headers);
    if matches!(cr, Some(ContentRange::Unsatisfied { total: 0 })) {
        return Ok(RangeFetchResponse {
            total_len: 0,
            strong_etag: etag,
            body: Vec::new(),
        });
    }
    Err(RangeTransportError::HttpStatus {
        code: 416,
        detail: "Range Not Satisfiable".to_owned(),
    })
}

fn validate_206_response_headers(
    headers: &HeaderMap,
    mode: &RangeFetchMode<'_>,
) -> Result<(Option<String>, usize, u64), RangeTransportError> {
    validate_identity_content_encoding(headers).map_err(RangeTransportError::InvalidResponse)?;

    let response_strong_etag = parse_strong_etag(headers);
    if let RangeFetchMode::BlockSpan {
        strong_etag: Some(expected),
        ..
    } = *mode
        && headers.contains_key(ETAG)
        && response_strong_etag.as_deref() != Some(expected)
    {
        return Err(RangeTransportError::PreconditionFailed(
            "response ETag mismatch during byte-range fetch".to_owned(),
        ));
    }

    let cr_raw = headers
        .get(CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            RangeTransportError::InvalidResponse(
                "206 response missing Content-Range header".to_owned(),
            )
        })?;
    let cr = parse_content_range(cr_raw).map_err(RangeTransportError::InvalidResponse)?;
    let (expected_len, total_len) = match *mode {
        RangeFetchMode::OpenProbe { probe_len } => match cr {
            ContentRange::Span {
                start: 0,
                end,
                total,
            } => {
                let expected = end.saturating_add(1);
                let cap = probe_len.max(1) as u64;
                if expected > cap {
                    return Err(RangeTransportError::InvalidResponse(format!(
                        "206 open probe returned {expected} B exceeding requested {cap} B"
                    )));
                }
                (expected as usize, total)
            }
            other => {
                return Err(RangeTransportError::InvalidResponse(format!(
                    "206 open probe Content-Range did not start at byte 0: {other:?}"
                )));
            }
        },
        RangeFetchMode::BlockSpan {
            byte_start,
            byte_end_incl,
            total_len,
            ..
        } => match cr {
            ContentRange::Span { start, end, total }
                if start == byte_start && end == byte_end_incl && total == total_len =>
            {
                ((byte_end_incl - byte_start + 1) as usize, total)
            }
            other => {
                return Err(RangeTransportError::InvalidResponse(format!(
                    "Content-Range mismatch: expected {byte_start}-{byte_end_incl}/{total_len}, got {other:?}"
                )));
            }
        },
    };

    Ok((response_strong_etag, expected_len, total_len))
}

fn classify_range_status(
    req_cfg: &RangeFetchRequest<'_>,
    status: reqwest::StatusCode,
    headers: &HeaderMap,
) -> Option<AttemptOutcome> {
    let code = status.as_u16();

    if code == 200 {
        return Some(AttemptOutcome::Fatal(
            RangeTransportError::RangeNotSupported(
                "server responded with 200 OK instead of 206 Partial Content".to_owned(),
            ),
        ));
    }

    if code == 412 && matches!(req_cfg.mode, RangeFetchMode::BlockSpan { .. }) {
        return Some(AttemptOutcome::Fatal(
            RangeTransportError::PreconditionFailed(
                "ETag mismatch during byte-range fetch".to_owned(),
            ),
        ));
    }

    if code == 416 && matches!(req_cfg.mode, RangeFetchMode::OpenProbe { .. }) {
        return Some(match handle_416_open_probe(headers) {
            Ok(probe) => AttemptOutcome::Done(probe),
            Err(err) => AttemptOutcome::Fatal(err),
        });
    }

    if matches!(code, 429 | 503) {
        record_rate_limit_cooldown(
            req_cfg.host_limiters,
            req_cfg.host,
            headers,
            DEFAULT_RATE_LIMIT_BACKOFF,
        );
        return Some(AttemptOutcome::Retry {
            sleep_backoff: false,
            err: RangeTransportError::HttpStatus {
                code,
                detail: "rate limited".to_owned(),
            },
        });
    }

    if status.is_server_error() || code == 408 {
        return Some(AttemptOutcome::Retry {
            sleep_backoff: true,
            err: RangeTransportError::HttpStatus {
                code,
                detail: status
                    .canonical_reason()
                    .unwrap_or("server error")
                    .to_owned(),
            },
        });
    }

    if code != 206 {
        return Some(AttemptOutcome::Fatal(RangeTransportError::HttpStatus {
            code,
            detail: status
                .canonical_reason()
                .unwrap_or("unexpected status")
                .to_owned(),
        }));
    }

    None
}

async fn read_range_body(
    resp: &mut reqwest::Response,
    expected_len: usize,
    total_len: u64,
    strong_etag: Option<String>,
    secrets: &[EncodedSecret],
    cancel: &CancelToken,
) -> AttemptOutcome {
    let body = match read_body_capped(resp, expected_len, expected_len, cancel).await {
        Ok(body) => body,
        Err(BodyReadError::Cancelled) => {
            return AttemptOutcome::Fatal(RangeTransportError::Cancelled);
        }
        Err(BodyReadError::TooLarge) => {
            return AttemptOutcome::Fatal(RangeTransportError::InvalidResponse(format!(
                "206 body exceeded Content-Range length ({expected_len} B)"
            )));
        }
        Err(BodyReadError::Network(err)) => {
            return AttemptOutcome::Retry {
                sleep_backoff: true,
                err: RangeTransportError::Network(redact_text_many(&err.to_string(), secrets)),
            };
        }
    };

    if body.len() != expected_len {
        return AttemptOutcome::Retry {
            sleep_backoff: true,
            err: RangeTransportError::InvalidResponse(format!(
                "truncated 206 body: expected {expected_len} B, received {} B",
                body.len()
            )),
        };
    }

    AttemptOutcome::Done(RangeFetchResponse {
        total_len,
        strong_etag,
        body,
    })
}

/// Execute a single HTTP Range attempt while owning the per-host semaphore
/// permit strictly inside this function's lexical scope so the permit is
/// guaranteed to drop before any retry backoff sleep or host cooldown wait.
async fn execute_range_attempt(
    req_cfg: &RangeFetchRequest<'_>,
    range_hdr: &str,
    cancel: &CancelToken,
) -> AttemptOutcome {
    let Ok(_permit) = req_cfg
        .host_limiters
        .acquire_permit(req_cfg.host, cancel)
        .await
    else {
        return AttemptOutcome::Fatal(RangeTransportError::Cancelled);
    };

    let mut req = req_cfg
        .client
        .get(req_cfg.url)
        .header(RANGE, range_hdr)
        .header(ACCEPT_ENCODING, "identity");
    if let RangeFetchMode::BlockSpan {
        strong_etag: Some(etag),
        ..
    } = req_cfg.mode
    {
        req = req.header(IF_MATCH, etag);
    }
    for (name, value) in req_cfg.extra_headers {
        req = req.header(name, value);
    }

    #[cfg(any(test, feature = "test-support"))]
    req_cfg.request_counter.fetch_add(1, Ordering::Relaxed);

    let resp_res = tokio::select! {
        biased;
        _ = cancel.cancelled() => return AttemptOutcome::Fatal(RangeTransportError::Cancelled),
        res = req.send() => res,
    };

    let mut resp = match resp_res {
        Ok(resp) => resp,
        Err(err) => {
            return AttemptOutcome::Retry {
                sleep_backoff: true,
                err: RangeTransportError::Network(redact_text_many(
                    &err.to_string(),
                    req_cfg.secrets,
                )),
            };
        }
    };

    if let Some(outcome) = classify_range_status(req_cfg, resp.status(), resp.headers()) {
        return outcome;
    }

    let (response_strong_etag, expected_len, total_len) =
        match validate_206_response_headers(resp.headers(), &req_cfg.mode) {
            Ok(parsed) => parsed,
            Err(err) => return AttemptOutcome::Fatal(err),
        };

    read_range_body(
        &mut resp,
        expected_len,
        total_len,
        response_strong_etag,
        req_cfg.secrets,
        cancel,
    )
    .await
}

pub(super) async fn fetch_range_http(
    req_cfg: &RangeFetchRequest<'_>,
    cancel: &CancelToken,
) -> Result<RangeFetchResponse, RangeTransportError> {
    let range_hdr = match req_cfg.mode {
        RangeFetchMode::OpenProbe { probe_len } => format!("bytes=0-{}", probe_len.max(1) - 1),
        RangeFetchMode::BlockSpan {
            byte_start,
            byte_end_incl,
            ..
        } => format!("bytes={byte_start}-{byte_end_incl}"),
    };

    let mut attempt = 0usize;
    loop {
        if cancel.is_cancelled() {
            return Err(RangeTransportError::Cancelled);
        }

        match execute_range_attempt(req_cfg, &range_hdr, cancel).await {
            AttemptOutcome::Done(resp) => return Ok(resp),
            AttemptOutcome::Fatal(err) => return Err(err),
            AttemptOutcome::Retry { sleep_backoff, err } => {
                if attempt >= req_cfg.max_retries || cancel.is_cancelled() {
                    return Err(err);
                }
                attempt += 1;
                if sleep_backoff {
                    sleep_cancel(Duration::from_millis(150 * (attempt as u64)), cancel).await?;
                }
            }
        }
    }
}
