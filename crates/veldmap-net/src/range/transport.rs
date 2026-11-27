//! Unified RFC 9110 HTTP Range GET transport shared by `HttpRangeSource::open_async`
//! (1-RTT adaptive `0..probe_len` probe) and block-span fetches (`If-Match`,
//! response `ETag` verification, `Content-Range` validation, `429`/`503`
//! cooldowns, and cancel-aware retries).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reqwest::header::{
    ACCEPT_ENCODING, CONTENT_RANGE, ETAG, HeaderMap, HeaderName, HeaderValue, IF_MATCH, RANGE,
};
use veldmap_core::sync::CancelToken;

use crate::http::{BodyReadError, HostLimiters, read_body_capped, record_rate_limit_cooldown};
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
    pub(super) request_counter: Option<&'a AtomicU64>,
}

pub(super) struct RangeFetchResponse {
    pub(super) total_len: u64,
    pub(super) strong_etag: Option<String>,
    pub(super) body: Vec<u8>,
    pub(super) requests_made: u64,
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

async fn sleep_cancel(dur: Duration, cancel: &CancelToken) -> Result<(), RangeTransportError> {
    tokio::select! {
        biased;
        _ = cancel.cancelled() => Err(RangeTransportError::Cancelled),
        _ = tokio::time::sleep(dur) => Ok(()),
    }
}

/// Attempt a retry when under `max_retries` and uncancelled; callers must
/// release their host semaphore permit BEFORE invoking this helper.
///
/// For `429`/`503` responses, pass `sleep_backoff = false` so the next loop
/// iteration's `acquire_permit` honors the recorded host cooldown directly
/// without an extra backoff sleep.
async fn try_retry(
    attempt: &mut usize,
    max_retries: usize,
    cancel: &CancelToken,
    sleep_backoff: bool,
    make_err: impl FnOnce() -> RangeTransportError,
) -> Result<(), RangeTransportError> {
    if *attempt < max_retries && !cancel.is_cancelled() {
        *attempt += 1;
        if sleep_backoff {
            sleep_cancel(Duration::from_millis(150 * (*attempt as u64)), cancel).await?;
        }
        return Ok(());
    }
    Err(make_err())
}

fn handle_416_open_probe(
    headers: &HeaderMap,
    requests_made: u64,
) -> Result<RangeFetchResponse, RangeTransportError> {
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
            requests_made,
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
    let mut requests_made = 0u64;
    loop {
        if cancel.is_cancelled() {
            return Err(RangeTransportError::Cancelled);
        }

        let permit = req_cfg
            .host_limiters
            .acquire_permit(req_cfg.host, cancel)
            .await
            .map_err(|_| RangeTransportError::Cancelled)?;

        let mut req = req_cfg
            .client
            .get(req_cfg.url)
            .header(RANGE, &range_hdr)
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

        requests_made += 1;
        if let Some(counter) = req_cfg.request_counter {
            counter.fetch_add(1, Ordering::Relaxed);
        }

        let resp_res = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(RangeTransportError::Cancelled),
            res = req.send() => res,
        };

        let mut resp = match resp_res {
            Ok(resp) => resp,
            Err(err) => {
                drop(permit);
                try_retry(&mut attempt, req_cfg.max_retries, cancel, true, || {
                    RangeTransportError::Network(redact_text_many(
                        &err.to_string(),
                        req_cfg.secrets,
                    ))
                })
                .await?;
                continue;
            }
        };

        let status = resp.status();
        let code = status.as_u16();

        if code == 200 {
            return Err(RangeTransportError::RangeNotSupported(
                "server responded with 200 OK instead of 206 Partial Content".to_owned(),
            ));
        }

        if code == 412 && matches!(req_cfg.mode, RangeFetchMode::BlockSpan { .. }) {
            return Err(RangeTransportError::PreconditionFailed(
                "ETag mismatch during byte-range fetch".to_owned(),
            ));
        }

        if code == 416 && matches!(req_cfg.mode, RangeFetchMode::OpenProbe { .. }) {
            return handle_416_open_probe(resp.headers(), requests_made);
        }

        if matches!(code, 429 | 503) {
            record_rate_limit_cooldown(
                req_cfg.host_limiters,
                req_cfg.host,
                resp.headers(),
                DEFAULT_RATE_LIMIT_BACKOFF,
            );
            drop(permit);
            try_retry(&mut attempt, req_cfg.max_retries, cancel, false, || {
                RangeTransportError::HttpStatus {
                    code,
                    detail: "rate limited".to_owned(),
                }
            })
            .await?;
            continue;
        }

        if status.is_server_error() || code == 408 {
            drop(permit);
            try_retry(&mut attempt, req_cfg.max_retries, cancel, true, || {
                RangeTransportError::HttpStatus {
                    code,
                    detail: status
                        .canonical_reason()
                        .unwrap_or("server error")
                        .to_owned(),
                }
            })
            .await?;
            continue;
        }

        if code != 206 {
            return Err(RangeTransportError::HttpStatus {
                code,
                detail: status
                    .canonical_reason()
                    .unwrap_or("unexpected status")
                    .to_owned(),
            });
        }

        let (response_strong_etag, expected_len, total_len) =
            validate_206_response_headers(resp.headers(), &req_cfg.mode)?;

        let body_res = read_body_capped(&mut resp, expected_len, expected_len, cancel).await;
        drop(permit);

        let body = match body_res {
            Ok(body) => body,
            Err(BodyReadError::Cancelled) => return Err(RangeTransportError::Cancelled),
            Err(BodyReadError::TooLarge) => {
                return Err(RangeTransportError::InvalidResponse(format!(
                    "206 body exceeded Content-Range length ({expected_len} B)"
                )));
            }
            Err(BodyReadError::Network(err)) => {
                try_retry(&mut attempt, req_cfg.max_retries, cancel, true, || {
                    RangeTransportError::Network(redact_text_many(
                        &err.to_string(),
                        req_cfg.secrets,
                    ))
                })
                .await?;
                continue;
            }
        };

        if body.len() != expected_len {
            try_retry(&mut attempt, req_cfg.max_retries, cancel, true, || {
                RangeTransportError::InvalidResponse(format!(
                    "truncated 206 body: expected {expected_len} B, received {} B",
                    body.len()
                ))
            })
            .await?;
            continue;
        }

        return Ok(RangeFetchResponse {
            total_len,
            strong_etag: response_strong_etag,
            body,
            requests_made,
        });
    }
}
