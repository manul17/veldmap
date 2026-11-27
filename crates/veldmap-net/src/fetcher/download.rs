//! Size-capped, cancel-aware HTTP tile download and error classification
//! helpers for [`crate::LiveTilePipeline`].

use std::time::Duration;

use veldmap_core::sync::CancelToken;

use crate::http::{BodyReadError, parse_retry_after_seconds, read_body_capped};

pub(super) enum FetchOutcome {
    Payload(Vec<u8>),
    /// HTTP `204 No Content` (valid empty/nodata tile, not an error).
    Empty204,
    /// Download was cancelled via [`CancelToken`] before completion.
    Cancelled,
}

/// Classification of a transient HTTP/network error so impossible combinations
/// (such as non-rate-limited errors carrying `Retry-After`) are unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TransientKind {
    Network { offline: bool },
    ServerError,
    RateLimited { retry_after: Option<Duration> },
}

pub(super) struct TransientFetchError {
    pub(super) msg: String,
    pub(super) kind: TransientKind,
}

pub(super) enum FetchFailure {
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

fn classify_http_error(status: reqwest::StatusCode, retry_after: Option<&str>) -> FetchFailure {
    let code = status.as_u16();
    let msg = format!("HTTP {code}");
    if matches!(code, 429 | 503) {
        return FetchFailure::Transient(TransientFetchError {
            msg,
            kind: TransientKind::RateLimited {
                retry_after: parse_retry_after_seconds(retry_after),
            },
        });
    }
    if code == 408 {
        // Request Timeout is transient, and receiving an HTTP response proves
        // the host is reachable (so do not set `offline = true`).
        return FetchFailure::Transient(TransientFetchError {
            msg,
            kind: TransientKind::ServerError,
        });
    }
    if status.is_client_error() {
        return FetchFailure::Terminal(msg);
    }
    FetchFailure::Transient(TransientFetchError {
        msg,
        kind: TransientKind::ServerError,
    })
}

pub(super) async fn download_capped(
    client: &reqwest::Client,
    url: &str,
    max_bytes: usize,
    cancel: &CancelToken,
) -> Result<FetchOutcome, FetchFailure> {
    if cancel.is_cancelled() {
        return Ok(FetchOutcome::Cancelled);
    }

    let resp_res = tokio::select! {
        biased;
        _ = cancel.cancelled() => return Ok(FetchOutcome::Cancelled),
        res = client.get(url).send() => res,
    };
    let mut resp = resp_res.map_err(map_reqwest_err)?;

    let status = resp.status();
    if status.as_u16() == 204 {
        return Ok(FetchOutcome::Empty204);
    }
    if !status.is_success() {
        let hdr = resp
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok());
        return Err(classify_http_error(status, hdr));
    }

    if let Some(len) = resp.content_length()
        && len > max_bytes as u64
    {
        return Err(FetchFailure::Terminal(format!(
            "tile payload {len} B exceeds limit {max_bytes} B"
        )));
    }

    let initial_cap = resp
        .content_length()
        .map_or(16 * 1024, |n| (n as usize).min(max_bytes));
    let buf = match read_body_capped(&mut resp, max_bytes, initial_cap, cancel).await {
        Ok(buf) => buf,
        Err(BodyReadError::Cancelled) => return Ok(FetchOutcome::Cancelled),
        Err(BodyReadError::TooLarge) => {
            return Err(FetchFailure::Terminal(format!(
                "tile payload exceeded limit {max_bytes} B"
            )));
        }
        Err(BodyReadError::Network(err)) => return Err(map_reqwest_err(err)),
    };

    if buf.is_empty() {
        return Err(FetchFailure::Terminal("empty tile response".to_owned()));
    }
    Ok(FetchOutcome::Payload(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_http_error_treats_client_errors_as_terminal_except_408_and_429() {
        for status in [
            reqwest::StatusCode::BAD_REQUEST,
            reqwest::StatusCode::METHOD_NOT_ALLOWED,
            reqwest::StatusCode::PRECONDITION_FAILED,
            reqwest::StatusCode::RANGE_NOT_SATISFIABLE,
            reqwest::StatusCode::UNPROCESSABLE_ENTITY,
        ] {
            assert!(
                matches!(classify_http_error(status, None), FetchFailure::Terminal(_)),
                "expected terminal for {status}"
            );
        }
        assert!(matches!(
            classify_http_error(reqwest::StatusCode::REQUEST_TIMEOUT, None),
            FetchFailure::Transient(TransientFetchError {
                kind: TransientKind::ServerError,
                ..
            })
        ));
        assert!(matches!(
            classify_http_error(reqwest::StatusCode::TOO_MANY_REQUESTS, Some("7")),
            FetchFailure::Transient(TransientFetchError {
                kind: TransientKind::RateLimited {
                    retry_after: Some(d),
                },
                ..
            }) if d == Duration::from_secs(7)
        ));
        assert!(matches!(
            classify_http_error(reqwest::StatusCode::INTERNAL_SERVER_ERROR, None),
            FetchFailure::Transient(TransientFetchError {
                kind: TransientKind::ServerError,
                ..
            })
        ));
    }
}
