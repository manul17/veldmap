//! Asynchronous STAC `/search` HTTP client sharing [`NetContext`].

use std::sync::Arc;
use std::time::Duration;

use reqwest::header::{ACCEPT, CONTENT_TYPE};
use serde_json::Value;
use veldmap_core::sync::CancelToken;

use crate::context::NetContext;
use crate::http::{
    BodyReadError, HostLimiters, link_cancel_token, parse_custom_headers, read_body_capped,
    record_rate_limit_cooldown,
};
use crate::redact::{redact_text_many, sanitize_url_for_display};
use crate::url::{extract_host, is_valid_http_endpoint, split_url_query_fragment};

use super::parse::{build_search_payload, parse_search_response};
use super::types::{StacCatalogConfig, StacError, StacSearchQuery, StacSearchResult};

/// Hard response body size cap for STAC JSON search responses (`16 MiB`).
const MAX_STAC_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Maximum error body snippet bytes read on non-2xx STAC responses.
const MAX_STAC_ERROR_SNIPPET_BYTES: usize = 512;

/// Additional protocol-owned headers forbidden in STAC custom auth headers.
const STAC_EXTRA_FORBIDDEN_HEADERS: &[&str] = &["content-type", "accept"];

/// Asynchronous STAC API search client sharing [`NetContext`]'s gzip-enabled
/// HTTP clients and per-host concurrency/cooldown limiters.
///
/// Intentionally holds cloned inner `Arc`s / clients rather than
/// `Arc<NetContext>` so spawned Tokio tasks can never drop `NetContext` on a
/// Tokio worker thread.
#[derive(Clone)]
pub struct StacClient {
    api_client: reqwest::Client,
    api_auth_client: reqwest::Client,
    host_limiters: Arc<HostLimiters>,
    shutdown: CancelToken,
}

impl StacClient {
    pub fn new(net: Arc<NetContext>) -> Self {
        Self {
            api_client: net.api_client().clone(),
            api_auth_client: net.api_auth_client().clone(),
            host_limiters: net.host_limiters(),
            shutdown: net.shutdown_token().clone(),
        }
    }

    /// Execute a STAC `/search` query asynchronously against `catalog`.
    pub async fn search_async(
        &self,
        catalog: &StacCatalogConfig,
        query: &StacSearchQuery,
        cancel: &CancelToken,
    ) -> Result<StacSearchResult, StacError> {
        let (linked_cancel, _cancel_guard) = link_cancel_token(&self.shutdown, cancel);
        if linked_cancel.is_cancelled() {
            return Err(StacError::Cancelled);
        }

        let endpoint = catalog.endpoint.trim();
        if !is_valid_http_endpoint(endpoint) {
            return Err(StacError::InvalidEndpoint(sanitize_url_for_display(
                endpoint,
            )));
        }
        let search_base = append_search_path(endpoint);
        let search_url = catalog.auth.apply_to_url(&search_base);
        let host = extract_host(&search_url);

        let raw_headers = catalog.auth.extra_headers();
        let parsed = parse_custom_headers(&raw_headers, STAC_EXTRA_FORBIDDEN_HEADERS)
            .map_err(|(name, reason)| StacError::InvalidAuthHeader { name, reason })?;
        let custom_headers = parsed.headers;
        let mut secrets = parsed.secrets;
        if let Some(secret) = catalog.auth.encoded_secret()
            && !secrets.contains(&secret)
        {
            secrets.push(secret);
        }
        let redact_msg = |msg: &str| redact_text_many(msg, &secrets);

        let body_json = build_search_payload(&catalog.collection, query);
        let body_bytes =
            serde_json::to_vec(&body_json).map_err(|e| StacError::InvalidJson(e.to_string()))?;

        let permit = self
            .host_limiters
            .acquire_permit(&host, &linked_cancel)
            .await
            .map_err(|_| StacError::Cancelled)?;

        let client = if custom_headers.is_empty() {
            &self.api_client
        } else {
            &self.api_auth_client
        };
        let mut req = client
            .post(&search_url)
            .header(ACCEPT, "application/geo+json, application/json")
            .header(CONTENT_TYPE, "application/json")
            .body(body_bytes);
        for (name, val) in custom_headers {
            req = req.header(name, val);
        }

        let resp_res = tokio::select! {
            biased;
            _ = linked_cancel.cancelled() => return Err(StacError::Cancelled),
            res = req.send() => res,
        };
        let mut resp = match resp_res {
            Ok(resp) => resp,
            Err(err) => {
                return Err(StacError::Network(redact_msg(&err.to_string())));
            }
        };

        let status = resp.status();
        let code = status.as_u16();
        if matches!(code, 429 | 503) {
            record_rate_limit_cooldown(
                &self.host_limiters,
                &host,
                resp.headers(),
                Duration::from_secs(5),
            );
        }
        if !status.is_success() {
            let detail = read_error_detail(&mut resp, status, &linked_cancel, &redact_msg).await?;
            return Err(StacError::HttpStatus { code, detail });
        }

        let raw = match read_body_capped(
            &mut resp,
            MAX_STAC_RESPONSE_BYTES,
            16 * 1024,
            &linked_cancel,
        )
        .await
        {
            Ok(bytes) => bytes,
            Err(BodyReadError::Cancelled) => return Err(StacError::Cancelled),
            Err(BodyReadError::TooLarge) => {
                return Err(StacError::ResponseTooLarge {
                    limit: MAX_STAC_RESPONSE_BYTES,
                });
            }
            Err(BodyReadError::Network(err)) => {
                return Err(StacError::Network(redact_msg(&err.to_string())));
            }
        };
        drop(permit);

        let value: Value = serde_json::from_slice(&raw)
            .map_err(|e| StacError::InvalidJson(redact_msg(&e.to_string())))?;
        parse_search_response(&value)
    }
}

async fn read_error_detail(
    resp: &mut reqwest::Response,
    status: reqwest::StatusCode,
    cancel: &CancelToken,
    redact_msg: &impl Fn(&str) -> String,
) -> Result<String, StacError> {
    let reason = status
        .canonical_reason()
        .unwrap_or("request failed")
        .to_owned();
    let snippet = match read_body_capped(
        resp,
        MAX_STAC_ERROR_SNIPPET_BYTES,
        MAX_STAC_ERROR_SNIPPET_BYTES,
        cancel,
    )
    .await
    {
        Ok(bytes) => {
            let text = String::from_utf8_lossy(&bytes);
            let trimmed = text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(redact_msg(trimmed))
            }
        }
        Err(BodyReadError::Cancelled) => return Err(StacError::Cancelled),
        Err(_) => None,
    };
    Ok(match snippet {
        Some(s) => format!("{reason} ({s})"),
        None => reason,
    })
}

fn append_search_path(endpoint: &str) -> String {
    let (base, query, frag) = split_url_query_fragment(endpoint);
    let trimmed_base = base.trim_end_matches('/');
    let mut out = if trimmed_base.ends_with("/search") {
        trimmed_base.to_owned()
    } else {
        format!("{trimmed_base}/search")
    };
    if let Some(q) = query {
        out.push('?');
        out.push_str(q);
    }
    if let Some(f) = frag {
        out.push('#');
        out.push_str(f);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::types::StacAuth;
    use super::*;

    #[test]
    fn append_search_path_and_sas_preserve_query_and_fragment() {
        assert_eq!(
            append_search_path("https://example.org/stac/v1/?api=1#top"),
            "https://example.org/stac/v1/search?api=1#top"
        );
        assert_eq!(
            append_search_path("https://example.org/stac/v1/search?api=1"),
            "https://example.org/stac/v1/search?api=1"
        );

        let sas = StacAuth::SasQuery("?sv=2024&sig=abc".to_owned());
        assert_eq!(
            sas.apply_to_url("https://blob.example/a.tif#band1"),
            "https://blob.example/a.tif?sv=2024&sig=abc#band1"
        );
        assert_eq!(
            sas.apply_to_url("https://blob.example/a.tif?x=1#band1"),
            "https://blob.example/a.tif?x=1&sv=2024&sig=abc#band1"
        );
    }
}
