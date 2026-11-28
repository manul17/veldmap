//! Asynchronous STAC `/search` HTTP client sharing [`NetContext`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use reqwest::Method;
use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderName, HeaderValue};
use serde_json::Value;
use tokio::sync::Notify;
use veldmap_core::sync::{CancelToken, lock_unpoisoned};

use crate::context::NetContext;
use crate::http::{BodyReadError, link_cancel_token, parse_custom_headers, read_body_capped};
use crate::limiter::{HostLimiters, record_rate_limit_cooldown};
use crate::redact::{EncodedSecret, redact_text_many, sanitize_url_for_display};
use crate::url::{
    extract_host, is_valid_http_endpoint, same_http_origin, split_url_query_fragment,
};

use super::parse::{build_search_payload, parse_search_response, parse_single_item_response};
use super::types::{
    StacAuth, StacCatalogConfig, StacError, StacItem, StacNextLink, StacNextMethod,
    StacSearchQuery, StacSearchResult,
};

/// Hard response body size cap for STAC JSON search responses (`16 MiB`).
const MAX_STAC_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Maximum error body snippet bytes read on non-2xx STAC responses.
const MAX_STAC_ERROR_SNIPPET_BYTES: usize = 512;

struct PreparedStacRequest {
    url: String,
    host: String,
    custom_headers: Vec<(HeaderName, HeaderValue)>,
    secrets: Vec<EncodedSecret>,
}

impl PreparedStacRequest {
    fn redact(&self, msg: &str) -> String {
        redact_text_many(msg, &self.secrets)
    }
}

fn is_https_downgrade(base_url: &str, target_url: &str) -> bool {
    let base_is_https = base_url
        .trim()
        .get(..8)
        .is_some_and(|p| p.eq_ignore_ascii_case("https://"));
    let target_is_http = target_url
        .trim()
        .get(..7)
        .is_some_and(|p| p.eq_ignore_ascii_case("http://"));
    base_is_https && target_is_http
}

fn prepare_request_auth(
    target_url: &str,
    auth: &StacAuth,
) -> Result<PreparedStacRequest, StacError> {
    let trimmed = target_url.trim();
    if !is_valid_http_endpoint(trimmed) {
        return Err(StacError::InvalidEndpoint(sanitize_url_for_display(
            trimmed,
        )));
    }
    let url = auth.apply_to_url(trimmed);
    let host = extract_host(&url);

    let raw_headers = auth.extra_headers();
    let parsed = parse_custom_headers(&raw_headers)
        .map_err(|(name, reason)| StacError::InvalidAuthHeader { name, reason })?;
    let custom_headers = parsed.headers;
    let mut secrets = parsed.secrets;
    if let Some(secret) = auth.redaction_secret()
        && !secrets.contains(&secret)
    {
        secrets.push(secret);
    }

    Ok(PreparedStacRequest {
        url,
        host,
        custom_headers,
        secrets,
    })
}

fn prepare_search_request(
    catalog: &StacCatalogConfig,
    query: &StacSearchQuery,
) -> Result<(PreparedStacRequest, Vec<u8>), StacError> {
    let endpoint = catalog.endpoint.trim();
    if !is_valid_http_endpoint(endpoint) {
        return Err(StacError::InvalidEndpoint(sanitize_url_for_display(
            endpoint,
        )));
    }
    let search_base = append_search_path(endpoint);
    let prepared = prepare_request_auth(&search_base, &catalog.auth)?;
    let body_json = build_search_payload(&catalog.collection, query);
    let body_bytes =
        serde_json::to_vec(&body_json).map_err(|e| StacError::InvalidJson(e.to_string()))?;
    Ok((prepared, body_bytes))
}

fn prepare_next_request(
    catalog: &StacCatalogConfig,
    prev_query: &StacSearchQuery,
    next_link: &StacNextLink,
) -> Result<(PreparedStacRequest, Method, Option<Vec<u8>>), StacError> {
    let endpoint = catalog.endpoint.trim();
    if !is_valid_http_endpoint(endpoint) {
        return Err(StacError::InvalidEndpoint(sanitize_url_for_display(
            endpoint,
        )));
    }
    let target = next_link.href.trim();
    if !is_valid_http_endpoint(target) {
        return Err(StacError::LinkRejected(format!(
            "invalid next-page URL '{}'",
            sanitize_url_for_display(target)
        )));
    }
    if is_https_downgrade(endpoint, target) {
        return Err(StacError::LinkRejected(
            "refusing HTTPS-to-HTTP downgrade on STAC next link".to_owned(),
        ));
    }
    let has_auth = !matches!(catalog.auth, StacAuth::None);
    if has_auth && !same_http_origin(endpoint, target) {
        return Err(StacError::LinkRejected(
            "refusing to send STAC credentials to off-origin next link".to_owned(),
        ));
    }

    let prepared = prepare_request_auth(target, &catalog.auth)?;
    match next_link.method {
        StacNextMethod::Get => Ok((prepared, Method::GET, None)),
        StacNextMethod::Post => {
            let body_val = if next_link.merge {
                let mut base = build_search_payload(&catalog.collection, prev_query);
                if let Some(raw_body) = &next_link.body {
                    let patch: Value = serde_json::from_str(raw_body)
                        .map_err(|e| StacError::InvalidJson(prepared.redact(&e.to_string())))?;
                    if let (Some(base_map), Some(patch_map)) =
                        (base.as_object_mut(), patch.as_object())
                    {
                        for (k, v) in patch_map {
                            base_map.insert(k.clone(), v.clone());
                        }
                    }
                }
                base
            } else if let Some(raw_body) = &next_link.body {
                serde_json::from_str(raw_body)
                    .map_err(|e| StacError::InvalidJson(prepared.redact(&e.to_string())))?
            } else {
                Value::Object(serde_json::Map::new())
            };
            let bytes =
                serde_json::to_vec(&body_val).map_err(|e| StacError::InvalidJson(e.to_string()))?;
            Ok((prepared, Method::POST, Some(bytes)))
        }
    }
}

/// Asynchronous STAC API search & item client sharing [`NetContext`]'s
/// gzip-enabled HTTP clients and per-host concurrency/cooldown limiters.
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

    /// Execute an HTTP GET/POST and read the capped response body while owning
    /// the per-host semaphore permit strictly inside this function's lexical
    /// scope so the permit drops before JSON parsing begins.
    async fn execute_stac_http(
        &self,
        prepared: &PreparedStacRequest,
        method: Method,
        body_bytes: Option<Vec<u8>>,
        cancel: &CancelToken,
    ) -> Result<Vec<u8>, StacError> {
        let _permit = self
            .host_limiters
            .acquire_permit(&prepared.host, cancel)
            .await
            .map_err(|_| StacError::Cancelled)?;

        let client = if prepared.custom_headers.is_empty() {
            &self.api_client
        } else {
            &self.api_auth_client
        };
        let mut req = client
            .request(method, &prepared.url)
            .header(ACCEPT, "application/geo+json, application/json");
        if let Some(body) = body_bytes {
            req = req.header(CONTENT_TYPE, "application/json").body(body);
        }
        for (name, val) in &prepared.custom_headers {
            req = req.header(name, val);
        }

        let resp_res = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(StacError::Cancelled),
            res = req.send() => res,
        };
        let mut resp = match resp_res {
            Ok(resp) => resp,
            Err(err) => {
                return Err(StacError::Network(prepared.redact(&err.to_string())));
            }
        };

        let status = resp.status();
        let code = status.as_u16();
        if matches!(code, 429 | 503) {
            record_rate_limit_cooldown(
                &self.host_limiters,
                &prepared.host,
                resp.headers(),
                Duration::from_secs(5),
            );
        }
        if !status.is_success() {
            let detail =
                read_error_detail(&mut resp, status, cancel, |msg| prepared.redact(msg)).await?;
            return Err(StacError::HttpStatus { code, detail });
        }

        match read_body_capped(&mut resp, MAX_STAC_RESPONSE_BYTES, 16 * 1024, cancel).await {
            Ok(bytes) => Ok(bytes),
            Err(BodyReadError::Cancelled) => Err(StacError::Cancelled),
            Err(BodyReadError::TooLarge) => Err(StacError::ResponseTooLarge {
                limit: MAX_STAC_RESPONSE_BYTES,
            }),
            Err(BodyReadError::Network(err)) => {
                Err(StacError::Network(prepared.redact(&err.to_string())))
            }
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

        let (prepared, body_bytes) = prepare_search_request(catalog, query)?;
        let raw = self
            .execute_stac_http(&prepared, Method::POST, Some(body_bytes), &linked_cancel)
            .await?;

        let value: Value = serde_json::from_slice(&raw)
            .map_err(|e| StacError::InvalidJson(prepared.redact(&e.to_string())))?;
        parse_search_response(&value, &prepared.url)
    }

    /// Follow a STAC `links[rel="next"]` pagination link (supporting both
    /// `GET` cursor URLs and `POST` `body` + `merge: true`, with HTTPS
    /// no-downgrade and same-origin auth guards).
    pub async fn search_next_async(
        &self,
        catalog: &StacCatalogConfig,
        prev_query: &StacSearchQuery,
        next_link: &StacNextLink,
        cancel: &CancelToken,
    ) -> Result<StacSearchResult, StacError> {
        let (linked_cancel, _cancel_guard) = link_cancel_token(&self.shutdown, cancel);
        if linked_cancel.is_cancelled() {
            return Err(StacError::Cancelled);
        }

        let (prepared, method, body_bytes) = prepare_next_request(catalog, prev_query, next_link)?;
        let raw = self
            .execute_stac_http(&prepared, method, body_bytes, &linked_cancel)
            .await?;

        let value: Value = serde_json::from_slice(&raw)
            .map_err(|e| StacError::InvalidJson(prepared.redact(&e.to_string())))?;
        parse_search_response(&value, &prepared.url)
    }

    /// Fetch a single STAC Item JSON document from `item_href` (used to
    /// refresh expired presigned COG asset URLs on HTTP 401/403 restore).
    ///
    /// When `auth` carries custom headers, uses `NetContext::api_auth_client()`
    /// which blocks cross-origin redirects and HTTPS downgrades.
    pub async fn fetch_item_async(
        &self,
        item_href: &str,
        auth: &StacAuth,
        cancel: &CancelToken,
    ) -> Result<StacItem, StacError> {
        let (linked_cancel, _cancel_guard) = link_cancel_token(&self.shutdown, cancel);
        if linked_cancel.is_cancelled() {
            return Err(StacError::Cancelled);
        }

        let prepared = prepare_request_auth(item_href, auth)?;
        let raw = self
            .execute_stac_http(&prepared, Method::GET, None, &linked_cancel)
            .await?;

        let value: Value = serde_json::from_slice(&raw)
            .map_err(|e| StacError::InvalidJson(prepared.redact(&e.to_string())))?;
        parse_single_item_response(&value)
    }
}

/// Bounded maximum entries retained in [`StacItemFlightCache`] per epoch.
const MAX_ITEM_FLIGHT_CACHE_ENTRIES: usize = 64;

#[derive(Clone)]
enum ItemFlightState {
    InFlight(Arc<Notify>),
    Done(Result<Arc<StacItem>, String>),
}

/// Panic- and cancel-safe RAII cleanup guard for a leader in
/// [`StacItemFlightCache`]: removes the `InFlight` slot and wakes all waiting
/// followers if the leader is cancelled or panics before resolving.
struct ItemFlightLeaderGuard {
    state: Arc<Mutex<ItemFlightCacheInner>>,
    key: (u64, String),
    notify: Arc<Notify>,
    settled: bool,
}

impl ItemFlightLeaderGuard {
    fn settle(mut self, result: Result<Arc<StacItem>, String>) {
        self.settled = true;
        {
            let mut inner = lock_unpoisoned(&self.state);
            if inner.entries.len() >= MAX_ITEM_FLIGHT_CACHE_ENTRIES {
                // Only evict completed entries; never evict active `InFlight`
                // singleflight slots with parked followers.
                inner
                    .entries
                    .retain(|_, state| matches!(state, ItemFlightState::InFlight(_)));
            }
            if matches!(
                inner.entries.get(&self.key),
                Some(ItemFlightState::InFlight(n)) if Arc::ptr_eq(n, &self.notify)
            ) {
                inner
                    .entries
                    .insert(self.key.clone(), ItemFlightState::Done(result));
            }
        }
        self.notify.notify_waiters();
    }
}

impl Drop for ItemFlightLeaderGuard {
    fn drop(&mut self) {
        if !self.settled {
            {
                let mut inner = lock_unpoisoned(&self.state);
                if matches!(
                    inner.entries.get(&self.key),
                    Some(ItemFlightState::InFlight(n)) if Arc::ptr_eq(n, &self.notify)
                ) {
                    inner.entries.remove(&self.key);
                }
            }
            self.notify.notify_waiters();
        }
    }
}

enum FlightSlotRole {
    Done(Result<Arc<StacItem>, String>),
    Follower(Arc<Notify>),
    Leader(Arc<Notify>),
}

#[derive(Default)]
struct ItemFlightCacheInner {
    last_epoch: u64,
    entries: HashMap<(u64, String), ItemFlightState>,
}

/// Async singleflight & per-epoch coalescing cache for [`StacClient::fetch_item_async`].
///
/// Deduplicates concurrent `401`/`403` `stac_item_href` refreshes across bands
/// of a restored `BandMath` recipe (or sibling layers) while respecting each
/// follower's independent [`CancelToken`] and avoiding lost wakeups or leader
/// cancellation deadlocks.
#[derive(Clone, Default)]
pub struct StacItemFlightCache {
    inner: Arc<Mutex<ItemFlightCacheInner>>,
}

impl StacItemFlightCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fetch `item_href` at most once per `(scope_epoch, item_href)`, coalescing
    /// concurrent callers onto one HTTP GET.
    pub async fn fetch_item_coalesced_async(
        &self,
        client: &StacClient,
        scope_epoch: u64,
        item_href: &str,
        auth: &StacAuth,
        cancel: &CancelToken,
    ) -> Result<Arc<StacItem>, String> {
        let key = (scope_epoch, item_href.to_owned());
        loop {
            if cancel.is_cancelled() {
                return Err(StacError::Cancelled.to_string());
            }
            let role = {
                let mut inner = lock_unpoisoned(&self.inner);
                if scope_epoch > inner.last_epoch {
                    inner.last_epoch = scope_epoch;
                    // Prune stale completed epochs while preserving active
                    // `InFlight` slots so parked followers always wake cleanly.
                    inner.entries.retain(|(ep, _), state| {
                        *ep >= scope_epoch || matches!(state, ItemFlightState::InFlight(_))
                    });
                }
                match inner.entries.get(&key).cloned() {
                    Some(ItemFlightState::Done(res)) => FlightSlotRole::Done(res),
                    Some(ItemFlightState::InFlight(notify)) => FlightSlotRole::Follower(notify),
                    None => {
                        let notify = Arc::new(Notify::new());
                        inner
                            .entries
                            .insert(key.clone(), ItemFlightState::InFlight(Arc::clone(&notify)));
                        FlightSlotRole::Leader(notify)
                    }
                }
            };

            let leader_notify = match role {
                FlightSlotRole::Done(res) => return res,
                FlightSlotRole::Follower(notify) => {
                    let notified = notify.notified();
                    tokio::pin!(notified);
                    notified.as_mut().enable();
                    // Re-check state after enabling notification so no wakeup is lost.
                    {
                        let inner = lock_unpoisoned(&self.inner);
                        match inner.entries.get(&key) {
                            Some(ItemFlightState::Done(res)) => return res.clone(),
                            None => continue,
                            Some(ItemFlightState::InFlight(current))
                                if !Arc::ptr_eq(current, &notify) =>
                            {
                                continue;
                            }
                            Some(ItemFlightState::InFlight(_)) => {}
                        }
                    }
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return Err(StacError::Cancelled.to_string()),
                        _ = notified => continue,
                    }
                }
                FlightSlotRole::Leader(notify) => notify,
            };

            // We own the singleflight slot as leader with the exact `Notify`
            // published to any followers.
            let guard = ItemFlightLeaderGuard {
                state: Arc::clone(&self.inner),
                key: key.clone(),
                notify: leader_notify,
                settled: false,
            };
            match client.fetch_item_async(item_href, auth, cancel).await {
                Ok(item) => {
                    let arc = Arc::new(item);
                    guard.settle(Ok(Arc::clone(&arc)));
                    return Ok(arc);
                }
                Err(err) if err.is_cancelled() => {
                    // Drop guard without settling Done so uncancelled followers
                    // can take over leadership cleanly.
                    drop(guard);
                    return Err(err.to_string());
                }
                Err(err) => {
                    let msg = err.to_string();
                    guard.settle(Err(msg.clone()));
                    return Err(msg);
                }
            }
        }
    }
}

async fn read_error_detail(
    resp: &mut reqwest::Response,
    status: reqwest::StatusCode,
    cancel: &CancelToken,
    redact_msg: impl Fn(&str) -> String,
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

    #[test]
    fn prepare_search_request_redacts_short_custom_api_key_and_sas_and_bearer() {
        let query = StacSearchQuery {
            bbox: Some([0.0, 0.0, 1.0, 1.0]),
            intersects: None,
            datetime: None,
            max_cloud_cover: None,
            sort_by: None,
            limit: 10,
        };

        let api_key_cfg = StacCatalogConfig {
            endpoint: "https://stac.example.org/v1".to_owned(),
            collection: "sentinel-2-l2a".to_owned(),
            auth: StacAuth::ApiKey {
                header_name: "X-Custom-Pass".to_owned(),
                value: "s3cr!".to_owned(),
            },
        };
        let (prepared_key, _) = prepare_search_request(&api_key_cfg, &query).expect("api key");
        assert!(prepared_key.secrets.iter().any(|s| s.raw() == "s3cr!"));

        let sas_cfg = StacCatalogConfig {
            auth: StacAuth::SasQuery("?sv=2024&sig=tok".to_owned()),
            ..api_key_cfg.clone()
        };
        let (prepared_sas, _) = prepare_search_request(&sas_cfg, &query).expect("sas");
        assert!(
            prepared_sas
                .secrets
                .iter()
                .any(|s| s.raw() == "sv=2024&sig=tok")
        );

        let bearer_cfg = StacCatalogConfig {
            auth: StacAuth::Bearer("abc".to_owned()),
            ..api_key_cfg
        };
        let (prepared_bearer, _) = prepare_search_request(&bearer_cfg, &query).expect("bearer");
        assert!(prepared_bearer.secrets.iter().any(|s| s.raw() == "abc"));
    }
}
