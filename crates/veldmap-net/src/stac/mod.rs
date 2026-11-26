//! Zero-auth and custom-header STAC API client (`StacClient`), Sentinel-2 L2A
//! & COP-DEM GLO-30 asset/band metadata parser (`raster:bands` `scale`/`offset`),
//! and built-in zero-auth onboarding sample scenes.

mod parse;
mod samples;

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use reqwest::header::{ACCEPT, CONTENT_TYPE};
use serde_json::Value;
use thiserror::Error;
use veldmap_core::geo::GeoJsonGeometry;
use veldmap_core::sync::CancelToken;

use crate::context::NetContext;
use crate::http::{
    BodyReadError, HostLimiters, link_cancel_token, parse_custom_headers, read_body_capped,
    record_rate_limit_cooldown,
};
use crate::redact::{
    EncodedSecret, redact_secret_label, redact_text_many, sanitize_url_for_display,
};
use crate::url::{
    extract_host, is_valid_http_endpoint, split_url_query_fragment, strip_http_scheme,
};

use self::parse::{build_search_payload, parse_search_response};
pub use self::samples::{SampleScene, builtin_sample_scenes};

/// Element84 AWS Earth Search v1 public zero-auth STAC API endpoint.
pub const AWS_EARTH_SEARCH_V1_ENDPOINT: &str = "https://earth-search.aws.element84.com/v1";

/// Sentinel-2 L2A Cloud-Optimized GeoTIFF collection ID on AWS Earth Search.
pub const COLLECTION_SENTINEL2_L2A: &str = "sentinel-2-l2a";

/// Copernicus DEM GLO-30 Cloud-Optimized GeoTIFF collection ID on AWS Earth Search.
pub const COLLECTION_COP_DEM_GLO30: &str = "cop-dem-glo-30";

/// Hard response body size cap for STAC JSON search responses (`16 MiB`).
const MAX_STAC_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Maximum error body snippet bytes read on non-2xx STAC responses.
const MAX_STAC_ERROR_SNIPPET_BYTES: usize = 512;

/// Maximum items requested in a single STAC search call.
pub const MAX_STAC_SEARCH_LIMIT: usize = 100;

/// Additional protocol-owned headers forbidden in STAC custom auth headers.
const STAC_EXTRA_FORBIDDEN_HEADERS: &[&str] = &["content-type", "accept"];

/// Errors returned by STAC catalog search requests.
#[derive(Debug, Error)]
pub enum StacError {
    #[error("STAC request cancelled")]
    Cancelled,
    #[error("invalid STAC endpoint URL: {0}")]
    InvalidEndpoint(String),
    #[error("invalid STAC auth header '{name}': {reason}")]
    InvalidAuthHeader { name: String, reason: String },
    #[error("STAC HTTP {code}: {detail}")]
    HttpStatus { code: u16, detail: String },
    #[error("STAC network error: {0}")]
    Network(String),
    #[error("STAC response exceeded {limit} B safety cap")]
    ResponseTooLarge { limit: usize },
    #[error("invalid STAC JSON response: {0}")]
    InvalidJson(String),
}

impl StacError {
    pub const fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

/// Optional authentication credentials for custom STAC catalogs and their COG
/// assets (Bearer token, custom API-key header, or SAS query parameter).
#[derive(Clone, Default, PartialEq, Eq)]
pub enum StacAuth {
    #[default]
    None,
    Bearer(String),
    ApiKey {
        header_name: String,
        value: String,
    },
    SasQuery(String),
}

impl fmt::Debug for StacAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("StacAuth::None"),
            Self::Bearer(tok) => f
                .debug_tuple("StacAuth::Bearer")
                .field(&redact_secret_label(Some(tok.as_str()), "<empty>"))
                .finish(),
            Self::ApiKey { header_name, value } => f
                .debug_struct("StacAuth::ApiKey")
                .field("header_name", header_name)
                .field(
                    "value",
                    &redact_secret_label(Some(value.as_str()), "<empty>"),
                )
                .finish(),
            Self::SasQuery(sas) => f
                .debug_tuple("StacAuth::SasQuery")
                .field(&redact_secret_label(Some(sas.as_str()), "<empty>"))
                .finish(),
        }
    }
}

impl StacAuth {
    /// Convert header-based auth into `(header_name, header_value)` pairs for
    /// [`crate::HttpRangeConfig::extra_headers`].
    pub fn extra_headers(&self) -> Vec<(String, String)> {
        match self {
            Self::Bearer(tok) if !tok.trim().is_empty() => {
                vec![("Authorization".to_owned(), format!("Bearer {}", tok.trim()))]
            }
            Self::ApiKey { header_name, value }
                if !header_name.trim().is_empty() && !value.trim().is_empty() =>
            {
                vec![(header_name.trim().to_owned(), value.trim().to_owned())]
            }
            _ => Vec::new(),
        }
    }

    /// Append a configured SAS query string to `url` when using [`Self::SasQuery`],
    /// preserving any existing `?query` and `#fragment`.
    pub fn apply_to_url(&self, url: &str) -> String {
        let Self::SasQuery(sas) = self else {
            return url.to_owned();
        };
        let token = sas.trim().trim_start_matches('?').trim_start_matches('&');
        if token.is_empty() {
            return url.to_owned();
        }
        let (base, query, frag) = split_url_query_fragment(url);
        let mut out = match query {
            Some(q) if !q.is_empty() => format!("{base}?{q}&{token}"),
            _ => format!("{base}?{token}"),
        };
        if let Some(f) = frag {
            out.push('#');
            out.push_str(f);
        }
        out
    }

    fn encoded_secret(&self) -> Option<EncodedSecret> {
        let raw = match self {
            Self::None => return None,
            Self::Bearer(v) | Self::ApiKey { value: v, .. } | Self::SasQuery(v) => v.trim(),
        };
        (!raw.is_empty()).then(|| EncodedSecret::new(raw.to_owned()))
    }
}

/// Target STAC API endpoint, collection filter, and optional authentication.
#[derive(Clone, PartialEq, Eq)]
pub struct StacCatalogConfig {
    pub endpoint: String,
    pub collection: String,
    pub auth: StacAuth,
}

impl fmt::Debug for StacCatalogConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StacCatalogConfig")
            .field("endpoint", &sanitize_url_for_display(&self.endpoint))
            .field("collection", &self.collection)
            .field("auth", &self.auth)
            .finish()
    }
}

impl StacCatalogConfig {
    /// AWS Earth Search v1 Sentinel-2 L2A Cloud-Optimized GeoTIFF catalog.
    pub fn aws_sentinel2_l2a() -> Self {
        Self {
            endpoint: AWS_EARTH_SEARCH_V1_ENDPOINT.to_owned(),
            collection: COLLECTION_SENTINEL2_L2A.to_owned(),
            auth: StacAuth::None,
        }
    }

    /// AWS Earth Search v1 Copernicus DEM GLO-30 COG catalog.
    pub fn aws_cop_dem_glo30() -> Self {
        Self {
            endpoint: AWS_EARTH_SEARCH_V1_ENDPOINT.to_owned(),
            collection: COLLECTION_COP_DEM_GLO30.to_owned(),
            auth: StacAuth::None,
        }
    }
}

/// Parameters for a STAC `/search` POST query.
#[derive(Clone, Debug)]
pub struct StacSearchQuery {
    /// `[min_lon, min_lat, max_lon, max_lat]` in WGS84 degrees.
    pub bbox: Option<[f64; 4]>,
    /// Optional GeoJSON geometry for exact polygon intersection filtering.
    pub intersects: Option<GeoJsonGeometry>,
    /// Optional RFC 3339 interval string (`"start/end"`, e.g. `"2024-06-01T00:00:00Z/2024-09-01T23:59:59Z"`).
    pub datetime: Option<String>,
    /// Optional upper bound on `eo:cloud_cover` percentage (`0.0..=100.0`).
    /// Callers querying optical collections (e.g. Sentinel-2 L2A) should set
    /// this explicitly; `None` omits the `eo:cloud_cover` query filter.
    pub max_cloud_cover: Option<f32>,
    /// Maximum items to return (`1..=100`).
    pub limit: usize,
}

impl Default for StacSearchQuery {
    fn default() -> Self {
        Self {
            bbox: None,
            intersects: None,
            datetime: None,
            max_cloud_cover: None,
            limit: 20,
        }
    }
}

/// Radiometric scaling and nodata metadata parsed from a STAC asset's
/// `raster:bands` extension array.
#[derive(Clone, Debug, PartialEq)]
pub struct StacRasterBandMeta {
    pub scale: Option<f64>,
    pub offset: Option<f64>,
    pub nodata: Option<f64>,
    pub data_type: Option<String>,
    pub spatial_resolution: Option<f64>,
    pub unit: Option<String>,
}

/// Single asset (e.g. Sentinel-2 `visual`, `red`/`B04`, `nir`/`B08`, `scl`, or
/// DEM `data`) attached to a [`StacItem`].
#[derive(Clone, PartialEq)]
pub struct StacAsset {
    pub key: String,
    pub title: Option<String>,
    pub href: String,
    pub media_type: Option<String>,
    pub roles: Vec<String>,
    pub eo_bands: Vec<String>,
    pub gsd: Option<f64>,
    pub raster_bands: Vec<StacRasterBandMeta>,
}

impl fmt::Debug for StacAsset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StacAsset")
            .field("key", &self.key)
            .field("title", &self.title)
            .field("href", &sanitize_url_for_display(&self.href))
            .field("media_type", &self.media_type)
            .field("roles", &self.roles)
            .field("eo_bands", &self.eo_bands)
            .field("gsd", &self.gsd)
            .field("raster_bands", &self.raster_bands)
            .finish()
    }
}

impl StacAsset {
    /// Return whether this asset points to an HTTP(S) Cloud-Optimized GeoTIFF
    /// or GeoTIFF raster.
    pub fn is_geotiff_http(&self) -> bool {
        if strip_http_scheme(&self.href).is_none() {
            return false;
        }
        if let Some(mt) = &self.media_type {
            let lower = mt.to_ascii_lowercase();
            if lower.contains("geotiff") || lower.contains("tiff") {
                return true;
            }
        }
        let path_part = self.href.split(['?', '#']).next().unwrap_or(&self.href);
        let lower = path_part.to_ascii_lowercase();
        lower.ends_with(".tif") || lower.ends_with(".tiff")
    }

    /// Primary `(scale, offset)` metadata from `raster:bands[0]`, if declared.
    pub fn primary_scale_offset(&self) -> Option<(f64, f64)> {
        let b = self.raster_bands.first()?;
        if b.scale.is_none() && b.offset.is_none() {
            return None;
        }
        Some((b.scale.unwrap_or(1.0), b.offset.unwrap_or(0.0)))
    }
}

/// Single parsed STAC Item returned from `/search`.
#[derive(Clone, Debug, PartialEq)]
pub struct StacItem {
    pub id: String,
    pub collection: Option<String>,
    pub datetime: Option<String>,
    pub cloud_cover: Option<f32>,
    pub platform: Option<String>,
    pub mgrs_tile: Option<String>,
    pub processing_level: Option<String>,
    pub bbox: Option<[f64; 4]>,
    pub geometry: Option<GeoJsonGeometry>,
    pub assets: Vec<StacAsset>,
}

impl StacItem {
    /// Return the best visual/preview or primary data GeoTIFF asset for
    /// one-click streaming.
    pub fn preferred_cog_asset(&self) -> Option<&StacAsset> {
        const PREFERRED_KEYS: &[&str] = &["visual", "tci", "data", "red", "B04", "nir", "B08"];
        for key in PREFERRED_KEYS {
            if let Some(asset) = self
                .assets
                .iter()
                .find(|a| a.key.eq_ignore_ascii_case(key) && a.is_geotiff_http())
            {
                return Some(asset);
            }
        }
        self.assets.iter().find(|a| a.is_geotiff_http())
    }

    /// Return all HTTP(S) GeoTIFF assets on this item ordered for analyst
    /// selection (visual/TCI first, then spectral bands, SCL, and DEM data).
    pub fn cog_assets(&self) -> Vec<&StacAsset> {
        self.assets.iter().filter(|a| a.is_geotiff_http()).collect()
    }
}

/// Parsed STAC `/search` response.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StacSearchResult {
    pub items: Vec<StacItem>,
    pub matched: Option<u64>,
}

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
        if let Some(s) = catalog.auth.encoded_secret() {
            secrets.push(s);
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
            let reason = status
                .canonical_reason()
                .unwrap_or("request failed")
                .to_owned();
            let snippet = match read_body_capped(
                &mut resp,
                MAX_STAC_ERROR_SNIPPET_BYTES,
                MAX_STAC_ERROR_SNIPPET_BYTES,
                &linked_cancel,
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
            let detail = match snippet {
                Some(s) => format!("{reason} ({s})"),
                None => reason,
            };
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
