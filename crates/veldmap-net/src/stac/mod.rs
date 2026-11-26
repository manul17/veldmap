//! Zero-auth and custom-header STAC API client (`StacClient`), Sentinel-2 L2A
//! & COP-DEM GLO-30 asset/band metadata parser (`raster:bands` `scale`/`offset`),
//! and built-in zero-auth onboarding sample scenes.

mod samples;

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderName, HeaderValue, RETRY_AFTER};
use serde_json::{Map, Value};
use thiserror::Error;
use veldmap_core::geo::{GeoJsonGeometry, export_geojson_geometry, parse_geojson};
use veldmap_core::sync::CancelToken;

use crate::context::NetContext;
use crate::http::parse_retry_after_seconds;
use crate::redact::{EncodedSecret, redact_secret_label, redact_text, sanitize_url_for_display};
use crate::url::{extract_host, parse_http_authority, strip_http_scheme};

pub use self::samples::{SampleScene, builtin_sample_scenes};

/// Element84 AWS Earth Search v1 public zero-auth STAC API endpoint.
pub const AWS_EARTH_SEARCH_V1_ENDPOINT: &str = "https://earth-search.aws.element84.com/v1";

/// Sentinel-2 L2A Cloud-Optimized GeoTIFF collection ID on AWS Earth Search.
pub const COLLECTION_SENTINEL2_L2A: &str = "sentinel-2-l2a";

/// Copernicus DEM GLO-30 Cloud-Optimized GeoTIFF collection ID on AWS Earth Search.
pub const COLLECTION_COP_DEM_GLO30: &str = "cop-dem-glo-30";

/// Hard response body size cap for STAC JSON search responses (`16 MiB`).
const MAX_STAC_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Maximum items requested in a single STAC search call.
pub const MAX_STAC_SEARCH_LIMIT: usize = 100;

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

    /// Append a configured SAS query string to `url` when using [`Self::SasQuery`].
    pub fn apply_to_url(&self, url: &str) -> String {
        let Self::SasQuery(sas) = self else {
            return url.to_owned();
        };
        let token = sas.trim().trim_start_matches('?').trim_start_matches('&');
        if token.is_empty() {
            return url.to_owned();
        }
        if url.contains('?') {
            format!("{url}&{token}")
        } else {
            format!("{url}?{token}")
        }
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
            max_cloud_cover: Some(25.0),
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
/// HTTP client and per-host concurrency/cooldown limiters.
#[derive(Clone)]
pub struct StacClient {
    net: Arc<NetContext>,
}

impl StacClient {
    pub fn new(net: Arc<NetContext>) -> Self {
        Self { net }
    }

    /// Execute a STAC `/search` query asynchronously against `catalog`.
    pub async fn search_async(
        &self,
        catalog: &StacCatalogConfig,
        query: &StacSearchQuery,
        cancel: &CancelToken,
    ) -> Result<StacSearchResult, StacError> {
        if cancel.is_cancelled() {
            return Err(StacError::Cancelled);
        }

        let endpoint = catalog.endpoint.trim().trim_end_matches('/');
        if strip_http_scheme(endpoint).is_none() || parse_http_authority(endpoint).is_none() {
            return Err(StacError::InvalidEndpoint(sanitize_url_for_display(
                endpoint,
            )));
        }
        let search_base = if endpoint.ends_with("/search") {
            endpoint.to_owned()
        } else {
            format!("{endpoint}/search")
        };
        let search_url = catalog.auth.apply_to_url(&search_base);
        let host = extract_host(&search_url);
        let secret = catalog.auth.encoded_secret();
        let redact_msg = |msg: &str| redact_text(msg, secret.as_ref());

        let body_json = build_search_payload(&catalog.collection, query);
        let body_bytes =
            serde_json::to_vec(&body_json).map_err(|e| StacError::InvalidJson(e.to_string()))?;

        let mut custom_headers = Vec::new();
        match &catalog.auth {
            StacAuth::None | StacAuth::SasQuery(_) => {}
            StacAuth::Bearer(tok) if !tok.trim().is_empty() => {
                let val =
                    HeaderValue::from_str(&format!("Bearer {}", tok.trim())).map_err(|_| {
                        StacError::InvalidAuthHeader {
                            name: "Authorization".to_owned(),
                            reason: "invalid bearer token header value".to_owned(),
                        }
                    })?;
                custom_headers.push((AUTHORIZATION, val));
            }
            StacAuth::ApiKey { header_name, value }
                if !header_name.trim().is_empty() && !value.trim().is_empty() =>
            {
                let name = HeaderName::from_bytes(header_name.trim().as_bytes()).map_err(|e| {
                    StacError::InvalidAuthHeader {
                        name: header_name.clone(),
                        reason: e.to_string(),
                    }
                })?;
                let val = HeaderValue::from_str(value.trim()).map_err(|_| {
                    StacError::InvalidAuthHeader {
                        name: header_name.clone(),
                        reason: "invalid API key header value".to_owned(),
                    }
                })?;
                custom_headers.push((name, val));
            }
            _ => {}
        }

        let permit = self
            .net
            .host_limiters()
            .acquire_permit(&host, cancel)
            .await
            .map_err(|_| StacError::Cancelled)?;

        let mut req = self
            .net
            .api_client()
            .post(&search_url)
            .header(ACCEPT, "application/geo+json, application/json")
            .header(CONTENT_TYPE, "application/json")
            .body(body_bytes);
        for (name, val) in custom_headers {
            req = req.header(name, val);
        }

        let resp_res = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                drop(permit);
                return Err(StacError::Cancelled);
            }
            res = req.send() => res,
        };
        let mut resp = match resp_res {
            Ok(resp) => resp,
            Err(err) => {
                drop(permit);
                return Err(StacError::Network(redact_msg(&err.to_string())));
            }
        };

        let status = resp.status();
        let code = status.as_u16();
        if matches!(code, 429 | 503) {
            let retry_after = resp
                .headers()
                .get(RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| parse_retry_after_seconds(Some(s)))
                .unwrap_or(Duration::from_secs(5));
            self.net
                .host_limiters()
                .record_cooldown(&host, Instant::now() + retry_after);
        }
        if !status.is_success() {
            drop(permit);
            return Err(StacError::HttpStatus {
                code,
                detail: status
                    .canonical_reason()
                    .unwrap_or("request failed")
                    .to_owned(),
            });
        }

        let mut raw = Vec::new();
        loop {
            let chunk_res = tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    drop(permit);
                    return Err(StacError::Cancelled);
                }
                chunk = resp.chunk() => chunk,
            };
            match chunk_res {
                Ok(Some(chunk)) => {
                    if raw.len().saturating_add(chunk.len()) > MAX_STAC_RESPONSE_BYTES {
                        drop(permit);
                        return Err(StacError::ResponseTooLarge {
                            limit: MAX_STAC_RESPONSE_BYTES,
                        });
                    }
                    raw.extend_from_slice(&chunk);
                }
                Ok(None) => break,
                Err(err) => {
                    drop(permit);
                    return Err(StacError::Network(redact_msg(&err.to_string())));
                }
            }
        }
        drop(permit);

        let value: Value = serde_json::from_slice(&raw)
            .map_err(|e| StacError::InvalidJson(redact_msg(&e.to_string())))?;
        parse_search_response(&value)
    }
}

fn build_search_payload(collection: &str, query: &StacSearchQuery) -> Value {
    let mut obj = Map::new();
    let col = collection.trim();
    if !col.is_empty() {
        obj.insert(
            "collections".to_owned(),
            Value::Array(vec![Value::String(col.to_owned())]),
        );
    }
    let limit = query.limit.clamp(1, MAX_STAC_SEARCH_LIMIT);
    obj.insert("limit".to_owned(), Value::from(limit as u64));

    if let Some(geom) = &query.intersects {
        let geom_str = export_geojson_geometry(geom);
        if let Ok(geom_val) = serde_json::from_str::<Value>(&geom_str) {
            obj.insert("intersects".to_owned(), geom_val);
        }
    } else if let Some([min_lon, min_lat, max_lon, max_lat]) = query.bbox {
        obj.insert(
            "bbox".to_owned(),
            serde_json::json!([min_lon, min_lat, max_lon, max_lat]),
        );
    }

    if let Some(dt) = &query.datetime {
        let trimmed = dt.trim();
        if !trimmed.is_empty() {
            obj.insert("datetime".to_owned(), Value::String(trimmed.to_owned()));
        }
    }

    if let Some(max_cc) = query.max_cloud_cover
        && max_cc.is_finite()
        && col != COLLECTION_COP_DEM_GLO30
    {
        let clamped = f64::from(max_cc.clamp(0.0, 100.0));
        obj.insert(
            "query".to_owned(),
            serde_json::json!({
                "eo:cloud_cover": { "lte": clamped }
            }),
        );
    }

    Value::Object(obj)
}

fn parse_search_response(val: &Value) -> Result<StacSearchResult, StacError> {
    let obj = val.as_object().ok_or_else(|| {
        StacError::InvalidJson("STAC response root is not a JSON object".to_owned())
    })?;
    let matched = obj
        .get("numberMatched")
        .and_then(Value::as_u64)
        .or_else(|| {
            obj.get("context")
                .and_then(|c| c.get("matched"))
                .and_then(Value::as_u64)
        });
    let features = obj
        .get("features")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            StacError::InvalidJson("STAC response missing 'features' array".to_owned())
        })?;

    let mut items = Vec::with_capacity(features.len());
    for feat in features {
        if let Some(item) = parse_stac_item(feat) {
            items.push(item);
        }
    }
    Ok(StacSearchResult { items, matched })
}

fn parse_stac_item(val: &Value) -> Option<StacItem> {
    let obj = val.as_object()?;
    let id = obj.get("id")?.as_str()?.to_owned();
    let collection = obj
        .get("collection")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let props = obj.get("properties").and_then(Value::as_object);

    let datetime = props
        .and_then(|p| p.get("datetime"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let cloud_cover = props
        .and_then(|p| p.get("eo:cloud_cover"))
        .and_then(Value::as_f64)
        .map(|v| v as f32);
    let platform = props
        .and_then(|p| p.get("platform"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let mgrs_tile = props.and_then(|p| {
        p.get("grid:code")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                let zone = p.get("mgrs:utm_zone").and_then(Value::as_u64)?;
                let lat_band = p.get("mgrs:latitude_band").and_then(Value::as_str)?;
                let sq = p.get("mgrs:grid_square").and_then(Value::as_str)?;
                Some(format!("{zone:02}{lat_band}{sq}"))
            })
            .or_else(|| {
                p.get("s2:mgrs_tile")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
    });
    let processing_level = props
        .and_then(|p| {
            p.get("s2:product_type")
                .or_else(|| p.get("processing:level"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .or_else(|| {
            collection.as_deref().and_then(|c| {
                if c.contains("l2a") {
                    Some("L2A (BOA)".to_owned())
                } else if c.contains("l1c") {
                    Some("L1C (TOA)".to_owned())
                } else {
                    None
                }
            })
        });

    let bbox = obj.get("bbox").and_then(Value::as_array).and_then(|arr| {
        if arr.len() >= 4 {
            let a = arr[0].as_f64()?;
            let b = arr[1].as_f64()?;
            let c = arr[2].as_f64()?;
            let d = arr[3].as_f64()?;
            if a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite() {
                Some([a, b, c, d])
            } else {
                None
            }
        } else {
            None
        }
    });

    let geometry = obj.get("geometry").and_then(|g| {
        if g.is_null() {
            return None;
        }
        let wrapped = serde_json::json!({
            "type": "Feature",
            "properties": {},
            "geometry": g,
        });
        let text = serde_json::to_string(&wrapped).ok()?;
        let fc = parse_geojson(&text).ok()?;
        fc.features.into_iter().next().map(|f| f.geometry)
    });

    let mut assets = Vec::new();
    if let Some(asset_map) = obj.get("assets").and_then(Value::as_object) {
        for (key, a_val) in asset_map {
            if let Some(asset) = parse_stac_asset(key, a_val) {
                assets.push(asset);
            }
        }
    }
    assets.sort_by_key(|a| (asset_sort_rank(&a.key), a.key.clone()));

    Some(StacItem {
        id,
        collection,
        datetime,
        cloud_cover,
        platform,
        mgrs_tile,
        processing_level,
        bbox,
        geometry,
        assets,
    })
}

fn asset_sort_rank(key: &str) -> u8 {
    match key.to_ascii_lowercase().as_str() {
        "visual" | "tci" => 0,
        "data" => 1,
        "red" | "b04" => 2,
        "green" | "b03" => 3,
        "blue" | "b02" => 4,
        "nir" | "b08" => 5,
        "nir08" | "b8a" => 6,
        "swir16" | "b11" => 7,
        "swir22" | "b12" => 8,
        "scl" => 9,
        _ => 20,
    }
}

fn parse_stac_asset(key: &str, val: &Value) -> Option<StacAsset> {
    let obj = val.as_object()?;
    let href = obj.get("href")?.as_str()?.to_owned();
    let title = obj.get("title").and_then(Value::as_str).map(str::to_owned);
    let media_type = obj.get("type").and_then(Value::as_str).map(str::to_owned);
    let roles = obj
        .get("roles")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    let eo_bands = obj
        .get("eo:bands")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|b| {
                    b.get("common_name")
                        .or_else(|| b.get("name"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .collect()
        })
        .unwrap_or_default();
    let gsd = obj.get("gsd").and_then(Value::as_f64);
    let raster_bands = obj
        .get("raster:bands")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(parse_raster_band_meta).collect())
        .unwrap_or_default();

    Some(StacAsset {
        key: key.to_owned(),
        title,
        href,
        media_type,
        roles,
        eo_bands,
        gsd,
        raster_bands,
    })
}

fn parse_raster_band_meta(val: &Value) -> Option<StacRasterBandMeta> {
    let obj = val.as_object()?;
    let nodata = obj.get("nodata").and_then(|v| {
        v.as_f64().or_else(|| {
            v.as_str()
                .and_then(|s| match s.trim().to_ascii_lowercase().as_str() {
                    "nan" => Some(f64::NAN),
                    other => other.parse::<f64>().ok(),
                })
        })
    });
    Some(StacRasterBandMeta {
        scale: obj.get("scale").and_then(Value::as_f64),
        offset: obj.get("offset").and_then(Value::as_f64),
        nodata,
        data_type: obj
            .get("data_type")
            .and_then(Value::as_str)
            .map(str::to_owned),
        spatial_resolution: obj.get("spatial_resolution").and_then(Value::as_f64),
        unit: obj.get("unit").and_then(Value::as_str).map(str::to_owned),
    })
}
