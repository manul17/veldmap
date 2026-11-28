//! STAC catalog configuration, authentication, query, error, and parsed
//! response model types.

use std::fmt;

use thiserror::Error;
use veldmap_core::geo::GeoJsonGeometry;

use crate::redact::{EncodedSecret, redact_secret_label, sanitize_url_for_display};
use crate::url::{split_url_query_fragment, strip_http_scheme};

/// Element84 AWS Earth Search v1 public zero-auth STAC API endpoint.
const AWS_EARTH_SEARCH_V1_ENDPOINT: &str = "https://earth-search.aws.element84.com/v1";

/// Sentinel-2 L2A Cloud-Optimized GeoTIFF collection ID on AWS Earth Search.
const COLLECTION_SENTINEL2_L2A: &str = "sentinel-2-l2a";

/// Copernicus DEM GLO-30 Cloud-Optimized GeoTIFF collection ID on AWS Earth Search.
const COLLECTION_COP_DEM_GLO30: &str = "cop-dem-glo-30";

/// Maximum items requested in a single STAC search call.
pub(super) const MAX_STAC_SEARCH_LIMIT: usize = 100;

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

    fn trimmed_sas_token(&self) -> Option<&str> {
        let Self::SasQuery(sas) = self else {
            return None;
        };
        let token = sas.trim().trim_start_matches('?').trim_start_matches('&');
        (!token.is_empty()).then_some(token)
    }

    /// Append a configured SAS query string to `url` when using [`Self::SasQuery`],
    /// preserving any existing `?query` and `#fragment`.
    pub fn apply_to_url(&self, url: &str) -> String {
        let Some(token) = self.trimmed_sas_token() else {
            return url.to_owned();
        };
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

    /// Return an explicit error-redaction secret for auth modes that may not be
    /// covered by generic header heuristics (`SasQuery`, which is appended to
    /// the URL rather than headers, and `ApiKey`, which may use a short value
    /// on a non-standard header name). `Bearer` is always covered by
    /// `parse_custom_headers` via the `Authorization` header.
    pub(crate) fn redaction_secret(&self) -> Option<EncodedSecret> {
        match self {
            Self::SasQuery(_) => self
                .trimmed_sas_token()
                .map(|token| EncodedSecret::new(token.to_owned())),
            Self::ApiKey { value, .. } => crate::redact::extract_explicit_header_secret(value),
            Self::None | Self::Bearer(_) => None,
        }
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
    fn is_geotiff_http(&self) -> bool {
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

    /// Return whether this asset represents a Sentinel-2 Scene Classification
    /// Layer (`SCL`) GeoTIFF.
    pub fn is_scl_asset(&self) -> bool {
        if !self.is_geotiff_http() {
            return false;
        }
        if self.key.eq_ignore_ascii_case("scl")
            || self
                .key
                .get(..4)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("scl_"))
        {
            return true;
        }
        if self.roles.iter().any(|r| r.eq_ignore_ascii_case("scl")) {
            return true;
        }
        if let Some(title) = &self.title {
            let lower = title.to_ascii_lowercase();
            if lower.contains("scene classification") || lower == "scl" {
                return true;
            }
        }
        false
    }

    fn is_visual_or_overview(&self) -> bool {
        self.key.eq_ignore_ascii_case("visual")
            || self.key.eq_ignore_ascii_case("tci")
            || self.roles.iter().any(|r| {
                r.eq_ignore_ascii_case("visual")
                    || r.eq_ignore_ascii_case("overview")
                    || r.eq_ignore_ascii_case("thumbnail")
            })
    }
}

/// Single parsed STAC Item returned from `/search`.
#[derive(Clone, Debug, PartialEq)]
pub struct StacItem {
    pub id: String,
    pub collection: Option<String>,
    /// Optional sanitized `links[rel="self"]` item URL for workspace metadata
    /// persistence.
    pub self_href: Option<String>,
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

    /// Find the highest-priority HTTP(S) GeoTIFF asset matching any alias in
    /// `aliases` (checking exact asset keys first in priority order, then
    /// single-band scientific `eo:bands` common names / band names while
    /// skipping 8-bit `visual`/`tci`/overview composites).
    pub fn find_band_asset(&self, aliases: &[&str]) -> Option<&StacAsset> {
        for alias in aliases {
            if let Some(asset) = self
                .assets
                .iter()
                .find(|a| a.is_geotiff_http() && a.key.eq_ignore_ascii_case(alias))
            {
                return Some(asset);
            }
        }
        for alias in aliases {
            if let Some(asset) = self.assets.iter().find(|a| {
                a.is_geotiff_http()
                    && !a.is_visual_or_overview()
                    && a.raster_bands.len() <= 1
                    && a.eo_bands.iter().any(|b| b.eq_ignore_ascii_case(alias))
            }) {
                return Some(asset);
            }
        }
        None
    }

    /// Return the item's Sentinel-2 Scene Classification Layer (`SCL`) GeoTIFF
    /// asset, if present.
    pub fn scl_asset(&self) -> Option<&StacAsset> {
        self.assets.iter().find(|a| a.is_scl_asset())
    }
}

/// Parsed STAC `/search` response.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StacSearchResult {
    pub items: Vec<StacItem>,
    pub matched: Option<u64>,
}
