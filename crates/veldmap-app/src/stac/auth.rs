//! Origin-aware authentication resolution for remote Cloud-Optimized GeoTIFFs
//! opened from STAC search results, built-in sample scenes, or direct URLs.

use veldmap_net::{StacAuth, same_http_origin};

use crate::storage::settings::StacSettings;

/// Authentication policy for opening a remote COG from the Cloud / STAC panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StacCogAuthMode {
    /// Zero-auth public scene (built-in sample scenes).
    None,
    /// Inherit auth mode & catalog origin from the STAC search that produced
    /// the displayed result list (decoupled from the live preset dropdown).
    FromSearchResult,
    /// Explicit user opt-in on a manually entered Direct COG URL.
    DirectCustom,
}

/// Return `true` if `url` already carries a cloud presigned-URL signature query
/// parameter (AWS S3, Azure SAS, GCS) so we never double-sign or attach
/// conflicting auth headers.
fn url_has_presigned_query(url: &str) -> bool {
    let before_frag = url.split('#').next().unwrap_or(url);
    let Some((_, query)) = before_frag.split_once('?') else {
        return false;
    };
    query.split('&').any(|pair| {
        let key = pair.split('=').next().unwrap_or("").trim();
        key.eq_ignore_ascii_case("sig")
            || key.eq_ignore_ascii_case("x-amz-signature")
            || key.eq_ignore_ascii_case("signature")
            || key.eq_ignore_ascii_case("x-goog-signature")
    })
}

/// Resolve the final URL and custom HTTP headers for opening a remote COG,
/// preventing accidental credential leaks onto public or cross-origin hosts and
/// avoiding dual-auth errors on already-presigned URLs.
pub(super) fn resolve_cog_auth(
    url: &str,
    mode: StacCogAuthMode,
    result_uses_custom_auth: bool,
    result_catalog_endpoint: Option<&str>,
    stac_settings: &StacSettings,
) -> (String, Vec<(String, String)>) {
    if mode == StacCogAuthMode::None || url_has_presigned_query(url) {
        return (url.to_owned(), Vec::new());
    }
    if mode == StacCogAuthMode::FromSearchResult && !result_uses_custom_auth {
        return (url.to_owned(), Vec::new());
    }

    let auth = stac_settings.to_stac_auth();
    match &auth {
        StacAuth::None => (url.to_owned(), Vec::new()),
        // SAS query tokens are intentionally used across storage domains (e.g.
        // STAC catalog on one host, COGs on `*.blob.core.windows.net`).
        StacAuth::SasQuery(_) => (auth.apply_to_url(url), Vec::new()),
        StacAuth::Bearer(_) | StacAuth::ApiKey { .. } => {
            if mode == StacCogAuthMode::FromSearchResult
                && let Some(origin_ep) = result_catalog_endpoint
                && !same_http_origin(origin_ep, url)
            {
                tracing::debug!("Skipping custom STAC auth headers for cross-origin asset URL");
                return (url.to_owned(), Vec::new());
            }
            (url.to_owned(), auth.extra_headers())
        }
    }
}
