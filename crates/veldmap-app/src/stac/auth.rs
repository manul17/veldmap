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
                && result_catalog_endpoint.is_none_or(|origin_ep| !same_http_origin(origin_ep, url))
            {
                tracing::debug!(
                    "Skipping custom STAC auth headers for cross-origin or unverified-origin asset URL"
                );
                return (url.to_owned(), Vec::new());
            }
            (url.to_owned(), auth.extra_headers())
        }
    }
}

/// Normalize `mode` against whether the active STAC search used custom auth so
/// public zero-auth catalog assets persist strictly as [`StacCogAuthMode::None`]
/// and never accidentally inherit custom credentials on reload.
pub(crate) const fn effective_persisted_auth_mode(
    mode: StacCogAuthMode,
    result_uses_custom_auth: bool,
) -> StacCogAuthMode {
    match mode {
        StacCogAuthMode::None => StacCogAuthMode::None,
        StacCogAuthMode::FromSearchResult => {
            if result_uses_custom_auth {
                StacCogAuthMode::FromSearchResult
            } else {
                StacCogAuthMode::None
            }
        }
        StacCogAuthMode::DirectCustom => StacCogAuthMode::DirectCustom,
    }
}

/// Trust boundary for persisted remote COG auth resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AuthTrust {
    /// Local user-private `session.json` managed by VeldMap on this machine.
    LocalSession,
    /// Portable/shareable `.veldmap.json` project file (potentially from an
    /// external source): credentials from private `settings.json` are attached
    /// strictly to URLs matching `stac_settings.custom_endpoint`'s origin.
    ImportedProject,
}

/// Downgrade an imported `.veldmap.json` remote asset's `auth_mode` against
/// `stac_settings` so untrusted project files can never escalate into
/// cross-origin credential attachment on load, retry, or subsequent
/// `session.json` restarts.
pub(crate) fn sanitize_imported_auth_mode(
    url: &str,
    mode: StacCogAuthMode,
    stac_settings: &StacSettings,
) -> StacCogAuthMode {
    if mode == StacCogAuthMode::None {
        return StacCogAuthMode::None;
    }
    let custom_ep = stac_settings.custom_endpoint.trim();
    if !custom_ep.is_empty() && same_http_origin(custom_ep, url) {
        // Downgrade `DirectCustom` to `FromSearchResult` so later changing
        // `custom_endpoint` will not send headers to a stale imported origin.
        StacCogAuthMode::FromSearchResult
    } else {
        StacCogAuthMode::None
    }
}

/// Resolve credentials from `settings.json` when restoring a persisted remote
/// COG layer without an active STAC search result list.
pub(crate) fn resolve_persisted_cog_auth(
    url: &str,
    mode: StacCogAuthMode,
    trust: AuthTrust,
    stac_settings: &StacSettings,
) -> (String, Vec<(String, String)>) {
    let custom_ep = stac_settings.custom_endpoint.trim();
    let origin_ep = (!custom_ep.is_empty()).then_some(custom_ep);
    let effective_mode = match trust {
        AuthTrust::LocalSession => mode,
        AuthTrust::ImportedProject => {
            let safe_mode = sanitize_imported_auth_mode(url, mode, stac_settings);
            if safe_mode == StacCogAuthMode::None {
                return (url.to_owned(), Vec::new());
            }
            safe_mode
        }
    };
    resolve_cog_auth(
        url,
        effective_mode,
        effective_mode != StacCogAuthMode::None,
        origin_ep,
        stac_settings,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::settings::StacAuthKind;

    #[test]
    fn persisted_from_search_result_never_leaks_headers_without_matching_custom_origin() {
        let mut settings = StacSettings {
            custom_endpoint: String::new(),
            custom_collection: String::new(),
            auth_kind: StacAuthKind::Bearer,
            auth_secret: "super-secret-token".to_owned(),
            auth_header_name: "X-API-Key".to_owned(),
        };

        // Empty custom_endpoint must NOT attach Bearer headers to arbitrary URLs.
        let (_, headers_empty) = resolve_persisted_cog_auth(
            "https://evil.example/data/scene.tif",
            StacCogAuthMode::FromSearchResult,
            AuthTrust::LocalSession,
            &settings,
        );
        assert!(
            headers_empty.is_empty(),
            "empty custom_endpoint must block FromSearchResult headers"
        );

        // Matching origin attaches headers; mismatched origin strips them.
        settings.custom_endpoint = "https://stac.corp.example/v1".to_owned();
        let (_, headers_same) = resolve_persisted_cog_auth(
            "https://stac.corp.example/assets/b04.tif",
            StacCogAuthMode::FromSearchResult,
            AuthTrust::LocalSession,
            &settings,
        );
        assert_eq!(headers_same.len(), 1);
        let (_, headers_other) = resolve_persisted_cog_auth(
            "https://other.example/assets/b04.tif",
            StacCogAuthMode::FromSearchResult,
            AuthTrust::LocalSession,
            &settings,
        );
        assert!(headers_other.is_empty());
    }

    #[test]
    fn imported_project_never_leaks_bearer_apikey_or_sas_to_cross_origin_urls() {
        for kind in [
            StacAuthKind::Bearer,
            StacAuthKind::ApiKey,
            StacAuthKind::SasQuery,
        ] {
            let settings = StacSettings {
                custom_endpoint: "https://stac.corp.example/v1".to_owned(),
                custom_collection: String::new(),
                auth_kind: kind,
                auth_secret: "sig=topsecret".to_owned(),
                auth_header_name: "X-API-Key".to_owned(),
            };
            for claimed_mode in [
                StacCogAuthMode::DirectCustom,
                StacCogAuthMode::FromSearchResult,
            ] {
                let attacker = "https://attacker.example/exfil.tif";
                let (url, headers) = resolve_persisted_cog_auth(
                    attacker,
                    claimed_mode,
                    AuthTrust::ImportedProject,
                    &settings,
                );
                assert_eq!(url, attacker, "must not append SAS query to attacker URL");
                assert!(
                    headers.is_empty(),
                    "must not attach headers to attacker URL for {kind:?}/{claimed_mode:?}"
                );
                assert_eq!(
                    sanitize_imported_auth_mode(attacker, claimed_mode, &settings),
                    StacCogAuthMode::None
                );

                // Same-origin imported URL downgrades DirectCustom -> FromSearchResult.
                let same = "https://stac.corp.example/cogs/b04.tif";
                assert_eq!(
                    sanitize_imported_auth_mode(same, claimed_mode, &settings),
                    StacCogAuthMode::FromSearchResult
                );
            }
        }
    }
}
