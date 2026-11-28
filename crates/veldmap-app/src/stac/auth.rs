//! Origin-aware authentication resolution for remote Cloud-Optimized GeoTIFFs
//! opened from STAC search results, built-in sample scenes, direct URLs, and
//! `401`/`403` `stac_item_href` restore refreshes across named Custom STAC
//! profiles.

use veldmap_net::{StacAuth, same_http_origin};

use crate::layers::{StacCogAuthMode, StacItemRefreshPolicy};
use crate::storage::settings::{StacProfile, StacSettings};

/// Return `true` if `url` already carries a cloud presigned-URL signature query
/// parameter (AWS S3, Azure SAS, GCS) so we never double-sign or attach
/// conflicting auth headers.
pub(crate) fn url_has_presigned_query(url: &str) -> bool {
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

fn apply_profile_cog_auth(
    url: &str,
    mode: StacCogAuthMode,
    origin_ep: Option<&str>,
    auth: &StacAuth,
) -> (String, Vec<(String, String)>) {
    match auth {
        StacAuth::None => (url.to_owned(), Vec::new()),
        // SAS query tokens are intentionally used across storage domains (e.g.
        // STAC catalog on one host, COGs on `*.blob.core.windows.net`).
        StacAuth::SasQuery(_) => (auth.apply_to_url(url), Vec::new()),
        StacAuth::Bearer(_) | StacAuth::ApiKey { .. } => {
            if mode == StacCogAuthMode::FromSearchResult
                && origin_ep.is_none_or(|ep| !same_http_origin(ep, url))
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

/// Resolve the final URL and custom HTTP headers for opening a remote COG,
/// preventing accidental credential leaks onto public or cross-origin hosts and
/// avoiding dual-auth errors on already-presigned URLs.
pub(super) fn resolve_cog_auth(
    url: &str,
    mode: StacCogAuthMode,
    result_uses_custom_auth: bool,
    result_catalog_endpoint: Option<&str>,
    result_auth: Option<&StacAuth>,
    stac_settings: &StacSettings,
) -> (String, Vec<(String, String)>) {
    if mode == StacCogAuthMode::None || url_has_presigned_query(url) {
        return (url.to_owned(), Vec::new());
    }
    if mode == StacCogAuthMode::FromSearchResult && !result_uses_custom_auth {
        return (url.to_owned(), Vec::new());
    }

    let (auth, origin_ep): (StacAuth, Option<&str>) = match mode {
        StacCogAuthMode::FromSearchResult => {
            if let Some(a) = result_auth {
                (a.clone(), result_catalog_endpoint)
            } else if let Some(profile) = stac_settings
                .profile_matching_origin(url)
                .or_else(|| {
                    result_catalog_endpoint.and_then(|ep| stac_settings.profile_matching_origin(ep))
                })
                .or_else(|| stac_settings.active_profile())
            {
                let ep = result_catalog_endpoint
                    .or_else(|| profile.is_configured().then_some(profile.endpoint.as_str()));
                (profile.to_stac_auth(), ep)
            } else {
                (StacAuth::None, None)
            }
        }
        StacCogAuthMode::DirectCustom => {
            let auth = stac_settings
                .profile_matching_origin(url)
                .map_or_else(|| stac_settings.to_stac_auth(), StacProfile::to_stac_auth);
            (auth, result_catalog_endpoint)
        }
        StacCogAuthMode::None => (StacAuth::None, None),
    };

    apply_profile_cog_auth(url, mode, origin_ep, &auth)
}

/// Normalize `mode` against whether the active STAC search used custom auth so
/// public zero-auth catalog assets persist strictly as [`StacCogAuthMode::None`]
/// and never accidentally inherit custom credentials on reload.
pub(super) const fn effective_persisted_auth_mode(
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
    /// strictly to URLs matching a configured Custom STAC profile's origin.
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
    if stac_settings.profile_matching_origin(url).is_some() {
        // Downgrade `DirectCustom` to `FromSearchResult` so later changing
        // profiles will not send headers to a stale imported origin.
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
    if effective_mode == StacCogAuthMode::None || url_has_presigned_query(url) {
        return (url.to_owned(), Vec::new());
    }
    let matched_profile = stac_settings.profile_matching_origin(url);
    if let Some(profile) = matched_profile {
        return apply_profile_cog_auth(
            url,
            effective_mode,
            Some(profile.endpoint.as_str()),
            &profile.to_stac_auth(),
        );
    }
    // No profile origin matched `url`: for `LocalSession`, `DirectCustom` or
    // `SasQuery` on the active profile may still apply (e.g. SAS query across
    // Azure Blob storage host), whereas `Bearer`/`ApiKey` with
    // `FromSearchResult` require an explicit matching profile origin.
    if let Some(active) = stac_settings.active_profile() {
        let origin_ep = active.is_configured().then_some(active.endpoint.as_str());
        return apply_profile_cog_auth(url, effective_mode, origin_ep, &active.to_stac_auth());
    }
    (url.to_owned(), Vec::new())
}

/// Build an origin-fenced [`StacItemRefreshPolicy`] for refreshing expired
/// presigned asset URLs via `stac_item_href` on HTTP `401`/`403` during
/// session/project restore.
///
/// Security invariants:
/// - For `AuthTrust::ImportedProject`, never sends private credentials from
///   `settings.json` to `stac_item_href` or the refreshed asset URL (only
///   unauthenticated public STAC item refresh is permitted).
/// - For `AuthTrust::LocalSession`, attaches a profile's credentials to the
///   `stac_item_href` GET strictly when `same_http_origin(profile.endpoint,
///   stac_item_href)` holds.
pub(crate) fn build_stac_item_refresh_policy(
    stac_item_href: Option<&str>,
    asset_key: Option<&str>,
    trust: AuthTrust,
    stac_settings: &StacSettings,
) -> Option<StacItemRefreshPolicy> {
    let item_href = stac_item_href?.trim();
    let key = asset_key?.trim();
    if item_href.is_empty() || key.is_empty() {
        return None;
    }
    match trust {
        AuthTrust::ImportedProject => Some(StacItemRefreshPolicy {
            item_auth: StacAuth::None,
            asset_auth_origin: None,
            asset_headers: Vec::new(),
        }),
        AuthTrust::LocalSession => {
            let matched = stac_settings.profile_matching_origin(item_href);
            let item_auth = matched.map_or(StacAuth::None, StacProfile::to_stac_auth);
            let (asset_auth_origin, asset_headers) = match matched {
                Some(profile) => {
                    let hdrs = item_auth.extra_headers();
                    if hdrs.is_empty() {
                        (None, Vec::new())
                    } else {
                        (Some(profile.endpoint.clone()), hdrs)
                    }
                }
                None => (None, Vec::new()),
            };
            Some(StacItemRefreshPolicy {
                item_auth,
                asset_auth_origin,
                asset_headers,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::settings::StacAuthKind;

    #[test]
    fn persisted_from_search_result_never_leaks_headers_without_matching_custom_origin() {
        let mut settings = StacSettings::from_single_profile(StacProfile {
            name: "Corp".to_owned(),
            endpoint: String::new(),
            collection: String::new(),
            auth_kind: StacAuthKind::Bearer,
            auth_secret: "super-secret-token".to_owned(),
            auth_header_name: "X-API-Key".to_owned(),
        });

        // Empty endpoint must NOT attach Bearer headers to arbitrary URLs.
        let (_, headers_empty) = resolve_persisted_cog_auth(
            "https://evil.example/data/scene.tif",
            StacCogAuthMode::FromSearchResult,
            AuthTrust::LocalSession,
            &settings,
        );
        assert!(
            headers_empty.is_empty(),
            "empty profile endpoint must block FromSearchResult headers"
        );

        // Matching origin attaches headers; mismatched origin strips them.
        settings.profiles[0].endpoint = "https://stac.corp.example/v1".to_owned();
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
            let settings = StacSettings::from_single_profile(StacProfile {
                name: "Corp".to_owned(),
                endpoint: "https://stac.corp.example/v1".to_owned(),
                collection: String::new(),
                auth_kind: kind,
                auth_secret: "sig=topsecret".to_owned(),
                auth_header_name: "X-API-Key".to_owned(),
            });
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

    #[test]
    fn refresh_policy_origin_fences_local_session_and_strips_imported_credentials() {
        let settings = StacSettings::from_single_profile(StacProfile {
            name: "Corp".to_owned(),
            endpoint: "https://stac.corp.example/v1".to_owned(),
            collection: "s2".to_owned(),
            auth_kind: StacAuthKind::Bearer,
            auth_header_name: "x-api-key".to_owned(),
            auth_secret: "corp-secret".to_owned(),
        });

        // LocalSession + matching origin -> attaches Bearer auth for item fetch.
        let p_same = build_stac_item_refresh_policy(
            Some("https://stac.corp.example/v1/collections/s2/items/abc"),
            Some("red"),
            AuthTrust::LocalSession,
            &settings,
        )
        .expect("policy");
        assert_eq!(p_same.item_auth, StacAuth::Bearer("corp-secret".to_owned()));
        assert_eq!(p_same.asset_headers.len(), 1);

        // LocalSession + off-origin item_href -> public unauthenticated refresh only.
        let p_off = build_stac_item_refresh_policy(
            Some("https://other.example/items/abc"),
            Some("red"),
            AuthTrust::LocalSession,
            &settings,
        )
        .expect("policy");
        assert_eq!(p_off.item_auth, StacAuth::None);
        assert!(p_off.asset_headers.is_empty());

        // ImportedProject -> NEVER attaches private credentials even if origin matches.
        let p_imp = build_stac_item_refresh_policy(
            Some("https://stac.corp.example/v1/collections/s2/items/abc"),
            Some("red"),
            AuthTrust::ImportedProject,
            &settings,
        )
        .expect("policy");
        assert_eq!(p_imp.item_auth, StacAuth::None);
        assert!(p_imp.asset_headers.is_empty());
    }
}
