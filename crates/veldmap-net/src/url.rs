//! Shared URL authority, origin comparison, canonicalization, and userinfo
//! parsing helpers for `veldmap-net` (used by live basemap validation,
//! `HttpRangeSource` fingerprinting, STAC origin checks, and URL credential
//! redaction).

use std::net::Ipv6Addr;

/// Result of splitting a post-scheme URL slice (`after_scheme`, immediately
/// following `"://"`) into its optional `userinfo`, `host_port`, and remaining
/// path/query/fragment `tail`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AuthorityParts<'a> {
    pub(crate) userinfo: Option<&'a str>,
    pub(crate) host_port: &'a str,
    pub(crate) tail: &'a str,
}

/// Parsed HTTP(S) authority components (`host_port` without `user:pass@`, and
/// bare `host` preserving bracketed IPv6 literals like `[::1]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ParsedAuthority<'a> {
    pub(crate) host_port: &'a str,
    pub(crate) host: &'a str,
}

/// Strip an ASCII case-insensitive `https://` or `http://` prefix without
/// ever slicing across a multi-byte UTF-8 character boundary.
pub(crate) fn strip_http_scheme(url: &str) -> Option<&str> {
    let bytes = url.as_bytes();
    if bytes.len() >= 8 && bytes[..8].eq_ignore_ascii_case(b"https://") {
        Some(&url[8..])
    } else if bytes.len() >= 7 && bytes[..7].eq_ignore_ascii_case(b"http://") {
        Some(&url[7..])
    } else {
        None
    }
}

/// Split `after_scheme` (the slice immediately following `"://"`) at the first
/// `/`, `?`, or `#` into authority and `tail`, then split optional `userinfo@`
/// using the rightmost `@` within the authority.
pub(crate) fn split_authority_and_userinfo(after_scheme: &str) -> AuthorityParts<'_> {
    let auth_end = after_scheme
        .find(['/', '?', '#'])
        .unwrap_or(after_scheme.len());
    let authority = &after_scheme[..auth_end];
    let tail = &after_scheme[auth_end..];
    let (userinfo, host_port) = match authority.rsplit_once('@') {
        Some((user, hp)) => (Some(user), hp),
        None => (None, authority),
    };
    AuthorityParts {
        userinfo,
        host_port,
        tail,
    }
}

fn valid_optional_port(port_suffix: &str) -> bool {
    if port_suffix.is_empty() {
        return true;
    }
    let Some(port_str) = port_suffix.strip_prefix(':') else {
        return false;
    };
    !port_str.is_empty()
        && port_str.bytes().all(|b| b.is_ascii_digit())
        && port_str.parse::<u16>().is_ok()
}

/// Parse the authority section of an HTTP(S) URL or template, handling
/// case-insensitive `http://`/`https://` schemes, `user:pass@` userinfo,
/// numeric `:port` validation, and bracketed IPv6 literals
/// (`http://[::1]:8080/...`).
pub(crate) fn parse_http_authority(url: &str) -> Option<ParsedAuthority<'_>> {
    let without_scheme = strip_http_scheme(url)?;
    let host_port = split_authority_and_userinfo(without_scheme).host_port;
    if host_port.is_empty() {
        return None;
    }
    let host = if let Some(rest) = host_port.strip_prefix('[') {
        let end = rest.find(']')?;
        let inner = &rest[..end];
        inner.parse::<Ipv6Addr>().ok()?;
        let after = &rest[end + 1..];
        if !valid_optional_port(after) {
            return None;
        }
        &host_port[..=end + 1]
    } else {
        let (h, port_suffix) = match host_port.find(':') {
            Some(colon) => (&host_port[..colon], &host_port[colon..]),
            None => (host_port, ""),
        };
        if h.is_empty() || !valid_optional_port(port_suffix) {
            return None;
        }
        h
    };
    Some(ParsedAuthority { host_port, host })
}

/// Extract the lowercased `host[:port]` authority key from an HTTP(S) URL.
pub(crate) fn extract_host(url: &str) -> String {
    parse_http_authority(url)
        .map(|auth| auth.host_port.to_ascii_lowercase())
        .unwrap_or_else(|| "unknown".to_owned())
}

/// Normalize an HTTP(S) URL into `scheme://host[:port]/path` with lowercased
/// scheme and authority, stripping any inline `user:pass@`, `?query`, and
/// `#fragment` so presigned URLs for the same immutable object share a stable
/// cache identity.
pub(crate) fn canonical_url_without_query(url: &str) -> Option<String> {
    let trimmed = url.trim();
    if trimmed.is_empty()
        || trimmed
            .bytes()
            .any(|b| b.is_ascii_whitespace() || b.is_ascii_control())
    {
        return None;
    }
    let after_scheme = strip_http_scheme(trimmed)?;
    let scheme = if trimmed.len() >= 8 && trimmed.as_bytes()[..8].eq_ignore_ascii_case(b"https://")
    {
        "https"
    } else {
        "http"
    };
    let authority = parse_http_authority(trimmed)?;
    let parts = split_authority_and_userinfo(after_scheme);
    let path_end = parts.tail.find(['?', '#']).unwrap_or(parts.tail.len());
    let path = &parts.tail[..path_end];
    let normalized_path = if path.is_empty() { "/" } else { path };
    Some(format!(
        "{scheme}://{}{normalized_path}",
        authority.host_port.to_ascii_lowercase()
    ))
}

/// Return whether `url` is a syntactically valid `http://` or `https://` URL
/// with a valid authority and no ASCII whitespace or control bytes.
pub fn is_valid_http_endpoint(url: &str) -> bool {
    let trimmed = url.trim();
    !trimmed.is_empty()
        && !trimmed
            .bytes()
            .any(|b| b.is_ascii_whitespace() || b.is_ascii_control())
        && strip_http_scheme(trimmed).is_some()
        && parse_http_authority(trimmed).is_some()
}

/// Split `url` into `(base_without_query_or_fragment, optional_query_without_qmark,
/// optional_fragment_without_hash)`.
pub(crate) fn split_url_query_fragment(url: &str) -> (&str, Option<&str>, Option<&str>) {
    let (before_frag, frag) = match url.split_once('#') {
        Some((b, f)) => (b, Some(f)),
        None => (url, None),
    };
    let (base, query) = match before_frag.split_once('?') {
        Some((b, q)) => (b, Some(q)),
        None => (before_frag, None),
    };
    (base, query, frag)
}

fn normalized_http_origin(url: &str) -> Option<(&'static str, String, u16)> {
    let trimmed = url.trim();
    if trimmed.is_empty()
        || trimmed
            .bytes()
            .any(|b| b.is_ascii_whitespace() || b.is_ascii_control())
    {
        return None;
    }
    let (scheme, default_port) =
        if trimmed.len() >= 8 && trimmed.as_bytes()[..8].eq_ignore_ascii_case(b"https://") {
            ("https", 443u16)
        } else if trimmed.len() >= 7 && trimmed.as_bytes()[..7].eq_ignore_ascii_case(b"http://") {
            ("http", 80u16)
        } else {
            return None;
        };
    let auth = parse_http_authority(trimmed)?;
    let port_suffix = &auth.host_port[auth.host.len()..];
    let port = if let Some(p) = port_suffix.strip_prefix(':') {
        p.parse::<u16>().ok()?
    } else {
        default_port
    };
    Some((scheme, auth.host.to_ascii_lowercase(), port))
}

/// Return whether two HTTP(S) URLs share the exact same `(scheme, host, port)`
/// origin (normalizing default ports `http -> 80` and `https -> 443`).
pub fn same_http_origin(a: &str, b: &str) -> bool {
    match (normalized_http_origin(a), normalized_http_origin(b)) {
        (Some(oa), Some(ob)) => oa == ob,
        _ => false,
    }
}

#[cfg(test)]
#[path = "url_tests.rs"]
mod tests;
