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

/// Return whether `s` contains ASCII whitespace or control bytes disallowed
/// in HTTP(S) URLs and URL templates.
pub(crate) fn has_invalid_url_bytes(s: &str) -> bool {
    s.bytes()
        .any(|b| b.is_ascii_whitespace() || b.is_ascii_control())
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

/// Parse and validate the authority section of an HTTP(S) URL or `{s}`
/// template, handling case-insensitive `http://`/`https://` schemes,
/// `user:pass@` userinfo, numeric `:port` validation, and bracketed IPv6
/// literals (`http://[::1]:8080/...`), and returning the bare `host`
/// (preserving IPv6 brackets).
pub(crate) fn parse_http_authority(url: &str) -> Option<&str> {
    let after_scheme = strip_http_scheme(url)?;
    let parts = split_authority_and_userinfo(after_scheme);
    let host_port = parts.host_port;
    if host_port.is_empty() {
        return None;
    }
    if let Some(rest) = host_port.strip_prefix('[') {
        let end = rest.find(']')?;
        let inner = &rest[..end];
        inner.parse::<Ipv6Addr>().ok()?;
        let after = &rest[end + 1..];
        if !valid_optional_port(after) {
            return None;
        }
        Some(&host_port[..=end + 1])
    } else {
        let (host, port_suffix) = match host_port.find(':') {
            Some(colon) => (&host_port[..colon], &host_port[colon..]),
            None => (host_port, ""),
        };
        if host.is_empty() || !valid_optional_port(port_suffix) {
            return None;
        }
        Some(host)
    }
}

/// Strictly parse a concrete `http://` or `https://` URL using
/// `reqwest::Url::parse` after enforcing an explicit `http://`/`https://`
/// scheme prefix and rejecting ASCII whitespace and control bytes (which
/// WHATWG URL parsing would otherwise silently normalize, strip, or
/// percent-encode).
fn parse_http_url(url: &str) -> Option<reqwest::Url> {
    let trimmed = url.trim();
    if trimmed.is_empty() || has_invalid_url_bytes(trimmed) {
        return None;
    }
    strip_http_scheme(trimmed)?;
    let parsed = reqwest::Url::parse(trimmed).ok()?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return None;
    }
    parsed.host_str()?;
    Some(parsed)
}

/// Extract the lowercased `host[:port]` authority key from an HTTP(S) URL
/// (stripping default `:80`/`:443` ports and normalizing IDNA hosts).
pub(crate) fn extract_host(url: &str) -> String {
    let Some(parsed) = parse_http_url(url) else {
        return "unknown".to_owned();
    };
    let Some(host) = parsed.host_str() else {
        return "unknown".to_owned();
    };
    match parsed.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_owned(),
    }
}

/// Normalize an HTTP(S) URL into `scheme://host[:port]/path` with lowercased
/// scheme and authority, default ports (`:80` for `http`, `:443` for `https`)
/// stripped, and any inline `user:pass@`, `?query`, and `#fragment` removed so
/// presigned URLs for the same immutable object share a stable cache identity.
pub(crate) fn canonical_url_without_query(url: &str) -> Option<String> {
    let mut parsed = parse_http_url(url)?;
    parsed.set_password(None).ok()?;
    parsed.set_username("").ok()?;
    parsed.set_query(None);
    parsed.set_fragment(None);
    Some(parsed.into())
}

/// Return whether `url` is a syntactically valid `http://` or `https://` URL
/// with a valid authority and no ASCII whitespace or control bytes.
pub fn is_valid_http_endpoint(url: &str) -> bool {
    parse_http_url(url).is_some()
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

/// Return whether two parsed URLs share the exact same `(scheme, host, port)`
/// origin (normalizing default ports and comparing hosts case-insensitively).
pub(crate) fn same_url_origin(a: &reqwest::Url, b: &reqwest::Url) -> bool {
    a.scheme() == b.scheme()
        && a.host_str()
            .zip(b.host_str())
            .is_some_and(|(ha, hb)| ha.eq_ignore_ascii_case(hb))
        && a.port_or_known_default() == b.port_or_known_default()
}

/// Return whether two HTTP(S) URLs share the exact same `(scheme, host, port)`
/// origin (normalizing default ports `http -> 80` and `https -> 443`).
pub fn same_http_origin(a: &str, b: &str) -> bool {
    let (Some(ua), Some(ub)) = (parse_http_url(a), parse_http_url(b)) else {
        return false;
    };
    same_url_origin(&ua, &ub)
}

#[cfg(test)]
#[path = "url_tests.rs"]
mod tests;
