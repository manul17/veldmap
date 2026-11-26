//! Shared URL authority and userinfo parsing helpers for `veldmap-net`
//! (used by live basemap validation/host extraction and URL credential redaction).

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_authority_and_userinfo_handles_credentials_ipv6_and_tails() {
        assert_eq!(
            split_authority_and_userinfo("user:p@ss@[::1]:8080/tiles?k=1#f"),
            AuthorityParts {
                userinfo: Some("user:p@ss"),
                host_port: "[::1]:8080",
                tail: "/tiles?k=1#f",
            }
        );
        assert_eq!(
            split_authority_and_userinfo("tiles.example.org?token=abc"),
            AuthorityParts {
                userinfo: None,
                host_port: "tiles.example.org",
                tail: "?token=abc",
            }
        );
        assert_eq!(
            split_authority_and_userinfo("tiles.example.org"),
            AuthorityParts {
                userinfo: None,
                host_port: "tiles.example.org",
                tail: "",
            }
        );
    }

    #[test]
    fn url_authority_parsing_and_host_extraction_handle_edge_cases() {
        assert_eq!(
            extract_host("HTTPS://GIBS.Earthdata.NASA.gov/wmts/1/2/3.jpg"),
            "gibs.earthdata.nasa.gov"
        );
        // Multi-byte UTF-8 straddling byte offsets 7..8 must never panic.
        assert_eq!(
            strip_http_scheme("http://\u{00e9}/{z}/{x}/{y}.png"),
            Some("\u{00e9}/{z}/{x}/{y}.png")
        );
        assert_eq!(
            strip_http_scheme("\u{00e9}\u{00e9}\u{00e9}\u{00e9}://example.com"),
            None
        );
        assert_eq!(extract_host("http://\u{00e9}/0/0/0.png"), "\u{00e9}");
        assert_eq!(
            extract_host("http://user:pass@127.0.0.1:8080/0/0/0.png?key=1"),
            "127.0.0.1:8080"
        );
        assert_eq!(extract_host("http://[::1]:8080/0/0/0.png"), "[::1]:8080");
        let ipv6 = parse_http_authority("http://user:pw@[::1]:8080/0/0/0.png").expect("ipv6");
        assert_eq!(ipv6.host, "[::1]");
        assert_eq!(ipv6.host_port, "[::1]:8080");

        // Reject invalid ports and malformed IPv6 brackets.
        assert!(parse_http_authority("http://example.com:notaport/0/0/0.png").is_none());
        assert!(parse_http_authority("http://example.com:70000/0/0/0.png").is_none());
        assert!(parse_http_authority("http://example.com:/0/0/0.png").is_none());
        assert!(parse_http_authority("http://[invalid]/0/0/0.png").is_none());
        assert!(parse_http_authority("http://[::1]:bad/0/0/0.png").is_none());
    }
}
