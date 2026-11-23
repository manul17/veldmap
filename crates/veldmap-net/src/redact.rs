//! Canonical HTTP(S) URL authority parsing, percent-encoding, and credential /
//! query-string redaction helpers shared across live basemaps and future
//! authenticated cloud archive streams.

/// Parsed HTTP(S) authority components (`host_port` without `user:pass@`, and
/// bare `host` preserving bracketed IPv6 literals like `[::1]`).
pub(crate) struct ParsedAuthority<'a> {
    pub host_port: &'a str,
    pub host: &'a str,
}

/// Parse the authority section of an HTTP(S) URL or template, handling
/// `user:pass@` userinfo and bracketed IPv6 literals (`http://[::1]:8080/...`).
pub(crate) fn parse_url_authority(url: &str) -> Option<ParsedAuthority<'_>> {
    let without_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let authority = without_scheme.split(['/', '?', '#']).next().unwrap_or("");
    let host_port = authority.rsplit_once('@').map_or(authority, |(_, hp)| hp);
    if host_port.is_empty() {
        return None;
    }
    let host = if let Some(rest) = host_port.strip_prefix('[') {
        let end = rest.find(']')?;
        let inner = &rest[..end];
        if inner.is_empty() {
            return None;
        }
        let after = &rest[end + 1..];
        if !after.is_empty() && !after.starts_with(':') {
            return None;
        }
        &host_port[..=end + 1]
    } else {
        let h = host_port.split(':').next().unwrap_or("");
        if h.is_empty() {
            return None;
        }
        h
    };
    Some(ParsedAuthority { host_port, host })
}

/// Extract the lowercased `host[:port]` authority key from an HTTP(S) URL.
pub(crate) fn extract_host(url: &str) -> String {
    parse_url_authority(url)
        .map(|auth| auth.host_port.to_ascii_lowercase())
        .unwrap_or_else(|| "unknown".to_owned())
}

/// Percent-encode a secret string into both uppercase and lowercase hex forms
/// for URL substitution and redaction matching.
pub(crate) fn percent_encode_variants(secret: &str) -> (String, String) {
    let mut upper = String::with_capacity(secret.len());
    let mut lower = String::with_capacity(secret.len());
    for &b in secret.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            upper.push(char::from(b));
            lower.push(char::from(b));
        } else {
            upper.push_str(&format!("%{b:02X}"));
            lower.push_str(&format!("%{b:02x}"));
        }
    }
    (upper, lower)
}

fn is_valid_url_scheme(scheme: &str) -> bool {
    scheme
        .bytes()
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic())
        && scheme
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
}

/// Redact `user:pass@` authority credentials and `?query` strings from a URL
/// or URL template so inline secrets never appear in `Debug` output, logs, or
/// UI tooltips.
pub fn sanitize_url_for_display(url: &str) -> String {
    let trimmed = url.trim();
    let scheme_end = trimmed
        .find("://")
        .filter(|&idx| is_valid_url_scheme(&trimmed[..idx]));
    let Some(scheme_idx) = scheme_end else {
        return match trimmed.split_once('?') {
            Some((base, _)) => format!("{base}?<redacted>"),
            None => trimmed.to_owned(),
        };
    };

    let scheme_prefix = &trimmed[..scheme_idx + 3];
    let rest = &trimmed[scheme_idx + 3..];
    let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..auth_end];
    let tail = &rest[auth_end..];

    let sanitized_authority = match authority.rsplit_once('@') {
        Some((_, host_port)) => format!("<redacted>@{host_port}"),
        None => authority.to_owned(),
    };

    let sanitized_tail = match tail.split_once('?') {
        Some((path, query_and_frag)) => match query_and_frag.split_once('#') {
            Some((_, frag)) => format!("{path}?<redacted>#{frag}"),
            None => format!("{path}?<redacted>"),
        },
        None => tail.to_owned(),
    };

    format!("{scheme_prefix}{sanitized_authority}{sanitized_tail}")
}

/// Redact `user:pass@` credentials and `?query` tokens inside any URLs
/// embedded within an arbitrary error or diagnostic string (e.g.
/// `reqwest::Error::to_string()`).
pub(crate) fn sanitize_urls_in_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = find_url_scheme_start(rest) {
        out.push_str(&rest[..pos]);
        let candidate = &rest[pos..];
        let raw_end = candidate
            .find(|c: char| c.is_whitespace() || matches!(c, '"' | '<' | '>' | '`'))
            .unwrap_or(candidate.len());
        let end = trim_trailing_prose_punct(&candidate[..raw_end]);
        out.push_str(&sanitize_url_for_display(&candidate[..end]));
        rest = &candidate[end..];
    }
    out.push_str(rest);
    out
}

fn find_url_scheme_start(s: &str) -> Option<usize> {
    let mut offset = 0usize;
    while let Some(rel) = s[offset..].find("://") {
        let sep = offset + rel;
        let prefix = &s[..sep];
        let start = prefix
            .char_indices()
            .rfind(|&(_, c)| !(c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')))
            .map_or(0, |(idx, c)| idx + c.len_utf8());
        let scheme = &s[start..sep];
        if is_valid_url_scheme(scheme) {
            return Some(start);
        }
        offset = sep + 3;
    }
    None
}

/// Strip trailing sentence/wrapper punctuation from a scanned URL token while
/// preserving balanced `[...]` IPv6 literals, balanced `(...)`, and internal
/// RFC 3986 sub-delimiters (`,`, `;`, `'`).
fn trim_trailing_prose_punct(token: &str) -> usize {
    let mut end = token.len();
    while end > 0 {
        let slice = &token[..end];
        let Some(last) = slice.chars().next_back() else {
            break;
        };
        let strip = match last {
            '.' | ',' | ';' | ':' | '\'' => true,
            ')' => slice.matches(')').count() > slice.matches('(').count(),
            ']' => slice.matches(']').count() > slice.matches('[').count(),
            _ => false,
        };
        if strip {
            end -= last.len_utf8();
        } else {
            break;
        }
    }
    end
}

/// Return `"<redacted>"` when `secret` is non-empty, or `empty_label` otherwise.
pub fn redact_secret_label(secret: Option<&str>, empty_label: &'static str) -> &'static str {
    if secret.is_some_and(|s| !s.is_empty()) {
        "<redacted>"
    } else {
        empty_label
    }
}

/// Format common Custom XYZ/TMS basemap fields with URL and API-key redaction.
pub fn fmt_redacted_custom_xyz_fields(
    debug_struct: &mut std::fmt::DebugStruct<'_, '_>,
    name: &str,
    url_template: &str,
    api_key: &str,
    tms_flip_y: bool,
    max_level: u8,
    attribution: &str,
) {
    debug_struct
        .field("name", &name)
        .field("url_template", &sanitize_url_for_display(url_template))
        .field("api_key", &redact_secret_label(Some(api_key), "<empty>"))
        .field("tms_flip_y", &tms_flip_y)
        .field("max_level", &max_level)
        .field("attribution", &attribution);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_host_ipv6_and_url_sanitization_handle_edge_cases() {
        assert_eq!(
            extract_host("https://GIBS.Earthdata.NASA.gov/wmts/1/2/3.jpg"),
            "gibs.earthdata.nasa.gov"
        );
        assert_eq!(
            extract_host("http://user:pass@127.0.0.1:8080/0/0/0.png?key=1"),
            "127.0.0.1:8080"
        );
        assert_eq!(extract_host("http://[::1]:8080/0/0/0.png"), "[::1]:8080");
        let ipv6 = parse_url_authority("http://user:pw@[::1]:8080/0/0/0.png").expect("ipv6");
        assert_eq!(ipv6.host, "[::1]");
        assert_eq!(ipv6.host_port, "[::1]:8080");

        assert_eq!(
            sanitize_url_for_display("https://alice:secret@tiles.example.com/1/2/3.png?token=xyz"),
            "https://<redacted>@tiles.example.com/1/2/3.png?<redacted>"
        );
        assert_eq!(
            sanitize_url_for_display(
                "  HTTPS://alice:p,a;s]s@tiles.example.com/1/2/3.png?token=xyz  "
            ),
            "HTTPS://<redacted>@tiles.example.com/1/2/3.png?<redacted>"
        );
        assert_eq!(
            sanitize_urls_in_text(
                "error sending request for url (https://u:p@host.test/1/2/3.png?token=abc): timed out"
            ),
            "error sending request for url (https://<redacted>@host.test/1/2/3.png?<redacted>): timed out"
        );
        assert_eq!(
            sanitize_urls_in_text(
                "error sending request for url (http://[::1]:8080/1/2/3.png?token=secret): timed out"
            ),
            "error sending request for url (http://[::1]:8080/1/2/3.png?<redacted>): timed out"
        );
        assert_eq!(
            sanitize_urls_in_text(
                "request failed: https://tiles.example/wms?bbox=0,0,1,1&layers=a;b&token=secret, retrying"
            ),
            "request failed: https://tiles.example/wms?<redacted>, retrying"
        );
        assert_eq!(
            sanitize_urls_in_text(
                "failed [HTTPS://alice:p,a;s]s@tiles.example/0/0/0.png?key=secret]"
            ),
            "failed [HTTPS://<redacted>@tiles.example/0/0/0.png?<redacted>]"
        );
        // Multi-byte UTF-8 punctuation immediately before scheme must not panic.
        assert_eq!(
            sanitize_urls_in_text("failed—https://u:p@host.test/0/0/0.png?token=secret"),
            "failed—https://<redacted>@host.test/0/0/0.png?<redacted>"
        );
    }
}
