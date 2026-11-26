//! Secret percent-encoding and credential / query-string URL redaction helpers
//! shared across live basemaps and future authenticated cloud archive streams.

use std::fmt::Write as _;

use crate::url::split_authority_and_userinfo;

/// Percent-encode a secret string into both uppercase and lowercase hex forms
/// for URL substitution and redaction matching.
fn percent_encode_variants(secret: &str) -> (String, String) {
    let mut upper = String::with_capacity(secret.len());
    let mut lower = String::with_capacity(secret.len());
    for &b in secret.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            upper.push(char::from(b));
            lower.push(char::from(b));
        } else {
            let _ = write!(upper, "%{b:02X}");
            let _ = write!(lower, "%{b:02x}");
        }
    }
    (upper, lower)
}

/// Pre-encoded API secret variants computed once at configuration construction
/// so per-tile URL expansion and error redaction do zero redundant
/// percent-encoding work.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct EncodedSecret {
    raw: String,
    encoded_upper: String,
    encoded_lower: String,
}

impl EncodedSecret {
    pub(crate) fn new(raw: String) -> Self {
        let (encoded_upper, encoded_lower) = percent_encode_variants(&raw);
        Self {
            raw,
            encoded_upper,
            encoded_lower,
        }
    }

    pub(crate) fn raw(&self) -> &str {
        &self.raw
    }

    pub(crate) fn encoded_upper(&self) -> &str {
        &self.encoded_upper
    }

    /// Redact raw and percent-encoded (upper/lower hex) forms of this secret
    /// from `text`.
    fn redact_in(&self, mut text: String) -> String {
        if self.raw.is_empty() {
            return text;
        }
        text = text.replace(&self.raw, "<redacted>");
        if self.encoded_upper != self.raw {
            text = text.replace(&self.encoded_upper, "<redacted>");
        }
        if self.encoded_lower != self.raw && self.encoded_lower != self.encoded_upper {
            text = text.replace(&self.encoded_lower, "<redacted>");
        }
        text
    }
}

/// Redact inline URL `user:pass@` authority credentials, `?query` tokens, and
/// any configured [`EncodedSecret`] (raw or percent-encoded) from `text`.
pub(crate) fn redact_text(text: &str, secret: Option<&EncodedSecret>) -> String {
    let out = sanitize_urls_in_text(text);
    match secret {
        Some(secret) => secret.redact_in(out),
        None => out,
    }
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
    let parts = split_authority_and_userinfo(&trimmed[scheme_idx + 3..]);

    let sanitized_authority = match parts.userinfo {
        Some(_) => format!("<redacted>@{}", parts.host_port),
        None => parts.host_port.to_owned(),
    };

    let sanitized_tail = match parts.tail.split_once('?') {
        Some((path, query_and_frag)) => match query_and_frag.split_once('#') {
            Some((_, frag)) => format!("{path}?<redacted>#{frag}"),
            None => format!("{path}?<redacted>"),
        },
        None => parts.tail.to_owned(),
    };

    format!("{scheme_prefix}{sanitized_authority}{sanitized_tail}")
}

/// Redact `user:pass@` credentials and `?query` tokens inside any URLs
/// embedded within an arbitrary error or diagnostic string (e.g.
/// `reqwest::Error::to_string()`).
fn sanitize_urls_in_text(text: &str) -> String {
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
pub(crate) fn redact_secret_label(secret: Option<&str>, empty_label: &'static str) -> &'static str {
    if secret.is_some_and(|s| !s.is_empty()) {
        "<redacted>"
    } else {
        empty_label
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_sanitization_handles_credentials_queries_ipv6_and_unicode_punctuation() {
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
