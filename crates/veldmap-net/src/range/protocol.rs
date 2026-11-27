//! RFC 9110 HTTP Range protocol helpers: strong-ETag validation, `Content-Range`
//! parsing, `Content-Encoding: identity` enforcement, and zero-extra-RTT remote
//! `SourceFingerprint` derivation.

use reqwest::header::HeaderMap;
use veldmap_core::tile::SourceFingerprint;

/// Domain separation prefix for remote HTTP source fingerprints so they can
/// never collide with local `len || head || mid || tail` digests.
const REMOTE_FINGERPRINT_DOMAIN: &[u8] = b"veldmap-remote-v1\0";

/// Head window size (`64 KiB`) hashed when a server omits or weakens `ETag`.
const FINGERPRINT_HEAD_FALLBACK_BYTES: usize = 64 * 1024;

/// Parsed RFC 9110 `Content-Range` header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ContentRange {
    /// `bytes <start>-<end>/<total>` with `start <= end < total`.
    Span { start: u64, end: u64, total: u64 },
    /// `bytes */<total>` (used for `416 Range Not Satisfiable`, e.g. 0-byte files).
    Unsatisfied { total: u64 },
}

/// Extract a validated strong RFC 9110 `ETag` (`"..."`, never weak `W/"..."`
/// or multi-quoted/control-byte values).
pub(super) fn parse_strong_etag(headers: &HeaderMap) -> Option<String> {
    let raw = headers.get(reqwest::header::ETAG)?.to_str().ok()?.trim();
    if raw.len() < 2 || !raw.starts_with('"') || !raw.ends_with('"') {
        return None;
    }
    let inner = &raw.as_bytes()[1..raw.len() - 1];
    // RFC 9110 §8.8.3 `etagc = %x21 / %x23-7E / obs-text` (forbids `"`, SP, CTL).
    if !inner
        .iter()
        .all(|&b| b == 0x21 || (0x23..=0x7E).contains(&b) || b >= 0x80)
    {
        return None;
    }
    Some(raw.to_owned())
}

/// Reject any response carrying a non-`identity` `Content-Encoding` header so
/// byte-range offsets are never silently corrupted by transparent compression.
pub(super) fn validate_identity_content_encoding(headers: &HeaderMap) -> Result<(), String> {
    for val in headers.get_all(reqwest::header::CONTENT_ENCODING) {
        let text = val
            .to_str()
            .map_err(|_| "non-ASCII Content-Encoding header on byte-range response".to_owned())?;
        for part in text.split(',') {
            let trimmed = part.trim();
            if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("identity") {
                return Err(format!(
                    "unexpected Content-Encoding '{trimmed}' on byte-range response (expected identity)"
                ));
            }
        }
    }
    Ok(())
}

/// Parse an RFC 9110 `Content-Range` header with an explicit total byte length.
pub(super) fn parse_content_range(raw: &str) -> Result<ContentRange, String> {
    let trimmed = raw.trim();
    let bytes = trimmed.as_bytes();
    if bytes.len() < 6 || !bytes[..5].eq_ignore_ascii_case(b"bytes") || bytes[5] != b' ' {
        return Err(format!("unsupported Content-Range unit: '{trimmed}'"));
    }
    let rest = trimmed[6..].trim();
    let (range_part, total_part) = rest
        .split_once('/')
        .ok_or_else(|| format!("malformed Content-Range (missing '/'): '{trimmed}'"))?;
    let total_str = total_part.trim();
    if total_str == "*" {
        return Err("Content-Range with unknown total length ('*') is not supported".to_owned());
    }
    let total: u64 = total_str
        .parse()
        .map_err(|_| format!("invalid Content-Range total length: '{total_str}'"))?;

    let range_str = range_part.trim();
    if range_str == "*" {
        return Ok(ContentRange::Unsatisfied { total });
    }
    let (start_str, end_str) = range_str
        .split_once('-')
        .ok_or_else(|| format!("malformed Content-Range span: '{range_str}'"))?;
    let start: u64 = start_str
        .trim()
        .parse()
        .map_err(|_| format!("invalid Content-Range start: '{start_str}'"))?;
    let end: u64 = end_str
        .trim()
        .parse()
        .map_err(|_| format!("invalid Content-Range end: '{end_str}'"))?;
    if start > end || end >= total {
        return Err(format!(
            "invalid Content-Range bounds {start}-{end}/{total}"
        ));
    }
    Ok(ContentRange::Span { start, end, total })
}

/// Derive a deterministic [`SourceFingerprint`] without issuing extra HTTP GETs:
/// - Strong ETag present: hashes `(canonical_url_without_query, len, strong_etag)`
/// - Weak/missing ETag: hashes `(canonical_url_without_query, len, blake3(head_64k))`
pub(super) fn fingerprint_remote_source(
    canonical_url_without_query: &str,
    content_length: u64,
    strong_etag: Option<&str>,
    head_bytes: &[u8],
) -> SourceFingerprint {
    let mut hasher = blake3::Hasher::new();
    hasher.update(REMOTE_FINGERPRINT_DOMAIN);
    hasher.update(&(canonical_url_without_query.len() as u64).to_le_bytes());
    hasher.update(canonical_url_without_query.as_bytes());
    hasher.update(&content_length.to_le_bytes());
    if let Some(etag) = strong_etag {
        hasher.update(&[1u8]);
        hasher.update(&(etag.len() as u64).to_le_bytes());
        hasher.update(etag.as_bytes());
    } else {
        hasher.update(&[0u8]);
        let prefix = &head_bytes[..head_bytes.len().min(FINGERPRINT_HEAD_FALLBACK_BYTES)];
        let head_hash = blake3::hash(prefix);
        hasher.update(head_hash.as_bytes());
    }
    SourceFingerprint::from_digest(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use reqwest::header::{CONTENT_ENCODING, ETAG, HeaderValue};

    use super::*;

    #[test]
    fn parses_strong_etag_and_rejects_weak_etags() {
        let mut h = HeaderMap::new();
        h.insert(ETAG, HeaderValue::from_static("\"abc-123\""));
        assert_eq!(parse_strong_etag(&h).as_deref(), Some("\"abc-123\""));

        h.insert(ETAG, HeaderValue::from_static("W/\"abc-123\""));
        assert_eq!(parse_strong_etag(&h), None);

        h.insert(ETAG, HeaderValue::from_static("w/\"abc-123\""));
        assert_eq!(parse_strong_etag(&h), None);

        h.insert(ETAG, HeaderValue::from_static("unquoted"));
        assert_eq!(parse_strong_etag(&h), None);

        h.insert(ETAG, HeaderValue::from_static("\"a\", \"b\""));
        assert_eq!(parse_strong_etag(&h), None);
    }

    #[test]
    fn parses_content_range_and_enforces_identity_encoding() {
        assert_eq!(
            parse_content_range("Bytes 0-65535/1048576").unwrap(),
            ContentRange::Span {
                start: 0,
                end: 65535,
                total: 1_048_576,
            }
        );
        assert_eq!(
            parse_content_range("bytes */0").unwrap(),
            ContentRange::Unsatisfied { total: 0 }
        );
        assert!(parse_content_range("bytes 0-10/*").is_err());
        assert!(parse_content_range("bytes 10-5/20").is_err());
        assert!(parse_content_range("bytes 0-20/20").is_err());

        let mut h = HeaderMap::new();
        assert!(validate_identity_content_encoding(&h).is_ok());
        h.insert(CONTENT_ENCODING, HeaderValue::from_static("identity"));
        assert!(validate_identity_content_encoding(&h).is_ok());
        h.insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        assert!(validate_identity_content_encoding(&h).is_err());
    }
}
