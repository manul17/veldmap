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
    assert_eq!(extract_host("http://\u{00e9}/0/0/0.png"), "xn--9ca");
    assert_eq!(
        extract_host("http://user:pass@127.0.0.1:8080/0/0/0.png?key=1"),
        "127.0.0.1:8080"
    );
    assert_eq!(extract_host("http://[::1]:8080/0/0/0.png"), "[::1]:8080");
    assert_eq!(
        parse_http_authority("http://user:pw@[::1]:8080/0/0/0.png"),
        Some("[::1]")
    );

    // Reject invalid ports and malformed IPv6 brackets.
    assert!(parse_http_authority("http://example.com:notaport/0/0/0.png").is_none());
    assert!(parse_http_authority("http://example.com:70000/0/0/0.png").is_none());
    assert!(parse_http_authority("http://example.com:/0/0/0.png").is_none());
    assert!(parse_http_authority("http://[invalid]/0/0/0.png").is_none());
    assert!(parse_http_authority("http://[::1]:bad/0/0/0.png").is_none());
}

#[test]
fn canonical_endpoint_origin_and_query_fragment_helpers_work() {
    assert_eq!(
        sanitize_url_for_persistence("  HTTPS://user:pw@Example.COM:443/path/a.tif?sig=abc#frag  ")
            .as_deref(),
        Some("https://example.com/path/a.tif")
    );
    assert_eq!(
        sanitize_url_for_persistence("https://user:pass@[::1]:443/a.tif?sig=1#f").as_deref(),
        Some("https://[::1]/a.tif")
    );
    assert_eq!(
        sanitize_url_for_persistence("http://user@example.com:80/a.tif").as_deref(),
        Some("http://example.com/a.tif")
    );
    assert_eq!(
        sanitize_url_for_persistence("https://@example.com/a.tif").as_deref(),
        Some("https://example.com/a.tif")
    );
    assert_eq!(
        sanitize_url_for_persistence("https://u%3A1:p%402@example.com/a.tif?token=1").as_deref(),
        Some("https://example.com/a.tif")
    );
    assert_eq!(
        sanitize_url_for_persistence("https://example.com:443/a.tif"),
        sanitize_url_for_persistence("https://example.com/a.tif")
    );
    assert_eq!(
        sanitize_url_for_persistence("http://example.com:80/a.tif").as_deref(),
        Some("http://example.com/a.tif")
    );
    assert_eq!(
        sanitize_url_for_persistence("https://example.com:8443/a.tif").as_deref(),
        Some("https://example.com:8443/a.tif")
    );
    assert_eq!(
        sanitize_url_for_persistence("https://example.com"),
        Some("https://example.com/".to_owned())
    );
    assert!(sanitize_url_for_persistence("https://example.com/a b.tif").is_none());
    assert!(sanitize_url_for_persistence("ftp://example.com/a.tif").is_none());
    // Reject slash-missing special-scheme URLs (which WHATWG Url::parse would
    // otherwise normalize while bypassing `://` credential redaction) and
    // internal ASCII control characters.
    for malformed in [
        "http:example.com/v1",
        "https:/example.com/v1",
        "https:user:secret@example.com/a.tif",
        "https://example.com/a\tb.tif",
        "https://example.com/a\0b.tif",
        "https://example.com/a\x7fb.tif",
    ] {
        assert!(
            sanitize_url_for_persistence(malformed).is_none(),
            "sanitize_url_for_persistence must reject {malformed:?}"
        );
        assert!(
            !is_valid_http_endpoint(malformed),
            "is_valid_http_endpoint must reject {malformed:?}"
        );
        assert!(
            !same_http_origin(malformed, "https://example.com/v1"),
            "same_http_origin must reject {malformed:?}"
        );
    }

    assert!(is_valid_http_endpoint(
        "https://earth-search.aws.element84.com/v1"
    ));
    assert!(!is_valid_http_endpoint("https://example.com/v1\r\nX: y"));
    assert!(!is_valid_http_endpoint("https://example.com/with space"));

    assert!(same_http_origin(
        "HTTPS://Example.COM:443/v1",
        "https://example.com/assets/b04.tif"
    ));
    assert!(same_http_origin(
        "http://127.0.0.1:80/a",
        "http://127.0.0.1/b"
    ));
    assert!(!same_http_origin(
        "https://stac.example.com/v1",
        "https://other-bucket.s3.amazonaws.com/a.tif"
    ));
    assert!(!same_http_origin(
        "https://example.com/a",
        "http://example.com/a"
    ));

    assert_eq!(
        split_url_query_fragment("https://host/v1?k=1#sec"),
        ("https://host/v1", Some("k=1"), Some("sec"))
    );
    assert_eq!(
        split_url_query_fragment("https://host/v1#sec"),
        ("https://host/v1", None, Some("sec"))
    );
}
