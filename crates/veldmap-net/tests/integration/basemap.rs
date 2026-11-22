use veldmap_core::tile::{GlobeScheme, PlanetaryScheme, QuadTileId};
use veldmap_net::{
    CustomBasemapConfig, LiveBasemapSpec, NetError, is_valid_iso_date, sanitize_url_for_display,
};

#[test]
fn gibs_and_eox_specs_enforce_polar_cap_and_max_level_contracts() {
    let gibs = LiveBasemapSpec::nasa_gibs_viirs("2024-07-01").expect("valid date");
    assert_eq!(gibs.scheme(), GlobeScheme::WebMercator);
    assert_eq!(gibs.max_level(), 9);
    let url = gibs
        .resolve_url(QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 3,
            x: 5,
            y: 2,
        })
        .expect("valid gibs tile");
    assert!(url.contains("/2024-07-01/GoogleMapsCompatible_Level9/3/2/5.jpg"));

    assert!(matches!(
        gibs.resolve_url(QuadTileId {
            scheme: PlanetaryScheme::PolarCap { north: true },
            level: 0,
            x: 0,
            y: 0,
        }),
        Err(NetError::PolarCapForbidden)
    ));

    assert!(matches!(
        gibs.resolve_url(QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level: 0,
            x: 0,
            y: 0,
        }),
        Err(NetError::SchemeMismatch { .. })
    ));
    assert!(matches!(
        gibs.resolve_url(QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 1,
            x: 2,
            y: 0,
        }),
        Err(NetError::TileOutOfRange { .. })
    ));

    assert!(matches!(
        gibs.resolve_url(QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 10,
            x: 0,
            y: 0,
        }),
        Err(NetError::LevelAboveMax {
            level: 10,
            max_level: 9
        })
    ));

    let eox = LiveBasemapSpec::eox_sentinel2_cloudless();
    assert_eq!(eox.max_level(), 14);
    assert!(eox.attribution().contains("CC BY 4.0"));
    assert!(
        eox.resolve_url(QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 14,
            x: 0,
            y: 0,
        })
        .is_ok()
    );
    assert!(matches!(
        eox.resolve_url(QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 15,
            x: 0,
            y: 0,
        }),
        Err(NetError::LevelAboveMax { .. })
    ));
}

#[test]
fn custom_xyz_tms_handles_subdomains_flip_y_ipv6_encoding_and_redaction() {
    let base_cfg = CustomBasemapConfig {
        name: "Custom",
        url_template: "https://alice:pw@{s}.tiles.example.org/{z}/{x}/{y}.png?token={key}&inline=secret42",
        api_key: "sec+ret/key 99",
        tms_flip_y: true,
        max_level: 6,
        attribution: "Example",
    };
    let spec = LiveBasemapSpec::custom(base_cfg.clone()).expect("build custom");
    let url = spec
        .resolve_url(QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 2,
            x: 1,
            y: 0,
        })
        .expect("resolve custom url");
    assert_eq!(
        url,
        "https://alice:pw@b.tiles.example.org/2/1/3.png?token=sec%2Bret%2Fkey%2099&inline=secret42"
    );

    // Debug output and redact_secrets must redact API keys, inline query tokens,
    // AND user:pass@ authority credentials.
    let dbg = format!("{spec:?}");
    assert!(!dbg.contains("sec+ret"));
    assert!(!dbg.contains("alice:pw"));
    assert!(!dbg.contains("secret42"), "query string must be redacted");
    assert!(!spec.cache_key().contains("sec"));
    assert_eq!(
        spec.redact_secrets(&url),
        "https://<redacted>@b.tiles.example.org/2/1/3.png?<redacted>"
    );
    assert_eq!(
        spec.redact_secrets("err: sec%2bret%2fkey%2099"),
        "err: <redacted>"
    );
    assert_eq!(
        sanitize_url_for_display("http://user:pw@[::1]:8080/{z}/{x}/{y}.png?tok=abc"),
        "http://<redacted>@[::1]:8080/{z}/{x}/{y}.png?<redacted>"
    );

    let higher_max = LiveBasemapSpec::custom(CustomBasemapConfig {
        max_level: 14,
        ..base_cfg.clone()
    })
    .expect("build higher max");
    assert_eq!(spec.cache_key(), higher_max.cache_key());
    assert_eq!(spec.source_id(), higher_max.source_id());

    // Bracketed IPv6 literal templates are accepted cleanly.
    let ipv6 = LiveBasemapSpec::custom(CustomBasemapConfig {
        url_template: "http://[::1]:8080/{z}/{x}/{y}.png",
        tms_flip_y: false,
        api_key: "",
        ..base_cfg.clone()
    })
    .expect("ipv6 template");
    assert_eq!(
        ipv6.resolve_url(QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 1,
            x: 0,
            y: 1,
        })
        .expect("ipv6 url"),
        "http://[::1]:8080/1/0/1.png"
    );

    let neg_y = LiveBasemapSpec::custom(CustomBasemapConfig {
        url_template: "https://tiles.example.org/{z}/{x}/{-y}.png",
        tms_flip_y: false,
        api_key: "",
        ..base_cfg.clone()
    })
    .expect("neg y");
    assert_eq!(
        neg_y
            .resolve_url(QuadTileId {
                scheme: PlanetaryScheme::WebMercator,
                level: 2,
                x: 1,
                y: 0,
            })
            .expect("neg y url"),
        "https://tiles.example.org/2/1/3.png"
    );

    for bad in [
        "ftp://example.com/{z}/{x}/{y}.png",
        "https://{z}/{x}/{y}.png",
        "https://{-y}/{z}/{x}/{y}.png",
        "https://example.com/{z}/{x}/{y}.png bad",
    ] {
        assert!(
            matches!(
                LiveBasemapSpec::custom(CustomBasemapConfig {
                    url_template: bad,
                    ..base_cfg.clone()
                }),
                Err(NetError::InvalidUrlTemplate(_))
            ),
            "expected InvalidUrlTemplate for {bad:?}"
        );
    }
}

#[test]
fn iso_date_validation_accepts_valid_and_rejects_malformed() {
    assert!(is_valid_iso_date("2024-07-01"));
    assert!(is_valid_iso_date("2024-02-29"));
    assert!(!is_valid_iso_date("2023-02-29"));
    assert!(!is_valid_iso_date("2024-02-31"));
    assert!(!is_valid_iso_date("2024-13-01"));
    assert!(!is_valid_iso_date("not-a-date"));
    assert!(matches!(
        LiveBasemapSpec::nasa_gibs_viirs("bad-date"),
        Err(NetError::InvalidDate(_))
    ));
}
