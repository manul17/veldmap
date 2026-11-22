use veldmap_core::tile::{QuadTileId, TileScheme};
use veldmap_net::{CustomBasemapConfig, LiveBasemapSpec, NetError, is_valid_iso_date};

#[test]
fn gibs_and_eox_specs_enforce_polar_cap_and_max_level_contracts() {
    let gibs = LiveBasemapSpec::nasa_gibs_viirs("2024-07-01").expect("valid date");
    assert_eq!(gibs.scheme, TileScheme::WebMercator);
    assert_eq!(gibs.max_level, 9);
    let url = gibs
        .resolve_url(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 3,
            x: 5,
            y: 2,
        })
        .expect("valid gibs tile");
    assert!(url.contains("/2024-07-01/GoogleMapsCompatible_Level9/3/2/5.jpg"));

    assert!(matches!(
        gibs.resolve_url(QuadTileId {
            scheme: TileScheme::PolarCap { north: true },
            level: 0,
            x: 0,
            y: 0,
        }),
        Err(NetError::PolarCapForbidden)
    ));

    assert!(matches!(
        gibs.resolve_url(QuadTileId {
            scheme: TileScheme::Geodetic,
            level: 0,
            x: 0,
            y: 0,
        }),
        Err(NetError::SchemeMismatch { .. })
    ));
    assert!(matches!(
        gibs.resolve_url(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 1,
            x: 2,
            y: 0,
        }),
        Err(NetError::TileOutOfRange { .. })
    ));

    assert!(matches!(
        gibs.resolve_url(QuadTileId {
            scheme: TileScheme::WebMercator,
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
    assert_eq!(eox.max_level, 14);
    assert!(eox.attribution.contains("CC BY 4.0"));
    assert!(
        eox.resolve_url(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 14,
            x: 0,
            y: 0,
        })
        .is_ok()
    );
    assert!(matches!(
        eox.resolve_url(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 15,
            x: 0,
            y: 0,
        }),
        Err(NetError::LevelAboveMax { .. })
    ));
}

#[test]
fn custom_xyz_tms_handles_subdomains_flip_y_encoding_and_redaction() {
    let base_cfg = CustomBasemapConfig {
        name: "Custom",
        url_template: "https://{s}.tiles.example.org/{z}/{x}/{y}.png?token={key}&extra=1",
        api_key: "sec+ret/key 99",
        tms_flip_y: true,
        max_level: 6,
        attribution: "Example",
    };
    let spec = LiveBasemapSpec::custom(base_cfg.clone()).expect("build custom");
    let url = spec
        .resolve_url(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 2,
            x: 1,
            y: 0,
        })
        .expect("resolve custom url");
    assert_eq!(
        url,
        "https://b.tiles.example.org/2/1/3.png?token=sec%2Bret%2Fkey%2099&extra=1"
    );
    let dbg = format!("{spec:?}");
    assert!(!dbg.contains("sec+ret"));
    assert!(!dbg.contains("extra=1"), "query string must be redacted");
    assert!(!spec.cache_key.contains("sec"));
    assert_eq!(
        spec.redact_secrets(&url),
        "https://b.tiles.example.org/2/1/3.png?token=<redacted>&extra=1"
    );
    assert_eq!(
        spec.redact_secrets("err: sec%2bret%2fkey%2099"),
        "err: <redacted>"
    );

    let higher_max = LiveBasemapSpec::custom(CustomBasemapConfig {
        max_level: 14,
        ..base_cfg.clone()
    })
    .expect("build higher max");
    assert_eq!(spec.cache_key, higher_max.cache_key);
    assert_eq!(spec.source_id, higher_max.source_id);

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
                scheme: TileScheme::WebMercator,
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
