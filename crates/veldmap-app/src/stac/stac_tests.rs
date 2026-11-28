use std::sync::Arc;

use veldmap_core::geo::{GeoJsonGeometry, GeoPoint, Polygon, Ring};
use veldmap_core::sync::CancelToken;
use veldmap_net::StacSearchResult;

use crate::render::DrawVectorId;
use crate::storage::settings::{StacAuthKind, StacSettings};

use super::aoi::{footprint_bounds_from_bbox, polygon_geometry_from_bbox};
use super::*;

fn test_session() -> StacSession {
    StacSession::new(None, Arc::new(|| {}), CancelToken::new())
}

#[test]
fn bbox_polygon_and_bounds_handle_antimeridian_and_non_finite() {
    assert!(polygon_geometry_from_bbox([f64::NAN, 0.0, 1.0, 1.0]).is_none());
    assert!(footprint_bounds_from_bbox([f64::INFINITY, 0.0, 1.0, 1.0]).is_none());

    // Antimeridian crossing: [170, -20, -170, 20] splits into 2 polygons
    // and centers on 180 deg (not 0 deg Prime Meridian).
    let geom = polygon_geometry_from_bbox([170.0, -20.0, -170.0, 20.0]).expect("antimeridian");
    match geom {
        GeoJsonGeometry::MultiPolygon(polys) => assert_eq!(polys.len(), 2),
        other => panic!("expected MultiPolygon, got {other:?}"),
    }
    let bounds = footprint_bounds_from_bbox([170.0, -20.0, -170.0, 20.0]).expect("bounds");
    assert!(
        bounds.lon.abs() > 179.0,
        "antimeridian midpoint must be near +/-180, got {}",
        bounds.lon
    );
    assert!(bounds.radius_deg < 30.0);

    // Degenerate zero-width half collapses cleanly to single Polygon.
    let half = polygon_geometry_from_bbox([170.0, 0.0, -180.0, 10.0]).expect("one-sided");
    assert!(matches!(half, GeoJsonGeometry::Polygon(_)));
}

#[test]
fn cancel_increments_seq_so_queued_search_result_is_ignored() {
    let mut session = test_session();
    session.seed_queued_search_for_test(
        5,
        "https://example.org/v1",
        StacSearchResult {
            items: Vec::new(),
            matched: Some(7),
        },
    );

    session.cancel();
    assert!(!session.poll_results());
    assert!(session.matched().is_none());
}

#[test]
fn resolve_cog_auth_enforces_origin_presigned_and_result_coupling() {
    let mut stac_settings = StacSettings {
        custom_endpoint: "https://stac.private.example/v1".to_owned(),
        auth_kind: StacAuthKind::Bearer,
        auth_secret: "secret-bearer".to_owned(),
        ..StacSettings::default()
    };

    let mut session = test_session();
    // Public search result must NEVER attach custom credentials.
    session.set_result_auth_context_for_test(false, None);
    let (u, h) = session.resolve_cog_auth(
        "https://sentinel-cogs.s3.amazonaws.com/a.tif",
        StacCogAuthMode::FromSearchResult,
        &stac_settings,
    );
    assert_eq!(u, "https://sentinel-cogs.s3.amazonaws.com/a.tif");
    assert!(h.is_empty());

    // Custom search result: same-origin gets Bearer header, cross-origin does not.
    session.set_result_auth_context_for_test(true, Some("https://stac.private.example/v1"));
    let (_, same_hdrs) = session.resolve_cog_auth(
        "https://stac.private.example/assets/a.tif",
        StacCogAuthMode::FromSearchResult,
        &stac_settings,
    );
    assert_eq!(same_hdrs.len(), 1);
    let (_, cross_hdrs) = session.resolve_cog_auth(
        "https://other-bucket.s3.amazonaws.com/a.tif",
        StacCogAuthMode::FromSearchResult,
        &stac_settings,
    );
    assert!(cross_hdrs.is_empty());

    // Already-presigned URL never gets dual auth or double SAS.
    stac_settings.auth_kind = StacAuthKind::SasQuery;
    stac_settings.auth_secret = "sv=2024&sig=custom".to_owned();
    let presigned = "https://blob.example/a.tif?X-Amz-Signature=existing";
    let (u2, h2) =
        session.resolve_cog_auth(presigned, StacCogAuthMode::DirectCustom, &stac_settings);
    assert_eq!(u2, presigned);
    assert!(h2.is_empty());

    // Unsigned cross-origin URL with SasQuery DOES receive SAS token.
    let (u3, h3) = session.resolve_cog_auth(
        "https://blob.example/a.tif",
        StacCogAuthMode::FromSearchResult,
        &stac_settings,
    );
    assert_eq!(u3, "https://blob.example/a.tif?sv=2024&sig=custom");
    assert!(h3.is_empty());
}

#[test]
fn preview_preserves_committed_aoi_until_finished_and_display_bbox_keeps_antimeridian() {
    let mut session = test_session();
    assert!(session.set_aoi_bbox([170.0, -10.0, -170.0, 10.0], "Pacific".to_owned()));
    assert_eq!(session.aoi_bbox(), Some([170.0, -10.0, -170.0, 10.0]));
    assert!(session.set_aoi_preview_bbox([10.0, 10.0, 20.0, 20.0]));
    // Preview must not clobber committed AOI geometry/bbox.
    assert_eq!(session.aoi_bbox(), Some([170.0, -10.0, -170.0, 10.0]));
    let rev_before = session.rev();
    assert!(session.clear_aoi_preview());
    assert_ne!(
        session.rev(),
        rev_before,
        "clearing AOI preview must bump rev so DrawSnapshot rebuilds"
    );
    let mut layers = Vec::new();
    session.push_draw_layers(&mut layers);
    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].id, DrawVectorId::StacAoi);
}

#[test]
fn ui_errors_do_not_bump_rev_and_set_aoi_geometry_caches_bbox() {
    let mut session = test_session();
    let rev0 = session.rev();
    session.set_error("transient error".to_owned());
    assert_eq!(session.last_error(), Some("transient error"));
    assert_eq!(session.rev(), rev0);
    session.clear_error();
    assert!(session.last_error().is_none());
    assert_eq!(session.rev(), rev0);

    let poly = GeoJsonGeometry::Polygon(Polygon::new(Ring {
        points: vec![
            GeoPoint::new(45.0, 7.0),
            GeoPoint::new(45.0, 9.0),
            GeoPoint::new(47.0, 9.0),
            GeoPoint::new(47.0, 7.0),
            GeoPoint::new(45.0, 7.0),
        ],
    }));
    session.set_aoi_geometry(poly, "Alps".to_owned());
    assert_ne!(session.rev(), rev0);
    assert_eq!(session.aoi_bbox(), Some([7.0, 45.0, 9.0, 47.0]));
}

#[test]
fn hovering_already_selected_item_skips_highlight_rebuild() {
    let mut session = test_session();
    let mk_item = |id: &str, bbox: [f64; 4]| veldmap_net::StacItem {
        id: id.to_owned(),
        collection: None,
        self_href: None,
        datetime: None,
        cloud_cover: None,
        platform: None,
        mgrs_tile: None,
        processing_level: None,
        bbox: Some(bbox),
        geometry: None,
        assets: Vec::new(),
    };
    session.seed_queued_search_for_test(
        1,
        "https://example.org/v1",
        StacSearchResult {
            items: vec![
                mk_item("a", [7.0, 45.0, 8.0, 46.0]),
                mk_item("b", [8.0, 46.0, 9.0, 47.0]),
            ],
            matched: Some(2),
        },
    );
    assert!(session.poll_results());
    let rev_selected_0 = session.rev();

    // Hovering and unhovering the already-selected item 0 notifies UI (true)
    // without rebuilding the highlight mesh or bumping rev.
    assert!(session.set_hovered(Some(0)));
    assert_eq!(session.rev(), rev_selected_0);
    assert!(session.set_hovered(None));
    assert_eq!(session.rev(), rev_selected_0);

    // Hovering a different item 1 DOES rebuild highlight and bump rev.
    assert!(session.set_hovered(Some(1)));
    assert_ne!(session.rev(), rev_selected_0);
}
