use super::*;
use crate::storage::settings::StacAuthKind;

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
    session.active_seq = 5;
    session.searching = true;
    session
        .tx
        .send(SearchDoneMsg {
            seq: 5,
            uses_custom_auth: false,
            catalog_endpoint: "https://example.org/v1".to_owned(),
            outcome: Ok(StacSearchResult {
                items: Vec::new(),
                matched: Some(7),
            }),
        })
        .expect("send");

    session.cancel();
    assert!(!session.poll_results());
    assert!(session.matched().is_none());
}

#[test]
fn resolve_cog_auth_enforces_origin_presigned_and_result_coupling() {
    let mut settings = AppSettings::default();
    settings.stac.custom_endpoint = "https://stac.private.example/v1".to_owned();
    settings.stac.auth_kind = StacAuthKind::Bearer;
    settings.stac.auth_secret = "secret-bearer".to_owned();

    let mut session = test_session();
    // Public search result must NEVER attach custom credentials.
    session.result_uses_custom_auth = false;
    let (u, h) = resolve_cog_auth(
        "https://sentinel-cogs.s3.amazonaws.com/a.tif",
        StacCogAuthMode::FromSearchResult,
        &session,
        &settings,
    );
    assert_eq!(u, "https://sentinel-cogs.s3.amazonaws.com/a.tif");
    assert!(h.is_empty());

    // Custom search result: same-origin gets Bearer header, cross-origin does not.
    session.result_uses_custom_auth = true;
    session.result_catalog_endpoint = Some("https://stac.private.example/v1".to_owned());
    let (_, same_hdrs) = resolve_cog_auth(
        "https://stac.private.example/assets/a.tif",
        StacCogAuthMode::FromSearchResult,
        &session,
        &settings,
    );
    assert_eq!(same_hdrs.len(), 1);
    let (_, cross_hdrs) = resolve_cog_auth(
        "https://other-bucket.s3.amazonaws.com/a.tif",
        StacCogAuthMode::FromSearchResult,
        &session,
        &settings,
    );
    assert!(cross_hdrs.is_empty());

    // Already-presigned URL never gets dual auth or double SAS.
    settings.stac.auth_kind = StacAuthKind::SasQuery;
    settings.stac.auth_secret = "sv=2024&sig=custom".to_owned();
    let presigned = "https://blob.example/a.tif?X-Amz-Signature=existing";
    let (u2, h2) = resolve_cog_auth(
        presigned,
        StacCogAuthMode::DirectCustom,
        &session,
        &settings,
    );
    assert_eq!(u2, presigned);
    assert!(h2.is_empty());

    // Unsigned cross-origin URL with SasQuery DOES receive SAS token.
    let (u3, h3) = resolve_cog_auth(
        "https://blob.example/a.tif",
        StacCogAuthMode::FromSearchResult,
        &session,
        &settings,
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
    assert!(session.clear_aoi_preview());
    let mut layers = Vec::new();
    session.push_draw_layers(&mut layers);
    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].id, DrawVectorId::StacAoi);
}
