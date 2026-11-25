use super::*;
use crate::geo::footprint::{Ring, covers_polygon};
use crate::geo::geodesic::ring_signed_area_m2;
use crate::geo::geodesy::WGS84;
use glam::Vec3;

#[test]
fn parses_feature_collection_with_holes_and_round_trips_rfc7946_winding() {
    // Exterior intentionally given CW, hole intentionally given CCW — exporter
    // must normalize exterior to CCW and hole to CW per RFC 7946.
    let raw = r#"{
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": { "name": "Donut Parcel" },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[39.0, 47.0], [39.0, 48.0], [40.0, 48.0], [40.0, 47.0], [39.0, 47.0]],
                        [[39.3, 47.3], [39.7, 47.3], [39.7, 47.7], [39.3, 47.7], [39.3, 47.3]]
                    ]
                }
            }
        ]
    }"#;

    let fc = parse_geojson(raw).expect("valid geojson");
    assert_eq!(fc.features.len(), 1);
    assert_eq!(fc.features[0].name.as_deref(), Some("Donut Parcel"));
    let GeoJsonGeometry::Polygon(ref poly) = fc.features[0].geometry else {
        panic!("expected polygon");
    };
    assert!(covers_polygon(poly, 47.1, 39.1));
    assert!(!covers_polygon(poly, 47.5, 39.5));
    assert!(fc.bounds().is_some());

    let exported = export_geojson_feature_collection(&fc);
    let reparsed = parse_geojson(&exported).expect("reparse exported geojson");
    let GeoJsonGeometry::Polygon(ref poly2) = reparsed.features[0].geometry else {
        panic!("expected polygon");
    };
    assert_eq!(poly2.exterior.points.first(), poly2.exterior.points.last());
    assert!(ring_signed_area_m2(&poly2.exterior.points) > 0.0);
    assert_eq!(poly2.holes.len(), 1);
    assert_eq!(poly2.holes[0].points.first(), poly2.holes[0].points.last());
    assert!(ring_signed_area_m2(&poly2.holes[0].points) < 0.0);
}

#[test]
fn rejects_invalid_geojson_nesting_and_out_of_range_positions() {
    let nested_fc =
        r#"{"type":"FeatureCollection","features":[{"type":"FeatureCollection","features":[]}]}"#;
    assert!(matches!(
        parse_geojson(nested_fc),
        Err(GeoJsonError::Structure(_))
    ));

    let nested_feature = r#"{"type":"Feature","properties":{},"geometry":{"type":"Feature","properties":{},"geometry":null}}"#;
    assert!(matches!(
        parse_geojson(nested_feature),
        Err(GeoJsonError::Structure(_))
    ));

    let huge_lon = r#"{"type":"Point","coordinates":[1e200, 10.0]}"#;
    assert!(matches!(
        parse_geojson(huge_lon),
        Err(GeoJsonError::Structure(_))
    ));

    let tiny_ring = r#"{"type":"Polygon","coordinates":[[[0,0],[1,1]]]}"#;
    assert!(matches!(
        parse_geojson(tiny_ring),
        Err(GeoJsonError::Structure(_))
    ));
}

#[test]
fn tessellates_holed_concave_and_continental_polygons_with_subdivision_and_rtc_clusters() {
    let continental = Polygon::with_holes(
        Ring {
            points: vec![
                GeoPoint::new(20.0, -20.0),
                GeoPoint::new(20.0, 90.0),
                GeoPoint::new(75.0, 90.0),
                GeoPoint::new(50.0, 35.0), // concave indentation
                GeoPoint::new(75.0, -20.0),
            ],
        },
        vec![Ring {
            points: vec![
                GeoPoint::new(30.0, 10.0),
                GeoPoint::new(40.0, 10.0),
                GeoPoint::new(40.0, 25.0),
                GeoPoint::new(30.0, 25.0),
            ],
        }],
    );

    let fc = GeoJsonFeatureCollection {
        features: vec![GeoJsonFeature::new(
            Some("Continental".to_string()),
            GeoJsonGeometry::Polygon(continental),
        )],
    };

    let max_chord = 4.0;
    let mesh = VectorMesh::from_collection(&fc, max_chord);
    assert!(mesh.fill_triangle_count() > 100);
    assert!(mesh.line_segment_count() > 10);

    for cluster in &mesh.clusters {
        for tri in cluster.fill_indices.chunks_exact(3) {
            let v0 = cluster.fill_vertices[tri[0] as usize];
            let v1 = cluster.fill_vertices[tri[1] as usize];
            let v2 = cluster.fill_vertices[tri[2] as usize];
            let p0 = Vec3::from_array(v0.rel_pos).as_dvec3();
            let p1 = Vec3::from_array(v1.rel_pos).as_dvec3();
            let p2 = Vec3::from_array(v2.rel_pos).as_dvec3();
            let n0 = Vec3::from_array(v0.normal).as_dvec3();
            let n1 = Vec3::from_array(v1.normal).as_dvec3();
            let n2 = Vec3::from_array(v2.normal).as_dvec3();
            assert!((n0.length() - 1.0).abs() < 1e-4);
            let face = (p1 - p0).cross(p2 - p0);
            assert!(face.dot(n0 + n1 + n2) >= 0.0, "inward triangle winding");

            let max_edge_m = p0.distance(p1).max(p1.distance(p2)).max(p2.distance(p0));
            assert!(
                max_edge_m < 550_000.0,
                "interior edge was not subdivided: {max_edge_m} m"
            );
        }
    }
}

#[test]
fn tessellates_polar_antimeridian_and_over_hemisphere_polygons_accurately() {
    let polar = Polygon::new(Ring {
        points: vec![
            GeoPoint::new(82.0, 0.0),
            GeoPoint::new(82.0, 90.0),
            GeoPoint::new(82.0, 180.0),
            GeoPoint::new(82.0, -90.0),
        ],
    });
    let antimeridian = Polygon::new(Ring {
        points: vec![
            GeoPoint::new(50.0, 172.0),
            GeoPoint::new(50.0, -172.0),
            GeoPoint::new(60.0, -172.0),
            GeoPoint::new(60.0, 172.0),
        ],
    });
    assert!(covers_polygon(&polar, 89.5, 45.0));
    assert!(covers_polygon(&polar, 90.0, 0.0));
    assert!(!covers_polygon(&polar, 70.0, 45.0));
    assert!(covers_polygon(&antimeridian, 55.0, 179.0));
    assert!(covers_polygon(&antimeridian, 55.0, -179.0));

    // >hemisphere RFC 7946 CCW cap: boundary at lat = -20 deg going East,
    // enclosing the Northern Hemisphere + equatorial band down to -20 deg
    // (radius 110 deg > 90 deg, covering ~67% of Earth's surface).
    let over_hemisphere = Polygon::new(Ring {
        points: vec![
            GeoPoint::new(-20.0, 0.0),
            GeoPoint::new(-20.0, 90.0),
            GeoPoint::new(-20.0, 180.0),
            GeoPoint::new(-20.0, -90.0),
        ],
    });
    let fc = GeoJsonFeatureCollection {
        features: vec![GeoJsonFeature::new(
            Some("Over-hemisphere cap".to_string()),
            GeoJsonGeometry::Polygon(over_hemisphere),
        )],
    };
    let mesh = VectorMesh::from_collection(&fc, 10.0);
    assert!(mesh.fill_triangle_count() > 50);

    // Verify triangles cover the North Pole (z > 0) and NOT the South Pole (z < -0.95 R):
    // if complement inversion happened, triangles would cluster around z = -R instead.
    let mut max_z = f64::NEG_INFINITY;
    let mut min_z = f64::INFINITY;
    for c in &mesh.clusters {
        for v in &c.fill_vertices {
            assert!(v.rel_pos.iter().all(|x| x.is_finite()));
            assert!(v.normal.iter().all(|x| x.is_finite()));
            let wz = c.origin_ecef_m.z + f64::from(v.rel_pos[2]);
            max_z = max_z.max(wz);
            min_z = min_z.min(wz);
        }
    }
    assert!(
        max_z > 5_000_000.0,
        "expected >hemisphere cap to cover Northern Hemisphere, max_z={max_z}"
    );
    assert!(
        min_z > -3_000_000.0,
        "over-hemisphere cap leaked into South Polar complement, min_z={min_z}"
    );
}

#[test]
fn per_cluster_rtc_preserves_submillimetre_precision_across_continents_and_tiny_drone_polygons() {
    // Two tiny ~5 cm x 5 cm survey markers placed 6,400 km apart (Moscow & Vladivostok).
    let d_deg = 5e-7; // ~5.5 cm
    let marker = |lat: f64, lon: f64| {
        GeoJsonFeature::new(
            None,
            GeoJsonGeometry::Polygon(Polygon::new(Ring {
                points: vec![
                    GeoPoint::new(lat, lon),
                    GeoPoint::new(lat, lon + d_deg),
                    GeoPoint::new(lat + d_deg, lon + d_deg),
                    GeoPoint::new(lat + d_deg, lon),
                ],
            })),
        )
    };
    let fc = GeoJsonFeatureCollection {
        features: vec![marker(55.75, 37.62), marker(43.115, 131.885)],
    };
    let mesh = VectorMesh::from_collection(&fc, DEFAULT_VECTOR_MAX_CHORD_DEG);
    // Must separate into 2 localized RTC clusters, preserve all 4 triangles, and
    // deduplicate shared vertices inside each cluster (4 corners vs 6 indices).
    assert_eq!(mesh.clusters.len(), 2);
    assert_eq!(mesh.fill_triangle_count(), 4);
    for c in &mesh.clusters {
        assert_eq!(c.fill_indices.len(), 6);
        assert_eq!(c.fill_vertices.len(), 4);
        for v in &c.fill_vertices {
            let offset_m = Vec3::from_array(v.rel_pos).as_dvec3().length();
            assert!(
                offset_m < 10.0,
                "expected localized RTC offset < 10 m, got {offset_m} m"
            );
        }
        // Verify reconstructed f64 ECEF stays on WGS84 within < 1 mm.
        let p0 = c.origin_ecef_m + Vec3::from_array(c.fill_vertices[0].rel_pos).as_dvec3();
        let geo = WGS84.from_ecef(p0);
        assert!(
            geo.height_m.abs() < 1e-3,
            "height drift: {} m",
            geo.height_m
        );
    }
}
