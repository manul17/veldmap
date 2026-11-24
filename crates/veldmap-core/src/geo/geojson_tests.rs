use super::*;
use crate::geo::footprint::{covers_polygon, frame_polygons};

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
    // Footprint covers exterior ring but excludes hole.
    assert!(covers_polygon(poly, 47.1, 39.1));
    assert!(!covers_polygon(poly, 47.5, 39.5));
    assert!(frame_polygons(std::slice::from_ref(poly)).is_some());

    // Export and re-parse: verify RFC 7946 winding (exterior CCW > 0, hole CW < 0) and closure.
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
fn tessellates_holed_concave_and_continental_hemispheric_polygons_with_interior_subdivision() {
    // 1. Large continental concave polygon spanning > 100 degrees across North Pole / high latitudes
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
        features: vec![GeoJsonFeature {
            name: Some("Continental".to_string()),
            properties: Map::new(),
            geometry: GeoJsonGeometry::Polygon(continental),
        }],
    };

    let max_chord = 4.0;
    let mesh = VectorMesh::from_collection(&fc, max_chord);
    assert!(!mesh.fill_indices.is_empty());
    assert!(!mesh.line_vertices.is_empty());
    // Interior subdivision must emit many more triangles than raw 9 boundary vertices (~9 tris).
    assert!(
        mesh.fill_indices.len() / 3 > 100,
        "expected subdivided interior triangles, got {}",
        mesh.fill_indices.len() / 3
    );

    // Every triangle must have outward CCW winding, unit outward normals, and
    // bounded chord length (no triangle spanning 60 deg straight through the Earth).
    for tri in mesh.fill_indices.chunks_exact(3) {
        let v0 = mesh.fill_vertices[tri[0] as usize];
        let v1 = mesh.fill_vertices[tri[1] as usize];
        let v2 = mesh.fill_vertices[tri[2] as usize];
        let p0 = DVec3::new(
            v0.rel_pos[0] as f64,
            v0.rel_pos[1] as f64,
            v0.rel_pos[2] as f64,
        );
        let p1 = DVec3::new(
            v1.rel_pos[0] as f64,
            v1.rel_pos[1] as f64,
            v1.rel_pos[2] as f64,
        );
        let p2 = DVec3::new(
            v2.rel_pos[0] as f64,
            v2.rel_pos[1] as f64,
            v2.rel_pos[2] as f64,
        );
        let n0 = DVec3::new(
            v0.normal[0] as f64,
            v0.normal[1] as f64,
            v0.normal[2] as f64,
        );
        let n1 = DVec3::new(
            v1.normal[0] as f64,
            v1.normal[1] as f64,
            v1.normal[2] as f64,
        );
        let n2 = DVec3::new(
            v2.normal[0] as f64,
            v2.normal[1] as f64,
            v2.normal[2] as f64,
        );
        assert!((n0.length() - 1.0).abs() < 1e-4);
        let face = (p1 - p0).cross(p2 - p0);
        assert!(face.dot(n0 + n1 + n2) >= 0.0, "inward triangle winding");

        // Edge chord length bounded by ~max_chord + slack (4 deg ~ 450 km, allow 550 km).
        let max_edge_m = p0.distance(p1).max(p1.distance(p2)).max(p2.distance(p0));
        assert!(
            max_edge_m < 600_000.0,
            "interior edge was not subdivided: {max_edge_m} m"
        );
    }
}

#[test]
fn tessellates_polar_antimeridian_and_over_hemispheric_polygons_without_blowup() {
    // Polar cap polygon surrounding the North Pole
    let polar = Polygon::new(Ring {
        points: vec![
            GeoPoint::new(82.0, 0.0),
            GeoPoint::new(82.0, 90.0),
            GeoPoint::new(82.0, 180.0),
            GeoPoint::new(82.0, -90.0),
        ],
    });
    // Antimeridian-crossing polygon
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

    let fc = GeoJsonFeatureCollection {
        features: vec![
            GeoJsonFeature {
                name: None,
                properties: Map::new(),
                geometry: GeoJsonGeometry::Polygon(polar),
            },
            GeoJsonFeature {
                name: None,
                properties: Map::new(),
                geometry: GeoJsonGeometry::Polygon(antimeridian),
            },
        ],
    };
    let mesh = VectorMesh::from_collection(&fc, 2.0);
    assert!(!mesh.fill_indices.is_empty());
    assert!(mesh.fill_vertices.iter().all(|v| {
        v.rel_pos.iter().all(|c| c.is_finite()) && v.normal.iter().all(|c| c.is_finite())
    }));
}
