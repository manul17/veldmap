//! User-defined GeoTIFF datum, linear-unit, geographic CRS, and projected CRS
//! resolution.

use veldmap_core::geo::{
    CrsSpec, Datum, DatumWarning, Ellipsoid, LinearUnit, Projection, System, WGS84, WGS84_DATUM,
    ellipsoid_from_epsg, known_datum_from_code,
};

use super::geokeys::{
    CT_ALBERS_EQUAL_AREA, CT_LAMBERT_AZIMUTHAL_EQUAL_AREA, CT_LAMBERT_CONF_CONIC_1SP,
    CT_LAMBERT_CONF_CONIC_2SP, CT_MERCATOR, CT_POLAR_STEREOGRAPHIC, CT_TRANSVERSE_MERCATOR,
    EPSG_UNIT_KILOMETRE, EPSG_WGS84_DATUM, GEOKEY_GEOG_ELLIPSOID, GEOKEY_GEOG_GEODETIC_DATUM,
    GEOKEY_GEOG_INV_FLATTENING, GEOKEY_GEOG_SEMI_MAJOR_AXIS, GEOKEY_GEOG_SEMI_MINOR_AXIS,
    GEOKEY_GEOG_TOWGS84, GEOKEY_PCS_CITATION, GEOKEY_PROJ_CENTER_EASTING, GEOKEY_PROJ_CENTER_LAT,
    GEOKEY_PROJ_CENTER_LONG, GEOKEY_PROJ_CENTER_NORTHING, GEOKEY_PROJ_COORD_TRANS,
    GEOKEY_PROJ_FALSE_EASTING, GEOKEY_PROJ_FALSE_NORTHING, GEOKEY_PROJ_FALSE_ORIGIN_EASTING,
    GEOKEY_PROJ_FALSE_ORIGIN_LAT, GEOKEY_PROJ_FALSE_ORIGIN_LONG, GEOKEY_PROJ_FALSE_ORIGIN_NORTHING,
    GEOKEY_PROJ_LINEAR_UNIT_SIZE, GEOKEY_PROJ_LINEAR_UNITS, GEOKEY_PROJ_NAT_ORIGIN_LAT,
    GEOKEY_PROJ_NAT_ORIGIN_LONG, GEOKEY_PROJ_SCALE_AT_CENTER, GEOKEY_PROJ_SCALE_AT_NAT_ORIGIN,
    GEOKEY_PROJ_STD_PARALLEL1, GEOKEY_PROJ_STD_PARALLEL2, GEOKEY_PROJ_STRAIGHT_VERT_POLE_LONG,
    GEOKEY_USER_DEFINED, GeoKeyParams, UNSAID_GEOKEY_CODES, WGS84_GEOGRAPHIC_CODES,
};

fn has_user_geographic_datum(geo_params: &GeoKeyParams) -> bool {
    let has_towgs84 = geo_params.get_f64_slice(GEOKEY_GEOG_TOWGS84).is_some();
    let has_user_geog = geo_params.geographic_crs == Some(GEOKEY_USER_DEFINED);
    let has_explicit_ellipsoid = geo_params.get_u16(GEOKEY_GEOG_ELLIPSOID).is_some()
        || geo_params.get_f64(GEOKEY_GEOG_SEMI_MAJOR_AXIS).is_some();
    has_towgs84 || (has_user_geog && has_explicit_ellipsoid)
}

fn non_wgs84_geographic_code(geo_params: &GeoKeyParams) -> Option<u16> {
    if let Some(code) = geo_params.geographic_crs.filter(|code| {
        !WGS84_GEOGRAPHIC_CODES.contains(code) && !UNSAID_GEOKEY_CODES.contains(code)
    }) {
        return Some(code);
    }
    geo_params
        .get_u16(GEOKEY_GEOG_GEODETIC_DATUM)
        .filter(|code| *code != EPSG_WGS84_DATUM && !UNSAID_GEOKEY_CODES.contains(code))
}

/// Resolve a non-WGS84 geographic CRS for tie-point grid bindings (`None`
/// implies default WGS 84 / EPSG:4326).
pub(super) fn resolve_geographic_crs(geo_params: &GeoKeyParams) -> Option<CrsSpec> {
    if has_user_geographic_datum(geo_params) {
        let (datum, warning) = resolve_user_datum(geo_params);
        return Some(CrsSpec::Geographic {
            datum,
            label: "User-defined geographic CRS".to_string(),
            warning,
        });
    }

    let code = non_wgs84_geographic_code(geo_params)?;
    let code_u32 = u32::from(code);
    if let Ok(crs) = CrsSpec::from_epsg(code_u32)
        && crs.is_geographic()
    {
        return Some(crs);
    }
    if let Some((datum, warning)) = known_datum_from_code(code_u32) {
        return Some(CrsSpec::Geographic {
            datum,
            label: format!("EPSG:{code}"),
            warning,
        });
    }

    tracing::warn!(
        foreign_datum_epsg = code,
        "Geographic raster uses foreign datum EPSG:{code} without a bundled shift; binding unshifted on WGS84"
    );
    Some(CrsSpec::Geographic {
        datum: WGS84_DATUM,
        label: format!("EPSG:{code}"),
        warning: Some(DatumWarning::UnshiftedForeignDatum {
            label: format!("EPSG:{code}"),
        }),
    })
}

fn parse_user_ellipsoid(geo_params: &GeoKeyParams) -> Option<Ellipsoid> {
    geo_params
        .get_u16(GEOKEY_GEOG_ELLIPSOID)
        .and_then(ellipsoid_from_epsg)
        .or_else(|| {
            let a = geo_params.get_f64(GEOKEY_GEOG_SEMI_MAJOR_AXIS)?;
            if !(a.is_finite() && a > 0.0) {
                return None;
            }
            if let Some(rf) = geo_params.get_f64(GEOKEY_GEOG_INV_FLATTENING)
                && rf.is_finite()
                && rf > 1.0
            {
                return Some(Ellipsoid {
                    semi_major_m: a,
                    flattening: 1.0 / rf,
                });
            }
            if let Some(b) = geo_params.get_f64(GEOKEY_GEOG_SEMI_MINOR_AXIS)
                && b.is_finite()
                && b > 0.0
                && b <= a
            {
                return Some(Ellipsoid {
                    semi_major_m: a,
                    flattening: (a - b) / a,
                });
            }
            None
        })
}

fn resolve_user_datum(geo_params: &GeoKeyParams) -> (Datum, Option<DatumWarning>) {
    let base_ellipsoid = parse_user_ellipsoid(geo_params);

    let datum_code = geo_params
        .geographic_crs
        .filter(|code| !UNSAID_GEOKEY_CODES.contains(code))
        .or_else(|| {
            geo_params
                .get_u16(GEOKEY_GEOG_GEODETIC_DATUM)
                .filter(|code| !UNSAID_GEOKEY_CODES.contains(code))
        });
    let known = datum_code.and_then(|code| known_datum_from_code(u32::from(code)));

    if let Some(vals) = geo_params.get_f64_slice(GEOKEY_GEOG_TOWGS84)
        && (vals.len() == 3 || vals.len() == 7)
        && vals.iter().all(|v| v.is_finite())
    {
        // Preserve ellipsoid from explicit GeoKey 2056/2057-2059, or fall back
        // to the ellipsoid of the referenced datum/geographic CRS (2048/2050)
        // before defaulting to WGS84.
        let ell = base_ellipsoid
            .or_else(|| known.as_ref().map(|(d, _)| d.ellipsoid))
            .unwrap_or(WGS84);
        let p7 = [
            vals[0],
            vals[1],
            vals[2],
            vals.get(3).copied().unwrap_or(0.0),
            vals.get(4).copied().unwrap_or(0.0),
            vals.get(5).copied().unwrap_or(0.0),
            vals.get(6).copied().unwrap_or(0.0),
        ];
        return (Datum::from_towgs84(ell, p7), None);
    }

    if let Some((datum, warning)) = known {
        return (datum, warning);
    }

    if let Some(ell) = base_ellipsoid {
        let non_wgs84 = (ell.semi_major_m - WGS84.semi_major_m).abs() > 1.0
            || (ell.flattening - WGS84.flattening).abs() > 1e-6;
        let warning = non_wgs84.then(|| DatumWarning::UnshiftedForeignDatum {
            label: "user-defined ellipsoid".to_string(),
        });
        return (
            Datum {
                ellipsoid: ell,
                ..WGS84_DATUM
            },
            warning,
        );
    }

    if let Some(code) = datum_code
        && !WGS84_GEOGRAPHIC_CODES.contains(&code)
        && code != EPSG_WGS84_DATUM
    {
        return (
            WGS84_DATUM,
            Some(DatumWarning::UnshiftedForeignDatum {
                label: format!("EPSG:{code}"),
            }),
        );
    }

    (WGS84_DATUM, None)
}

fn resolve_linear_unit_m(geo_params: &GeoKeyParams) -> Option<f64> {
    let Some(code) = geo_params.get_u16(GEOKEY_PROJ_LINEAR_UNITS) else {
        return Some(1.0);
    };
    if UNSAID_GEOKEY_CODES.contains(&code) {
        if code == GEOKEY_USER_DEFINED {
            let size = geo_params.get_f64(GEOKEY_PROJ_LINEAR_UNIT_SIZE)?;
            return (size.is_finite() && size > 0.0).then_some(size);
        }
        return Some(1.0);
    }
    if let Some(unit) = LinearUnit::from_geotiff_code(code) {
        return Some(unit.to_metres());
    }
    if code == EPSG_UNIT_KILOMETRE {
        return Some(1000.0);
    }
    None
}

struct ProjectedParams {
    datum: Datum,
    unit_m: f64,
    lon_0: f64,
    lat_0: f64,
    scale: f64,
    false_easting_m: f64,
    false_northing_m: f64,
    std_parallel_1: Option<f64>,
    std_parallel_2: Option<f64>,
}

impl ProjectedParams {
    fn is_valid(&self) -> bool {
        self.unit_m.is_finite()
            && self.unit_m > 0.0
            && self.lon_0.is_finite()
            && self.lat_0.is_finite()
            && self.scale.is_finite()
            && self.scale > 0.0
            && self.false_easting_m.is_finite()
            && self.false_northing_m.is_finite()
            && self.std_parallel_1.is_none_or(f64::is_finite)
            && self.std_parallel_2.is_none_or(f64::is_finite)
    }
}

fn build_user_transverse_mercator(
    geo_params: &GeoKeyParams,
    p: &ProjectedParams,
    warning: Option<DatumWarning>,
) -> Option<(CrsSpec, Option<DatumWarning>)> {
    let sys = System {
        datum: p.datum,
        central_meridian_deg: p.lon_0,
        origin_lat_deg: p.lat_0,
        scale: p.scale,
        false_easting_m: p.false_easting_m,
        false_northing_m: p.false_northing_m,
        unit_to_metres: p.unit_m,
    };
    let proj = match geo_params.get_ascii(GEOKEY_PCS_CITATION) {
        Some(citation) if !citation.trim().is_empty() => {
            Projection::transverse_named(sys, citation)
        }
        _ => Projection::from(sys),
    };
    Some((CrsSpec::Projected(proj), warning))
}

fn polar_stereographic_args(geo_params: &GeoKeyParams, p: &ProjectedParams) -> Vec<String> {
    let pole_lon = geo_params
        .get_f64(GEOKEY_PROJ_STRAIGHT_VERT_POLE_LONG)
        .filter(|v| v.is_finite())
        .unwrap_or(p.lon_0);
    let explicit_scale = geo_params.get_f64(GEOKEY_PROJ_SCALE_AT_NAT_ORIGIN);
    let nat_lat = geo_params.get_f64(GEOKEY_PROJ_NAT_ORIGIN_LAT);
    let lat_hint = p.std_parallel_1.or(nat_lat).unwrap_or(p.lat_0);
    let pole_lat = if lat_hint < 0.0 { -90.0 } else { 90.0 };

    let mut args = vec![
        "+proj=stere".to_string(),
        format!("+lat_0={pole_lat}"),
        format!("+lon_0={pole_lon}"),
    ];
    if let Some(ts) = p.std_parallel_1
        && ts.abs() < 90.0
    {
        args.push(format!("+lat_ts={ts}"));
    } else if explicit_scale.is_none_or(|s| (s - 1.0).abs() < 1e-12)
        && let Some(nl) = nat_lat
        && nl.is_finite()
        && nl.abs() > 0.0
        && nl.abs() < 90.0
    {
        args.push(format!("+lat_ts={nl}"));
    } else {
        args.push(format!("+k_0={}", p.scale));
    }
    args
}

fn mercator_args(p: &ProjectedParams) -> Vec<String> {
    let mut args = vec!["+proj=merc".to_string(), format!("+lon_0={}", p.lon_0)];
    if let Some(lat_ts) = p.std_parallel_1
        && lat_ts.abs() < 90.0
    {
        args.push(format!("+lat_ts={lat_ts}"));
    } else {
        args.push(format!("+k_0={}", p.scale));
    }
    args
}

fn lcc_1sp_args(p: &ProjectedParams) -> Vec<String> {
    let lat1 = p.std_parallel_1.unwrap_or(p.lat_0);
    vec![
        "+proj=lcc".to_string(),
        format!("+lat_0={}", p.lat_0),
        format!("+lon_0={}", p.lon_0),
        format!("+k_0={}", p.scale),
        format!("+lat_1={lat1}"),
    ]
}

fn standard_proj_args(proj_name: &str, p: &ProjectedParams) -> Vec<String> {
    let mut args = vec![
        format!("+proj={proj_name}"),
        format!("+lat_0={}", p.lat_0),
        format!("+lon_0={}", p.lon_0),
        format!("+k_0={}", p.scale),
    ];
    if let Some(lat1) = p.std_parallel_1 {
        args.push(format!("+lat_1={lat1}"));
    }
    if let Some(lat2) = p.std_parallel_2 {
        args.push(format!("+lat_2={lat2}"));
    }
    args
}

fn append_proj4_trailer(parts: &mut Vec<String>, p: &ProjectedParams) {
    parts.extend([
        format!("+x_0={}", p.false_easting_m),
        format!("+y_0={}", p.false_northing_m),
        format!("+to_meter={}", p.unit_m),
        format!("+a={}", p.datum.ellipsoid.semi_major_m),
        format!("+b={}", p.datum.ellipsoid.semi_minor_m()),
    ]);
    if !p.datum.is_wgs84() {
        let t = p.datum.to_towgs84();
        parts.push(format!(
            "+towgs84={},{},{},{},{},{},{}",
            t[0], t[1], t[2], t[3], t[4], t[5], t[6]
        ));
    }
    parts.push("+no_defs".to_string());
}

fn build_user_proj4_crs(
    ct: u16,
    geo_params: &GeoKeyParams,
    p: &ProjectedParams,
    warning: Option<DatumWarning>,
) -> Option<(CrsSpec, Option<DatumWarning>)> {
    let (proj_name, mut parts) = match ct {
        CT_POLAR_STEREOGRAPHIC => ("stere", polar_stereographic_args(geo_params, p)),
        CT_MERCATOR => ("merc", mercator_args(p)),
        CT_LAMBERT_CONF_CONIC_1SP => ("lcc", lcc_1sp_args(p)),
        CT_LAMBERT_CONF_CONIC_2SP => ("lcc", standard_proj_args("lcc", p)),
        CT_LAMBERT_AZIMUTHAL_EQUAL_AREA => ("laea", standard_proj_args("laea", p)),
        CT_ALBERS_EQUAL_AREA => ("aea", standard_proj_args("aea", p)),
        _ => return None,
    };
    append_proj4_trailer(&mut parts, p);

    let proj_str = parts.join(" ");
    let label = geo_params
        .get_ascii(GEOKEY_PCS_CITATION)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("User-Defined ({proj_name})"));
    let crs = CrsSpec::from_proj_string_labeled(&proj_str, Some(label)).ok()?;
    let final_warning = warning.or_else(|| crs.datum_warning().cloned());
    Some((crs, final_warning))
}

pub(super) fn build_user_defined_projected_crs(
    geo_params: &GeoKeyParams,
) -> Option<(CrsSpec, Option<DatumWarning>)> {
    let ct = geo_params.get_u16(GEOKEY_PROJ_COORD_TRANS)?;
    let unit_m = resolve_linear_unit_m(geo_params)?;
    let (datum, warning) = resolve_user_datum(geo_params);
    let lon_0 = geo_params
        .get_f64(GEOKEY_PROJ_NAT_ORIGIN_LONG)
        .or_else(|| geo_params.get_f64(GEOKEY_PROJ_FALSE_ORIGIN_LONG))
        .or_else(|| geo_params.get_f64(GEOKEY_PROJ_CENTER_LONG))
        .unwrap_or(0.0);
    let lat_0 = geo_params
        .get_f64(GEOKEY_PROJ_NAT_ORIGIN_LAT)
        .or_else(|| geo_params.get_f64(GEOKEY_PROJ_FALSE_ORIGIN_LAT))
        .or_else(|| geo_params.get_f64(GEOKEY_PROJ_CENTER_LAT))
        .unwrap_or(0.0);
    let scale = geo_params
        .get_f64(GEOKEY_PROJ_SCALE_AT_NAT_ORIGIN)
        .or_else(|| geo_params.get_f64(GEOKEY_PROJ_SCALE_AT_CENTER))
        .unwrap_or(1.0);
    // GeoTIFF false easting/northing are stored in the CRS linear unit.
    let false_easting_m = geo_params
        .get_f64(GEOKEY_PROJ_FALSE_EASTING)
        .or_else(|| geo_params.get_f64(GEOKEY_PROJ_FALSE_ORIGIN_EASTING))
        .or_else(|| geo_params.get_f64(GEOKEY_PROJ_CENTER_EASTING))
        .unwrap_or(0.0)
        * unit_m;
    let false_northing_m = geo_params
        .get_f64(GEOKEY_PROJ_FALSE_NORTHING)
        .or_else(|| geo_params.get_f64(GEOKEY_PROJ_FALSE_ORIGIN_NORTHING))
        .or_else(|| geo_params.get_f64(GEOKEY_PROJ_CENTER_NORTHING))
        .unwrap_or(0.0)
        * unit_m;
    let std_parallel_1 = geo_params.get_f64(GEOKEY_PROJ_STD_PARALLEL1);
    let std_parallel_2 = geo_params.get_f64(GEOKEY_PROJ_STD_PARALLEL2);

    let params = ProjectedParams {
        datum,
        unit_m,
        lon_0,
        lat_0,
        scale,
        false_easting_m,
        false_northing_m,
        std_parallel_1,
        std_parallel_2,
    };
    if !params.is_valid() {
        return None;
    }
    if ct == CT_TRANSVERSE_MERCATOR {
        build_user_transverse_mercator(geo_params, &params, warning)
    } else {
        build_user_proj4_crs(ct, geo_params, &params, warning)
    }
}
