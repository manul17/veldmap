//! Pure domain CRS override specification decoupled from `session.json`
//! serialization types.

use veldmap_core::geo::{CrsSpec, CrsTextKind, CustomTmParams};

/// Domain CRS override applied to a local raster layer or pending file open.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CrsOverride {
    Epsg(u32),
    Wkt(String),
    Proj(String),
    CustomTm(CustomTmParams),
}

impl CrsOverride {
    /// Parse and validate a user-entered CRS text (WKT1/WKT2, PROJ.4, or
    /// `EPSG:<code>` / bare numeric EPSG shorthand), stripping any leading
    /// UTF-8 BOM and surrounding whitespace once before storing.
    pub(crate) fn parse_text(input: &str) -> Result<Self, String> {
        let trimmed = input.trim().trim_start_matches('\u{FEFF}').trim();
        let (_, kind) = CrsSpec::parse_text(trimmed).map_err(|e| e.to_string())?;
        match kind {
            CrsTextKind::Wkt => Ok(Self::Wkt(trimmed.to_owned())),
            CrsTextKind::Epsg(code) => Ok(Self::Epsg(code)),
            CrsTextKind::Proj => Ok(Self::Proj(trimmed.to_owned())),
        }
    }

    /// Resolve this override into a live [`CrsSpec`].
    pub(crate) fn to_crs_spec(&self) -> Result<CrsSpec, String> {
        match self {
            Self::Epsg(code) => CrsSpec::from_epsg(*code).map_err(|e| e.to_string()),
            Self::Wkt(wkt) => CrsSpec::from_wkt(wkt).map_err(|e| e.to_string()),
            Self::Proj(proj) => CrsSpec::from_user_string(proj).map_err(|e| e.to_string()),
            Self::CustomTm(params) => {
                let proj = params.to_projection().map_err(|e| e.to_string())?;
                Ok(CrsSpec::Projected(proj))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_text_strips_bom_whitespace_and_classifies_epsg_wkt_and_proj() {
        assert_eq!(
            CrsOverride::parse_text(" \u{FEFF} EPSG:32637 \n").expect("epsg prefix"),
            CrsOverride::Epsg(32637)
        );
        assert_eq!(
            CrsOverride::parse_text("\u{FEFF}4326").expect("bare epsg"),
            CrsOverride::Epsg(4326)
        );

        let proj_raw = "\u{FEFF} +proj=utm +zone=37 +datum=WGS84 +units=m +no_defs ";
        let proj = CrsOverride::parse_text(proj_raw).expect("proj");
        assert_eq!(
            proj,
            CrsOverride::Proj("+proj=utm +zone=37 +datum=WGS84 +units=m +no_defs".to_owned())
        );
        assert!(proj.to_crs_spec().is_ok());

        let wkt_raw = concat!(
            "\u{FEFF} GEOGCS[\"WGS 84\",",
            "DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],",
            "PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]] "
        );
        let wkt = CrsOverride::parse_text(wkt_raw).expect("wkt");
        match &wkt {
            CrsOverride::Wkt(s) => {
                assert!(!s.starts_with('\u{FEFF}'));
                assert!(s.starts_with("GEOGCS"));
            }
            other => panic!("expected Wkt, got {other:?}"),
        }
        assert!(wkt.to_crs_spec().is_ok());

        assert!(CrsOverride::parse_text("   ").is_err());
    }
}
