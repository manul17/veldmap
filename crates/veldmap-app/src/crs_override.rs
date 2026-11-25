//! Pure domain CRS override specification decoupled from `session.json`
//! serialization types.

use veldmap_core::geo::{CrsSpec, CustomTmParams};

/// Domain CRS override applied to a local raster layer or pending file open.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CrsOverride {
    Epsg(u32),
    Wkt(String),
    Proj(String),
    CustomTm(CustomTmParams),
}

impl CrsOverride {
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
