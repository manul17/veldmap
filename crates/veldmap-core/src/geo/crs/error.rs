//! Error types for parsing and instantiating CRS definitions.

use thiserror::Error;

/// Errors produced when parsing or instantiating a CRS definition.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CrsError {
    #[error("unsupported or unknown EPSG code: EPSG:{0}")]
    UnsupportedEpsg(u32),
    #[error("failed to parse WKT CRS definition: {0}")]
    InvalidWkt(String),
    #[error("failed to parse PROJ CRS string: {0}")]
    InvalidProj(String),
    #[error("CRS definition is missing inverse or forward transform support")]
    NonInvertible,
    #[error("invalid custom Transverse Mercator parameters: {0}")]
    InvalidCustomTm(String),
}
