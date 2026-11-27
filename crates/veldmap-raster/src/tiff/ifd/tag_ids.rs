//! Centralized Classic TIFF, BigTIFF, GeoTIFF, YCbCr, and GDAL tag ID
//! constants shared across IFD wire reading, IFD schema parsing, GeoKeys, and
//! open-time metadata extraction.

pub(in crate::tiff) const TAG_NEW_SUBFILE_TYPE: u16 = 254;
pub(in crate::tiff) const TAG_IMAGE_WIDTH: u16 = 256;
pub(in crate::tiff) const TAG_IMAGE_LENGTH: u16 = 257;
pub(in crate::tiff) const TAG_BITS_PER_SAMPLE: u16 = 258;
pub(in crate::tiff) const TAG_COMPRESSION: u16 = 259;
pub(in crate::tiff) const TAG_PHOTOMETRIC: u16 = 262;
pub(in crate::tiff) const TAG_STRIP_OFFSETS: u16 = 273;
pub(in crate::tiff) const TAG_SAMPLES_PER_PIXEL: u16 = 277;
pub(in crate::tiff) const TAG_ROWS_PER_STRIP: u16 = 278;
pub(in crate::tiff) const TAG_STRIP_BYTE_COUNTS: u16 = 279;
pub(in crate::tiff) const TAG_PLANAR_CONFIG: u16 = 284;
pub(in crate::tiff) const TAG_PREDICTOR: u16 = 317;
pub(in crate::tiff) const TAG_COLOR_MAP: u16 = 320;
pub(in crate::tiff) const TAG_TILE_WIDTH: u16 = 322;
pub(in crate::tiff) const TAG_TILE_LENGTH: u16 = 323;
pub(in crate::tiff) const TAG_TILE_OFFSETS: u16 = 324;
pub(in crate::tiff) const TAG_TILE_BYTE_COUNTS: u16 = 325;
pub(in crate::tiff) const TAG_EXTRA_SAMPLES: u16 = 338;
pub(in crate::tiff) const TAG_SAMPLE_FORMAT: u16 = 339;
pub(in crate::tiff) const TAG_JPEG_TABLES: u16 = 347;
pub(in crate::tiff) const TAG_YCBCR_COEFFICIENTS: u16 = 529;
pub(in crate::tiff) const TAG_YCBCR_SUBSAMPLING: u16 = 530;
pub(in crate::tiff) const TAG_REFERENCE_BLACK_WHITE: u16 = 532;
pub(in crate::tiff) const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
pub(in crate::tiff) const TAG_MODEL_TIEPOINT: u16 = 33922;
pub(in crate::tiff) const TAG_MODEL_TRANSFORMATION: u16 = 34264;
pub(in crate::tiff) const TAG_GEO_KEY_DIRECTORY: u16 = 34735;
pub(in crate::tiff) const TAG_GEO_DOUBLE_PARAMS: u16 = 34736;
pub(in crate::tiff) const TAG_GEO_ASCII_PARAMS: u16 = 34737;
pub(in crate::tiff) const TAG_GDAL_METADATA: u16 = 42112;
pub(in crate::tiff) const TAG_GDAL_NODATA: u16 = 42113;
