//! Centralized Classic TIFF, BigTIFF, GeoTIFF, YCbCr, and GDAL tag ID
//! constants shared across IFD wire reading, IFD schema parsing, GeoKeys, and
//! open-time metadata extraction.

pub(super) const TAG_NEW_SUBFILE_TYPE: u16 = 254;
pub(super) const TAG_IMAGE_WIDTH: u16 = 256;
pub(super) const TAG_IMAGE_LENGTH: u16 = 257;
pub(super) const TAG_BITS_PER_SAMPLE: u16 = 258;
pub(super) const TAG_COMPRESSION: u16 = 259;
pub(super) const TAG_PHOTOMETRIC: u16 = 262;
pub(super) const TAG_STRIP_OFFSETS: u16 = 273;
pub(super) const TAG_SAMPLES_PER_PIXEL: u16 = 277;
pub(super) const TAG_ROWS_PER_STRIP: u16 = 278;
pub(super) const TAG_STRIP_BYTE_COUNTS: u16 = 279;
pub(super) const TAG_PLANAR_CONFIG: u16 = 284;
pub(super) const TAG_DATETIME: u16 = 306;
pub(super) const TAG_PREDICTOR: u16 = 317;
pub(super) const TAG_COLOR_MAP: u16 = 320;
pub(super) const TAG_TILE_WIDTH: u16 = 322;
pub(super) const TAG_TILE_LENGTH: u16 = 323;
pub(super) const TAG_TILE_OFFSETS: u16 = 324;
pub(super) const TAG_TILE_BYTE_COUNTS: u16 = 325;
pub(super) const TAG_EXTRA_SAMPLES: u16 = 338;
pub(super) const TAG_SAMPLE_FORMAT: u16 = 339;
pub(super) const TAG_JPEG_TABLES: u16 = 347;
pub(super) const TAG_YCBCR_COEFFICIENTS: u16 = 529;
pub(super) const TAG_YCBCR_SUBSAMPLING: u16 = 530;
pub(super) const TAG_REFERENCE_BLACK_WHITE: u16 = 532;
pub(super) const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
pub(super) const TAG_MODEL_TIEPOINT: u16 = 33922;
pub(super) const TAG_MODEL_TRANSFORMATION: u16 = 34264;
pub(super) const TAG_GEO_KEY_DIRECTORY: u16 = 34735;
pub(super) const TAG_GEO_DOUBLE_PARAMS: u16 = 34736;
pub(super) const TAG_GEO_ASCII_PARAMS: u16 = 34737;
pub(super) const TAG_GDAL_METADATA: u16 = 42112;
pub(super) const TAG_GDAL_NODATA: u16 = 42113;
