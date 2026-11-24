//! Composite tile cache keys, payload encoding tags, and SQLite column
//! encoders for [`super::TileDiskCache`].

use veldmap_core::tile::{BandKey, PixelFormat, TileKey, TileScheme, TileStreamSpec};

use crate::StorageError;

/// Validate a disk cache source key: non-empty, `<= 128` bytes, not `.` or `..`,
/// closed safe ASCII alphabet (`a-z`, `A-Z`, `0-9`, `-`, `_`, `.`).
pub(crate) fn valid_source_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 128
        && key != "."
        && key != ".."
        && key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

/// Wire/disk payload encoding tag stored alongside each cached tile blob.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TileBlobEncoding {
    /// Raw upstream encoded image bytes (PNG/JPEG/WebP from live tile servers).
    #[default]
    EncodedImage = 0,
    /// Lossless QOI-encoded 8-bit display RGBA tiles (Slice 3A-1b).
    Qoi = 1,
    /// Byte-shuffled LZ4 scientific/categorical tile payloads (Slice 3B).
    Lz4Raw = 2,
}

impl TileBlobEncoding {
    pub(super) fn from_i64(raw: i64) -> Option<Self> {
        match raw {
            0 => Some(Self::EncodedImage),
            1 => Some(Self::Qoi),
            2 => Some(Self::Lz4Raw),
            _ => None,
        }
    }
}

fn encode_scheme(scheme: TileScheme) -> i64 {
    match scheme {
        TileScheme::WebMercator => 1,
        TileScheme::Geodetic => 2,
        TileScheme::PolarCap { north: true } => 3,
        TileScheme::PolarCap { north: false } => 4,
        TileScheme::RasterPyramid { max_level } => 100 + i64::from(max_level),
    }
}

fn encode_band(band: BandKey) -> (i64, i64) {
    match band {
        BandKey::Display => (0, 0),
        BandKey::Index(idx) => (1, i64::from(idx)),
        BandKey::NamedHash(hash) => (2, hash as i64),
    }
}

fn encode_format(format: PixelFormat) -> i64 {
    match format {
        PixelFormat::Rgba8UnormSrgb => 0,
        PixelFormat::R32Float => 1,
        PixelFormat::R8Uint => 2,
        PixelFormat::R16Uint => 3,
    }
}

/// Composite lookup key for one encoded tile stored in [`super::TileDiskCache`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DiskTileKey {
    pub source_key: String,
    pub band: BandKey,
    pub scheme: TileScheme,
    pub format: PixelFormat,
    pub gutter: u8,
    pub decode_rev: u32,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl DiskTileKey {
    /// Construct a display-track (`TileStreamSpec::DISPLAY`) disk cache key.
    pub fn new(
        source_key: impl Into<String>,
        scheme: TileScheme,
        level: u8,
        x: u32,
        y: u32,
    ) -> Self {
        Self::from_stream(source_key, TileStreamSpec::DISPLAY, scheme, level, x, y)
    }

    /// Construct a disk cache key from a [`TileStreamSpec`] and tile coordinates.
    fn from_stream(
        source_key: impl Into<String>,
        spec: TileStreamSpec,
        scheme: TileScheme,
        level: u8,
        x: u32,
        y: u32,
    ) -> Self {
        Self {
            source_key: source_key.into(),
            band: spec.band,
            scheme,
            format: spec.format,
            gutter: spec.gutter,
            decode_rev: spec.decode_rev,
            level,
            x,
            y,
        }
    }

    /// Construct a full multi-band / multi-format disk cache key from a
    /// [`TileKey`] and its stable ASCII `source_key` / fingerprint string.
    pub fn from_tile_key(source_key: impl Into<String>, key: &TileKey) -> Self {
        Self::from_stream(
            source_key,
            key.stream_spec(),
            key.scheme,
            key.level,
            key.x,
            key.y,
        )
    }
}

/// Pre-encoded 10-column composite key shared across all `tile_meta` queries.
pub(super) struct EncodedKey<'a> {
    pub(super) source_key: &'a str,
    pub(super) band_kind: i64,
    pub(super) band_id: i64,
    pub(super) scheme: i64,
    pub(super) format: i64,
    pub(super) gutter: i64,
    pub(super) decode_rev: i64,
    pub(super) level: i64,
    pub(super) x: i64,
    pub(super) y: i64,
}

impl<'a> EncodedKey<'a> {
    pub(super) fn from_key(key: &'a DiskTileKey) -> Result<Self, StorageError> {
        if !valid_source_key(&key.source_key) {
            return Err(StorageError::InvalidSourceKey(key.source_key.clone()));
        }
        let (band_kind, band_id) = encode_band(key.band);
        Ok(Self {
            source_key: &key.source_key,
            band_kind,
            band_id,
            scheme: encode_scheme(key.scheme),
            format: encode_format(key.format),
            gutter: i64::from(key.gutter),
            decode_rev: i64::from(key.decode_rev),
            level: i64::from(key.level),
            x: i64::from(key.x),
            y: i64::from(key.y),
        })
    }

    pub(super) fn key_params(&self) -> [&dyn rusqlite::ToSql; 10] {
        [
            &self.source_key,
            &self.band_kind,
            &self.band_id,
            &self.scheme,
            &self.format,
            &self.gutter,
            &self.decode_rev,
            &self.level,
            &self.x,
            &self.y,
        ]
    }

    pub(super) fn upsert_params<'p>(
        &'p self,
        encoding: &'p i64,
        new_bytes: &'p i64,
        ts_ms: &'p i64,
    ) -> [&'p dyn rusqlite::ToSql; 13] {
        let [k0, k1, k2, k3, k4, k5, k6, k7, k8, k9] = self.key_params();
        [
            k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, encoding, new_bytes, ts_ms,
        ]
    }
}
