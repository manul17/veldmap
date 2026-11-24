//! Source, band, pixel-format, and stream-partitioning identifiers
//! (`SourceFingerprint`, `SourceId`, `BandKey`, `PixelFormat`,
//! `TileStreamSpec`, `StreamKey`, `fnv1a_64`).

use std::fmt;

use super::key::TileKey;

/// Deterministic 64-bit FNV-1a hash used for stable `Copy` source/band keys.
pub const fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    hash
}

/// Opaque 256-bit content fingerprint uniquely identifying a raster byte stream
/// (sampled BLAKE3 digest for local files, or metadata digest for remote COGs).
///
/// Formats as 64 lowercase ASCII hex characters, which always satisfies strict
/// disk-cache `source_key` validation (`<= 128` safe ASCII bytes).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceFingerprint([u8; 32]);

impl SourceFingerprint {
    /// Wrap a 32-byte digest into an opaque [`SourceFingerprint`].
    pub const fn from_digest(digest: [u8; 32]) -> Self {
        Self(digest)
    }

    /// Borrow the raw 32-byte digest.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Encode as 64 lowercase hexadecimal ASCII characters.
    pub fn to_hex(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(64);
        for &byte in &self.0 {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        out
    }
}

impl fmt::Display for SourceFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for SourceFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hex = self.to_hex();
        write!(f, "SourceFingerprint({}..)", &hex[..12])
    }
}

/// Compact identifier for a tile data source (bundled basemap, live XYZ/TMS
/// layer, local GeoTIFF, or cloud COG).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(pub u64);

impl SourceId {
    /// Reserved source ID for the bundled offline global basemap.
    pub const BUNDLED_BASEMAP: Self = Self(1);

    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Derive a deterministic `SourceId` from a stable ASCII/UTF-8 source key
    /// (e.g. a live basemap layer ID or normalized URL template).
    ///
    /// Sets the high bit so derived IDs never collide with small reserved
    /// constants such as [`Self::BUNDLED_BASEMAP`].
    pub const fn from_stable_key(key: &str) -> Self {
        Self(fnv1a_64(key.as_bytes()) | (1u64 << 63))
    }

    /// Derive a deterministic [`SourceId`] directly from the first 8 bytes of
    /// a uniform 256-bit [`SourceFingerprint`] digest, setting the high bit so
    /// it never collides with reserved constants.
    pub const fn from_fingerprint(fingerprint: &SourceFingerprint) -> Self {
        let b = fingerprint.as_bytes();
        let low = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        Self(low | (1u64 << 63))
    }
}

impl From<&SourceFingerprint> for SourceId {
    fn from(fingerprint: &SourceFingerprint) -> Self {
        Self::from_fingerprint(fingerprint)
    }
}

/// Deterministic, `Copy` band selector avoiding heap allocations in `TileKey`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BandKey {
    /// Display-ready color composite (e.g. sRGB basemap or RGB orthophoto).
    Display,
    /// Zero-based numeric band index in a multi-band raster.
    Index(u16),
    /// Deterministic 64-bit FNV-1a hash of a named band/variable (e.g. `"B08"`,
    /// `"SCL"`, or NetCDF variable name).
    NamedHash(u64),
}

impl BandKey {
    /// Compute a deterministic 64-bit FNV-1a hash for a band or variable name.
    pub const fn from_name(name: &str) -> Self {
        Self::NamedHash(fnv1a_64(name.as_bytes()))
    }
}

/// Pixel storage format for GPU and CPU tile caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PixelFormat {
    /// 8-bit sRGB RGBA display imagery (4 bytes/pixel).
    Rgba8UnormSrgb,
    /// 32-bit IEEE float scientific samples (4 bytes/pixel, NaN = nodata).
    R32Float,
    /// 8-bit unsigned integer categorical/QA masks (1 byte/pixel).
    R8Uint,
    /// 16-bit unsigned integer categorical/QA masks (2 bytes/pixel).
    R16Uint,
}

impl PixelFormat {
    /// Exact unpacked byte size of a single pixel in this format.
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgba8UnormSrgb | Self::R32Float => 4,
            Self::R8Uint => 1,
            Self::R16Uint => 2,
        }
    }
}

/// Source-independent stream-partitioning descriptor (`band`, `format`,
/// `gutter`, `decode_rev`) shared across `TileKey`, `DiskTileKey`,
/// `PyramidMeta`, and `ChunkKey` so cache/key comparisons stay in lockstep.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileStreamSpec {
    pub band: BandKey,
    pub format: PixelFormat,
    /// Border gutter width in texels (`0` for standard tiles, `1` for DEM
    /// relief tiles) so guttered and unguttered tiles never alias in caches.
    pub gutter: u8,
    pub decode_rev: u32,
}

impl TileStreamSpec {
    /// Canonical display stream (`BandKey::Display`, `Rgba8UnormSrgb`,
    /// `gutter = 0`, `decode_rev = 0`).
    pub const DISPLAY: Self = Self {
        band: BandKey::Display,
        format: PixelFormat::Rgba8UnormSrgb,
        gutter: 0,
        decode_rev: 0,
    };

    /// Construct a stream descriptor.
    pub const fn new(band: BandKey, format: PixelFormat, gutter: u8, decode_rev: u32) -> Self {
        Self {
            band,
            format,
            gutter,
            decode_rev,
        }
    }
}

/// Identifies one independent raster pyramid stream (`source`, `band`,
/// `format`, `gutter`) independent of `decode_rev`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StreamKey {
    pub source: SourceId,
    pub band: BandKey,
    pub format: PixelFormat,
    pub gutter: u8,
}

impl StreamKey {
    /// Construct a [`StreamKey`] for `(source, band, format, gutter)`.
    pub const fn new(source: SourceId, band: BandKey, format: PixelFormat, gutter: u8) -> Self {
        Self {
            source,
            band,
            format,
            gutter,
        }
    }

    /// Extract the [`StreamKey`] identifying `key`'s band/format/gutter stream.
    pub const fn from_tile_key(key: &TileKey) -> Self {
        Self {
            source: key.source,
            band: key.band,
            format: key.format,
            gutter: key.gutter,
        }
    }
}
