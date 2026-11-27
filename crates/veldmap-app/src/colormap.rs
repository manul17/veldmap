//! Scientific continuous colormaps, categorical class palettes, and exact
//! IEC 61966-2-1 sRGB-to-linear transfer functions shared between GPU uniform
//! LUTs, draped vector styling, and CPU `egui` colorbar / histogram widgets.

/// Number of LUT entries in `PaletteUniform` (`256 * 16 B = 4096 B`, well
/// within baseline WebGPU's 16 KiB uniform buffer limit).
pub(crate) const PALETTE_LUT_LEN: usize = 256;

/// Continuous scientific colormap presets for `R32Float` layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ColormapId {
    Grayscale,
    #[default]
    Viridis,
    Inferno,
    Plasma,
    Turbo,
    RdBu,
    RdYlGn,
    BrBG,
    Terrain,
}

impl ColormapId {
    pub(crate) const ALL: [Self; 9] = [
        Self::Grayscale,
        Self::Viridis,
        Self::Inferno,
        Self::Plasma,
        Self::Turbo,
        Self::RdBu,
        Self::RdYlGn,
        Self::BrBG,
        Self::Terrain,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Grayscale => "Grayscale",
            Self::Viridis => "Viridis",
            Self::Inferno => "Inferno",
            Self::Plasma => "Plasma",
            Self::Turbo => "Turbo",
            Self::RdBu => "RdBu (Diverging)",
            Self::RdYlGn => "RdYlGn (NDVI)",
            Self::BrBG => "BrBG (NDWI)",
            Self::Terrain => "Terrain",
        }
    }
}

/// Discrete categorical palette presets for `R8Uint` / `R16Uint` layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CategoricalPaletteId {
    /// Embedded source palette (e.g., TIFF `ColorMap` tag 320), falling back
    /// to [`Self::DistinctClasses`] when absent.
    #[default]
    SourceColorMap,
    /// ESA Sentinel-2 L2A Scene Classification Layer (`SCL`, classes 0–11).
    Sentinel2Scl,
    /// High-contrast deterministic distinct class colors (256 entries).
    DistinctClasses,
}

impl CategoricalPaletteId {
    pub(crate) const ALL: [Self; 3] = [
        Self::SourceColorMap,
        Self::Sentinel2Scl,
        Self::DistinctClasses,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::SourceColorMap => "Source ColorMap",
            Self::Sentinel2Scl => "Sentinel-2 SCL",
            Self::DistinctClasses => "Distinct Classes",
        }
    }
}

/// Convert an sRGB channel to exact linear light (IEC 61966-2-1), clamping
/// out-of-range values to `[0.0, 1.0]` and sanitizing non-finite inputs to `0.0`.
#[inline]
fn srgb_to_linear_f32(c: f32) -> f32 {
    let s = if c.is_finite() {
        c.clamp(0.0, 1.0)
    } else {
        0.0
    };
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert a normalized `[r, g, b, a]` sRGB color to linear-light RGB with
/// clamped finite alpha scaled by `opacity`.
#[inline]
pub(crate) fn srgb_rgba_f32_to_linear(rgba: [f32; 4], opacity: f32) -> [f32; 4] {
    let a = if rgba[3].is_finite() {
        rgba[3].clamp(0.0, 1.0)
    } else {
        0.0
    };
    [
        srgb_to_linear_f32(rgba[0]),
        srgb_to_linear_f32(rgba[1]),
        srgb_to_linear_f32(rgba[2]),
        (a * opacity).clamp(0.0, 1.0),
    ]
}

fn sample_stops_srgb(stops: &[[u8; 3]], t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    let last = stops.len() - 1;
    let scaled = t * (last as f32);
    let i0 = (scaled.floor() as usize).min(last);
    let i1 = (i0 + 1).min(last);
    let f = scaled - (i0 as f32);
    let a = stops[i0];
    let b = stops[i1];
    let lerp = |u: u8, v: u8| (f32::from(u) * (1.0 - f) + f32::from(v) * f).round() as u8;
    [lerp(a[0], b[0]), lerp(a[1], b[1]), lerp(a[2], b[2]), 255]
}

const GRAYSCALE_STOPS: &[[u8; 3]] = &[[0, 0, 0], [255, 255, 255]];

const VIRIDIS_STOPS: &[[u8; 3]] = &[
    [68, 1, 84],
    [72, 40, 120],
    [62, 74, 137],
    [49, 104, 142],
    [38, 130, 142],
    [31, 158, 137],
    [53, 183, 121],
    [109, 205, 89],
    [180, 222, 44],
    [253, 231, 37],
];

const INFERNO_STOPS: &[[u8; 3]] = &[
    [0, 0, 4],
    [27, 12, 65],
    [74, 12, 107],
    [120, 28, 109],
    [165, 44, 96],
    [207, 68, 70],
    [237, 105, 37],
    [251, 155, 6],
    [247, 209, 61],
    [252, 255, 164],
];

const PLASMA_STOPS: &[[u8; 3]] = &[
    [13, 8, 135],
    [70, 3, 159],
    [114, 1, 168],
    [156, 23, 158],
    [189, 55, 134],
    [216, 87, 107],
    [237, 121, 83],
    [251, 159, 58],
    [253, 202, 38],
    [240, 249, 33],
];

const TURBO_STOPS: &[[u8; 3]] = &[
    [48, 18, 59],
    [69, 91, 205],
    [62, 156, 254],
    [24, 215, 203],
    [72, 248, 130],
    [164, 252, 60],
    [226, 220, 56],
    [254, 163, 49],
    [239, 89, 17],
    [194, 36, 3],
    [122, 4, 3],
];

const RDBU_STOPS: &[[u8; 3]] = &[
    [103, 0, 31],
    [178, 24, 43],
    [214, 96, 77],
    [244, 165, 130],
    [247, 247, 247],
    [146, 197, 222],
    [67, 147, 195],
    [33, 102, 172],
    [5, 48, 97],
];

/// ColorBrewer diverging Red–Yellow–Green palette (EO standard for NDVI).
const RDYLGN_STOPS: &[[u8; 3]] = &[
    [165, 0, 38],
    [215, 48, 39],
    [244, 109, 67],
    [253, 174, 97],
    [254, 224, 139],
    [255, 255, 191],
    [217, 239, 139],
    [166, 217, 106],
    [102, 189, 99],
    [26, 152, 80],
    [0, 104, 55],
];

/// ColorBrewer diverging Brown–White–Blue-Green palette (EO standard for NDWI).
const BRBG_STOPS: &[[u8; 3]] = &[
    [84, 48, 5],
    [140, 81, 10],
    [191, 129, 45],
    [223, 194, 125],
    [246, 232, 195],
    [245, 245, 245],
    [199, 234, 229],
    [128, 205, 193],
    [53, 151, 143],
    [1, 102, 94],
    [0, 60, 48],
];

const TERRAIN_STOPS: &[[u8; 3]] = &[
    [32, 74, 135],
    [52, 132, 168],
    [62, 158, 92],
    [140, 192, 92],
    [224, 212, 124],
    [196, 150, 98],
    [145, 104, 84],
    [176, 172, 170],
    [248, 248, 250],
];

/// Evaluate a continuous scientific colormap in sRGB `[r, g, b, 255]` at `t` in `[0, 1]`.
pub(crate) fn sample_colormap_srgb(colormap: ColormapId, t: f32, invert: bool) -> [u8; 4] {
    let u = if invert {
        1.0 - t.clamp(0.0, 1.0)
    } else {
        t.clamp(0.0, 1.0)
    };
    let stops = match colormap {
        ColormapId::Grayscale => GRAYSCALE_STOPS,
        ColormapId::Viridis => VIRIDIS_STOPS,
        ColormapId::Inferno => INFERNO_STOPS,
        ColormapId::Plasma => PLASMA_STOPS,
        ColormapId::Turbo => TURBO_STOPS,
        ColormapId::RdBu => RDBU_STOPS,
        ColormapId::RdYlGn => RDYLGN_STOPS,
        ColormapId::BrBG => BRBG_STOPS,
        ColormapId::Terrain => TERRAIN_STOPS,
    };
    sample_stops_srgb(stops, u)
}

/// Build a 256-entry linear-light RGBA LUT for a scientific [`ColormapId`].
pub(crate) fn scientific_lut_linear(colormap: ColormapId) -> [[f32; 4]; PALETTE_LUT_LEN] {
    let mut lut = [[0.0f32; 4]; PALETTE_LUT_LEN];
    for (i, slot) in lut.iter_mut().enumerate() {
        let t = (i as f32) / ((PALETTE_LUT_LEN - 1) as f32);
        *slot = srgb_rgba_to_linear(sample_colormap_srgb(colormap, t, false));
    }
    lut
}

fn srgb_rgba_to_linear(rgba: [u8; 4]) -> [f32; 4] {
    [
        srgb_to_linear_f32(f32::from(rgba[0]) / 255.0),
        srgb_to_linear_f32(f32::from(rgba[1]) / 255.0),
        srgb_to_linear_f32(f32::from(rgba[2]) / 255.0),
        f32::from(rgba[3]) / 255.0,
    ]
}

/// Deterministic golden-ratio HSV palette for arbitrary discrete class IDs.
fn distinct_class_srgb(idx: usize) -> [u8; 4] {
    if idx == 0 {
        return [24, 28, 34, 255];
    }
    let h = ((idx as f32) * 0.618_034) % 1.0;
    let s = if idx.is_multiple_of(3) { 0.72 } else { 0.85 };
    let v = if idx.is_multiple_of(2) { 0.88 } else { 0.74 };
    let h6 = h * 6.0;
    let sector = (h6.floor() as i32) % 6;
    let f = h6 - h6.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match sector {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
        255,
    ]
}

/// Official ESA Sentinel-2 L2A Scene Classification Layer (`SCL`) class names (0..=11).
pub(crate) const SENTINEL2_SCL_NAMES: [&str; 12] = [
    "No Data",
    "Saturated / Defective",
    "Dark Area / Cast Shadows",
    "Cloud Shadows",
    "Vegetation",
    "Not Vegetated",
    "Water",
    "Unclassified",
    "Cloud Medium Prob.",
    "Cloud High Prob.",
    "Thin Cirrus",
    "Snow / Ice",
];

/// Default Sentinel-2 L2A `SCL` cloud/shadow/nodata mask bitmask
/// (classes `0, 1, 3, 8, 9, 10`).
pub(crate) const DEFAULT_SCL_MASK_BITS: u32 =
    (1 << 0) | (1 << 1) | (1 << 3) | (1 << 8) | (1 << 9) | (1 << 10);

/// Look up the official ESA Sentinel-2 L2A `SCL` class name for `class_id`.
pub(crate) fn sentinel2_scl_class_name(class_id: u16) -> Option<&'static str> {
    SENTINEL2_SCL_NAMES.get(usize::from(class_id)).copied()
}

/// Official ESA Sentinel-2 L2A Scene Classification Layer (`SCL`) colors (0..=11).
const SENTINEL2_SCL_SRGB: [[u8; 4]; 12] = [
    [0, 0, 0, 0],         // 0: No Data
    [255, 0, 4, 255],     // 1: Saturated or Defective
    [47, 47, 47, 255],    // 2: Dark Area / Cast Shadows
    [100, 50, 0, 255],    // 3: Cloud Shadows
    [0, 160, 0, 255],     // 4: Vegetation
    [255, 230, 90, 255],  // 5: Not Vegetated
    [0, 0, 255, 255],     // 6: Water
    [128, 128, 128, 255], // 7: Unclassified
    [192, 192, 192, 255], // 8: Cloud Medium Probability
    [255, 255, 255, 255], // 9: Cloud High Probability
    [100, 200, 255, 255], // 10: Thin Cirrus
    [255, 150, 255, 255], // 11: Snow or Ice
];

/// Build a 256-entry linear-light RGBA LUT for a categorical palette.
pub(crate) fn categorical_lut_linear(
    palette: CategoricalPaletteId,
    custom_color_map: Option<&[[u8; 4]]>,
) -> [[f32; 4]; PALETTE_LUT_LEN] {
    let mut lut = [[0.0f32; 4]; PALETTE_LUT_LEN];
    match palette {
        CategoricalPaletteId::SourceColorMap => {
            if let Some(cmap) = custom_color_map
                && !cmap.is_empty()
            {
                for (i, slot) in lut.iter_mut().enumerate() {
                    let srgb = cmap
                        .get(i)
                        .copied()
                        .unwrap_or_else(|| distinct_class_srgb(i));
                    *slot = srgb_rgba_to_linear(srgb);
                }
                return lut;
            }
            for (i, slot) in lut.iter_mut().enumerate() {
                *slot = srgb_rgba_to_linear(distinct_class_srgb(i));
            }
        }
        CategoricalPaletteId::Sentinel2Scl => {
            for (i, slot) in lut.iter_mut().enumerate() {
                let srgb = SENTINEL2_SCL_SRGB
                    .get(i)
                    .copied()
                    .unwrap_or_else(|| distinct_class_srgb(i));
                *slot = srgb_rgba_to_linear(srgb);
            }
        }
        CategoricalPaletteId::DistinctClasses => {
            for (i, slot) in lut.iter_mut().enumerate() {
                *slot = srgb_rgba_to_linear(distinct_class_srgb(i));
            }
        }
    }
    lut
}
