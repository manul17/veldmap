//! Curated searchable CRS presets for the UI CRS picker.

use std::sync::OnceLock;

use super::super::projection::{MSK61_EPSG_BASE, MSK61_ZONES};
use super::CrsSpec;

/// Searchable CRS preset entry for the UI CRS picker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrsPreset {
    pub epsg: u32,
    pub name: String,
    pub(super) category: &'static str,
}

impl CrsPreset {
    /// Display code label (`EPSG:<code>` for standard registry codes, or
    /// `MSK-61:z<n>` for non-standard regional pseudo-codes).
    pub fn code_label(&self) -> String {
        if let Some(zone) = self
            .epsg
            .checked_sub(MSK61_EPSG_BASE)
            .filter(|z| MSK61_ZONES.contains(z))
        {
            format!("MSK-61:z{zone}")
        } else {
            format!("EPSG:{}", self.epsg)
        }
    }
}

struct PresetCatalog {
    presets: Vec<CrsPreset>,
    /// Pre-lowercased searchable text per preset (same order as `presets`),
    /// avoiding per-keystroke heap allocations in [`search_crs_presets`].
    search_keys: Vec<String>,
}

fn preset_catalog() -> &'static PresetCatalog {
    static CATALOG: OnceLock<PresetCatalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        let mut presets = Vec::with_capacity(170);
        presets.push(CrsPreset {
            epsg: 4326,
            name: "WGS 84 (Geographic lon/lat)".to_string(),
            category: "Global",
        });
        presets.push(CrsPreset {
            epsg: 3857,
            name: "WGS 84 / Pseudo-Mercator (Web Mercator)".to_string(),
            category: "Global",
        });
        presets.push(CrsPreset {
            epsg: 4284,
            name: "Pulkovo 1942 (Geographic lon/lat)".to_string(),
            category: "Regional",
        });
        presets.push(CrsPreset {
            epsg: 3035,
            name: "ETRS89-extended / LAEA Europe".to_string(),
            category: "Regional",
        });
        presets.push(CrsPreset {
            epsg: 27700,
            name: "OSGB36 / British National Grid".to_string(),
            category: "Regional",
        });
        presets.push(CrsPreset {
            epsg: 2154,
            name: "RGF93 / Lambert-93 (France)".to_string(),
            category: "Regional",
        });
        presets.push(CrsPreset {
            epsg: 25832,
            name: "ETRS89 / UTM zone 32N".to_string(),
            category: "Regional",
        });
        presets.push(CrsPreset {
            epsg: 25833,
            name: "ETRS89 / UTM zone 33N".to_string(),
            category: "Regional",
        });
        presets.push(CrsPreset {
            epsg: 3413,
            name: "WGS 84 / NSIDC Sea Ice Polar Stereographic North".to_string(),
            category: "Polar",
        });
        presets.push(CrsPreset {
            epsg: 3031,
            name: "WGS 84 / Antarctic Polar Stereographic".to_string(),
            category: "Polar",
        });
        presets.push(CrsPreset {
            epsg: 26910,
            name: "NAD83 / UTM zone 10N".to_string(),
            category: "North America",
        });
        presets.push(CrsPreset {
            epsg: 26918,
            name: "NAD83 / UTM zone 18N".to_string(),
            category: "North America",
        });
        presets.push(CrsPreset {
            epsg: 2263,
            name: "NAD83 / New York Long Island (ftUS)".to_string(),
            category: "North America",
        });
        for zone in MSK61_ZONES {
            presets.push(CrsPreset {
                epsg: MSK61_EPSG_BASE + zone,
                name: format!("MSK-61 / Pulkovo 1942 zone {zone}"),
                category: "MSK / Pulkovo",
            });
        }
        for zone in 2..=32 {
            presets.push(CrsPreset {
                epsg: 28_400 + zone,
                name: format!("Pulkovo 1942 / Gauss-Kruger zone {zone}"),
                category: "MSK / Pulkovo",
            });
        }
        for zone in 1..=60 {
            presets.push(CrsPreset {
                epsg: 32_600 + zone,
                name: format!("WGS 84 / UTM zone {zone}N"),
                category: "UTM North",
            });
        }
        for zone in 1..=60 {
            presets.push(CrsPreset {
                epsg: 32_700 + zone,
                name: format!("WGS 84 / UTM zone {zone}S"),
                category: "UTM South",
            });
        }

        let search_keys = presets
            .iter()
            .map(|preset| {
                format!(
                    "epsg:{} {} {} {}",
                    preset.epsg,
                    preset.code_label().to_ascii_lowercase(),
                    preset.name.to_ascii_lowercase(),
                    preset.category.to_ascii_lowercase()
                )
            })
            .collect();

        PresetCatalog {
            presets,
            search_keys,
        }
    })
}

/// Curated searchable CRS preset catalog (WGS84, Web Mercator, UTM 1–60 N/S,
/// Pulkovo 1942 GK 2–32, MSK-61 1–3, and major international projected CRSs).
pub fn builtin_crs_presets() -> &'static [CrsPreset] {
    &preset_catalog().presets
}

/// Search curated presets (plus direct numeric EPSG lookup in `crs-definitions`)
/// matching `query` case-insensitively.
pub fn search_crs_presets(query: &str, limit: usize) -> Vec<CrsPreset> {
    if limit == 0 {
        return Vec::new();
    }
    let catalog = preset_catalog();
    let q = query.trim().to_ascii_lowercase();
    let mut results: Vec<CrsPreset> = catalog
        .presets
        .iter()
        .zip(&catalog.search_keys)
        .filter(|(_, key)| q.is_empty() || key.contains(&q))
        .take(limit)
        .map(|(preset, _)| preset.clone())
        .collect();

    // Allow direct lookup of any EPSG code in `crs-definitions` even if not in the curated list.
    let digits = q.strip_prefix("epsg:").unwrap_or(&q).trim();
    if let Ok(code) = digits.parse::<u32>()
        && !results.iter().any(|p| p.epsg == code)
        && CrsSpec::from_epsg(code).is_ok()
    {
        results.insert(
            0,
            CrsPreset {
                epsg: code,
                name: format!("EPSG:{code} (crs-definitions)"),
                category: "EPSG Catalog",
            },
        );
        results.truncate(limit);
    }
    results
}
