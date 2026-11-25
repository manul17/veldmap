//! Curated searchable CRS presets for the UI CRS picker.

use std::sync::OnceLock;

use super::projection::{
    GK_PULKOVO_EPSG_BASE, GK_PULKOVO_ZONES, MSK61_EPSG_BASE, MSK61_ZONES, UTM_NORTH_EPSG_BASE,
    UTM_SOUTH_EPSG_BASE, UTM_ZONES,
};
use super::{CrsSpec, parse_epsg_shorthand};

/// Searchable CRS preset entry for the UI CRS picker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrsPreset {
    pub epsg: u32,
    pub name: String,
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

const STATIC_PRESETS: &[(u32, &str, &str)] = &[
    (4326, "WGS 84 (Geographic lon/lat)", "Global"),
    (3857, "WGS 84 / Pseudo-Mercator (Web Mercator)", "Global"),
    (4284, "Pulkovo 1942 (Geographic lon/lat)", "Regional"),
    (3035, "ETRS89-extended / LAEA Europe", "Regional"),
    (27700, "OSGB36 / British National Grid", "Regional"),
    (2154, "RGF93 / Lambert-93 (France)", "Regional"),
    (25832, "ETRS89 / UTM zone 32N", "Regional"),
    (25833, "ETRS89 / UTM zone 33N", "Regional"),
    (
        3413,
        "WGS 84 / NSIDC Sea Ice Polar Stereographic North",
        "Polar",
    ),
    (3031, "WGS 84 / Antarctic Polar Stereographic", "Polar"),
    (26910, "NAD83 / UTM zone 10N", "North America"),
    (26918, "NAD83 / UTM zone 18N", "North America"),
    (2263, "NAD83 / New York Long Island (ftUS)", "North America"),
];

fn preset_catalog() -> &'static PresetCatalog {
    static CATALOG: OnceLock<PresetCatalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        let mut entries: Vec<(CrsPreset, &'static str)> = Vec::with_capacity(170);
        entries.extend(STATIC_PRESETS.iter().map(|&(epsg, name, category)| {
            (
                CrsPreset {
                    epsg,
                    name: name.to_string(),
                },
                category,
            )
        }));
        for zone in MSK61_ZONES {
            entries.push((
                CrsPreset {
                    epsg: MSK61_EPSG_BASE + zone,
                    name: format!("MSK-61 / Pulkovo 1942 zone {zone}"),
                },
                "MSK / Pulkovo",
            ));
        }
        for zone in GK_PULKOVO_ZONES {
            entries.push((
                CrsPreset {
                    epsg: GK_PULKOVO_EPSG_BASE + zone,
                    name: format!("Pulkovo 1942 / Gauss-Kruger zone {zone}"),
                },
                "MSK / Pulkovo",
            ));
        }
        for zone in UTM_ZONES {
            entries.push((
                CrsPreset {
                    epsg: UTM_NORTH_EPSG_BASE + zone,
                    name: format!("WGS 84 / UTM zone {zone}N"),
                },
                "UTM North",
            ));
        }
        for zone in UTM_ZONES {
            entries.push((
                CrsPreset {
                    epsg: UTM_SOUTH_EPSG_BASE + zone,
                    name: format!("WGS 84 / UTM zone {zone}S"),
                },
                "UTM South",
            ));
        }

        let search_keys = entries
            .iter()
            .map(|(preset, category)| {
                format!(
                    "epsg:{} {} {} {}",
                    preset.epsg,
                    preset.code_label().to_ascii_lowercase(),
                    preset.name.to_ascii_lowercase(),
                    category.to_ascii_lowercase()
                )
            })
            .collect();
        let presets = entries.into_iter().map(|(preset, _)| preset).collect();

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
    if let Some(code) = parse_epsg_shorthand(&q)
        && !results.iter().any(|p| p.epsg == code)
        && CrsSpec::from_epsg(code).is_ok()
    {
        results.insert(
            0,
            CrsPreset {
                epsg: code,
                name: format!("EPSG:{code} (crs-definitions)"),
            },
        );
        results.truncate(limit);
    }
    results
}
