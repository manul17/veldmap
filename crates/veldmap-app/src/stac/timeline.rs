//! Multi-temporal STAC AOI timeline grouping (`mgrs_tile`, preferring L2A
//! surface reflectance when multiple processing levels exist) and chronological
//! epoch request construction.

use std::collections::BTreeMap;

use veldmap_net::StacItem;

use crate::layers::{
    BandMathAssetOpenSpec, BandMathPreset, StacCogAuthMode, TimelineEpochSourceSpec,
    TimelineEpochSpec, TimelineOpenRequest,
};
use crate::storage::settings::StacSettings;

use super::auth::{AuthTrust, build_stac_item_refresh_policy};
use super::{StacItemView, StacRecipeView};

/// Asset/recipe mode available when constructing a STAC multi-temporal timeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum StacTimelineMode {
    VisualCog,
    Recipe(BandMathPreset),
}

impl StacTimelineMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::VisualCog => "Visual / Preferred COG",
            Self::Recipe(preset) => preset.short_label(),
        }
    }
}

/// UI-facing summary of one MGRS-tile (or AOI) candidate group inside STAC
/// search results with `>= 2` chronologically ordered scenes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StacTimelineGroupView {
    pub(crate) group_key: String,
    pub(crate) scene_count: usize,
    pub(crate) date_span: String,
    pub(crate) available_modes: Vec<StacTimelineMode>,
}

fn group_key_for_item(item: &StacItem) -> String {
    item.mgrs_tile
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("AOI")
        .to_owned()
}

fn short_date_for_item(item: &StacItem) -> String {
    item.datetime
        .as_deref()
        .map(|dt| dt.split('T').next().unwrap_or(dt))
        .unwrap_or(&item.id)
        .to_owned()
}

fn sort_key_for_item(item: &StacItem) -> String {
    item.datetime
        .as_deref()
        .map(|dt| format!("0:{dt}:{}", item.id))
        .unwrap_or_else(|| format!("1:{}", item.id))
}

fn filter_group_items<'a>(
    items: &'a [StacItem],
    views: &'a [StacItemView],
    group_key: &str,
) -> Vec<(&'a StacItem, &'a StacItemView)> {
    let mut matched: Vec<(&StacItem, &StacItemView)> = items
        .iter()
        .zip(views.iter())
        .filter(|(item, _)| group_key_for_item(item) == group_key)
        .collect();

    // Default to L2A / surface reflectance when a group mixes processing levels
    // (e.g. L1C + L2A) so dual-epoch radiometric diffs stay physically comparable.
    let l2a_count = matched
        .iter()
        .filter(|(_, v)| v.is_surface_reflectance)
        .count();
    if l2a_count >= 2 && l2a_count < matched.len() {
        matched.retain(|(_, v)| v.is_surface_reflectance);
    }

    matched.sort_by_key(|(a, _)| sort_key_for_item(a));
    matched
}

/// Group STAC search results by `mgrs_tile` (or `"AOI"` fallback) into
/// candidate multi-temporal timeline groups with `>= 2` chronologically
/// ordered scenes and their supported visual/recipe modes.
pub(super) fn build_timeline_groups(
    items: &[StacItem],
    views: &[StacItemView],
) -> Vec<StacTimelineGroupView> {
    let mut keys: BTreeMap<String, usize> = BTreeMap::new();
    for item in items {
        *keys.entry(group_key_for_item(item)).or_insert(0) += 1;
    }

    let mut groups = Vec::new();
    for (group_key, raw_count) in keys {
        if raw_count < 2 {
            continue;
        }
        let filtered = filter_group_items(items, views, &group_key);
        if filtered.len() < 2 {
            continue;
        }
        let mut available_modes = Vec::new();
        let visual_count = filtered
            .iter()
            .filter(|(_, v)| v.preferred_cog.is_some())
            .count();
        if visual_count >= 2 {
            available_modes.push(StacTimelineMode::VisualCog);
        }
        for preset in [
            BandMathPreset::Ndvi,
            BandMathPreset::Nbr,
            BandMathPreset::Ndwi,
            BandMathPreset::Ndsi,
            BandMathPreset::TrueColor,
            BandMathPreset::FalseColorNir,
            BandMathPreset::Swir,
        ] {
            let recipe_count = filtered
                .iter()
                .filter(|(_, v)| v.recipes.iter().any(|r| r.preset == preset))
                .count();
            if recipe_count >= 2 {
                available_modes.push(StacTimelineMode::Recipe(preset));
            }
        }
        if available_modes.is_empty() {
            continue;
        }
        let (Some((first_item, _)), Some((last_item, _))) = (filtered.first(), filtered.last())
        else {
            continue;
        };
        let first_date = short_date_for_item(first_item);
        let last_date = short_date_for_item(last_item);
        let date_span = if first_date == last_date {
            first_date
        } else {
            format!("{first_date} … {last_date}")
        };
        groups.push(StacTimelineGroupView {
            group_key,
            scene_count: filtered.len(),
            date_span,
            available_modes,
        });
    }
    groups.sort_by(|a, b| {
        b.scene_count
            .cmp(&a.scene_count)
            .then_with(|| a.group_key.cmp(&b.group_key))
    });
    groups
}

fn find_recipe(view: &StacItemView, preset: BandMathPreset) -> Option<&StacRecipeView> {
    view.recipes.iter().find(|r| r.preset == preset)
}

/// Construct a [`TimelineOpenRequest`] from currently loaded STAC search
/// results for `group_key` and `mode`, resolving each COG URL and header set
/// through `resolve_auth` (honoring the active search session's profile auth).
pub(super) fn build_stac_timeline_request(
    items: &[StacItem],
    views: &[StacItemView],
    group_key: &str,
    mode: StacTimelineMode,
    auth_mode: StacCogAuthMode,
    stac_settings: &StacSettings,
    resolve_auth: impl Fn(&str, StacCogAuthMode) -> (String, Vec<(String, String)>),
) -> Result<TimelineOpenRequest, String> {
    let filtered = filter_group_items(items, views, group_key);
    let mut epochs = Vec::new();
    for (item, view) in filtered {
        let date_label = short_date_for_item(item);
        let sort_key = sort_key_for_item(item);
        let source = match mode {
            StacTimelineMode::VisualCog => {
                let Some(asset) = &view.preferred_cog else {
                    continue;
                };
                let (url, extra_headers) = resolve_auth(asset.href.as_str(), auth_mode);
                let refresh_policy = build_stac_item_refresh_policy(
                    view.self_href.as_deref(),
                    Some(asset.key.as_str()),
                    AuthTrust::LocalSession,
                    stac_settings,
                );
                TimelineEpochSourceSpec::RemoteSingle {
                    url,
                    auth_mode,
                    extra_headers,
                    external_bands: asset.external_bands.clone(),
                    is_scl: asset.is_scl,
                    stac_item_href: view.self_href.clone(),
                    asset_key: Some(asset.key.clone()),
                    refresh_policy,
                }
            }
            StacTimelineMode::Recipe(preset) => {
                let Some(recipe) = find_recipe(view, preset) else {
                    continue;
                };
                let bands = recipe
                    .bands
                    .iter()
                    .map(|b| {
                        let (url, extra_headers) = resolve_auth(b.href.as_str(), auth_mode);
                        let refresh_policy = build_stac_item_refresh_policy(
                            b.stac_item_href.as_deref(),
                            b.asset_key.as_deref(),
                            AuthTrust::LocalSession,
                            stac_settings,
                        );
                        BandMathAssetOpenSpec::Remote {
                            label: b.label.clone(),
                            url,
                            auth_mode,
                            extra_headers,
                            external_bands: b.external_bands.clone(),
                            stac_item_href: b.stac_item_href.clone(),
                            asset_key: b.asset_key.clone(),
                            refresh_policy,
                            band_index: 0,
                        }
                    })
                    .collect();
                let scl = recipe.scl.as_ref().map(|s| {
                    let (url, extra_headers) = resolve_auth(s.href.as_str(), auth_mode);
                    let refresh_policy = build_stac_item_refresh_policy(
                        s.stac_item_href.as_deref(),
                        s.asset_key.as_deref(),
                        AuthTrust::LocalSession,
                        stac_settings,
                    );
                    BandMathAssetOpenSpec::Remote {
                        label: s.label.clone(),
                        url,
                        auth_mode,
                        extra_headers,
                        external_bands: s.external_bands.clone(),
                        stac_item_href: s.stac_item_href.clone(),
                        asset_key: s.asset_key.clone(),
                        refresh_policy,
                        band_index: 0,
                    }
                });
                TimelineEpochSourceSpec::BandMath { bands, scl }
            }
        };
        epochs.push(TimelineEpochSpec {
            label: item.id.clone(),
            date_label,
            sort_key,
            cloud_cover: item.cloud_cover,
            source,
        });
    }

    if epochs.len() < 2 {
        return Err(format!(
            "group '{group_key}' has fewer than 2 scenes supporting {}",
            mode.label()
        ));
    }

    let recipe_preset = match mode {
        StacTimelineMode::VisualCog => None,
        StacTimelineMode::Recipe(p) => Some(p),
    };
    let title = format!("{} · {} Timeline", group_key, mode.label());
    Ok(TimelineOpenRequest {
        title,
        epochs,
        recipe_preset,
        band_math_style: None,
        single_style: None,
        visible: true,
        opacity: 1.0,
        active_idx: None,
        compare: None,
        max_cloud_cover: None,
        diff_colormap: None,
        diff_invert_colormap: false,
        diff_stretch_min: None,
        diff_stretch_max: None,
        fly_to: true,
        restore_id: None,
    })
}

#[cfg(test)]
mod tests {
    use veldmap_net::StacAsset;

    use super::super::views::build_item_view;
    use super::*;

    fn make_s2_item(id: &str, dt: &str, tile: &str, level: &str, cloud: f32) -> StacItem {
        let mk_asset = |key: &str| StacAsset {
            key: key.to_owned(),
            title: Some(key.to_owned()),
            href: format!("https://example.com/{id}/{key}.tif"),
            media_type: Some("image/tiff; application=geotiff; profile=cloud-optimized".to_owned()),
            roles: vec!["data".to_owned()],
            gsd: Some(10.0),
            eo_bands: Vec::new(),
            raster_bands: Vec::new(),
        };
        StacItem {
            id: id.to_owned(),
            collection: Some("sentinel-2-l2a".to_owned()),
            self_href: Some(format!("https://stac.example.com/items/{id}")),
            datetime: Some(dt.to_owned()),
            cloud_cover: Some(cloud),
            platform: Some("sentinel-2a".to_owned()),
            processing_level: Some(level.to_owned()),
            mgrs_tile: Some(tile.to_owned()),
            bbox: Some([7.0, 46.0, 8.0, 47.0]),
            geometry: None,
            assets: vec![
                mk_asset("visual"),
                mk_asset("red"),
                mk_asset("nir"),
                mk_asset("scl"),
            ],
        }
    }

    #[test]
    fn timeline_grouping_clusters_by_mgrs_tile_prefers_l2a_and_sorts_chronologically() {
        let items = vec![
            make_s2_item("s2_aug", "2024-08-20T10:00:00Z", "32TMS", "L2A", 5.0),
            make_s2_item("s2_l1c", "2024-07-01T10:00:00Z", "32TMS", "L1C", 2.0),
            make_s2_item("s2_may", "2024-05-10T10:00:00Z", "32TMS", "L2A", 12.0),
            make_s2_item("s2_other_tile", "2024-06-01T10:00:00Z", "32TMT", "L2A", 0.0),
        ];
        let views: Vec<StacItemView> = items.iter().map(build_item_view).collect();
        let groups = build_timeline_groups(&items, &views);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].group_key, "32TMS");
        assert_eq!(groups[0].scene_count, 2); // L1C filtered out because >=2 L2A exist
        assert_eq!(groups[0].date_span, "2024-05-10 … 2024-08-20");
        assert!(
            groups[0]
                .available_modes
                .contains(&StacTimelineMode::VisualCog)
        );
        assert!(
            groups[0]
                .available_modes
                .contains(&StacTimelineMode::Recipe(BandMathPreset::Ndvi))
        );

        let req = build_stac_timeline_request(
            &items,
            &views,
            "32TMS",
            StacTimelineMode::Recipe(BandMathPreset::Ndvi),
            StacCogAuthMode::None,
            &StacSettings::default(),
            |u, _| (u.to_owned(), Vec::new()),
        )
        .expect("build request");
        assert_eq!(req.epochs.len(), 2);
        assert_eq!(req.epochs[0].date_label, "2024-05-10");
        assert_eq!(req.epochs[1].date_label, "2024-08-20");
    }
}
