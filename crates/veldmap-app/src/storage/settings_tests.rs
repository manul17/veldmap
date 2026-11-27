use veldmap_storage::test_support::TestTempDir;

use super::*;

#[test]
fn settings_round_trip_and_redact_api_key_userinfo_and_query_in_debug() {
    let dir = TestTempDir::new("settings-rt");
    let paths = StoragePaths::with_root(dir.path());

    let mut settings = AppSettings {
        active_basemap: BasemapSelection::CustomXyz,
        gibs_date: " 2024-08-15 ".to_owned(),
        custom_xyz: CustomXyzSettings {
            name: "My Tile Server".to_owned(),
            url_template:
                "https://alice:pw@tiles.example.com/{z}/{x}/{y}.png?inline=leak-me&key={key}"
                    .to_owned(),
            api_key: "super-secret-token-123".to_owned(),
            tms_flip_y: true,
            max_level: 99, // clamped to CUSTOM_MAX_LEVEL_MAX (20)
            attribution: "Example".to_owned(),
        },
        disk_cache_max_mb: 256,
        show_graticule: false,
        ..AppSettings::default()
    };

    let dbg = format!("{settings:?}");
    assert!(!dbg.contains("super-secret-token-123"));
    assert!(!dbg.contains("alice:pw"));
    assert!(!dbg.contains("leak-me"));
    assert!(dbg.contains("<redacted>"));

    settings.save(&paths).expect("save settings");
    let loaded = AppSettings::load_or_default(&paths);
    settings.normalize();
    assert_eq!(loaded, settings);
    assert_eq!(loaded.gibs_date, "2024-08-15");
    assert_eq!(loaded.custom_xyz.max_level, CUSTOM_MAX_LEVEL_MAX);
}

#[test]
fn corrupt_json_backs_up_file_and_unknown_basemap_variant_is_forward_compatible() {
    let dir = TestTempDir::new("settings-recovery");
    let paths = StoragePaths::with_root(dir.path());
    let settings_path = paths.settings_path();
    std::fs::create_dir_all(settings_path.parent().expect("parent")).expect("mkdir");

    std::fs::write(&settings_path, b"{not valid json").expect("write corrupt");
    let loaded = AppSettings::load_or_default(&paths);
    assert_eq!(loaded, AppSettings::default());
    let backup = settings_path.with_extension("json.corrupt");
    assert_eq!(
        std::fs::read(&backup).expect("backup preserved"),
        b"{not valid json"
    );

    std::fs::write(
        &settings_path,
        br#"{"active_basemap":"future_provider_v9","disk_cache_max_mb":999999,"gibs_date":"   "}"#,
    )
    .expect("write forward-compat json");
    let compat = AppSettings::load_or_default(&paths);
    assert_eq!(compat.active_basemap, BasemapSelection::BundledBlueMarble);
    assert_eq!(compat.disk_cache_max_mb, MAX_DISK_CACHE_MAX_MB);
    assert_eq!(compat.gibs_date, DEFAULT_GIBS_DATE);
    assert!(compat.show_graticule);

    // Non-string enum JSON still triggers `.corrupt` backup instead of silently succeeding.
    let _ = std::fs::remove_file(&backup);
    std::fs::write(&settings_path, br#"{"active_basemap":123}"#).expect("write non-string json");
    let invalid_type = AppSettings::load_or_default(&paths);
    assert_eq!(invalid_type, AppSettings::default());
    assert_eq!(
        std::fs::read(&backup).expect("non-string backup preserved"),
        br#"{"active_basemap":123}"#
    );
}

#[test]
fn basemap_draft_preserves_unrelated_app_settings_and_validates_partial_edits() {
    let mut settings = AppSettings {
        show_graticule: false,
        ..AppSettings::default()
    };
    let mut draft = AppSettingsDraft::from_settings(&settings);
    draft.active_basemap = BasemapSelection::EoxSentinel2Cloudless;
    draft.disk_cache_max_mb = 1024;

    // Applying basemap draft must leave unrelated `show_graticule` intact.
    draft.apply_into(&mut settings);
    assert!(!settings.show_graticule);
    assert_eq!(
        settings.active_basemap,
        BasemapSelection::EoxSentinel2Cloudless
    );
    assert_eq!(settings.disk_cache_max_mb, 1024);

    // Applying draft with an invalid GIBS date while GIBS VIIRS is inactive
    // preserves the existing valid GIBS date instead of persisting a typo.
    let prev_date = settings.gibs_date.clone();
    draft.gibs_date = "not-a-date".to_owned();
    draft.disk_cache_max_mb = 2048;
    draft.apply_into(&mut settings);
    assert_eq!(settings.gibs_date, prev_date);
    assert_eq!(settings.disk_cache_max_mb, 2048);

    // Saving a valid Custom XYZ URL while CustomXyz is inactive persists it.
    draft.custom_xyz.url_template = "https://tiles.example.org/{z}/{x}/{y}.png".to_owned();
    draft.apply_into(&mut settings);
    assert_eq!(
        settings.custom_xyz.url_template,
        "https://tiles.example.org/{z}/{x}/{y}.png"
    );

    // A half-typed invalid URL while CustomXyz is inactive must preserve
    // the existing saved Custom XYZ configuration.
    let saved_custom = settings.custom_xyz.clone();
    draft.custom_xyz.url_template = "https://incomplete-no-placeholders".to_owned();
    draft.custom_xyz.name = "Half-typed".to_owned();
    draft.apply_into(&mut settings);
    assert_eq!(settings.custom_xyz, saved_custom);

    // Intentionally clearing the URL to empty while CustomXyz is inactive
    // IS applied so the user can clear a saved Custom XYZ source.
    draft.custom_xyz.url_template = "   ".to_owned();
    draft.apply_into(&mut settings);
    assert_eq!(settings.custom_xyz.url_template, "   ");
}

#[test]
fn gpu_tile_cache_cap_clamps_persists_and_loads_legacy_settings() {
    let mut settings = AppSettings::default();
    assert_eq!(settings.gpu_tile_cache_max_mb, None);
    assert_eq!(
        settings.effective_gpu_tile_cache_bytes(512 * 1024 * 1024),
        512 * 1024 * 1024
    );

    settings.gpu_tile_cache_max_mb = Some(16);
    settings.normalize();
    assert_eq!(settings.gpu_tile_cache_max_mb, Some(MIN_GPU_CACHE_MAX_MB));
    assert_eq!(
        settings.effective_gpu_tile_cache_bytes(512 * 1024 * 1024),
        (MIN_GPU_CACHE_MAX_MB as usize) * 1024 * 1024
    );

    settings.gpu_tile_cache_max_mb = Some(99_999);
    settings.normalize();
    assert_eq!(settings.gpu_tile_cache_max_mb, Some(MAX_GPU_CACHE_MAX_MB));
    assert!(settings.effective_gpu_tile_cache_bytes(512 * 1024 * 1024) >= 128 * 1024 * 1024);

    let mut draft = AppSettingsDraft::from_settings(&settings);
    draft.gpu_tile_cache_max_mb = Some(768);
    draft.apply_into(&mut settings);
    assert_eq!(settings.gpu_tile_cache_max_mb, Some(768));

    // Legacy settings JSON without `gpu_tile_cache_max_mb` deserializes cleanly to None (Auto).
    let dir = TestTempDir::new("settings-legacy-gpu");
    let paths = StoragePaths::with_root(dir.path());
    let settings_path = paths.settings_path();
    std::fs::create_dir_all(settings_path.parent().expect("parent")).expect("mkdir");
    std::fs::write(
        &settings_path,
        r#"{"active_basemap":"bundled_blue_marble"}"#,
    )
    .expect("write legacy settings");
    let legacy = AppSettings::load_or_default(&paths);
    assert_eq!(legacy.gpu_tile_cache_max_mb, None);
    assert_eq!(legacy.stac, StacSettings::default());
}

#[test]
fn stac_settings_round_trip_and_redact_secrets_in_debug() {
    let dir = TestTempDir::new("settings-stac");
    let paths = StoragePaths::with_root(dir.path());

    let mut settings = AppSettings::default();
    let mut draft = AppSettingsDraft::from_settings(&settings);
    draft.stac = StacSettings {
        custom_endpoint: "https://user:pw@stac.example.org/v1?sig=hidden-query".to_owned(),
        custom_collection: "sentinel-2-l2a".to_owned(),
        auth_kind: StacAuthKind::Bearer,
        auth_header_name: "x-api-key".to_owned(),
        auth_secret: "stac-bearer-secret-xyz".to_owned(),
    };
    draft.apply_into(&mut settings);

    let dbg = format!("{settings:?}");
    assert!(!dbg.contains("stac-bearer-secret-xyz"));
    assert!(!dbg.contains("user:pw"));
    assert!(!dbg.contains("hidden-query"));
    assert!(dbg.contains("<redacted>"));

    settings.save(&paths).expect("save stac settings");
    let loaded = AppSettings::load_or_default(&paths);
    assert_eq!(loaded.stac, settings.stac);
    assert_eq!(
        loaded.stac.to_stac_auth(),
        StacAuth::Bearer("stac-bearer-secret-xyz".to_owned())
    );
}

#[test]
fn stac_settings_forward_compat_normalization_validation_and_secret_scrub() {
    // Unknown future StacAuthKind deserializes forward-compatibly to None.
    let dir = TestTempDir::new("settings-stac-compat");
    let paths = StoragePaths::with_root(dir.path());
    let settings_path = paths.settings_path();
    std::fs::create_dir_all(settings_path.parent().expect("parent")).expect("mkdir");
    std::fs::write(
        &settings_path,
        br#"{"stac":{"custom_endpoint":"  https://stac.example/v1  ","auth_kind":"sigv4_future","auth_secret":"leftover"}}"#,
    )
    .expect("write future auth");
    let loaded = AppSettings::load_or_default(&paths);
    assert_eq!(loaded.stac.auth_kind, StacAuthKind::None);
    assert_eq!(loaded.stac.custom_endpoint, "https://stac.example/v1");
    assert!(
        loaded.stac.auth_secret.is_empty(),
        "switching/falling back to None must scrub auth_secret"
    );

    // Invalid STAC endpoint or forbidden header fails validation and does not
    // clobber previously saved valid STAC settings via AppSettingsDraft.
    let mut settings = loaded;
    let saved = settings.stac.clone();
    let mut draft = AppSettingsDraft::from_settings(&settings);
    draft.stac.custom_endpoint = "ftp://not-http.example/v1".to_owned();
    assert!(draft.stac.validation_error().is_some());
    draft.apply_into(&mut settings);
    assert_eq!(settings.stac, saved);

    draft.stac.custom_endpoint = "https://stac.example/v1".to_owned();
    draft.stac.auth_kind = StacAuthKind::ApiKey;
    draft.stac.auth_header_name = "Range".to_owned(); // protocol-owned
    assert!(draft.stac.validation_error().is_some());
    draft.stac.auth_header_name = "x-api-key".to_owned();
    assert!(draft.stac.validation_error().is_none());
}
