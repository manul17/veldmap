use super::*;
use crate::test_support::TestTempDir;

#[test]
fn write_atomic_durable_creates_overwrites_and_leaves_no_temp_files() {
    let dir = TestTempDir::new("atomic");
    let target = dir.path().join("nested").join("settings.json");

    write_atomic_durable(&target, b"{\"v\":1}\n").expect("first write");
    assert_eq!(fs::read(&target).expect("read v1"), b"{\"v\":1}\n");

    // Overwrite existing file cleanly (including Windows replace semantics).
    write_atomic_durable(&target, b"{\"v\":2,\"ok\":true}\n").expect("second write");
    assert_eq!(
        fs::read(&target).expect("read v2"),
        b"{\"v\":2,\"ok\":true}\n"
    );

    let entries: Vec<_> = fs::read_dir(target.parent().expect("parent"))
        .expect("list dir")
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        entries,
        vec!["settings.json"],
        "stray temp files remained: {entries:?}"
    );
}

#[test]
fn private_write_sets_owner_only_mode_and_cleans_up_on_rename_failure() {
    let dir = TestTempDir::new("atomic-private");
    let secret_path = dir.path().join("secret.json");
    write_atomic_impl(&secret_path, b"{\"k\":1}\n", true).expect("private write");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&secret_path)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "private file must be 0600, got {mode:o}");
    }

    // Renaming onto an existing directory must fail and clean up its temp file.
    let dir_target = dir.path().join("target_is_dir");
    fs::create_dir_all(&dir_target).expect("create blocking dir");
    assert!(write_atomic_durable(&dir_target, b"payload").is_err());

    let leftovers: Vec<_> = fs::read_dir(dir.path())
        .expect("list dir")
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".tmp."))
        .collect();
    assert!(leftovers.is_empty(), "temp file leaked: {leftovers:?}");

    // Invalid path with no filename fails cleanly.
    assert!(matches!(
        write_atomic_durable(Path::new(""), b"x"),
        Err(StorageError::InvalidPath(_))
    ));
}

#[test]
fn read_json_or_default_handles_missing_and_valid_files() {
    #[derive(Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct Sample {
        count: u32,
    }

    let dir = TestTempDir::new("json");
    let path = dir.path().join("sample.json");

    let missing: Sample = read_json_or_default(&path).expect("missing returns default");
    assert_eq!(missing, Sample { count: 0 });

    write_json_durable_private(&path, &Sample { count: 42 }).expect("write json");
    let loaded: Sample = read_json_or_default(&path).expect("read json");
    assert_eq!(loaded, Sample { count: 42 });

    // Corrupt JSON is backed up to `<path>.corrupt` and falls back to Default.
    fs::write(&path, b"{not valid json").expect("write corrupt");
    let fallback: Sample = read_json_or_default_with_backup(&path);
    assert_eq!(fallback, Sample { count: 0 });
    let backup = path.with_extension("json.corrupt");
    assert_eq!(
        fs::read(&backup).expect("backup preserved"),
        b"{not valid json"
    );
}

#[test]
fn write_json_durable_and_read_json_limited_enforce_limits_without_renaming_corrupt_files() {
    #[derive(Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct Project {
        name: String,
    }

    let dir = TestTempDir::new("json-limited");
    let path = dir.path().join("study.veldmap.json");

    assert!(matches!(
        read_json_limited::<Project>(&path, 1024),
        Err(StorageError::Io(err)) if err.kind() == std::io::ErrorKind::NotFound
    ));

    let payload = Project {
        name: "alpha".to_owned(),
    };
    write_json_durable(&path, &payload).expect("write durable json");
    let loaded: Project = read_json_limited(&path, 1024).expect("read limited json");
    assert_eq!(loaded, payload);

    assert!(matches!(
        read_json_limited::<Project>(&path, 4),
        Err(StorageError::FileTooLarge { .. })
    ));

    fs::write(&path, b"{broken").expect("write invalid json");
    assert!(matches!(
        read_json_limited::<Project>(&path, 1024),
        Err(StorageError::Json(_))
    ));
    assert!(path.exists(), "invalid project file must remain in place");
    assert!(
        !append_path_suffix(&path, ".corrupt").exists(),
        "read_json_limited must not create a .corrupt backup"
    );
}

#[test]
fn read_json_or_default_backs_up_invalid_utf8_and_rejects_oversized_or_non_regular() {
    #[derive(Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct Sample {
        count: u32,
    }

    let dir = TestTempDir::new("json-bounded");

    // Invalid UTF-8 bytes must still produce `StorageError::Json` and trigger
    // `.corrupt` backup in `read_json_or_default_with_backup`.
    let bad_utf8 = dir.path().join("bad_utf8.json");
    fs::write(&bad_utf8, [0xFF, 0xFE, 0xFD]).expect("write invalid utf8");
    assert!(matches!(
        read_json_or_default::<Sample>(&bad_utf8),
        Err(StorageError::Json(_))
    ));
    let fallback: Sample = read_json_or_default_with_backup(&bad_utf8);
    assert_eq!(fallback, Sample::default());
    let backup = append_path_suffix(&bad_utf8, ".corrupt");
    assert_eq!(
        fs::read(&backup).expect("invalid utf8 backup preserved"),
        vec![0xFF, 0xFE, 0xFD]
    );

    // Non-regular paths (directories) are rejected with `InvalidPath` and
    // never renamed to `.corrupt`.
    let sub_dir = dir.path().join("not_a_file.json");
    fs::create_dir_all(&sub_dir).expect("create dir");
    assert!(matches!(
        read_json_or_default::<Sample>(&sub_dir),
        Err(StorageError::InvalidPath(_))
    ));
    let dir_fallback: Sample = read_json_or_default_with_backup(&sub_dir);
    assert_eq!(dir_fallback, Sample::default());
    assert!(sub_dir.is_dir(), "directory must not be renamed");
    assert!(!append_path_suffix(&sub_dir, ".corrupt").exists());

    // Oversized files are rejected with `FileTooLarge`.
    let big = dir.path().join("big.json");
    fs::write(&big, b"{\"count\":123456}").expect("write json");
    assert!(matches!(
        read_json_or_default_limited::<Sample>(&big, 8),
        Err(StorageError::FileTooLarge { limit_bytes: 8, .. })
    ));
}
