//! Local dated-folder raster scanner and deterministic chronological date
//! parser for multi-temporal `Timeline` ingestion.
//!
//! Date extraction priority:
//! 1. ISO `YYYY-MM-DD` or compact `YYYYMMDD` embedded in filename stem.
//! 2. TIFF `DateTime` ASCII tag (`306`, `YYYY:MM:DD HH:MM:SS` or `YYYY-MM-DD`)
//!    via lightweight base-IFD header read (`veldmap_raster::read_tiff_datetime`).
//! 3. Lexicographic filename fallback.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use veldmap_core::sync::CancelToken;
use veldmap_raster::read_tiff_datetime;
use veldmap_storage::LocalFileSource;

/// Upper bound on local raster files admitted from one folder scan.
pub(crate) const MAX_FOLDER_TIMELINE_EPOCHS: usize = 200;

/// Parsed date metadata for one scanned local raster epoch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScannedTimelineFile {
    pub(crate) path: PathBuf,
    pub(crate) date_label: String,
    pub(crate) sort_key: String,
}

fn is_valid_ymd(year: u32, month: u32, day: u32) -> bool {
    if !(1970..=2100).contains(&year) || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return false;
    }
    let leap = (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400);
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    day <= max_day
}

fn parse_ascii_digits(bytes: &[u8]) -> Option<u32> {
    if bytes.is_empty() || !bytes.iter().all(u8::is_ascii_digit) {
        return None;
    }
    std::str::from_utf8(bytes).ok()?.parse::<u32>().ok()
}

fn boundary_ok(bytes: &[u8], start: usize, end: usize) -> bool {
    let prev_ok = start == 0 || !bytes[start - 1].is_ascii_digit();
    let next_ok = end >= bytes.len() || !bytes[end].is_ascii_digit();
    prev_ok && next_ok
}

/// Extract an ISO date `YYYY-MM-DD` from a filename or TIFF `DateTime` string.
///
/// Recognizes `YYYY-MM-DD`, `YYYY:MM:DD`, `YYYY_MM_DD`, and compact `YYYYMMDD`
/// (with digit-boundary guards so longer numeric IDs are not misparsed).
pub(crate) fn extract_iso_date(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    // Pass 1: separated dates (`YYYY-MM-DD`, `YYYY:MM:DD`, `YYYY_MM_DD`).
    if bytes.len() >= 10 {
        for i in 0..=bytes.len() - 10 {
            let sep1 = bytes[i + 4];
            let sep2 = bytes[i + 7];
            if (sep1 == b'-' || sep1 == b':' || sep1 == b'_')
                && sep1 == sep2
                && boundary_ok(bytes, i, i + 10)
                && let (Some(y), Some(m), Some(d)) = (
                    parse_ascii_digits(&bytes[i..i + 4]),
                    parse_ascii_digits(&bytes[i + 5..i + 7]),
                    parse_ascii_digits(&bytes[i + 8..i + 10]),
                )
                && is_valid_ymd(y, m, d)
            {
                return Some(format!("{y:04}-{m:02}-{d:02}"));
            }
        }
    }
    // Pass 2: compact `YYYYMMDD`.
    if bytes.len() >= 8 {
        for i in 0..=bytes.len() - 8 {
            if boundary_ok(bytes, i, i + 8)
                && let (Some(y), Some(m), Some(d)) = (
                    parse_ascii_digits(&bytes[i..i + 4]),
                    parse_ascii_digits(&bytes[i + 4..i + 6]),
                    parse_ascii_digits(&bytes[i + 6..i + 8]),
                )
                && is_valid_ymd(y, m, d)
            {
                return Some(format!("{y:04}-{m:02}-{d:02}"));
            }
        }
    }
    None
}

fn is_supported_timeline_raster(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| {
            ext.eq_ignore_ascii_case("tif")
                || ext.eq_ignore_ascii_case("tiff")
                || ext.eq_ignore_ascii_case("png")
                || ext.eq_ignore_ascii_case("jpg")
                || ext.eq_ignore_ascii_case("jpeg")
        })
}

fn is_tiff_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("tif") || ext.eq_ignore_ascii_case("tiff"))
}

fn probe_tiff_date(path: &Path, cancel: &CancelToken) -> Option<String> {
    if !is_tiff_ext(path) {
        return None;
    }
    let src = LocalFileSource::open(path).ok()?;
    let raw = read_tiff_datetime(Arc::new(src), cancel)?;
    extract_iso_date(&raw)
}

/// Scan a local directory non-recursively for raster files, ordering them
/// deterministically by filename ISO date -> TIFF `DateTime` tag `306` ->
/// lexicographic filename fallback.
pub(crate) fn scan_dated_raster_folder(
    dir: &Path,
    cancel: &CancelToken,
) -> Result<Vec<ScannedTimelineFile>, String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("failed to read folder '{}': {e}", dir.display()))?;
    let mut paths = Vec::new();
    for entry in entries {
        if cancel.is_cancelled() {
            return Err("folder scan cancelled".to_owned());
        }
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        if path.is_file() && is_supported_timeline_raster(&path) {
            paths.push(path);
        }
    }
    paths.sort();
    paths.truncate(MAX_FOLDER_TIMELINE_EPOCHS);

    if paths.len() < 2 {
        return Err(format!(
            "folder '{}' needs at least 2 raster files to build a timeline (found {})",
            dir.display(),
            paths.len()
        ));
    }

    let mut out = Vec::with_capacity(paths.len());
    for path in paths {
        if cancel.is_cancelled() {
            return Err("folder scan cancelled".to_owned());
        }
        let fname = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("raster")
            .to_owned();
        let stem = path.file_stem().and_then(|n| n.to_str()).unwrap_or(&fname);
        let parsed = extract_iso_date(stem).or_else(|| probe_tiff_date(&path, cancel));
        let (date_label, sort_key) = match parsed {
            Some(iso) => (iso.clone(), format!("0:{iso}:{fname}")),
            None => (stem.to_owned(), format!("1:{fname}")),
        };
        out.push(ScannedTimelineFile {
            path,
            date_label,
            sort_key,
        });
    }

    out.sort_by(|a, b| a.sort_key.cmp(&b.sort_key));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_minimal_tiff_with_datetime(path: &Path, datetime: Option<&str>) {
        // Classic little-endian TIFF with 1 IFD containing ImageWidth(256),
        // ImageLength(257), and optional DateTime(306) ASCII tag.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // first IFD at offset 8

        let dt_bytes: Option<Vec<u8>> = datetime.map(|s| {
            let mut b = s.as_bytes().to_vec();
            b.push(0);
            b
        });
        let entry_count: u16 = if dt_bytes.is_some() { 3 } else { 2 };
        buf.extend_from_slice(&entry_count.to_le_bytes());
        let data_offset = 8u32 + 2 + u32::from(entry_count) * 12 + 4;

        // Tag 256: ImageWidth = 16 (SHORT=3, count=1)
        buf.extend_from_slice(&256u16.to_le_bytes());
        buf.extend_from_slice(&3u16.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&16u32.to_le_bytes());

        // Tag 257: ImageLength = 16
        buf.extend_from_slice(&257u16.to_le_bytes());
        buf.extend_from_slice(&3u16.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&16u32.to_le_bytes());

        if let Some(ref dt) = dt_bytes {
            // Tag 306: DateTime (ASCII=2)
            buf.extend_from_slice(&306u16.to_le_bytes());
            buf.extend_from_slice(&2u16.to_le_bytes());
            buf.extend_from_slice(&(dt.len() as u32).to_le_bytes());
            buf.extend_from_slice(&data_offset.to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // next IFD = 0
        if let Some(dt) = dt_bytes {
            buf.extend_from_slice(&dt);
        }
        std::fs::write(path, buf).expect("write synthetic tiff");
    }

    #[test]
    fn extract_iso_date_parses_hyphenated_colon_compact_and_rejects_invalid() {
        assert_eq!(
            extract_iso_date("S2A_32TMS_2024-07-15_B08"),
            Some("2024-07-15".to_owned())
        );
        assert_eq!(
            extract_iso_date("S2A_32TMS_20240229_L2A"),
            Some("2024-02-29".to_owned())
        );
        assert_eq!(
            extract_iso_date("2023:11:04 09:42:10"),
            Some("2023-11-04".to_owned())
        );
        assert_eq!(extract_iso_date("S2A_20230229_not_leap"), None);
        assert_eq!(extract_iso_date("scene_123456789012"), None);
        assert_eq!(extract_iso_date("orthophoto_v2"), None);
    }

    #[test]
    fn scan_dated_raster_folder_orders_filename_then_tiff_tag_306_then_lexicographic() {
        let dir =
            std::env::temp_dir().join(format!("veldmap_timeline_scan_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");

        // 1. Filename compact YYYYMMDD (2024-08-01)
        write_minimal_tiff_with_datetime(&dir.join("scene_20240801.tif"), None);
        // 2. No filename date, but TIFF Tag 306 = 2024:05:10 12:00:00 (must sort BEFORE 2024-08-01!)
        write_minimal_tiff_with_datetime(
            &dir.join("z_unnamed_scene.tif"),
            Some("2024:05:10 12:00:00"),
        );
        // 3. Filename ISO YYYY-MM-DD (2024-06-20)
        write_minimal_tiff_with_datetime(&dir.join("mid_2024-06-20.tif"), None);
        // 4. No filename date and no TIFF Tag 306 -> lexicographic fallback after dated items
        write_minimal_tiff_with_datetime(&dir.join("alpha_undated.tif"), None);
        write_minimal_tiff_with_datetime(&dir.join("beta_undated.tif"), None);
        // Ignored non-raster file
        std::fs::write(dir.join("notes.txt"), b"ignore").expect("write txt");

        let scanned =
            scan_dated_raster_folder(&dir, &CancelToken::new()).expect("scan folder succeeds");
        let labels: Vec<&str> = scanned.iter().map(|s| s.date_label.as_str()).collect();
        assert_eq!(
            labels,
            vec![
                "2024-05-10",
                "2024-06-20",
                "2024-08-01",
                "alpha_undated",
                "beta_undated",
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
