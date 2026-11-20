// Extracted from master:veldmodules/data-provider/src/s3.rs

#[cfg(test)]
mod tests {
    use super::*;

    fn test_identity() -> Identity {
        let credentials = aws_credential_types::Credentials::new("AK", "SK", None, None, "test");
        Identity::new(credentials, None)
    }

    /// Подпись — в заголовках, и у объекта, и у листинга: адрес объекта чист,
    /// заголовок sha256 пустого тела на месте, а подписи в строке запроса нет
    /// — хранилище такую не принимает.
    #[test]
    fn подпись_лежит_в_заголовках_а_не_в_адресе() {
        let identity = test_identity();
        let request = object(&identity, "eodata/Sentinel-2/T31TGK/TCI_10m.jp2");
        let url = Url::parse(&request.url).expect("адрес разбирается");
        assert_eq!(url.path(), "/eodata/Sentinel-2/T31TGK/TCI_10m.jp2");
        assert_eq!(url.query(), None, "у объекта строки запроса нет: {}", request.url);
        assert!(request.headers.keys().any(|name| name.eq_ignore_ascii_case("authorization")), "{:?}", request.headers);
        assert_eq!(request.headers.get("x-amz-content-sha256").map(String::as_str), Some(EMPTY_SHA256));
        assert!(request.headers.keys().any(|name| name.eq_ignore_ascii_case("x-amz-date")));

        let listing = listing(&identity, "eodata/Sentinel-2/", "");
        assert!(listing.headers.keys().any(|name| name.eq_ignore_ascii_case("authorization")), "{:?}", listing.headers);
        assert!(!listing.url.contains("X-Amz-Signature"), "{}", listing.url);
        assert!(listing.url.contains("list-type=2"));
    }

    /// Подпись заново — на тот же адрес и своя; чужой адрес и адрес без TLS
    /// не подписываются.
    #[test]
    fn переподпись_даёт_тот_же_адрес_и_только_своему_хранилищу() {
        let identity = test_identity();
        let request = object(&identity, "eodata/Sentinel-2/T31TGK/TCI_10m.jp2");
        let again = resign(&identity, &request.url).expect("свой адрес подписывается");
        assert_eq!(again.url, request.url);
        assert!(again.headers.keys().any(|name| name.eq_ignore_ascii_case("authorization")), "{:?}", again.headers);
        assert!(resign(&identity, "https://example.com/eodata/x.jp2").is_none(), "чужой адрес");
        assert!(resign(&identity, "http://eodata.dataspace.copernicus.eu/eodata/x.jp2").is_none(), "без TLS ключи не уходят");
    }

    #[test]
    fn product_root_climbs_to_the_container() {
        // Файл в глубине .SAFE — корень поднимается до самого продукта.
        assert_eq!(
            product_root(
                "eodata/Sentinel-2/MSI/L1C/2026/08/12/S2B_MSIL1C_X.SAFE/GRANULE/L1C_T24LVP/IMG_DATA/T24LVP_TCI.jp2"
            ),
            Some("eodata/Sentinel-2/MSI/L1C/2026/08/12/S2B_MSIL1C_X.SAFE")
        );
        // Сам продукт (папкой, со слэшем листинга) — он и есть корень.
        assert_eq!(
            product_root("eodata/Sentinel-1/SAR/IW_GRDH_1S-COG/2026/08/12/S1C_IW_GRDH_1SDV_15A3_COG.SAFE/"),
            Some("eodata/Sentinel-1/SAR/IW_GRDH_1S-COG/2026/08/12/S1C_IW_GRDH_1SDV_15A3_COG.SAFE")
        );
        // Без суффикса корень находится по дате — так лежат Landsat и RTC, и
        // файл внутри такого продукта поднимается до него же.
        assert_eq!(
            product_root("eodata/Sentinel-1-RTC/2024/01/05/S1A_IW_GRDH_RTC/"),
            Some("eodata/Sentinel-1-RTC/2024/01/05/S1A_IW_GRDH_RTC")
        );
        assert_eq!(
            product_root("eodata/Landsat-9/OLI-2/L2SP/2026/08/11/LC09_L2SP_02_T1/LC09_B1.TIF"),
            Some("eodata/Landsat-9/OLI-2/L2SP/2026/08/11/LC09_L2SP_02_T1")
        );
        // Путь к снимкам — ещё не снимок: ни суффикса, ни даты над ним.
        assert_eq!(product_root("eodata/Sentinel-2/MSI/L2A/2026/08"), None);
        assert_eq!(product_root("eodata/CLMS/Vegetation/ndvi_1999_v3.0.1.nc"), None);
    }

    #[test]
    fn single_object_is_told_by_container_suffix() {
        // Одиночные форматы: вспомогательные архивы, климатика.
        assert!(is_single_object(
            "eodata/Sentinel-2/AUX/GIP_R2ABCA/2026/08/13/S2C_OPER_GIP_R2ABCA_MPC_B00.TGZ"
        ));
        assert!(is_single_object("eodata/CLMS/Vegetation/ndvi_1999_v3.0.1.nc"));
        assert!(is_single_object("eodata/Envisat/MER_FRS_1P.N1"));
        // Каталоги: .SAFE и гранулы без контейнерного суффикса.
        assert!(!is_single_object(
            "eodata/Sentinel-1/SAR/IW_GRDH_1S/2026/08/12/S1C_IW_GRDH_1SSV_008959_011C6E_EDC8.SAFE"
        ));
        assert!(!is_single_object("eodata/Sentinel-1-RTC/2024/01/05/S1A_IW_GRDH_RTC"));
        assert!(!is_single_object("eodata/Landsat-5/TM/GTC_1P/1985/02/02/LS05_TM_GTC_1P_4A43"));
    }

    /// Пустышка-«папка» файлом не становится: хранилище кладёт рядом с ключами
    /// объект нулевого размера со слэшем на конце, и в обходе вглубь он —
    /// единственное, что притворяется файлом. Сама папка при этом остаётся в
    /// листинге общим префиксом.
    #[test]
    fn folder_placeholder_is_not_a_file() {
        let body = br#"<?xml version="1.0" encoding="UTF-8"?>
        <ListBucketResult>
          <Contents><Key>Landsat-5/1988/P.TIFF/</Key><Size>0</Size></Contents>
          <Contents><Key>Landsat-5/1988/P.TIFF/B1.TIF</Key><Size>10</Size></Contents>
          <CommonPrefixes><Prefix>Landsat-5/1988/P.TIFF/</Prefix></CommonPrefixes>
        </ListBucketResult>"#;
        let listing = parse_listing(body, "eodata/Landsat-5/1988/").expect("листинг разобран");
        let keys: Vec<&str> = listing.entries.iter().map(|e| e.identifier.as_str()).collect();
        assert_eq!(keys, vec![
            "eodata/Landsat-5/1988/P.TIFF/B1.TIF",
            "eodata/Landsat-5/1988/P.TIFF/",
        ], "папка осталась одна — общим префиксом");
    }

    /// Уровень от поддерева отличает один разделитель, а всё пустое до
    /// хранилища не доезжает: у корня бакета префикса нет, а токен появляется
    /// только со второй страницы.
    #[test]
    fn a_level_differs_from_a_subtree_by_the_delimiter_alone() {
        assert_eq!(
            query("Sentinel-2/MSI/", "", true),
            vec![
                ("delimiter", "/"),
                ("list-type", "2"),
                ("max-keys", PAGE_SIZE),
                ("prefix", "Sentinel-2/MSI/"),
            ]
        );
        assert_eq!(
            query("Sentinel-2/MSI/", "", false),
            vec![("list-type", "2"), ("max-keys", PAGE_SIZE), ("prefix", "Sentinel-2/MSI/")]
        );
        assert_eq!(query("", "", false), vec![("list-type", "2"), ("max-keys", PAGE_SIZE)]);
        assert_eq!(
            query("", "1/N7Ab", false),
            vec![("continuation-token", "1/N7Ab"), ("list-type", "2"), ("max-keys", PAGE_SIZE)]
        );
    }

    /// Дата — это ровно три разряда нужной длины и только из цифр: «08» месяцем
    /// быть может, а «MSI» или «2026-08» — нет.
    #[test]
    fn date_triple_is_matched_by_shape() {
        assert!(under_date("eodata/Sentinel-2/MSI/L2A/2026/08/12"));
        assert!(!under_date("eodata/Sentinel-2/MSI/L2A/2026/08"));
        assert!(!under_date("eodata/Sentinel-2/MSI/L2A/26/08/12"));
        assert!(!under_date("eodata/Sentinel-2/MSI/L2A/2026/8/12"));
        assert!(!under_date("eodata/Sentinel-2/MSI/L2A/2026/AU/12"));
    }
}
