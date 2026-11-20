// Extracted from master:veldmodules/data-provider/src/imagery.rs

#[cfg(test)]
mod tests {
    /// Растры из [`scan`] — тестам нужны только они.
    /// Растры удалённого продукта — того, что ещё не скачан. Скачанный
    /// спрашивается отдельно, там свой ответ.
    fn scan_rasters(product: &str, keys: &[String], measured: &[String]) -> Vec<(String, Role)> {
        scan(product, keys, measured, false).rasters
    }

    use super::*;

    fn keys(paths: &[&str]) -> Vec<String> {
        paths.iter().map(|path| path.to_string()).collect()
    }

    #[test]
    fn sentinel2_l2a_yields_pvi_and_r10m_tci() {
        let keys = keys(&[
            "eodata/…/S2C_MSIL2A_T40WFC.SAFE/GRANULE/L2A_T40WFC_A/QI_DATA/T40WFC_20260812_PVI.jp2",
            "eodata/…/S2C_MSIL2A_T40WFC.SAFE/GRANULE/L2A_T40WFC_A/IMG_DATA/R10m/T40WFC_20260812_TCI_10m.jp2",
            "eodata/…/S2C_MSIL2A_T40WFC.SAFE/GRANULE/L2A_T40WFC_A/IMG_DATA/R60m/T40WFC_20260812_TCI_60m.jp2",
            "eodata/…/S2C_MSIL2A_T40WFC.SAFE/GRANULE/L2A_T40WFC_A/IMG_DATA/R10m/T40WFC_20260812_B08_10m.jp2",
            "eodata/…/S2C_MSIL2A_T40WFC.SAFE/MTD_MSIL2A.xml",
        ]);
        let rasters = scan_rasters("eodata/…/S2C_MSIL2A_T40WFC.SAFE", &keys, &[]);
        assert_eq!(rasters.len(), 2);
        assert!(rasters[0].0.ends_with("_PVI.jp2") && rasters[0].1 == Role::Preview);
        assert!(rasters[1].0.ends_with("_TCI_10m.jp2") && rasters[1].1 == Role::Detailed);
    }

    #[test]
    fn sentinel2_l1c_falls_back_to_flat_tci() {
        let keys = keys(&[
            "eodata/…/S2B_MSIL1C_T33UUP.SAFE/GRANULE/L1C_T33UUP_A/QI_DATA/T33UUP_PVI.jp2",
            "eodata/…/S2B_MSIL1C_T33UUP.SAFE/GRANULE/L1C_T33UUP_A/IMG_DATA/T33UUP_20260601_TCI.jp2",
            "eodata/…/S2B_MSIL1C_T33UUP.SAFE/GRANULE/L1C_T33UUP_A/IMG_DATA/T33UUP_20260601_B02.jp2",
        ]);
        let rasters = scan_rasters("eodata/…/S2B_MSIL1C_T33UUP.SAFE", &keys, &[]);
        assert_eq!(rasters.len(), 2);
        assert!(rasters[1].0.ends_with("_TCI.jp2") && rasters[1].1 == Role::Detailed);
    }

    /// Полосный гигант старых GRD показывается только скачанным, и решает это
    /// его ГРУБЫЙ край, а не подробный. Подробный тайл у него дёшев — 512 строк
    /// на всю ширину, тайлер читает такой уровень окном; а грубому нужна каждая
    /// строка файла, и он стои́т целого прохода. Спрашивают же именно грубый:
    /// канва просмотра открывает подробный растр и вписывает его в окно.
    #[test]
    fn sentinel1_plain_grd_yields_measurement_only_when_downloaded() {
        let keys = keys(&[
            "eodata/…/S1C_IW_GRDH.SAFE/preview/quick-look.png",
            "eodata/…/S1C_IW_GRDH.SAFE/measurement/s1c-iw-grd-vv.tiff",
            "eodata/…/S1C_IW_GRDH.SAFE/manifest.safe",
        ]);

        let remote = scan_rasters("eodata/…/S1C_IW_GRDH.SAFE", &keys, &[]);
        assert_eq!(remote.len(), 1, "по сети предложен проход по всему файлу");
        assert!(remote[0].0.ends_with("quick-look.png") && remote[0].1 == Role::Preview);

        let local = scan("eodata/…/S1C_IW_GRDH.SAFE", &keys, &[], true).rasters;
        assert_eq!(local.len(), 2, "скачанный снимок остался при одном квиклуке");
        assert!(local[0].0.ends_with("quick-look.png") && local[0].1 == Role::Preview);
        assert!(local[1].0.ends_with("-vv.tiff") && local[1].1 == Role::Detailed);
    }

    /// А тайловый COG предлагается независимо от того, скачан он или нет:
    /// читается он точечно, и приближение стои́т своих тайлов, а не файла.
    #[test]
    fn sentinel1_cog_measurement_needs_no_disk() {
        let keys = keys(&[
            "eodata/…/S1C_IW_GRDH_COG.SAFE/preview/quick-look.png",
            "eodata/…/S1C_IW_GRDH_COG.SAFE/measurement/s1c-iw-grd-vv-cog.tiff",
        ]);
        for downloaded in [false, true] {
            let rasters = scan("eodata/…/S1C_IW_GRDH_COG.SAFE", &keys, &[], downloaded).rasters;
            assert_eq!(rasters.len(), 2, "скачан={downloaded}");
            assert!(rasters[1].0.ends_with("-vv-cog.tiff") && rasters[1].1 == Role::Detailed);
        }
    }

    #[test]
    fn sentinel1_grd_cog_yields_copol_measurement() {
        let keys = keys(&[
            "eodata/…/S1C_IW_GRDH_1SDV_COG.SAFE/preview/quick-look.png",
            "eodata/…/S1C_IW_GRDH_1SDV_COG.SAFE/measurement/s1c-iw-grd-vh-20260812t153507-002-cog.tiff",
            "eodata/…/S1C_IW_GRDH_1SDV_COG.SAFE/measurement/s1c-iw-grd-vv-20260812t153507-001-cog.tiff",
            "eodata/…/S1C_IW_GRDH_1SDV_COG.SAFE/manifest.safe",
        ]);
        let rasters = scan_rasters("eodata/…/S1C_IW_GRDH_1SDV_COG.SAFE", &keys, &[]);
        assert_eq!(rasters.len(), 2);
        assert!(rasters[0].0.ends_with("quick-look.png") && rasters[0].1 == Role::Preview);
        // Из двух поляризаций подробной выбрана ко- (vv), не первая по списку.
        assert!(rasters[1].0.contains("-vv-") && rasters[1].1 == Role::Detailed);
    }

    /// Раскладка незнакомая, а читаемый файл в продукте один — он и есть
    /// снимок. Так лежит климатика: каталог из единственного .nc.
    #[test]
    fn a_lone_readable_file_is_the_product_itself() {
        let clms = keys(&[
            "eodata/CLMS/lst/c_gls_LST_202608161000_GLOBE_GEO_V3.0.1_nc/c_gls_LST.nc",
        ]);
        let rasters =
            scan_rasters("eodata/CLMS/lst/c_gls_LST_202608161000_GLOBE_GEO_V3.0.1_nc", &clms, &[]);
        assert_eq!(rasters.len(), 1);
        assert!(rasters[0].0.ends_with(".nc") && rasters[0].1 == Role::Detailed);
    }

    /// Раскладка не узнана, а читаемых файлов несколько — показывается первый
    /// по алфавиту, и обзорная картинка отдельно от него. Промолчать было бы
    /// хуже: под это правило подпадает вся оптика Landsat.
    #[test]
    fn an_unknown_layout_still_shows_what_it_has() {
        let bands = keys(&[
            "eodata/Landsat-5/LT51780121988065ESA00/LT51780121988065ESA00_thumb_large.jpg",
            "eodata/Landsat-5/LT51780121988065ESA00/LT51780121988065ESA00_B2.TIF",
            "eodata/Landsat-5/LT51780121988065ESA00/LT51780121988065ESA00_B1.TIF",
        ]);
        let rasters = scan_rasters("eodata/Landsat-5/LT51780121988065ESA00", &bands, &[]);
        assert_eq!(rasters.len(), 2);
        assert!(rasters[0].0.ends_with("_thumb_large.jpg") && rasters[0].1 == Role::Preview);
        assert!(rasters[1].0.ends_with("_B1.TIF") && rasters[1].1 == Role::Detailed);
    }

    /// Обзорная картинка архива ESA зовётся `BP` — Browse Product, — и слово
    /// это перечислено: иначе она становится подробным растром, потому что по
    /// алфавиту `LS05…BP.PNG` стои́т раньше полос `LT51…_B1.TIF` из соседнего
    /// каталога. Раскладка настоящая: так лежит Landsat-5 в бакете CDSE.
    #[test]
    fn a_browse_product_is_a_quicklook_not_the_picture() {
        const PRODUCT: &str = "eodata/Landsat-5/TM/L1G/1988/03/05/LS05_RKSE_TM__GEO_1P_52FE";
        let scene = keys(&[
            "eodata/…/LS05_RKSE_TM__GEO_1P_52FE/LS05_RKSE_TM__GEO_1P_52FE.BP.PNG",
            "eodata/…/LS05_RKSE_TM__GEO_1P_52FE/LS05_RKSE_TM__GEO_1P_52FE.BP.XML",
            "eodata/…/LS05_RKSE_TM__GEO_1P_52FE/LS05_RKSE_TM__GEO_1P_52FE.JPG",
            "eodata/…/LS05_RKSE_TM__GEO_1P_52FE/LS05_RKSE_TM__GEO_1P_52FE.TIFF/LT51780121988065ESA00_B2.TIF",
            "eodata/…/LS05_RKSE_TM__GEO_1P_52FE/LS05_RKSE_TM__GEO_1P_52FE.TIFF/LT51780121988065ESA00_B1.TIF",
        ]);
        let rasters = scan_rasters(PRODUCT, &scene, &[]);
        assert_eq!(rasters.len(), 2, "{:?}", rasters);
        assert!(rasters[0].0.ends_with(".BP.PNG") && rasters[0].1 == Role::Preview);
        assert!(rasters[1].0.ends_with("_B1.TIF") && rasters[1].1 == Role::Detailed);

        // Словом, а не куском: две буквы подстрокой ловят что угодно.
        assert!(a_quicklook("x/scene.BP.PNG"));
        assert!(!a_quicklook("x/S3A_SL_2_LST_bpm_in.nc"));

        // Второй картинки в списке довольно, чтобы слово `BP` перестало
        // помогать: `LS05_…52FE.JPG` не названо ничем и по алфавиту стои́т
        // раньше полос (`S` < `T`). Различает их формат, а не имя.
        assert!(a_picture_format("x/LS05_RKSE_TM__GEO_1P_52FE.JPG"));
        assert!(!a_picture_format("x/LT51780121988065ESA00_B1.TIF"));
        assert!(!a_picture_format("x/LST_in.nc"));
    }

    /// Гранула OLCI без манифеста: первым по алфавиту стои́т файл координат, а
    /// не измерение — у неё все имена строчные, и уступки регистра, что
    /// выручает SLSTR, здесь нет. Имена настоящие, с гранулы `OL_2_LRR`.
    ///
    /// Те же координаты, найденные [`geolocation`], садятся потом под растр —
    /// одно и то же знание, спрошенное с двух сторон.
    #[test]
    fn a_coordinate_file_never_becomes_the_measurement() {
        const PRODUCT: &str = "eodata/…/S3A_OL_2_LRR____20260824T161836_PS1_O_NR_003.SEN3";
        let granule = keys(&[
            "eodata/…/S3A_OL_2_LRR.SEN3/quicklook.jpg",
            "eodata/…/S3A_OL_2_LRR.SEN3/geo_coordinates.nc",
            "eodata/…/S3A_OL_2_LRR.SEN3/gifapar.nc",
            "eodata/…/S3A_OL_2_LRR.SEN3/iwv.nc",
            "eodata/…/S3A_OL_2_LRR.SEN3/otci.nc",
            "eodata/…/S3A_OL_2_LRR.SEN3/tie_geo_coordinates.nc",
            "eodata/…/S3A_OL_2_LRR.SEN3/time_coordinates.nc",
            "eodata/…/S3A_OL_2_LRR.SEN3/xfdumanifest.xml",
        ]);
        let rasters = scan_rasters(PRODUCT, &granule, &[]);
        assert_eq!(rasters.len(), 2, "{:?}", rasters);
        assert!(rasters[1].0.ends_with("/gifapar.nc"), "{:?}", rasters[1]);

        // То же самое говорит и манифест — правило с ним не спорит, а
        // повторяет его там, где его нет.
        let said = vec!["gifapar.nc".to_string()];
        assert!(scan_rasters(PRODUCT, &granule, &said)[1].0.ends_with("/gifapar.nc"));
    }

    /// Гранула SLSTR без манифеста: рядом с измерением лежит подсобный файл, и
    /// по алфавиту он первый — прописные идут раньше строчных, так что спор
    /// решается на `a` против `i`. Имена настоящие, с гранулы `SL_2_LST`.
    ///
    /// Обычно такую гранулу разбирает манифест — он измерение называет прямо, —
    /// но манифест бывает и недоступен, и тогда весь ответ в именах файлов.
    #[test]
    fn an_ancillary_file_never_becomes_the_measurement() {
        const PRODUCT: &str = "eodata/…/S3A_SL_2_LST____20260824T174507_0540_PS1_O_NR_005.SEN3";
        let granule = keys(&[
            "eodata/…/S3A_SL_2_LST.SEN3/quicklook.jpg",
            "eodata/…/S3A_SL_2_LST.SEN3/LST_ancillary_ds.nc",
            "eodata/…/S3A_SL_2_LST.SEN3/LST_in.nc",
            "eodata/…/S3A_SL_2_LST.SEN3/flags_in.nc",
            "eodata/…/S3A_SL_2_LST.SEN3/geodetic_in.nc",
            "eodata/…/S3A_SL_2_LST.SEN3/xfdumanifest.xml",
        ]);
        let rasters = scan_rasters(PRODUCT, &granule, &[]);
        assert_eq!(rasters.len(), 2, "{:?}", rasters);
        assert!(rasters[0].0.ends_with("quicklook.jpg") && rasters[0].1 == Role::Preview);
        assert!(rasters[1].0.ends_with("/LST_in.nc") && rasters[1].1 == Role::Detailed);

        // А названный манифестом побеждает и без всяких ярусов.
        let said = vec!["LST_ancillary_ds.nc".to_string()];
        let by_manifest = scan_rasters(PRODUCT, &granule, &said);
        assert!(by_manifest[1].0.ends_with("LST_ancillary_ds.nc"), "{:?}", by_manifest);
    }

    /// Полосы CLMS — измерение, погрешность, флаги качества и время съёмки —
    /// все четыре читаемые растры одного размера, и первой по алфавиту стои́т
    /// погрешность: на шар легла бы она — разброс в кельвинах вместо самой
    /// температуры, а по виду это неотличимо, маска «нет данных» у полос общая.
    ///
    /// Спасает то, что имя файла у CLMS — это имя продукта с приставкой полосы:
    /// измерение узнаётся тем, что повторяет величину продукта.
    #[test]
    fn a_clms_product_shows_its_measurement_not_its_error_bar() {
        const PRODUCT: &str = "eodata/…/c_gls_LST_202608271600_GLOBE_GEO_V3.0.1_cog";
        let band = |what: &str| {
            format!("{}/c_gls_LST-{}_202608271600_GLOBE_GEO_V3.0.1.tiff", PRODUCT, what)
        };
        let bands: Vec<String> =
            ["ERRORBAR", "LST", "QFLAG", "TDELTA"].iter().map(|what| band(what)).collect();

        let rasters = scan_rasters(PRODUCT, &bands, &[]);
        assert_eq!(rasters.len(), 1, "{:?}", rasters);
        assert_eq!(rasters[0], (band("LST"), Role::Detailed));

        // Повтор считается по последнему сегменту пути, а не по всему: путь
        // назвал величину ещё до полосы, и со всем путём в счёте `LST` вышло бы
        // два против двух — измерение сравнялось бы с соседями.
        let deep = format!("eodata/CLMS/x_lst_y/2026/08/27/{}", file_name(PRODUCT));
        assert!(names_the_measurand(&band("LST"), &deep));
        assert!(!names_the_measurand(&band("ERRORBAR"), &deep));

        // Дата и версия повторяются в именах полос сами собой, и словами не
        // считаются: иначе на правило отвечала бы всякая полоса, у которой в
        // хвосте стои́т лишний номер.
        let numbered =
            format!("{}/c_gls_LST-QFLAG_202608271600_GLOBE_GEO_V3.0.1-1.tiff", PRODUCT);
        assert!(!names_the_measurand(&numbered, PRODUCT));

        // Полосе, которая величину продукта не повторяет, правило не отвечает
        // ничего, и выбор остаётся за алфавитом — как у Landsat, где полосы
        // зовутся `_B1`…`_B7`.
        let scene = "eodata/Landsat-5/LT51780121988065ESA00";
        assert!(!names_the_measurand(&format!("{}/LT51780121988065ESA00_B1.TIF", scene), scene));
    }

    /// Цветной снимок целиком идёт вперёд отдельных полос — по имени, потому
    /// что до открытия о файлах больше ничего и не известно. Слово целиком, а
    /// не кусок: `otci.nc` — индекс хлорофилла, а не цветной снимок.
    #[test]
    fn a_true_colour_file_wins_over_single_bands() {
        let granule = keys(&[
            "eodata/Sentinel-3/SR/x.SEN3/Oa01_radiance.nc",
            "eodata/Sentinel-3/SR/x.SEN3/rgb_TCI.nc",
        ]);
        let rasters = scan_rasters("eodata/Sentinel-3/SR/x.SEN3", &granule, &[]);
        assert_eq!(rasters.len(), 1);
        assert!(rasters[0].0.ends_with("rgb_TCI.nc"));

        assert!(a_whole_picture("x/T35UNV_20260818_TCI_10m.jp2"));
        assert!(a_whole_picture("x/scene_true_color.tif"));
        assert!(a_whole_picture("x/L2_TRUECOLOR.png"));
        // Пустые токены разборщик отбрасывает, и двойной разделитель пару не
        // разрывает: у ESA двойное подчёркивание — обычное дело.
        assert!(a_whole_picture("x/S3A_OL__TRUE__COLOR.png"));
        assert!(!a_whole_picture("eodata/…/S3A_OL_2_LFR.SEN3/otci.nc"));
        assert!(!a_whole_picture("eodata/…/S3A_OL_2_LFR.SEN3/rc_gifapar.nc"));
    }

    /// Подкаталог `preview/` — часть раскладки, и в нём лежит показ: обзорная
    /// картинка, шкала прибора, логотип миссии. Подробным растром оттуда не
    /// берётся ничего — иначе у Sentinel-1 OCN на шар ложится логотип
    /// Copernicus, а измерение остаётся неоткрытым.
    #[test]
    fn a_logo_never_becomes_the_detailed_raster() {
        let ocn = keys(&[
            "eodata/…/S1C_EW_OCN__2SDH.SAFE/preview/icons/logo.png",
            "eodata/…/S1C_EW_OCN__2SDH.SAFE/preview/quick-look-l2-owi.png",
            "eodata/…/S1C_EW_OCN__2SDH.SAFE/preview/owi-colorbar.png",
            "eodata/…/S1C_EW_OCN__2SDH.SAFE/measurement/s1c-ew-ocn-hh-20260818t193953-001.nc",
            "eodata/…/S1C_EW_OCN__2SDH.SAFE/measurement/s1c-ew1-osw-hh-20260818t193953-002.nc",
            "eodata/…/S1C_EW_OCN__2SDH.SAFE/manifest.safe",
        ]);
        let rasters = scan_rasters("eodata/…/S1C_EW_OCN__2SDH.SAFE", &ocn, &[]);
        assert_eq!(rasters.len(), 2, "{:?}", rasters);
        assert!(
            rasters[0].0.ends_with("quick-look-l2-owi.png") && rasters[0].1 == Role::Preview,
            "{:?}", rasters[0]
        );
        assert!(
            rasters[1].0.ends_with("-ocn-hh-20260818t193953-001.nc")
                && rasters[1].1 == Role::Detailed,
            "{:?}", rasters[1]
        );
    }

    /// Растр лежит только в `preview/` — тогда у продукта есть обзорная
    /// картинка и нет подробного. Это ответ, а не отказ: показать её честнее,
    /// чем промолчать.
    #[test]
    fn a_product_with_only_a_quicklook_still_has_one() {
        let only = keys(&[
            "eodata/…/X.SAFE/preview/quick-look.png",
            "eodata/…/X.SAFE/preview/icons/logo.png",
            "eodata/…/X.SAFE/annotation/report.xml",
        ]);
        let rasters = scan_rasters("eodata/…/X.SAFE", &only, &[]);
        assert_eq!(rasters.len(), 1, "{:?}", rasters);
        assert!(rasters[0].0.ends_with("quick-look.png") && rasters[0].1 == Role::Preview);
    }

    /// Список читаемого — один: чем отбирают растр, тем и называют форматы в
    /// отказе. Разойдясь, они говорят одному человеку разное.
    #[test]
    fn the_refusal_names_what_the_scan_accepts() {
        let said = readable();
        // Каждое расширение, которое отбирает `is_raster`, названо в отказе —
        // своим словом, а не самим суффиксом. Иначе человек читает «открываются
        // …», не находит там своего файла и решает, что формат не тот.
        for suffix in RASTER_SUFFIXES {
            assert!(is_raster(&format!("x/y.{}", suffix)), "{} не отбирается", suffix);
            let named = readable().to_ascii_lowercase();
            assert!(
                named.contains(&suffix.to_ascii_lowercase())
                    || ["tif", "tiff", "jp2", "j2k", "jpg", "jpeg", "nc", "h5"].contains(&suffix),
                "{} не назван в «{}»",
                suffix,
                said
            );
        }
        // А у тех, чьё слово с суффиксом не совпадает, названо само слово.
        for word in ["TIFF", "JPEG 2000", "JPEG", "NetCDF", "HDF5"] {
            assert!(said.contains(word), "{} не назван: {}", word, said);
        }
        assert!(said.contains("HDF5"), "h5 отбирается, а не назван: {}", said);
        assert!(said.contains("NetCDF") && said.contains("JPEG 2000") && said.contains("WebP"));
        // Синонимы названы по разу: «TIFF» не повторяется за tif и tiff.
        assert_eq!(said.matches("TIFF").count(), 1, "{}", said);
        assert_eq!(said.matches("JPEG 2000").count(), 1, "{}", said);
    }

    /// Продукт-файл: растр он или нет, видно по нему самому.
    #[test]
    fn single_object_is_its_own_raster_when_it_is_one() {
        assert_eq!(
            single("eodata/CLMS/dem/tile.TIF"),
            Some(("eodata/CLMS/dem/tile.TIF".to_string(), Role::Detailed))
        );
        assert_eq!(single("eodata/Sentinel-2/AUX/GIP_R2ABCA/S2A_OPER_B00.TGZ"), None);
        assert!(single("eodata/Sentinel-5P/TROPOMI/L2__NO2___/S5P_NRTI.nc").is_some());
    }

    /// Значок «на глобус» обещает показ, поэтому «нет» здесь твёрдое, а «да» —
    /// «похоже»: у папки содержимое неизвестно до листинга.
    #[test]
    fn only_a_definite_no_takes_the_globe_away() {
        // Сырьё уровня 0 — изображения в нём нет, сколько ни листай. И отказ
        // называет именно уровень: по «нет растра» его не отличить от снимка
        // незнакомой раскладки, а поступают с ними по-разному.
        let raw = unviewable("eodata/…/S1C_IW_RAW__0SDV.SAFE", true, Some(0), "IW_RAW__0S");
        assert!(raw.contains("уровня 0"), "отказ не назвал уровень: {}", raw);
        // Архив калибровочных таблиц — один файл, и он не растр. Отказ
        // называет и то, что лежит, и то, что подошло бы.
        let archive = unviewable("eodata/…/S2A_OPER_GIP_R2EQOG_B03.TGZ", false, Some(1), "");
        assert!(archive.contains(".tgz"), "отказ не назвал файла: {}", archive);
        assert!(archive.contains("TIFF"), "отказ не назвал подходящего: {}", archive);
        // Гранула Sentinel-5P — один файл, и он читается.
        assert!(unviewable("eodata/…/S5P_NRTI_L2__NO2___.nc", false, Some(2), "L2__NO2___").is_empty());
        // Папка: что внутри, скажет листинг.
        assert!(unviewable("eodata/…/S2C_MSIL2A_T40WFC.SAFE", true, Some(2), "S2MSI2A").is_empty());
        // Уровень неизвестен (листинг хранилища) — судим по одному ключу.
        assert!(unviewable("eodata/…/tile.TIF", false, None, "").is_empty());

        // Служебные данные гаснут типом продукта — до скачивания. Папкой, а
        // не файлом: у файла отказ дало бы и расширение, и проверка вышла бы
        // пустой.
        let aux = unviewable("eodata/…/S1A_AUX_PP1.SAFE", true, None, "AUX_PP1");
        assert!(aux.contains("служебные"), "AUX обязан гаснуть типом: {aux}");
        assert!(
            unviewable("eodata/…/S1A_AUX_PP1.SAFE", true, None, "").is_empty(),
            "без типа гасить нечем — проверка обязана держаться на нём"
        );

        // Спектральный куб Sentinel-5P: уровень у него первый, и правилом
        // уровня 0 он не ловится — гасит его тип продукта.
        for kind in ["L1B_RA_BD1", "L1B_RA_BD8", "L1B_IR_SIR", "L1B_IR_UVN", "L1B_ENG_DB"] {
            let cube = unviewable("eodata/…/S5P_NRTI.nc", false, Some(1), kind);
            assert!(cube.contains("куб"), "тип {kind} обязан гаснуть: {cube}");
        }

        // А снимаемое типом не гасится, каким бы длинным он ни был: цена
        // ошибки здесь односторонняя. SLC в этом списке не случайно — растр у
        // него комплексный, но квиклук рядом показывается.
        // `L2__*` того же спутника в этом списке нарочно: гаснет уровень 1B, а
        // не миссия, и соседняя буква в типе решает всё.
        for kind in
            ["IW_GRDH_1S", "S2MSI1C", "SL_2_LST___", "L2__CO____", "IW_SLC__1S", "L2__O3__PR"]
        {
            assert!(
                unviewable("eodata/…/product.SAFE", true, Some(2), kind).is_empty(),
                "тип {kind} погашен, а он показуемый"
            );
        }
    }

    /// Пустой ответ объясняется тем, что в продукте лежит: сырьё уровня 0 и
    /// незнакомая раскладка — разные ответы, и путать их нельзя.
    #[test]
    fn empty_answer_names_what_is_inside() {
        let raw = keys(&[
            "eodata/…/S1C_IW_RAW__0SDV_20260816T050356_009016_0C2C.SAFE/manifest.safe",
            "eodata/…/S1C_IW_RAW__0SDV_20260816T050356_009016_0C2C.SAFE/s1c-iw-raw-s-vv.dat",
        ]);
        let said = nothing_here(
            "eodata/…/S1C_IW_RAW__0SDV_20260816T050356_009016_0C2C.SAFE",
            &raw,
        );
        assert!(said.contains("уровня 0"), "{}", said);

        let unknown = keys(&["eodata/…/X.SEN3/report.xml", "eodata/…/X.SEN3/data.dat"]);
        let said = nothing_here("eodata/…/X.SEN3", &unknown);
        assert!(said.contains(".xml") && said.contains(".dat"), "содержимое названо: {}", said);
        assert!(!said.contains("уровня 0"), "{}", said);

        // Листинг пустой — это про хранилище, а не про раскладку.
        assert!(nothing_here("eodata/…/X.SAFE", &[]).contains("ни одного файла"));
    }

    /// Sentinel-3 держит координаты отдельным файлом, и его-то и надо назвать.
    /// У OLCI это опорная сетка — полмегабайта против тридцати мегабайт
    /// поотсчётного файла, а решётка привязки всё равно берётся по узлам.
    #[test]
    fn olci_points_at_the_cheap_tie_grid() {
        let olci: Vec<String> = [
            "eodata/…/S3A_OL_2_LFR.SEN3/gifapar.nc",
            "eodata/…/S3A_OL_2_LFR.SEN3/geo_coordinates.nc",
            "eodata/…/S3A_OL_2_LFR.SEN3/tie_geo_coordinates.nc",
        ]
        .iter()
        .map(|key| key.to_string())
        .collect();
        assert_eq!(
            geolocation(&olci, &olci[0]),
            Some("eodata/…/S3A_OL_2_LFR.SEN3/tie_geo_coordinates.nc".to_string())
        );
        // Опорной сетки нет — остаются поотсчётные координаты.
        assert_eq!(
            geolocation(&olci[..2], &olci[0]),
            Some("eodata/…/S3A_OL_2_LFR.SEN3/geo_coordinates.nc".to_string())
        );
    }

    /// Названный файл ложится подробным, выбор раскладки уходит за ним
    /// запасным, превью остаётся; названное не из продукта или не растр —
    /// как не названное.
    #[test]
    fn a_wanted_file_lies_detailed_and_the_layouts_choice_becomes_a_spare() {
        let keys: Vec<String> = ["p/quicklook.jpg", "p/S1_radiance_an.nc", "p/F2_BT_in.nc", "p/xfdumanifest.xml"]
            .iter()
            .map(|key| key.to_string())
            .collect();
        let scan = || Scan {
            rasters: vec![("p/quicklook.jpg".into(), Role::Preview), ("p/S1_radiance_an.nc".into(), Role::Detailed)],
            alternates: vec!["p/F1_BT_fn.nc".into()],
            guessed: true,
        };
        let (chosen, refused) = preferring(scan(), "p/F2_BT_in.nc", &keys);
        assert_eq!(chosen.rasters, vec![("p/quicklook.jpg".to_string(), Role::Preview), ("p/F2_BT_in.nc".to_string(), Role::Detailed)]);
        assert_eq!(chosen.alternates, vec!["p/S1_radiance_an.nc".to_string(), "p/F1_BT_fn.nc".to_string()]);
        assert!(chosen.guessed && refused.is_none());

        let (same, _) = preferring(scan(), "p/S1_radiance_an.nc", &keys);
        assert_eq!(same.rasters, scan().rasters);
        assert_eq!(same.alternates, vec!["p/F1_BT_fn.nc".to_string()], "названный выбор раскладки не стал своим же запасным");
        // Отказ — словами и растрами раскладки: манифест, чужой файл, квиклук.
        for (wanted, word) in [("p/xfdumanifest.xml", "не растр"), ("q/other.nc", "нет в продукте"), ("p/quicklook.jpg", "квиклук")] {
            let (fallback, refused) = preferring(scan(), wanted, &keys);
            assert_eq!(fallback.rasters, scan().rasters, "{wanted} лёг растром");
            assert!(refused.as_deref().is_some_and(|said| said.contains(word)), "{wanted}: {refused:?}");
        }
        let (plain, refused) = preferring(scan(), "", &keys);
        assert_eq!((plain.alternates, refused), (scan().alternates, None));
    }

    /// Гранула SLSTR уровня 1: измерений двадцать восемь, и первые четыре
    /// довода у всех одинаковы — решает густота сетки, а не алфавит.
    /// Полукилометровый видимый канал надирного обзора берёт верх над
    /// километровым тепловым, который стои́т раньше по алфавиту; косой обзор
    /// той же сетки — после надирного. Имена настоящие, с гранулы `SL_1_RBT`.
    #[test]
    fn the_densest_grid_wins_among_the_swath_measurements() {
        const PRODUCT: &str = "eodata/…/S3A_SL_1_RBT____20260824T174507_0540_PS1_O_NR_005.SEN3";
        let names = [
            "quicklook.jpg", "F1_BT_fn.nc", "F1_BT_fo.nc", "F1_BT_in.nc", "F1_BT_io.nc",
            "F2_BT_in.nc", "S1_radiance_an.nc", "S1_radiance_ao.nc", "S4_radiance_bn.nc",
            "S5_radiance_cn.nc", "S7_BT_in.nc", "S8_BT_in.nc", "S9_BT_io.nc", "met_tx.nc",
            "cartesian_an.nc", "flags_an.nc", "indices_an.nc", "time_an.nc", "geodetic_an.nc",
            "geodetic_tx.nc", "viscal.nc", "xfdumanifest.xml",
        ];
        let granule: Vec<String> = names.iter().map(|name| format!("eodata/…/S3A_SL_1_RBT.SEN3/{name}")).collect();
        let rasters = scan_rasters(PRODUCT, &granule, &[]);
        assert_eq!(rasters.len(), 2, "{:?}", rasters);
        assert!(rasters[0].0.ends_with("quicklook.jpg") && rasters[0].1 == Role::Preview);
        assert!(rasters[1].0.ends_with("/S1_radiance_an.nc"), "{:?}", rasters[1]);

        // Запасной — лучший файл сетки грубее: километровый надир; опорная
        // сетка запасным не идёт.
        let found = scan(PRODUCT, &granule, &[], false);
        assert_eq!(found.alternates, vec![format!("eodata/…/S3A_SL_1_RBT.SEN3/F1_BT_fn.nc")]);

        // Названные манифестом измерения — те же двадцать восемь; ответ тот же.
        let measured: Vec<String> = names
            .iter()
            .filter(|name| name.contains("radiance") || name.contains("_BT_"))
            .map(|name| name.to_string())
            .collect();
        assert!(scan_rasters(PRODUCT, &granule, &measured)[1].0.ends_with("/S1_radiance_an.nc"));

        // Косой обзор той же полукилометровой сетки берёт верх над километровым
        // надиром: сетка прежде обзора. Без полукилометровых сеток вовсе —
        // километровый надир, по алфавиту среди равных.
        let half_km = |key: &String| ["_an.nc", "_ao.nc", "_bn.nc", "_bo.nc", "_cn.nc", "_co.nc"].iter().any(|tail| key.ends_with(tail));
        let nadir_half_km = |key: &String| ["_an.nc", "_bn.nc", "_cn.nc"].iter().any(|tail| key.ends_with(tail));
        let no_nadir: Vec<String> = granule.iter().filter(|key| !nadir_half_km(key)).cloned().collect();
        assert!(scan_rasters(PRODUCT, &no_nadir, &[])[1].0.ends_with("/S1_radiance_ao.nc"), "{:?}", scan_rasters(PRODUCT, &no_nadir, &[]));
        let coarse: Vec<String> = granule.iter().filter(|key| !half_km(key)).cloned().collect();
        assert!(scan_rasters(PRODUCT, &coarse, &[])[1].0.ends_with("/F1_BT_fn.nc"), "{:?}", scan_rasters(PRODUCT, &coarse, &[]));
        assert_eq!(grid_rank("x/S1_radiance_an.nc"), (0, 0));
        assert_eq!(grid_rank("x/S1_radiance_ao.nc"), (0, 1));
        assert_eq!(grid_rank("x/F1_BT_fn.nc"), (1, 0));
        assert_eq!(grid_rank("x/met_tx.nc"), (2, 2));
        assert_eq!(grid_rank("x/gifapar.nc"), (1, 0), "имя без сетки — километровый надир");

        // У гранулы без хвостов сеток запасных нет: второго ответа имена не дают.
        let olci: Vec<String> = ["quicklook.jpg", "gifapar.nc", "iwv.nc", "otci.nc"]
            .iter()
            .map(|name| format!("eodata/…/S3A_OL_2_LRR.SEN3/{name}"))
            .collect();
        assert!(scan("eodata/…/S3A_OL_2_LRR.SEN3", &olci, &[], false).alternates.is_empty());
    }

    /// У SLSTR сеток несколько, и опорная общая им всем: растр километровой
    /// сетки `in` привязывается `geodetic_tx.nc` — на порядок более дешёвым,
    /// чем поотсчётный `geodetic_in.nc`, — а полукилометровой `an` первым
    /// отвечает её поотсчётный файл: опорная сетка сдвинула бы его на пиксели.
    #[test]
    fn slstr_takes_the_cheap_tie_grid_of_the_swath() {
        let slstr: Vec<String> = [
            "eodata/…/S3A_SL_2_LST.SEN3/LST_in.nc",
            "eodata/…/S3A_SL_2_LST.SEN3/geodetic_in.nc",
            "eodata/…/S3A_SL_2_LST.SEN3/geodetic_tx.nc",
            "eodata/…/S3A_SL_2_LST.SEN3/geo_coordinates.nc",
        ]
        .iter()
        .map(|key| key.to_string())
        .collect();
        assert_eq!(
            geolocation(&slstr, &slstr[0]),
            Some("eodata/…/S3A_SL_2_LST.SEN3/geodetic_tx.nc".to_string())
        );
        // Растру самой опорной сетки достаётся тот же файл: своей сетки у него
        // нет другой.
        assert_eq!(
            geolocation(&slstr, "eodata/…/S3A_SL_2_LST.SEN3/met_tx.nc"),
            Some("eodata/…/S3A_SL_2_LST.SEN3/geodetic_tx.nc".to_string())
        );
        // Имя не сеточное — опорная сетка ему не достаётся: под правило
        // подпадали бы и служебный `LST_ancillary_ds.nc`, и `chl_nn.nc` OLCI.
        assert_eq!(
            geolocation(&slstr, "eodata/…/S3A_SL_2_LST.SEN3/LST_ancillary_ds.nc"),
            Some("eodata/…/S3A_SL_2_LST.SEN3/geo_coordinates.nc".to_string())
        );
        // Опорной в продукте нет — остаются поотсчётные координаты своей
        // сетки, а не первый попавшийся файл координат.
        let lean: Vec<String> = slstr.iter().filter(|key| !key.ends_with("geodetic_tx.nc")).cloned().collect();
        assert_eq!(
            geolocation(&lean, &lean[0]),
            Some("eodata/…/S3A_SL_2_LST.SEN3/geodetic_in.nc".to_string())
        );
        // Полукилометровая сетка берёт свой поотсчётный файл прежде опорной, а
        // без него — опорную.
        let rbt: Vec<String> = ["S1_radiance_an.nc", "geodetic_an.nc", "geodetic_tx.nc"]
            .iter()
            .map(|name| format!("eodata/…/S3A_SL_1_RBT.SEN3/{name}"))
            .collect();
        assert_eq!(geolocation(&rbt, &rbt[0]), Some(rbt[1].clone()));
        assert_eq!(geolocation(&rbt[..1].iter().chain(&rbt[2..]).cloned().collect::<Vec<_>>(), &rbt[0]), Some(rbt[2].clone()));
        // Хвост из двух букв сеткой ещё не делает: так кончаются и служебный
        // `LST_ancillary_ds.nc`, и `chl_nn.nc` у OLCI.
        assert_eq!(grid_tag("x/LST_ancillary_ds.nc"), None);
        assert_eq!(grid_tag("x/chl_nn.nc"), None);
        assert_eq!(grid_tag("x/F1_BT_fn.nc"), Some("fn"));
        assert_eq!(grid_tag("x/geodetic_tx.nc"), Some("tx"));
    }

    /// У SYNERGY файл координат называется своим третьим именем, и знать о нём
    /// больше некому: ни одного из имён OLCI и SLSTR в продукте нет.
    #[test]
    fn synergy_calls_its_coordinates_by_a_third_name() {
        let syn: Vec<String> = [
            "eodata/…/S3B_SY_2_SYN.SEN3/Syn_Oa01_reflectance.nc",
            "eodata/…/S3B_SY_2_SYN.SEN3/geolocation.nc",
            "eodata/…/S3B_SY_2_SYN.SEN3/tiepoints_olci.nc",
        ]
        .iter()
        .map(|key| key.to_string())
        .collect();
        assert_eq!(
            geolocation(&syn, &syn[0]),
            Some("eodata/…/S3B_SY_2_SYN.SEN3/geolocation.nc".to_string())
        );
    }

    /// Координаты берутся только у соседей по каталогу: у продукта бывает
    /// несколько подкаталогов, и чужая сетка хуже, чем никакой.
    #[test]
    fn coordinates_come_from_the_same_folder() {
        let keys: Vec<String> = [
            "eodata/…/product.SEN3/measurement/band.nc",
            "eodata/…/product.SEN3/geo_coordinates.nc",
        ]
        .iter()
        .map(|key| key.to_string())
        .collect();
        assert_eq!(geolocation(&keys, &keys[0]), None);
        // И у растра, который координат не просит, соседей не ищут вовсе.
        let s2 = vec!["eodata/…/GRANULE/IMG_DATA/T36UXV_TCI_10m.jp2".to_string()];
        assert_eq!(geolocation(&s2, &s2[0]), None);
    }
}
