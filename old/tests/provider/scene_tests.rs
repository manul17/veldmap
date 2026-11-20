// Extracted from master:veldmodules/data-provider/src/scene.rs

#[cfg(test)]
mod tests {
    use super::*;

    /// Факты без единой связи — с них начинается каждый набор в тестах.
    fn bare(platform: &str, kind: &str, level: u32, size: u64) -> Facts {
        Facts {
            platform: platform.to_string(),
            instrument: String::new(),
            datatake: String::new(),
            tile: String::new(),
            named_tile: String::new(),
            slice: String::new(),
            orbit: String::new(),
            second: 100,
            level: Some(level),
            kind: kind.to_string(),
            size,
            framed: true,
        }
    }

    /// Части одного слайса Sentinel-1: номер слайса у них общий, а секунда
    /// начала — своя у каждой, как их и отдаёт каталог.
    fn facts(kind: &str, level: u32, size: u64) -> Facts {
        Facts {
            datatake: "73311".to_string(),
            slice: "23".to_string(),
            second: 1_786_000_000,
            ..bare("SENTINEL-1", kind, level, size)
        }
    }

    fn product(name: &str) -> DataProduct {
        DataProduct {
            identifier: format!("eodata/{}", name),
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Каталог называет дататейк не всему архиву, а имя — всегда. Без этого
    /// одна съёмка Sentinel-1 до осени 2023 года распадалась на три строки:
    /// секунды у сырья, комплексного и обработанного расходятся.
    #[test]
    fn a_datatake_is_read_from_the_name_when_the_catalogue_keeps_quiet() {
        // Живые имена: обычное, с двойным подчёркиванием в типе и с `_COG`.
        assert_eq!(
            datatake_in_name("S1B_IW_GRDH_1SDV_20190601T001108_20190601T001137_016495_01F0AB_C0FF.SAFE"),
            Some("127147".to_string())
        );
        assert_eq!(
            datatake_in_name("S1D_WV_SLC__1SSV_20260818T000819_20260818T001004_004172_007A39_CEFE.SAFE"),
            Some("31289".to_string())
        );
        assert_eq!(
            datatake_in_name(
                "S1C_IW_GRDH_1SDV_20260818T001104_20260818T001133_009042_011F30_E912_COG.SAFE"
            ),
            Some("73520".to_string())
        );
        // Не Sentinel-1 — поля стоя́т иначе, и выдумывать дататейк нельзя.
        assert_eq!(
            datatake_in_name("S2A_MSIL2A_20260818T012111_N0512_R031_T54RWV_20260818T042717.SAFE"),
            None
        );
        assert_eq!(
            datatake_in_name(
                "S3A_SL_2_LST____20260818T000122_20260818T000422_20260818T011237_0179_143_045_0000_PS1_O_NR_005.SEN3"
            ),
            None
        );
        assert_eq!(datatake_in_name("что угодно"), None);
    }

    /// «Одна ли это съёмка» отвечают в двух местах — сведением выдачи и
    /// обратным ходом по ключу, — и ответ у них обязан быть один. У части без
    /// номера слайса (OCN) голый ключ расходится с ключом её слайса, и она
    /// оставалась одиночкой ровно тогда, когда о ней спрашивали по ключу.
    #[test]
    fn the_same_scene_is_the_same_from_both_sides() {
        let mut grd = bare("SENTINEL-1", "IW_GRDH_1S", 1, 1_600_000_000);
        grd.datatake = "73290".to_string();
        grd.slice = "17".to_string();
        grd.second = 1_786_000_000;
        let mut ocn = bare("SENTINEL-1", "IW_OCN__2S", 2, 69_000_000);
        ocn.datatake = "73290".to_string();
        ocn.second = 1_786_000_000;

        let set = vec![
            (grd.clone(), product("S1C_IW_GRDH_1SDV_A.SAFE")),
            (ocn.clone(), product("S1C_IW_OCN__2SDV_B.SAFE")),
        ];
        // Сведение выдачи кладёт их в один снимок.
        assert_eq!(group(set.clone()).len(), 1, "поиск развёл части по снимкам");
        // И обратный ход по ключу отбирает те же самые две.
        assert_eq!(
            same_scene(&ocn, "S1C_IW_OCN__2SDV_B.SAFE", set).len(),
            2,
            "по ключу часть осталась одиночкой"
        );
    }

    /// Пять упаковок одной съёмки Sentinel-1 — один снимок, и показывается
    /// тайловая: сырьё нечем показать, полосной уровень 1 читается только
    /// целиком, а уровень 2 — уже не картинка.
    #[test]
    fn one_datatake_becomes_one_scene_shown_by_its_tiled_packaging() {
        let scenes = group(vec![
            (facts("IW_RAW__0S", 0, 1_218_000_000), product("raw")),
            (facts("IW_SLC__1S", 1, 6_630_000_000), product("slc")),
            (facts("IW_GRDH_1S", 1, 1_641_000_000), product("grd")),
            (facts("IW_GRDH_1S-COG", 1, 739_000_000), product("cog")),
            (facts("IW_OCN__2S", 2, 69_000_000), product("ocn")),
        ]);
        assert_eq!(scenes.len(), 1, "снимок один");
        assert_eq!(scenes[0].name, "cog");
        let order: Vec<&str> = scenes[0].parts.iter().map(|p| p.product_type.as_str()).collect();
        assert_eq!(order, ["IW_GRDH_1S-COG", "IW_GRDH_1S", "IW_SLC__1S", "IW_OCN__2S", "IW_RAW__0S"]);
        assert!(scenes[0].parts[0].shown, "показывается первая");
        assert_eq!(scenes[0].parts.iter().filter(|p| p.shown).count(), 1);
    }

    /// Тайловой упаковки нет — берётся уровень 1, а не самый маленький файл:
    /// продукт уровня 2 бывает меньше всех и картинкой не является.
    #[test]
    fn without_a_tiled_packaging_level_one_wins_over_the_smallest() {
        let scenes = group(vec![
            (facts("IW_OCN__2S", 2, 69_000_000), product("ocn")),
            (facts("IW_SLC__1S", 1, 6_630_000_000), product("slc")),
            (facts("IW_GRDH_1S", 1, 1_641_000_000), product("grd")),
        ]);
        assert_eq!(scenes[0].name, "grd");
    }

    /// Съёмки одной плитки за разные пролёты — разные снимки, и пустой
    /// дататейк их не сливает.
    #[test]
    fn a_degree_tile_is_read_from_the_name_when_the_catalogue_is_silent() {
        // Так называет свои клетки Sentinel-1 RTC: ни tileId, ни дататейка, ни
        // слайса у них нет, а виток с секундой у соседних клеток общие.
        assert_eq!(tile_in_name("N03W004_2018_01_19_010910"), Some("N03W004".to_string()));
        assert_eq!(tile_in_name("S11E024_2018_01_19_010906"), Some("S11E024".to_string()));
        // Имена прочих миссий начинаются с платформы — под правило не подпадают.
        assert_eq!(tile_in_name("S1A_IW_GRDH_1SDV_20260817T070301"), None);
        assert_eq!(tile_in_name("S2B_MSIL1C_20260812T093549"), None);
        assert_eq!(tile_in_name("N03W04_2018_01_19"), None, "долгота трёхзначная");
    }

    /// Пара времён стои́т не только в именах Sentinel-1, и поле за ней у чужих
    /// значит другое. У COP-DEM там нули, и все его клетки одной даты слились
    /// бы в одну строку списка.
    #[test]
    fn a_datatake_is_read_only_from_a_sentinel_one_name() {
        assert_eq!(
            datatake_in_name("DEM1_SAR_DTE_30_20101216T101933_20140316T000041_ADS_000000_rLrO"),
            None
        );
        assert_eq!(datatake_in_name("S2B_MSIL1C_20260812T093549_N0511_R036_T33UVP_x"), None);
        // А своё имя читается по-прежнему.
        assert_eq!(
            datatake_in_name("S1B_IW_GRDH_1SDV_20190601T001108_20190601T001137_016495_01F0AB_C0FF"),
            Some("127147".to_string())
        );
    }

    /// Клетки одного витка — разные снимки: пролёт нарезан на них уже после
    /// съёмки, и время у всех от неё одно.
    #[test]
    fn degree_tiles_of_one_pass_stay_apart() {
        let rtc = |tile: &str| Facts {
            named_tile: tile.to_string(),
            orbit: "118".to_string(),
            ..bare("SENTINEL-1", "RTC", 2, 100)
        };
        assert_ne!(
            acquisition(&rtc("N03W004")).expect("ключ"),
            acquisition(&rtc("N01W006")).expect("ключ")
        );
    }

    #[test]
    fn same_tile_at_another_pass_is_another_scene() {
        let tiled = |tile: &str, second: i64| Facts {
            tile: tile.to_string(),
            second,
            ..bare("SENTINEL-2", "S2MSI1C", 1, 800)
        };
        let scenes = group(vec![
            (tiled("43XDB", 100), product("a")),
            (tiled("43XDB", 200), product("b")),
            (tiled("43WFU", 100), product("c")),
        ]);
        assert_eq!(scenes.len(), 3);
    }

    /// Один виток и одна секунда — одна съёмка: пятнадцать продуктов
    /// Sentinel-5P об одной пятиминутной грануле становятся одним снимком.
    /// Показывается самый дешёвый уровень 2 — уровень 1 у этой миссии не
    /// картинка, а спектры.
    #[test]
    fn one_orbit_and_one_second_are_one_scene() {
        let tropomi = |kind: &str, level: u32, size: u64| Facts {
            instrument: "TROPOMI".to_string(),
            orbit: "45811".to_string(),
            ..bare("SENTINEL-5P", kind, level, size)
        };
        let scenes = group(vec![
            (tropomi("L2__NO2___", 2, 63_906_515), product("no2")),
            (tropomi("L2__AER_AI", 2, 25_905_862), product("aer")),
            (tropomi("L2__SO2___", 2, 119_595_377), product("so2")),
        ]);
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0].name, "aer");
        assert_eq!(scenes[0].parts.len(), 3);
    }

    /// Виток тот же, а гранула другая — снимка два: за полтора часа пролёта их
    /// два десятка, и склеивать их в один было бы враньём.
    #[test]
    fn another_granule_of_the_same_orbit_is_another_scene() {
        let granule = |second: i64| Facts {
            instrument: "TROPOMI".to_string(),
            orbit: "45811".to_string(),
            second,
            ..bare("SENTINEL-5P", "L2__NO2___", 2, 63_000_000)
        };
        let scenes = group(vec![(granule(100), product("a")), (granule(400), product("b"))]);
        assert_eq!(scenes.len(), 2);
    }

    /// Виток тот же и секунда та же, а прибор другой — снимка два: Sentinel-3
    /// снимает двумя разом, и это две разные съёмки.
    #[test]
    fn two_instruments_of_one_orbit_stay_apart() {
        let aboard = |instrument: &str, kind: &str| Facts {
            instrument: instrument.to_string(),
            orbit: "1234".to_string(),
            ..bare("SENTINEL-3", kind, 2, 40)
        };
        let scenes = group(vec![
            (aboard("SLSTR", "SL_2_WST___"), product("wst")),
            (aboard("OLCI", "OL_2_WFR___"), product("wfr")),
        ]);
        assert_eq!(scenes.len(), 2);
        assert!(scenes.iter().all(|scene| scene.parts.is_empty()), "часть одна");
    }

    /// Каталог не сказал ничего — ни дататейка, ни плитки, ни витка: снимок
    /// остаётся сам по себе. Догадываться тут не о чем.
    #[test]
    fn products_the_catalogue_did_not_link_stay_apart() {
        let scenes = group(vec![
            (bare("SENTINEL-3", "SL_2_WST___", 2, 40), product("wst")),
            (bare("SENTINEL-3", "OL_2_WFR___", 2, 40), product("wfr")),
        ]);
        assert_eq!(scenes.len(), 2);
    }

    /// Каталог их не связал, а имена расходятся одним хвостом — значит это
    /// одни и те же величины в двух укладках, и снимок у них один. Показывается
    /// COG: его читают окном, а NetCDF — только целиком.
    #[test]
    fn the_same_name_with_another_packaging_suffix_is_one_scene() {
        let clms = |size: u64| bare("PROBA-V", "land_surface_temperature", 2, size);
        let scenes = group(vec![
            (clms(27_069_429), product("c_gls_LST_202608160500_GLOBE_GEO_V3.0.1_cog")),
            (clms(23_912_978), product("c_gls_LST_202608160500_GLOBE_GEO_V3.0.1_nc")),
        ]);
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0].parts.len(), 2);
        assert!(scenes[0].name.ends_with("_cog"), "показывается тайловая: {}", scenes[0].name);
    }

    /// Без контура — не снимок: калибровочные таблицы каталог отдаёт наравне со
    /// съёмкой и по свежести её обгоняют.
    #[test]
    fn products_without_a_footprint_are_not_scenes() {
        let mut table = facts("GIP_R2ABCA", 1, 6_700);
        table.framed = false;
        table.datatake = String::new();
        let scenes = group(vec![
            (table, product("gipp")),
            (facts("IW_GRDH_1S", 1, 1_641_000_000), product("grd")),
        ]);
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0].name, "grd");
    }

    /// Спросили о сырье — отвечать надо обработанным снимком той же съёмки:
    /// изображения в сырье нет вовсе.
    #[test]
    fn a_question_about_level_zero_is_answered_by_the_processed_scene() {
        let products = vec![
            (facts("IW_RAW__0S", 0, 1_218_000_000), product("raw")),
            (facts("IW_GRDH_1S-COG", 1, 739_000_000), product("cog")),
        ];
        let answer = about(products, "eodata/raw").expect("спрошенное среди них есть");
        assert_eq!(answer.name, "cog");
        assert!(answer.parts.iter().find(|part| part.name == "cog").unwrap().shown);
    }

    /// Спросили о полосной укладке — отвечать надо тайловой: величина та же, а
    /// читается она окном, а не целиком.
    #[test]
    fn a_question_about_the_striped_packaging_is_answered_by_the_tiled_one() {
        let products = vec![
            (facts("IW_GRDH_1S", 1, 1_641_000_000), product("grd")),
            (facts("IW_GRDH_1S-COG", 1, 739_000_000), product("cog")),
        ];
        assert_eq!(about(products, "eodata/grd").unwrap().name, "cog");
    }

    /// А вот другая измеренная величина — не ответ: нажали на угарный газ,
    /// показаться должен угарный газ, даже если озон рядом дешевле.
    #[test]
    fn a_question_about_one_quantity_is_not_answered_by_another() {
        let tropomi = |kind: &str, size: u64| Facts {
            instrument: "TROPOMI".to_string(),
            orbit: "45811".to_string(),
            ..bare("SENTINEL-5P", kind, 2, size)
        };
        let products = vec![
            (tropomi("L2__CO____", 33_821_960), product("co")),
            (tropomi("L2__O3__PR", 3_000_000), product("o3pr")),
        ];
        let answer = about(products, "eodata/co").expect("спрошенное среди них есть");
        assert_eq!(answer.name, "co", "спрошенное и показывается");
        // Соседи при этом никуда не делись — их порядок общий, переехала
        // только пометка.
        let shown: Vec<&str> = answer
            .parts
            .iter()
            .filter(|part| part.shown)
            .map(|part| part.name.as_str())
            .collect();
        assert_eq!(shown, ["co"]);
        assert_eq!(answer.parts.len(), 2);
    }

    /// Уровень — последняя цифра слова: первая у Sentinel-2 принадлежит имени
    /// прибора, и по ней L1C оказался бы уровнем 2.
    #[test]
    fn level_is_the_last_digit_of_the_catalogue_word() {
        assert_eq!(level("LEVEL0"), Some(0));
        assert_eq!(level("LEVEL1"), Some(1));
        assert_eq!(level("LEVEL2"), Some(2));
        assert_eq!(level("S2MSI1C"), Some(1));
        assert_eq!(level("S2MSI2A"), Some(2));
        assert_eq!(level(""), None, "каталог промолчал — это отдельный ответ");
    }

    /// Части одного слайса сводятся номером слайса, а не секундой: у сырья
    /// нарезка своя, и начинается его слайс на четыре секунды раньше
    /// обработанного. По секунде одна съёмка разошлась бы на три строки.
    #[test]
    fn one_slice_is_one_scene_however_its_parts_start() {
        let part = |kind: &str, level: u32, size: u64, second: i64| Facts {
            second: 1_786_000_000 + second,
            ..facts(kind, level, size)
        };
        let scenes = group(vec![
            (part("IW_RAW__0S", 0, 1_218_000_000, -4), product("raw")),
            (part("IW_SLC__1S", 1, 6_630_000_000, -1), product("slc")),
            (part("IW_GRDH_1S", 1, 1_641_000_000, 0), product("grd")),
            (part("IW_GRDH_1S-COG", 1, 739_000_000, 0), product("cog")),
        ]);
        assert_eq!(scenes.len(), 1, "слайс один");
        assert_eq!(scenes[0].parts.len(), 4);
        assert_eq!(scenes[0].name, "cog");

        // А соседний слайс того же дататейка — другая съёмка: их разделяет
        // двадцать пять секунд, и окно до них не дотягивается.
        let next = group(vec![
            (part("IW_GRDH_1S", 1, 1_641_000_000, 0), product("here")),
            (Facts { slice: "24".to_string(), ..part("IW_GRDH_1S", 1, 1_641_000_000, 25) },
                product("next")),
        ]);
        assert_eq!(next.len(), 2);
    }

    /// Часть без номера слайса прикладывается к тому слайсу, рядом с которым
    /// снята: `OCN` каталог номером не снабжает вовсе, а начинается он в ту же
    /// секунду, что и снимок, из которого посчитан.
    #[test]
    fn a_part_without_a_slice_number_joins_the_one_beside_it() {
        let ocn = Facts { slice: String::new(), ..facts("IW_OCN__2S", 2, 69_000_000) };
        let scenes = group(vec![
            (facts("IW_GRDH_1S", 1, 1_641_000_000), product("grd")),
            (ocn, product("ocn")),
        ]);
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0].parts.len(), 2);

        // Далеко от всякого слайса — сам по себе: приложить его не к чему.
        let far = Facts {
            slice: String::new(),
            second: 1_786_000_600,
            ..facts("IW_OCN__2S", 2, 69_000_000)
        };
        let apart = group(vec![
            (facts("IW_GRDH_1S", 1, 1_641_000_000), product("grd")),
            (far, product("ocn")),
        ]);
        assert_eq!(apart.len(), 2);
    }

    /// Дататейк, которому каталог не назвал ни одного номера слайса, всё равно
    /// держит свои части вместе — их связывает он сам.
    ///
    /// Такие в каталоге есть: у съёмки 191698 (5 марта 2019) номера слайса нет
    /// ни у комплексного, ни у производной величины, и начинаются они в одну
    /// секунду. Приложить их не к чему — слайсов, рядом с которыми они сняты,
    /// не существует, — так что держит их ключ «по съёмке».
    #[test]
    fn a_datatake_without_any_slice_numbers_still_holds_its_parts() {
        let slc = Facts {
            datatake: "191698".to_string(),
            slice: String::new(),
            ..facts("IW_SLC__1S", 1, 7_800_000_000)
        };
        let ocn = Facts {
            datatake: "191698".to_string(),
            slice: String::new(),
            ..facts("IW_OCN__2S", 2, 69_000_000)
        };
        let scenes = group(vec![(slc.clone(), product("slc")), (ocn, product("ocn"))]);
        assert_eq!(scenes.len(), 1, "одна секунда на двоих — один ключ");
        assert_eq!(scenes[0].parts.len(), 2);

        // И держится это не совпадением секунд: у сырья, комплексного и
        // обработанного они расходятся на две-четыре, и по секунде такая
        // съёмка разъехалась бы по строкам. Дататейк без номеров слайса
        // размечается по времени тем же окном, что и прикладывание.
        let nameless = |kind: &str, level: u32, second: i64| Facts {
            datatake: "191698".to_string(),
            slice: String::new(),
            second,
            ..facts(kind, level, 1_500_000_000)
        };
        let together = group(vec![
            (nameless("IW_SLC__1S", 1, 1_786_000_000), product("slc")),
            (nameless("IW_RAW__0S", 0, 1_786_000_004), product("raw")),
        ]);
        assert_eq!(together.len(), 1, "четыре секунды — это одна съёмка");
        assert_eq!(together[0].parts.len(), 2);

        // А соседний слайс того же дататейка — уже другая съёмка: слайсы стоя́т
        // друг от друга на двадцать пять секунд и больше.
        let apart = group(vec![
            (nameless("IW_SLC__1S", 1, 1_786_000_000), product("slc")),
            (nameless("IW_SLC__1S", 1, 1_786_000_030), product("next")),
        ]);
        assert_eq!(apart.len(), 2, "разные слайсы одного дататейка — разные съёмки");
    }

    /// Уровень, не названный каталогом, не делает продукт снимком. Служебная
    /// аннотация к съёмке приезжает неделей позже, весит впятеро меньше — и по
    /// прежнему правилу «не сказано, значит уровень 1» вставала бы впереди
    /// самого снимка, а без тайловой упаковки и показывалась бы вместо него.
    #[test]
    fn an_unnamed_level_never_outranks_the_image() {
        let annotation = Facts { level: None, ..facts("IW_ETA__AX", 1, 59_345_163) };
        let scenes = group(vec![
            (annotation, product("eta")),
            (facts("IW_GRDH_1S", 1, 1_986_444_206), product("grd")),
        ]);
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0].name, "grd", "показывается снимок, а не аннотация к нему");
        assert!(scenes[0].parts[0].shown);
        assert_eq!(scenes[0].parts[1].product_type, "IW_ETA__AX");
    }

    /// Хвост укладки узнаётся независимо от регистра и от того, стои́т ли он в
    /// самом конце имени или перед расширением. Иначе одна и та же съёмка,
    /// уложенная двумя способами, показывалась бы двумя строками.
    #[test]
    fn packaging_is_stripped_wherever_it_stands() {
        assert_eq!(stem("c_gls_LST_202608160500_GLOBE_GEO_V3.0.1_cog"), "c_gls_LST_202608160500_GLOBE_GEO_V3.0.1");
        assert_eq!(stem("c_gls_LST_202608160500_GLOBE_GEO_V3.0.1_nc"), "c_gls_LST_202608160500_GLOBE_GEO_V3.0.1");
        // Верхний регистр и расширение после укладки.
        assert_eq!(stem("EW02_WV1_MS4_OR_TOU_1234_675b_COG.DIMA"), "EW02_WV1_MS4_OR_TOU_1234_675b");
        assert_eq!(stem("EW02_WV1_MS4_OR_TOU_1234_675b.DIMA"), "EW02_WV1_MS4_OR_TOU_1234_675b");
        // Укладка в конце против расширения у близнеца — тоже одна съёмка.
        assert_eq!(stem("EW02_WV1_MS4_OR_TOU_1234_5415_COG"), "EW02_WV1_MS4_OR_TOU_1234_5415");
        assert_eq!(stem("EW02_WV1_MS4_OR_TOU_1234_5415.DIMA"), "EW02_WV1_MS4_OR_TOU_1234_5415");
        // Версия с точками расширением не считается: иначе от имени остался бы огрызок.
        assert_eq!(without_extension("thing_V3.0.1"), "thing_V3.0.1");
    }
}
