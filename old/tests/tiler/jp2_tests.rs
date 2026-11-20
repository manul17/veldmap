// Extracted from master:veldmodules/image-tiler/src/adapters/jp2.rs

#[cfg(test)]
mod tests {
    use super::*;

    /// Коробка JP2: длина, тип, содержимое. `extended` — длина восьмёркой
    /// после типа, как её пишет настоящая гранула Sentinel-2.
    fn box_of(kind: &[u8; 4], body: &[u8], extended: bool) -> Vec<u8> {
        let mut out = Vec::new();
        match extended {
            false => out.extend_from_slice(&((body.len() + 8) as u32).to_be_bytes()),
            true => out.extend_from_slice(&1u32.to_be_bytes()),
        }
        out.extend_from_slice(kind);
        if extended {
            out.extend_from_slice(&((body.len() + 16) as u64).to_be_bytes());
        }
        out.extend_from_slice(body);
        out
    }

    /// GMLJP2 так, как он лежит в грануле: две вложенные `asoc` с метками,
    /// внешняя — с длиной восьмёркой.
    fn gmljp2(gml: &str) -> Vec<u8> {
        let inner = [
            box_of(b"lbl ", b"gml.root-instance", false),
            box_of(b"xml ", gml.as_bytes(), false),
        ]
        .concat();
        let outer = [box_of(b"lbl ", b"gml.data", false), box_of(b"asoc", &inner, false)].concat();
        box_of(b"asoc", &outer, true)
    }

    /// Настоящий GML из гранулы `S2B_MSIL2A_20260901T104619_..._T42XWQ`,
    /// урезанный до того, что читается. Числа не выдуманы.
    const REAL_GML: &str = r#"<gml:FeatureCollection>
      <gml:RectifiedGridCoverage><gml:domainSet><gml:RectifiedGrid dimension="2">
        <gml:limits><gml:GridEnvelope>
          <gml:low>1 1</gml:low><gml:high>10980 10980</gml:high>
        </gml:GridEnvelope></gml:limits>
        <gml:origin><gml:Point gml:id="P0001" srsName="urn:ogc:def:crs:EPSG::32642">
          <gml:pos>499985 8999995</gml:pos></gml:Point></gml:origin>
        <gml:offsetVector srsName="urn:ogc:def:crs:EPSG::32642">10 0</gml:offsetVector>
        <gml:offsetVector srsName="urn:ogc:def:crs:EPSG::32642">0 -10</gml:offsetVector>
      </gml:RectifiedGrid></gml:domainSet></gml:RectifiedGridCoverage></gml:FeatureCollection>"#;

    /// Привязка Sentinel-2 читается из файла, и угол выходит круглым.
    ///
    /// Круглый угол — не украшение, а проверка полпикселя: начало решётки в GML
    /// названо центром первого отсчёта, и снятые пять метров дают ровно ту
    /// сотку, по которой нарезана плитка MGRS. Не снятые — сдвинули бы снимок
    /// на полпикселя, и заметить это было бы нечем.
    #[test]
    fn привязка_гранулы_читается_и_угол_выходит_круглым() {
        let head = gmljp2(REAL_GML);
        let said = gml_placement(&head, 10980, 10980).expect("гранула несёт привязку");

        assert_eq!(said.epsg, 32642, "UTM 42 северная");
        assert_eq!(said.affine, [10.0, 0.0, 499_980.0, 0.0, -10.0, 9_000_000.0]);
    }

    /// Читается решётка, а не весь документ.
    ///
    /// В GML рядом с ней лежит габарит, а у файлов побогаче — и чужие сетки с
    /// теми же именами тегов. Ищи мы по всему тексту, первым нашлось бы
    /// постороннее: система, границы и шаг уехали бы от чужого элемента, а
    /// числа при этом остались бы правдоподобными.
    ///
    /// Приманки здесь собраны нарочно — по одной на каждое читаемое поле.
    #[test]
    fn читается_решётка_а_не_весь_документ() {
        let with_decoys = format!(
            r#"<gml:FeatureCollection>
                 <gml:boundedBy><gml:Envelope srsName="urn:ogc:def:crs:EPSG::4326">
                   <gml:pos>81.0 66.0</gml:pos></gml:Envelope></gml:boundedBy>
                 <gml:GridEnvelope><gml:low>0 0</gml:low><gml:high>1 1</gml:high></gml:GridEnvelope>
                 <gml:offsetVector srsName="urn:ogc:def:crs:EPSG::4326">1 0</gml:offsetVector>
                 <gml:offsetVector srsName="urn:ogc:def:crs:EPSG::4326">0 -1</gml:offsetVector>
                 {}
               </gml:FeatureCollection>"#,
            REAL_GML
        );
        let said = gml_placement(&gmljp2(&with_decoys), 10980, 10980).expect("решётка на месте");
        assert_eq!(said.epsg, 32642, "система взята у решётки");
        assert_eq!(said.affine, [10.0, 0.0, 499_980.0, 0.0, -10.0, 9_000_000.0], "и шаг с началом");
    }

    /// Система, чей порядок осей неизвестен, привязки не даёт: у зон
    /// Гаусса-Крюгера север пишется первым, и взятая как есть пара положила бы
    /// снимок накрест — числа при этом настоящие, и поймать это нечем.
    #[test]
    fn система_с_неизвестным_порядком_осей_не_берётся() {
        let gk = REAL_GML.replace("EPSG::32642", "EPSG::28408");
        assert!(gml_placement(&gmljp2(&gk), 10980, 10980).is_none());

        let web = REAL_GML.replace("EPSG::32642", "EPSG::3857");
        assert!(gml_placement(&gmljp2(&web), 10980, 10980).is_some(), "веб-Меркатор известен");
    }

    /// Переставленные векторы сдвига возвращаются на места. У квадратной
    /// гранулы сверка размера решётки такую перестановку не ловит вовсе, а
    /// файлы с обоими порядками записи существуют.
    #[test]
    fn переставленные_векторы_возвращаются_на_места() {
        let swapped = REAL_GML
            .replace(">10 0</gml:offsetVector>", ">\u{1}</gml:offsetVector>")
            .replace(">0 -10</gml:offsetVector>", ">10 0</gml:offsetVector>")
            .replace(">\u{1}</gml:offsetVector>", ">0 -10</gml:offsetVector>");
        let said = gml_placement(&gmljp2(&swapped), 10980, 10980).expect("привязка читается");
        assert_eq!(said.affine, [10.0, 0.0, 499_980.0, 0.0, -10.0, 9_000_000.0]);
    }

    /// Не-число проходит всякое сравнение — оно не равно ничему, включая себя.
    /// Дальше по нему считается варп-сетка, и там его уже не поймать.
    #[test]
    fn не_число_в_привязке_не_берётся() {
        let broken = REAL_GML.replace("499985 8999995", "NaN 8999995");
        assert!(gml_placement(&gmljp2(&broken), 10980, 10980).is_none());
    }

    /// Решётка не про этот растр — привязки нет. Натянутая, она положила бы
    /// снимок мимо себя, и молча: числа-то настоящие, просто чужие.
    #[test]
    fn чужая_решётка_не_берётся() {
        let head = gmljp2(REAL_GML);
        assert!(gml_placement(&head, 10980, 10980).is_some(), "своя берётся");
        assert!(gml_placement(&head, 5490, 10980).is_none(), "ширина не та");
        assert!(gml_placement(&head, 10980, 5490).is_none(), "высота не та");
    }

    /// Ни коробки, ни формы — молчание, а не отказ: JP2 без привязки обычен, и
    /// такой снимок ложится по контуру каталога.
    #[test]
    fn без_коробки_привязки_нет_и_это_не_беда() {
        assert!(gml_placement(&[], 10980, 10980).is_none(), "пусто");
        assert!(gml_placement(b"not a jp2 at all", 10980, 10980).is_none(), "мусор");

        let empty = gmljp2("<gml:FeatureCollection>RectifiedGrid без чисел</gml:FeatureCollection>");
        assert!(gml_placement(&empty, 10980, 10980).is_none(), "форма не та");
    }


    /// Бокс с телом — как их пишет контейнер.
    fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    fn jp2_head(width: u32, height: u32) -> Vec<u8> {
        let mut ihdr_body = height.to_be_bytes().to_vec();
        ihdr_body.extend_from_slice(&width.to_be_bytes());
        ihdr_body.extend_from_slice(&[0, 3, 7, 7, 0, 0]); // каналы, бит, прочее

        let mut head = JP2_MAGIC.to_vec();
        head.extend_from_slice(&[0x0D, 0x0A, 0x87, 0x0A]); // хвост сигнатурного бокса
        head.extend_from_slice(&boxed(b"ftyp", b"jp2 \x00\x00\x00\x00jp2 "));
        head.extend_from_slice(&boxed(b"jp2h", &boxed(b"ihdr", &ihdr_body)));
        head
    }

    #[test]
    fn container_header_yields_dims() {
        assert_eq!(header_dims(&jp2_head(10980, 5490)), Ok((10980, 5490)));
    }

    /// Кодстрим: SOC, SIZ с холстом и сеткой тайлов, необязательный сегмент
    /// между ними и COD, потом SOT. Именно так и лежит настоящий файл, и
    /// именно поэтому COD ищется обходом, а не по смещению.
    fn codestream_bytes(canvas: (u32, u32), origin: (u32, u32), tile: (u32, u32), levels: u8)
    -> Vec<u8> {
        let mut out = CODESTREAM_MAGIC.to_vec(); // SOC + маркер SIZ
        let mut siz = 0u16.to_be_bytes().to_vec(); // Rsiz
        for value in [canvas.0, canvas.1, origin.0, origin.1, tile.0, tile.1, origin.0, origin.1] {
            siz.extend_from_slice(&value.to_be_bytes());
        }
        siz.extend_from_slice(&3u16.to_be_bytes()); // Csiz
        siz.extend_from_slice(&[7, 1, 1, 7, 1, 1, 7, 1, 1]); // Ssiz/XRsiz/YRsiz ×3
        out.extend_from_slice(&((siz.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&siz);

        // Необязательный сегмент перед COD — обход обязан через него перешагнуть.
        out.extend_from_slice(&[0xFF, 0x55, 0x00, 0x04, 0x00, 0x00]); // TLM

        let cod = [0u8, 0, 0, 1, 0, levels, 5, 5, 0, 0, 0];
        out.extend_from_slice(&[0xFF, 0x52]);
        out.extend_from_slice(&((cod.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&cod);
        out.extend_from_slice(&[0xFF, 0x90]); // SOT — дальше главного заголовка нет
        out
    }

    /// Что решает чтение по сети, читается из той же головы: записи TLM
    /// считаются по всем сегментам и по ширине полей из Stlm, PLT ищется в
    /// заголовке первого тайл-парта, прогрессия берётся из COD.
    #[test]
    fn tile_part_index_and_plt_are_read_from_the_head() {
        let mut raw = codestream_bytes((2048, 2048), (0, 0), (1024, 1024), 4);
        let sot_at = raw.len() - 2;
        raw.truncate(sot_at);
        // Второй TLM: Ztlm 1, Stlm 0x40 (Ttlm нет, Ptlm по 4 байта), три записи.
        raw.extend_from_slice(&[0xFF, 0x55, 0x00, 0x10, 0x01, 0x40]);
        raw.extend_from_slice(&[0; 12]);
        // SOT первого тайл-парта, его заголовок с PLT, потом SOD.
        raw.extend_from_slice(&[0xFF, 0x90, 0x00, 0x0A, 0, 0, 0, 0, 0, 0, 0, 1]);
        raw.extend_from_slice(&[0xFF, 0x58, 0x00, 0x03, 0x00]);
        raw.extend_from_slice(&[0xFF, 0x93]);
        let layout = codestream(&raw).expect("раскладка читается");
        assert_eq!(layout.tlm_parts, Some(3), "первый TLM пуст, второй перечисляет три тайл-парта");
        assert_eq!(layout.plt_first, Some(true));
        assert_eq!((layout.progression, layout.layers), (0, 1));

        // Без PLT до SOD — «нет»; голова, кончившаяся раньше SOD, — «не видно».
        let without = codestream_bytes((2048, 2048), (0, 0), (1024, 1024), 4);
        let mut bare = without.clone();
        bare.extend_from_slice(&[0x00, 0x0A, 0, 0, 0, 0, 0, 0, 0, 1, 0xFF, 0x93]);
        assert_eq!(codestream(&bare).unwrap().plt_first, Some(false));
        assert_eq!(codestream(&without).unwrap().plt_first, None);
        assert_eq!(codestream(&without).unwrap().tlm_parts, Some(0), "пустой TLM — сегмент есть, записей нет");
    }

    /// Раскладка читается из кодстрима: холст, сетка тайлов, компоненты,
    /// разрешения. Число разрешений на единицу больше записанных ступеней —
    /// файл пишет ступени, а нам нужны уровни, которые декодер отдаст.
    #[test]
    fn codestream_layout_is_read_from_the_main_header() {
        let raw = codestream_bytes((10980, 10980), (0, 0), (1024, 1024), 5);
        let layout = codestream(&raw).expect("раскладка читается");

        assert_eq!(layout.canvas, (10980, 10980));
        assert_eq!(layout.origin, (0, 0));
        assert_eq!(layout.tile, (1024, 1024));
        assert_eq!(layout.components, 3);
        assert_eq!(layout.resolutions, 6, "ступеней 5 — значит разрешений 6");
        assert_eq!(layout.grid(), (11, 11));
    }

    /// Тот же кодстрим внутри коробки `jp2c`: у контейнера главный заголовок
    /// лежит не с начала файла, и найти его — половина работы.
    #[test]
    fn codestream_is_found_inside_the_container() {
        let mut head = jp2_head(700, 40);
        head.extend_from_slice(&boxed(b"jp2c", &codestream_bytes((700, 40), (0, 0), (700, 40), 4)));
        let layout = codestream(&head).expect("раскладка читается и в контейнере");

        assert_eq!(layout.canvas, (700, 40));
        assert_eq!(layout.resolutions, 5);
        assert_eq!(layout.grid(), (1, 1), "один тайл на весь холст");
    }

    /// Нарезка считается от начала сетки, а не от нуля: у файла со смещением
    /// холста они разные, и деление нацело здесь солгало бы.
    #[test]
    fn the_tile_grid_counts_from_its_own_origin() {
        let raw = codestream_bytes((1000, 600), (40, 40), (256, 256), 3);
        let layout = codestream(&raw).expect("раскладка читается");

        assert_eq!(layout.origin, (40, 40));
        assert_eq!(layout.grid(), (4, 3), "(1000-40)/256 → 4, (600-40)/256 → 3");
    }

    /// Кодстрим без COD — не раскладка «по умолчанию», а отказ: число
    /// разрешений решает, до какого уровня действует фактор декодера, и
    /// придумать его нельзя.
    #[test]
    fn a_codestream_without_cod_is_refused() {
        let mut raw = codestream_bytes((700, 40), (0, 0), (700, 40), 4);
        let cod = raw.windows(2).position(|pair| pair == [0xFF, 0x52]).expect("COD на месте");
        raw[cod + 1] = 0x58; // RGN — сегмент есть, но не тот
        assert!(codestream(&raw).is_err());
    }

    #[test]
    fn raw_codestream_yields_dims() {
        // SOC + SIZ с холстом 700×40 без смещения.
        let mut head = CODESTREAM_MAGIC.to_vec();
        head.extend_from_slice(&41u16.to_be_bytes()); // Lsiz
        head.extend_from_slice(&0u16.to_be_bytes()); // Rsiz
        for value in [700u32, 40, 0, 0] {
            head.extend_from_slice(&value.to_be_bytes());
        }
        assert_eq!(header_dims(&head), Ok((700, 40)));
    }

    #[test]
    fn garbage_is_refused() {
        assert!(header_dims(b"PNG not really").is_err());
        assert!(header_dims(JP2_MAGIC).is_err()); // сигнатура есть, заголовка нет
    }

    /// Шаг за боксом не выходит за голову, какую бы длину бокс ни объявил.
    ///
    /// Проверяется правило, а не отсутствие паники, и это не придирка к форме:
    /// паника здесь бывает только там, где `usize` тридцатидвухбитный, то есть
    /// в самом модуле, а тест бежит нативно на шестидесяти четырёх — сложение
    /// «начало плюс четыре гигабайта» в нём не переполняется по построению, и
    /// порченый разбор прошёл бы мимо. Правило же одинаково на любой
    /// разрядности, и держит оно ровно то, ради чего написано.
    #[test]
    fn a_box_never_steps_past_the_head() {
        // Длина обычного бокса: на тридцати двух битах ей уже не остаётся
        // места под «плюс начало».
        assert_eq!(step(0, u64::from(u32::MAX), 4096), 4096);
        assert_eq!(step(4088, u64::from(u32::MAX), 4096), 4096);
        // Расширенная длина приезжает из файла целым `u64`, так что переполнить
        // можно и его — от `at` в этой сумме места не остаётся вовсе.
        assert_eq!(step(4088, u64::MAX, 4096), 4096);
        assert_eq!(step(4088, u64::MAX - 1, 4096), 4096);
        // Обычный бокс шагает ровно на свою длину.
        assert_eq!(step(16, 32, 4096), 48);
    }

    /// Ключ «нет данных»: ноль всеми цветовыми каналами — прозрачность, и
    /// только у файла, которому он назначен; знаковые отсчёты сдвигаются в
    /// беззнаковые, разрядность до байта дотягивается до байта.
    #[test]
    fn samples_become_rgba_by_the_file_rules() {
        let planes = [vec![0, 200, 0], vec![0, 100, 0], vec![0, 50, 7]];
        let tile = Tile { x0: 0, y0: 0, width: 3, height: 1, planes: planes.iter().map(|p| p.as_slice()).collect() };
        let header = Header { x0: 0, y0: 0, width: 3, height: 1, components: 3, precision: 8, signed: false, uneven: false };
        let keyed = rgba(&tile, header, &Mapping::identity(Some(0.0)));
        assert_eq!(&keyed[0..4], &[0, 0, 0, 0], "ноль всеми каналами — поле");
        assert_eq!(&keyed[4..8], &[200, 100, 50, 255]);
        assert_eq!(&keyed[8..12], &[0, 0, 7, 255], "ноль не всеми каналами — цвет");
        let plain = rgba(&tile, header, &Mapping::identity(None));
        assert_eq!(&plain[0..4], &[0, 0, 0, 255], "без ключа поле — чёрный цвет");

        let signed = [vec![-8, 7]];
        let tile = Tile { x0: 0, y0: 0, width: 2, height: 1, planes: signed.iter().map(|p| p.as_slice()).collect() };
        let header = Header { precision: 4, signed: true, components: 1, width: 2, ..header };
        let grey = rgba(&tile, header, &Mapping::identity(None));
        assert_eq!(&grey[0..4], &[0, 0, 0, 255], "минимум знакового — ноль");
        assert_eq!(&grey[4..8], &[240, 240, 240, 255], "максимум четырёх бит — почти байт");
    }

    /// Раскладка драйвера из главного заголовка: чанк — тайл кодстрима, копии —
    /// пока тайл делится на два, глубина — по плоскостям декодера.
    #[test]
    fn the_layout_follows_the_codestream() {
        let cs = Codestream {
            canvas: (10980, 10980),
            origin: (0, 0),
            tile: (1024, 1024),
            tile_origin: (0, 0),
            components: 3,
            precision: 8,
            resolutions: 5,
            progression: 0,
            layers: 1,
            tlm_parts: Some(121),
            plt_first: Some(true),
        };
        let layout = Layout::of(&cs, 130 << 20, Format::Jp2, Some(0.0), None);
        assert_eq!(layout.grid.chunk, (1024, 1024));
        assert_eq!(layout.tiles, (11, 11));
        assert_eq!(layout.grid.overviews.len(), 4, "четыре копии при пяти разрешениях");
        assert_eq!((layout.grid.overviews[3].width, layout.grid.overviews[3].chunk), (687, (64, 64)));
        assert_eq!(layout.grid.depth, 3 * SAMPLE_BYTES);

        // Копии кончаются там, где тайл перестаёт делиться на два: у 300
        // это фактор 3, у нечётного тайла копий нет вовсе.
        let even = Codestream { tile: (300, 300), canvas: (600, 600), ..cs };
        assert_eq!(Layout::of(&even, 1, Format::J2k, None, None).grid.overviews.len(), 2);
        let odd = Codestream { tile: (301, 301), canvas: (602, 602), ..cs };
        assert!(Layout::of(&odd, 1, Format::J2k, None, None).grid.overviews.is_empty());
    }
}
