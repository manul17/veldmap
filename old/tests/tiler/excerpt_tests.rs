// Extracted from master:veldmodules/image-tiler/src/adapters/excerpt.rs

#[cfg(test)]
mod tests {
    use super::*;

    /// Кодирование гранулы Sentinel-2: три компонента без подвыборки, четыре
    /// ступени, прецинкты 256² на всех разрешениях.
    fn granule() -> Coding {
        let comp = Component { dx: 1, dy: 1, precincts: vec![(8, 8); 5] };
        Coding {
            canvas: (10980, 10980),
            origin: (0, 0),
            tile: (1024, 1024),
            tile_origin: (0, 0),
            comps: vec![comp.clone(), comp.clone(), comp],
            layers: 1,
            progression: 0,
            marked: false,
            changes: false,
        }
    }

    /// Счёт пакетов сходится с PLT настоящей гранулы (замер 3 сентября 2026):
    /// у полного тайла по разрешениям 3, 3, 3, 12, 48 пакетов, у краевого
    /// 740² — 3, 3, 3, 12, 27.
    #[test]
    fn packets_are_counted_as_the_granule_writes_them() {
        let coding = granule();
        assert_eq!((coding.across(), coding.down()), (11, 11));
        let by_resolution = |tile: u32| -> Vec<u64> {
            (0..5).map(|r| coding.packets(tile, r) - if r == 0 { 0 } else { coding.packets(tile, r - 1) }).collect()
        };
        assert_eq!(by_resolution(0), [3, 3, 3, 12, 48]);
        assert_eq!(by_resolution(120), [3, 3, 3, 12, 27]);
        assert_eq!(coding.packets(0, u32::MAX), 69);
        assert_eq!(coding.levels(), 4);
        assert!(coding.prefixed());

        // Прецинкты по умолчанию — один на разрешение, а два слоя удваивают.
        let mut plain = granule();
        for comp in &mut plain.comps {
            comp.precincts = vec![(15, 15); 5];
        }
        plain.layers = 2;
        assert_eq!(plain.packets(0, u32::MAX), 30);
        assert!(!plain.prefixed(), "LRCP с двумя слоями не кладёт разрешения префиксом");
        plain.progression = 1;
        assert!(plain.prefixed(), "у RLCP разрешение — внешний цикл при любых слоях");
    }

    /// Главный заголовок для индекса: SIZ на один компонент, COD, TLM с
    /// данными записями — до первого SOT, который приносит тайл-парт.
    fn main_header(canvas: u32, tile: u32, levels: u8, tlm: &[(u16, u32)]) -> Vec<u8> {
        main_header_with(canvas, tile, levels, tlm, 0, &[])
    }

    /// Тот же заголовок со стилем кодирования `scod` в COD и лишним сегментом
    /// `extra` (маркер, длина, тело) после него.
    fn main_header_with(canvas: u32, tile: u32, levels: u8, tlm: &[(u16, u32)], scod: u8, extra: &[u8]) -> Vec<u8> {
        let mut out = vec![0xFF, 0x4F, 0xFF, 0x51];
        let mut siz = 0u16.to_be_bytes().to_vec();
        for value in [canvas, canvas, 0, 0, tile, tile, 0, 0] {
            siz.extend_from_slice(&value.to_be_bytes());
        }
        siz.extend_from_slice(&1u16.to_be_bytes());
        siz.extend_from_slice(&[7, 1, 1]);
        out.extend_from_slice(&((siz.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&siz);
        let cod = [scod, 0, 0, 1, 0, levels, 4, 4, 0, 1];
        out.extend_from_slice(&[0xFF, 0x52]);
        out.extend_from_slice(&((cod.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&cod);
        out.extend_from_slice(extra);
        let mut body = vec![0u8, 0x60]; // Ztlm 0, Stlm: Ttlm u16, Ptlm u32
        for (tile, len) in tlm {
            body.extend_from_slice(&tile.to_be_bytes());
            body.extend_from_slice(&len.to_be_bytes());
        }
        out.extend_from_slice(&[0xFF, 0x55]);
        out.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&body);
        out
    }

    /// Длина пакета по правилу PLT: семь бит на байт, старший — «дальше».
    fn varint(value: u32) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        let mut rest = value;
        loop {
            out.insert(0, (rest & 0x7F) as u8 | if out.is_empty() { 0 } else { 0x80 });
            rest >>= 7;
            if rest == 0 {
                return out;
            }
        }
    }

    /// Тайл-парт: SOT, PLT с данными длинами, SOD, данные.
    fn tile_part(tile: u16, ordinal: u8, count: u8, packets: &[u32]) -> Vec<u8> {
        tile_part_with(tile, ordinal, count, packets, &[])
    }

    /// Тот же тайл-парт с лишним сегментом `extra` (маркер, длина, тело)
    /// в заголовке перед PLT.
    fn tile_part_with(tile: u16, ordinal: u8, count: u8, packets: &[u32], extra: &[u8]) -> Vec<u8> {
        let mut plt = vec![0u8];
        for n in packets {
            plt.extend_from_slice(&varint(*n));
        }
        let data: Vec<u8> = packets.iter().flat_map(|n| (1..=*n).map(|i| i as u8).collect::<Vec<u8>>()).collect();
        let len = 12 + extra.len() + 4 + plt.len() + 2 + data.len();
        let mut out = vec![0xFF, 0x90, 0x00, 0x0A];
        out.extend_from_slice(&tile.to_be_bytes());
        out.extend_from_slice(&(len as u32).to_be_bytes());
        out.extend_from_slice(&[ordinal, count]);
        out.extend_from_slice(extra);
        out.extend_from_slice(&[0xFF, 0x58]);
        out.extend_from_slice(&((plt.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&plt);
        out.extend_from_slice(&[0xFF, 0x93]);
        out.extend_from_slice(&data);
        out
    }

    /// Файл из главного заголовка и тайл-партов; TLM в заголовке — по ним.
    fn file(canvas: u32, tile: u32, levels: u8, parts: &[(u16, Vec<u8>)]) -> (Vec<u8>, Index) {
        let tlm: Vec<(u16, u32)> = parts.iter().map(|(tile, bytes)| (*tile, bytes.len() as u32)).collect();
        let mut bytes = main_header(canvas, tile, levels, &tlm);
        for (_, part) in parts {
            bytes.extend_from_slice(part);
        }
        bytes.extend_from_slice(&[0xFF, 0xD9]);
        let index = Index::of(&bytes, bytes.len() as u64).expect("индекс строится");
        (bytes, index)
    }

    /// Индекс складывает адреса из длин TLM от первого SOT и требует, чтобы
    /// они сошлись с концом кодстрима; без TLM индекса нет.
    #[test]
    fn the_index_follows_tlm_and_checks_the_sum() {
        let parts = [(0u16, tile_part(0, 0, 1, &[3, 4, 5])), (1, tile_part(1, 0, 1, &[6, 7, 8]))];
        assert_eq!(varint(70_000), [0x84, 0xA2, 0x70], "три семёрки бит со старшими «дальше»");
        let (bytes, index) = file(128, 64, 2, &parts);
        let sot = bytes.windows(2).position(|pair| pair == [0xFF, 0x90]).unwrap() as u64;
        assert_eq!(index.parts(), 2);
        assert_eq!(index.parts_of(1), [Part { tile: 1, offset: sot + parts[0].1.len() as u64, len: parts[1].1.len() as u64 }]);
        assert!(index.parts_of(2).is_empty());

        let mut lying = bytes.clone();
        lying.extend_from_slice(&[0; 7]);
        let why = Index::of(&lying, lying.len() as u64).expect_err("сумма не сошлась");
        assert!(why.contains("расходится"), "{why}");
        let short = &bytes[..sot as usize - 3];
        assert!(Index::of(short, bytes.len() as u64).expect_err("SOT не в голове").contains("не влез"));
    }

    /// Выдержка режет тайл-парт после пакетов нужного разрешения, дописывает
    /// пустые пакеты за отрезанные, переписывает Psot и TNsot — и не трогает
    /// ничего, когда нужен весь тайл.
    #[test]
    fn an_excerpt_ends_after_the_level_and_pads_the_rest() {
        // Один компонент, две ступени: по пакету на разрешение, три пакета
        // в тайле; у тайла 1 они разложены по двум тайл-партам.
        let parts = [
            (0u16, tile_part(0, 0, 1, &[3, 4, 5])),
            (1, tile_part(1, 0, 2, &[6, 7])),
            (1, tile_part(1, 1, 2, &[8])),
        ];
        let (bytes, index) = file(128, 64, 2, &parts);
        let probes = |tile: u32| -> Vec<Vec<u8>> {
            index.parts_of(tile).iter().map(|part| bytes[part.offset as usize..(part.offset + part.len) as usize].to_vec()).collect()
        };
        let header = &bytes[..bytes.windows(2).position(|pair| pair == [0xFF, 0x90]).unwrap()];

        // Фактор 1: два грубых пакета — первый тайл-парт целиком, второго
        // нет, вместо его пакета пустой; TNsot стал единицей.
        let cut = assemble(&index, 1, 1, probes(1)).expect("собирается");
        let Segment::Bytes(part) = &cut[1] else { panic!("тайл-парт переписан") };
        let expect_len = parts[1].1.len() as u32 + 1;
        assert_eq!(&part[6..10], &expect_len.to_be_bytes(), "Psot — свои байты плюс пустой пакет");
        assert_eq!(part[11], 1, "TNsot — сколько тайл-партов осталось");
        assert_eq!(&part[12..], &parts[1].1[12..], "данные первого тайл-парта целы");
        assert_eq!(cut[2], Segment::Bytes(Rc::from(&[0u8][..])), "пустой пакет — нулевой байт");
        assert_eq!(cut[3], Segment::Bytes(Rc::from(&[0xFFu8, 0xD9][..])));
        assert_eq!(cut.len(), 4);
        assert_eq!(cut[0], Segment::Bytes(Rc::from(header)));

        // Фактор 2: один пакет из первого тайл-парта, два пустых.
        let cut = assemble(&index, 1, 2, probes(1)).expect("собирается");
        let Segment::Bytes(part) = &cut[1] else { panic!() };
        let sod = 12 + 4 + 3 + 2;
        assert_eq!(part.len(), sod + 6, "заголовок и первый пакет");
        assert_eq!(&part[6..10], &((sod + 6 + 2) as u32).to_be_bytes());
        assert_eq!(cut[2], Segment::Bytes(Rc::from(&[0u8, 0][..])));

        // Фактор 0: всё нужно — оба тайл-парта как есть, без дописок.
        let whole = assemble(&index, 1, 0, probes(1)).expect("собирается");
        assert_eq!(whole.len(), 4);
        for (segment, (_, part)) in whole[1..3].iter().zip(&parts[1..]) {
            assert_eq!(segment, &Segment::Bytes(Rc::from(part.as_slice())));
        }

        // Проба не с того места — отказ по имени, а не тихая выдержка.
        let mut wrong = probes(1);
        wrong[0][5] = 0;
        assert!(assemble(&index, 1, 1, wrong).expect_err("TLM соврал").contains("не там"));
    }

    /// Хвост нужного, не попавший в пробу, идёт куском файла — ровно до
    /// конца нужного, а не до конца тайл-парта; читатель отдаёт выдержку
    /// байт в байт как переписанный поток.
    #[test]
    fn what_the_probe_missed_comes_from_the_file() {
        let parts = [(0u16, tile_part(0, 0, 1, &[70_000, 10, 10]))];
        let (bytes, index) = file(64, 64, 2, &parts);
        let part = index.parts_of(0)[0];
        assert!(part.len > PROBE, "тайл-парт длиннее пробы");
        let sod = 12 + 4 + (1 + 3 + 1 + 1) + 2;
        let probe = bytes[part.offset as usize..(part.offset + PROBE) as usize].to_vec();

        let cut = assemble(&index, 0, 1, vec![probe]).expect("собирается");
        let keep = sod + 70_000 + 10;
        assert_eq!(cut.len(), 5);
        assert_eq!(cut[2], Segment::File { offset: part.offset + PROBE, len: keep - PROBE });
        assert_eq!(cut[3], Segment::Bytes(Rc::from(&[0u8][..])));

        let source = bytes.clone();
        let fetch: Fetch = Box::new(move |offset, size| Ok(source[offset as usize..(offset + size) as usize].to_vec()));
        let mut reader = Reader::over_fetch(cut, Rc::new(Cell::new(0)), fetch);
        let mut got = Vec::new();
        reader.read_to_end(&mut got).unwrap();
        let mut expected = bytes[..part.offset as usize].to_vec();
        let mut head = bytes[part.offset as usize..(part.offset + keep) as usize].to_vec();
        head[6..10].copy_from_slice(&((keep + 1) as u32).to_be_bytes());
        expected.extend_from_slice(&head);
        expected.extend_from_slice(&[0, 0xFF, 0xD9]);
        assert_eq!(got, expected);
    }

    /// Читатель отдаёт куски подряд, читает файл окнами не за конец куска и
    /// принимает seek в любую сторону.
    #[test]
    fn the_reader_splices_segments_and_never_reads_past_a_piece() {
        let disk: Vec<u8> = (0..=255u8).cycle().take(1_000_000).collect();
        let asked = Rc::new(std::cell::RefCell::new(Vec::new()));
        let log = asked.clone();
        let from = disk.clone();
        let fetch: Fetch = Box::new(move |offset, size| {
            log.borrow_mut().push((offset, size));
            Ok(from[offset as usize..(offset + size) as usize].to_vec())
        });
        let segments = vec![
            Segment::Bytes(Rc::from(&b"head"[..])),
            Segment::File { offset: 1000, len: 600_000 },
            Segment::Bytes(Rc::from(&[0u8, 0][..])),
            Segment::File { offset: 5, len: 3 },
        ];
        let bytes = Rc::new(Cell::new(0));
        let mut reader = Reader::over_fetch(segments, bytes.clone(), fetch);
        assert_eq!(reader.len(), 4 + 600_000 + 2 + 3);
        let mut got = Vec::new();
        reader.read_to_end(&mut got).unwrap();
        let mut expected = b"head".to_vec();
        expected.extend_from_slice(&disk[1000..601_000]);
        expected.extend_from_slice(&[0, 0]);
        expected.extend_from_slice(&disk[5..8]);
        assert_eq!(got, expected);
        assert_eq!(
            *asked.borrow(),
            vec![(1000, 262_144), (263_144, 262_144), (525_288, 75_712), (5, 3)],
            "окна не длиннее WINDOW и не дальше конца куска"
        );
        assert_eq!(bytes.get(), 600_003);

        reader.seek(SeekFrom::Start(2)).unwrap();
        let mut two = [0u8; 4];
        reader.read_exact(&mut two).unwrap();
        assert_eq!(&two, &[b'a', b'd', disk[1000], disk[1001]], "стык кусков читается сплошь");
        reader.seek(SeekFrom::End(-3)).unwrap();
        let mut tail = Vec::new();
        reader.read_to_end(&mut tail).unwrap();
        assert_eq!(tail, &disk[5..8]);
        reader.seek(SeekFrom::Start(10)).unwrap();
        assert!(reader.seek(SeekFrom::Current(-100)).is_err(), "до начала выдержки seek нет");
    }

    /// Заголовок тайла, переопределяющий кодирование (COD, COC, POC, PPT),
    /// отменяет обрезку: счёт пакетов ведётся по главному заголовку, и с
    /// чужими прецинктами он соврёт. Тайл-парт идёт целиком, без дописок.
    #[test]
    fn a_tile_header_that_overrides_coding_keeps_the_part_whole() {
        let cod = [0xFF, 0x52, 0x00, 0x0C, 0, 0, 0, 1, 0, 2, 4, 4, 0, 1];
        let parts = [(0u16, tile_part_with(0, 0, 1, &[3, 4, 5], &cod))];
        let (bytes, index) = file(64, 64, 2, &parts);
        let part = index.parts_of(0)[0];
        let probe = bytes[part.offset as usize..(part.offset + part.len) as usize].to_vec();
        let whole = assemble(&index, 0, 2, vec![probe]).expect("собирается");
        assert_eq!(whole.len(), 3, "заголовок, тайл-парт целиком, EOC");
        assert_eq!(whole[1], Segment::Bytes(Rc::from(parts[0].1.as_slice())));
    }

    /// Порядок пакетов, в котором разрешение не внешний цикл, префикса не
    /// даёт: POC и PPM в главном заголовке, SOP и EPH в стиле кодирования —
    /// каждый порознь запрещает обрезку, и берётся это из самого заголовка.
    #[test]
    fn changes_and_markers_forbid_the_prefix() {
        let part = tile_part(0, 0, 1, &[3, 4, 5]);
        let index_with = |scod: u8, extra: &[u8]| {
            let mut bytes = main_header_with(64, 64, 2, &[(0, part.len() as u32)], scod, extra);
            bytes.extend_from_slice(&part);
            bytes.extend_from_slice(&[0xFF, 0xD9]);
            Index::of(&bytes, bytes.len() as u64).expect("индекс строится")
        };
        assert!(index_with(0, &[]).coding().prefixed(), "LRCP с одним слоем — префикс");
        let poc = [0xFF, 0x5F, 0x00, 0x09, 0, 0, 0, 0, 0, 0, 0];
        assert!(!index_with(0, &poc).coding().prefixed(), "POC ломает порядок");
        assert!(!index_with(0, &[0xFF, 0x60, 0x00, 0x03, 0]).coding().prefixed(), "PPM выносит заголовки пакетов");
        assert!(!index_with(0x02, &[]).coding().prefixed(), "SOP меняет вид пустого пакета");
        assert!(!index_with(0x04, &[]).coding().prefixed(), "EPH тоже");
    }

    /// Индекс отказывает по имени, а не молчит: сегменты TLM не по порядку,
    /// запись про тайл, которого нет, обрывок записи.
    #[test]
    fn a_broken_tlm_is_refused_by_name() {
        let parts = [(0u16, tile_part(0, 0, 1, &[3, 4, 5]))];
        let (bytes, _) = file(64, 64, 2, &parts);
        let tlm_at = bytes.windows(2).position(|pair| pair == [0xFF, 0x55]).unwrap();

        // Второй TLM с Ztlm 0 после первого с Ztlm 0 — не по порядку.
        let mut twice = bytes.clone();
        twice.splice(tlm_at..tlm_at, [0xFF, 0x55, 0x00, 0x04, 0x00, 0x60]);
        assert!(Index::of(&twice, twice.len() as u64).expect_err("порядок").contains("не по порядку"));

        // Тайл 9 в кодстриме на один тайл.
        let mut foreign = bytes.clone();
        foreign[tlm_at + 6..tlm_at + 8].copy_from_slice(&9u16.to_be_bytes());
        assert!(Index::of(&foreign, foreign.len() as u64).expect_err("чужой тайл").contains("не про этот"));

        // Лишний байт в теле TLM — ни одна ширина записи его не делит.
        let mut torn = bytes.clone();
        torn.splice(tlm_at + 6..tlm_at + 6, [0u8]);
        torn[tlm_at + 3] += 1;
        assert!(Index::of(&torn, torn.len() as u64).expect_err("обрывок").contains("обрывком"));
    }

    /// PLT не по порядку Zplt — длин нет, и тайл-парт идёт целиком.
    #[test]
    fn plt_segments_out_of_order_give_no_lengths() {
        let mut part = tile_part(0, 0, 1, &[3, 4, 5]);
        // Второй PLT с Zplt 0 перед SOD — тем же номером, что и первый.
        let sod = part.windows(2).position(|pair| pair == [0xFF, 0x93]).unwrap();
        part.splice(sod..sod, [0xFF, 0x58, 0x00, 0x04, 0x00, 0x01]);
        let head = head_of(&part).expect("до SOD дошли");
        assert!(head.plt.is_none(), "склеивать не по порядку нельзя");
        let ordered = head_of(&tile_part(0, 0, 1, &[3, 4, 5])).unwrap();
        assert_eq!(ordered.plt, Some(vec![3, 4, 5]));
    }
}
