use super::*;
use crate::io::testing::CountingByteSource;

#[test]
fn buffered_reader_caches_window_and_counting_source_tracks_spans() {
    use std::io::{Read, Seek, SeekFrom};

    struct ChunkySource(Vec<u8>);
    impl ByteSource for ChunkySource {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }
        fn read_at(
            &self,
            offset: u64,
            buf: &mut [u8],
            cancel: &CancelToken,
        ) -> Result<usize, IoError> {
            if cancel.is_cancelled() {
                return Err(IoError::Cancelled);
            }
            if offset >= self.0.len() as u64 || buf.is_empty() {
                return Ok(0);
            }
            // Simulate short reads (at most 5 bytes per call).
            let start = offset as usize;
            let n = (self.0.len() - start).min(buf.len()).min(5);
            buf[..n].copy_from_slice(&self.0[start..start + n]);
            Ok(n)
        }
    }

    let payload: Vec<u8> = (0..128u8).collect();
    let base: Arc<dyn ByteSource> = Arc::new(ChunkySource(payload.clone()));
    let counting = Arc::new(CountingByteSource::new(base));
    let cancel = CancelToken::new();

    // `read_exact_at` loops across short reads cleanly.
    let mut exact = [0u8; 17];
    counting
        .read_exact_at(10, &mut exact, &cancel)
        .expect("read_exact_at succeeds across short reads");
    assert_eq!(&exact, &payload[10..27]);
    assert!(counting.read_calls() >= 4);

    // `BufferedByteReader` satisfies small sequential & backward reads from its window.
    counting.reset();
    let mut reader = BufferedByteReader::with_capacity(
        Arc::clone(&counting) as Arc<dyn ByteSource>,
        cancel.clone(),
        64,
    );
    let mut two = [0u8; 2];
    reader.read_exact(&mut two).expect("read first two");
    assert_eq!(two, [0, 1]);
    let calls_after_first = counting.read_calls();
    reader
        .seek(SeekFrom::Start(0))
        .expect("seek back inside window");
    let mut four = [0u8; 4];
    reader.read_exact(&mut four).expect("read inside window");
    assert_eq!(four, [0, 1, 2, 3]);
    assert_eq!(counting.read_calls(), calls_after_first);

    // Cancelled reads fail promptly (including `std::io::Read::read_exact`,
    // which must NOT spin on `ErrorKind::Interrupted`), and `set_cancel`
    // rebinds cleanly for pooled reuse.
    let reader_cancel = CancelToken::new();
    let mut cancel_reader = BufferedByteReader::with_capacity(
        Arc::new(ChunkySource(payload.clone())),
        reader_cancel.clone(),
        16,
    );
    let mut head = [0u8; 4];
    cancel_reader.read_exact(&mut head).expect("read head");
    assert_eq!(head, [0, 1, 2, 3]);
    reader_cancel.cancel();
    let mut after_cancel = [0u8; 8];
    let err = cancel_reader
        .read_exact(&mut after_cancel)
        .expect_err("cancelled read_exact must abort without spinning");
    assert_ne!(err.kind(), std::io::ErrorKind::Interrupted);
    assert!(
        err.get_ref()
            .and_then(|e| e.downcast_ref::<IoError>())
            .is_some_and(IoError::is_cancelled)
    );
    cancel_reader.set_cancel(CancelToken::new());
    cancel_reader
        .read_exact(&mut after_cancel)
        .expect("reader recovers after set_cancel");
    assert_eq!(after_cancel, payload[4..12]);

    // Out-of-window backward seek, SeekFrom::Current/End, and large-read bypass.
    let mut nav_reader = BufferedByteReader::with_capacity(
        Arc::clone(&counting) as Arc<dyn ByteSource>,
        CancelToken::new(),
        16,
    );
    let mut large = vec![0u8; 32];
    nav_reader
        .read_exact(&mut large)
        .expect("large bypass read");
    assert_eq!(large, &payload[0..32]);
    assert_eq!(
        nav_reader
            .seek(SeekFrom::Current(-16))
            .expect("seek current"),
        16
    );
    let mut mid = [0u8; 4];
    nav_reader
        .read_exact(&mut mid)
        .expect("read after seek back");
    assert_eq!(mid, [16, 17, 18, 19]);
    assert_eq!(nav_reader.seek(SeekFrom::End(-4)).expect("seek end"), 124);
    nav_reader.read_exact(&mut mid).expect("read tail");
    assert_eq!(mid, [124, 125, 126, 127]);

    // Faulty ByteSource returning `n > buf.len()` fails cleanly instead of panicking.
    struct OverlongSource;
    impl ByteSource for OverlongSource {
        fn len(&self) -> u64 {
            64
        }
        fn read_at(&self, _: u64, buf: &mut [u8], _: &CancelToken) -> Result<usize, IoError> {
            Ok(buf.len() + 1)
        }
    }
    let overlong_tok = CancelToken::new();
    let mut small = [0u8; 4];
    assert!(matches!(
        OverlongSource.read_exact_at(0, &mut small, &overlong_tok),
        Err(IoError::InvalidInput(_))
    ));
    let mut bad_reader =
        BufferedByteReader::with_capacity(Arc::new(OverlongSource), overlong_tok, 8);
    assert!(bad_reader.read_exact(&mut small).is_err());

    cancel.cancel();
    assert!(
        counting
            .read_at(0, &mut two, &cancel)
            .is_err_and(|e| e.is_cancelled())
    );
}
