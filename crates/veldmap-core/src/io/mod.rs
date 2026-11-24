//! Pure-`std` synchronous caller-buffered positioned I/O (`ByteSource`,
//! `BufferedByteReader`, `IoError`) and opt-in test doubles (`testing`,
//! enabled under `#[cfg(test)]` or `feature = "test-support"`).

pub mod byte_source;
#[cfg(any(test, feature = "test-support"))]
pub mod testing;

pub use byte_source::{BufferedByteReader, ByteSource, DEFAULT_READER_BUFFER_BYTES, IoError};
