//! Pure-`std` synchronous caller-buffered positioned I/O (`ByteSource`,
//! `BufferedByteReader`, `IoError`) and explicit test doubles (`testing`).

pub mod byte_source;
pub mod testing;

pub use byte_source::{BufferedByteReader, ByteSource, DEFAULT_READER_BUFFER_BYTES, IoError};
