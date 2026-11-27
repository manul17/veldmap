//! Pure `#![forbid(unsafe_code)]` domain logic for VeldMap.
//!
//! Permanently free of UI, GPU, codec, SQLite, networking, and OS-probing
//! dependencies (`egui`, `eframe`, `wgpu`, `tokio`, `reqwest`, `image`,
//! `tiff`, `rusqlite`, `memmap2`).
//!
//! ## Modules
//! - [`budget`] — Pure host-RAM-scaled [`budget::MemoryBudget`] math.
//! - [`geo`] — WGS84 geodesy, Krueger & pure-Rust `proj4rs`/`proj4wkt` CRS
//!   projections, MGRS/DD/DMS parsing, Karney geodesics, GeoJSON/WKT &
//!   spherical tessellation, encapsulated raster [`geo::Frame`] bindings,
//!   nadir `f64` Reverse-Z [`geo::Camera`], and oblate culling.
//! - [`io`] — Synchronous caller-buffered [`io::ByteSource`] &
//!   [`io::BufferedByteReader`] random-access I/O traits.
//! - [`sync`] — Zero-dependency [`sync::CancelToken`], [`sync::Wakeup`], and
//!   unpoisoned mutex helpers.
//! - [`tile`] — Unified [`tile::TileKey`]/[`tile::StreamKey`], LRU cache,
//!   adaptive planetary quadtree + Polar Stereographic caps, and overlay
//!   culling/warp-mesh generation.

#![forbid(unsafe_code)]

pub use glam;

pub const APP_NAME: &str = "VeldMap";

pub mod budget;
pub mod geo;
pub mod io;
pub mod sync;
pub mod tile;
