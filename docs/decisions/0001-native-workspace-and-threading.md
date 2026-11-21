# 0001 — Native 2-Crate Workspace, Internal Module DAG, and Frame-Driven Threading

- **Status:** Accepted
- **Date:** 2026-11-21
- **Scope:** Workspace layout, dependency exclusions, `veldmap-core` internal module DAG, frame-driven rendering loop, and background worker shutdown order.

## Context

VeldMap is a native desktop analytical Earth-observation workstation built in Rust with `eframe`/`egui` and `wgpu`. Its predecessor architecture routed components through a WASI/Wasmtime event bus and protobuf IPC, adding serialization overhead, `!Send` constraints, and multi-process complexity without desktop benefit. At the same time, keeping domain logic decoupled from GPU and UI crates remains essential so core geodesy, tiling, raster decoding, and streaming code can be tested headlessly and reused cleanly in future companions.

## Decision

### 1. Two-Crate Native Workspace

The repository contains two workspace crates:

- `crates/veldmap-core`: Pure, UI-free and GPU-free domain logic.
- `crates/veldmap-app`: Native desktop application built on `eframe`/`egui` and `wgpu` (offscreen 3D globe renderer, 2D inspector, GPU tile cache, background workers, and UI).

Re-evaluate extracting `veldmap-raster` into a third workspace crate at Slice 3C-1 if pure-Rust codec compile times degrade inner-loop iteration speed.

### 2. Strict Acyclic Module DAG in `veldmap-core`

`veldmap-core` enforces an acyclic internal dependency graph through isolated Cargo feature flags (verified in CI via isolated feature builds and intra-feature import checks):

1. **Leaf (`--no-default-features`; minimal dependencies: `glam`, `thiserror`, `blake3`):**
   - `geo`: WGS84 geodesy, projections/datums, MGRS, `DMat4`/`DVec3` camera and frustum culling, warp/georeference math, geodesic measurements.
   - `tile`: `TileKey` (`Copy + Eq + Hash`), `TileScheme`, `BandKey`, `PixelFormat`, pyramid math, generic GPU-free O(1) LRU `TileCache<K, V>`, pure `TileScheduler`.
   - `io` / `mem`: strictly synchronous caller-buffered `ByteSource` trait, `Signer` trait, zero-dependency async-waking `CancelToken`, lock-free `MemoryBudget` + `Shrinkable`, pure `StoragePaths`.
2. **Orthogonal Mid-Layer (each depends only on the leaf layer):**
   - `cache` (`fs` feature): raw-byte tile disk store, two-tier atomic writes, incremental LRU eviction.
   - `raster` (`codecs` feature): pure-Rust decoders, level tables, grid/cascade/resampling, radiometry, pure synchronous range-planners.
   - `net` (`net` feature; `tokio` + `reqwest` over `rustls`): `HttpRange` (`ByteSource` + pinned prefetch + `Shrinkable` 64 KiB block pool), retry/ETag/range helpers.
   - `provider` (`net` feature): STAC/COG discovery, CDSE OData/S3/SigV4 (`Signer`), manifests.
3. **Upper Layer (`fs`, optional `net`):**
   - `library`: local catalog and storage layout (`fs`), plus resumable downloads (`fs` + `net`) parameterized over `Signer`.

Intra-feature boundaries enforced by CI include `cache ↛ library` and `net ↛ provider`.

### 3. Explicit Dependency Exclusions

The following technologies are explicitly excluded from `veldmap-core` and banned in [deny.toml](../../deny.toml):

- WASI, Wasmtime, ConnectRPC, protobuf/gRPC, browser WASM shims, and Emscripten.
- SQLite and ORMs (`rusqlite`, `sqlx`).
- Memory-mapped file I/O (`memmap2`); safe positioned I/O is used instead.
- C geospatial and TLS bindings (`gdal`, `proj`, `openssl`/`openssl-sys`); pure-Rust codecs and `rustls` are used instead.
- UI/GPU crates (`egui`, `eframe`, `wgpu`) inside `veldmap-core`.

### 4. Frame-Driven Streaming Engine & Shutdown Order

VeldMap uses a direct, frame-driven loop with no internal pub/sub event bus:

- **Main thread (`eframe` + `wgpu`):** Owns `AppState`, camera, timeline, `GpuTileCache` (directly on `VeldmapApp`, outside `CallbackResources`), and `PendingUploads`. It never blocks on disk I/O, network I/O, or decoding. Immutable `Arc<GpuTile>` draw snapshots are passed into `CallbackTrait` renderers.
- **Background workers:** A multi-threaded `tokio` runtime handles HTTP/STAC/CDSE requests, downloads, and Stage 1 async range preflight (producing memory-only pinned batches). CPU-bound decoding is isolated in separate bounded **local CPU** and **blocking remote** worker pools so slow network reads never starve local SSD decodes.
- **Deterministic shutdown order:**
  1. Cancel the root `CancelToken`.
  2. Drop the completion receiver (or drain an unbounded completion channel) and join the decode worker pools so no worker blocks trying to send completions.
  3. Drop the `tokio::Runtime`.

## Consequences

- `veldmap-core` compiles and tests quickly without graphics drivers or OS windowing libraries.
- `veldmap-app` uses `eframe`'s re-exported `egui` and `egui_wgpu` modules, avoiding duplicate version glue and honoring `WGPU_BACKEND` (excluding GL/GLES/WebGPU) and `WGPU_POWER_PREF`.
- Every future slice extends the module DAG along these exact boundaries without introducing circular dependencies or blocking calls on the UI thread.
