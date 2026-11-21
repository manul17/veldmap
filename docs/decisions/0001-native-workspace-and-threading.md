# 0001 — Native 2-Crate Workspace, Internal Module DAG, and Frame-Driven Threading

- **Status:** Accepted (Sections 1 & 3 active in Slice 1; Sections 2 & 4 define committed forward contracts exercised in Slices 2–4)
- **Date:** 2026-11-21
- **Scope:** Workspace layout, dependency exclusions, `veldmap-core` internal module DAG, frame-driven rendering loop, and background worker shutdown order.

## Context

VeldMap is a native desktop analytical Earth-observation workstation built in Rust with `eframe`/`egui` and `wgpu`. Its predecessor architecture routed components through a WASI/Wasmtime event bus and protobuf IPC, adding serialization overhead, `!Send` constraints, and multi-process complexity without desktop benefit. At the same time, keeping domain logic decoupled from GPU and UI crates remains essential so core geodesy, tiling, raster decoding, and streaming code can be tested headlessly and reused cleanly in future companions.

## Decision

### 1. Two-Crate Native Workspace (Active)

The repository contains two workspace crates:

- `crates/veldmap-core`: Pure domain logic with zero `egui`, `eframe`, or `wgpu` dependencies.
- `crates/veldmap-app`: Native desktop application built on `eframe`/`egui` and `wgpu`, structured with both `[lib]` (`veldmap_app`) and `[[bin]]` (`veldmap`) targets so offscreen rendering, HUD, and `naga` shader codegen tests execute headlessly without opening a window.

Re-evaluate extracting `veldmap-raster` into a third workspace crate at Slice 3C-1 if pure-Rust codec compile times degrade inner-loop iteration speed.

### 2. Strict Acyclic Module DAG in `veldmap-core` (Target Layout — Feature Matrix Enforced Starting Slice 2A)

`veldmap-core` organizes modules into an acyclic internal dependency graph enforced through isolated Cargo feature flags (verified in CI via isolated feature builds and intra-feature import checks):

1. **Leaf (`--no-default-features`; minimal dependencies: `glam`, `thiserror`, plus `blake3` when `ContentId` lands):**
   - `geo` *(implemented)*: WGS84 geodesy, projections/datums, MGRS, `DMat4`/`DVec3` camera and frustum culling, footprint/outline math, sagitta helpers.
   - `tile` *(Slice 2A)*: `TileKey` (`Copy + Eq + Hash`), `TileScheme`, `BandKey`, `PixelFormat`, pyramid math, generic GPU-free O(1) LRU `TileCache<K, V>`, pure `TileScheduler`.
   - `io` / `mem` *(Slices 2B, 3A-1)*: strictly synchronous caller-buffered `ByteSource` trait, `Signer` trait, zero-dependency async-waking `CancelToken`, lock-free `MemoryBudget` + `Shrinkable`, pure `StoragePaths`.
2. **Orthogonal Mid-Layer (each depends only on the leaf layer):**
   - `cache` (`fs` feature, *Slice 2B*): raw-byte tile disk store, two-tier atomic writes, incremental LRU eviction.
   - `raster` (`codecs` feature, *Slice 3A-1*): pure-Rust decoders, level tables, grid/cascade/resampling, radiometry, pure synchronous range-planners.
   - `net` (`net` feature, *Slices 2B, 4A*; `tokio` + `reqwest` over `rustls`): tile fetcher, `HttpRange` (`ByteSource` + pinned prefetch + `Shrinkable` 64 KiB block pool), retry/ETag/range helpers.
   - `provider` (`net` feature, *Slices 4A, 4B*): STAC/COG discovery, CDSE OData/S3/SigV4 (`Signer`), manifests.
3. **Upper Layer (`fs`, optional `net`):**
   - `library` *(Slice 4B)*: local catalog and storage layout (`fs`), plus resumable downloads (`fs` + `net`) parameterized over `Signer`.

Intra-feature boundaries enforced by CI include `cache ↛ library` and `net ↛ provider`. When Slice 4A wires `veldmap-core::geo::outlines` into `wgpu`, either keep GPU vertex packing (`bytemuck`) inside `veldmap-app` or explicitly admit `bytemuck` as a leaf dependency rather than using `unsafe` casts.

### 3. Explicit Dependency Exclusions (Active)

The following technologies are explicitly excluded from `veldmap-core` and banned in [deny.toml](../../deny.toml):

- WASI, Wasmtime, ConnectRPC, protobuf/gRPC, browser WASM shims, and Emscripten.
- SQLite and ORMs (`rusqlite`, `sqlx`).
- Memory-mapped file I/O (`memmap2`, permitted only via Wayland SHM wrappers in `winit`/`smithay-client-toolkit`); safe positioned I/O is used instead.
- C geospatial and TLS bindings (`gdal`, `proj`, `openssl`/`openssl-sys`); pure-Rust codecs and `rustls` are used instead.
- UI/GPU crates (`egui`, `eframe`, `wgpu`) inside `veldmap-core`.

### 4. Frame-Driven Streaming Engine & Shutdown Order (Committed Contract — Slices 2–3A)

VeldMap uses a direct, frame-driven loop with no internal pub/sub event bus:

- **Main thread (`eframe` + `wgpu`):** Owns `AppState`, camera, timeline, `GpuTileCache` (directly on `VeldmapApp`, outside `CallbackResources`), and `PendingUploads`. It never blocks on disk I/O, network I/O, or decoding. Immutable `Arc<GpuTile>` draw snapshots are passed into `CallbackTrait` renderers.
- **Background workers:** A multi-threaded `tokio` runtime handles HTTP/STAC/CDSE requests, downloads, and Stage 1 async range preflight (producing memory-only pinned batches). CPU-bound decoding is isolated in separate custom bounded `std::thread` pools (**local CPU** vs **blocking remote**, never sharing `rayon`'s global pool) so slow network reads never starve local SSD decodes and pools join deterministically on shutdown.
- **Deterministic shutdown order:**
  1. Cancel the root `CancelToken`.
  2. Drop the completion receiver (or drain an unbounded completion channel) and join the decode worker pools so no worker blocks trying to send completions.
  3. Drop the `tokio::Runtime`.

## Consequences

- `veldmap-core` compiles and tests quickly without graphics drivers or OS windowing libraries.
- `veldmap-app` uses `eframe`'s re-exported `egui` and `egui_wgpu` modules, avoiding duplicate version glue and honoring `WGPU_BACKEND` (excluding GL/GLES/WebGPU) and `WGPU_POWER_PREF`.
- Every future slice extends the module DAG along these exact boundaries without introducing circular dependencies or blocking calls on the UI thread.
