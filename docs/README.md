# VeldMap Documentation

Build instructions, design decisions, domain glossary, and current limitations. Module-level implementation details live in Rust `//!` docstrings (`cargo doc`).

## Index

- [Building & Running](building.md) — Rust toolchain, platform dependencies, runtime environment variables, and verification commands.
- [Design Decisions](decisions.md) — Implemented architectural tradeoffs ("why not X") and cross-slice accuracy/safety invariants.
- [Glossary](glossary.md) — Canonical English domain vocabulary shared across code, UI, and documentation.
- [Known Limitations](limitations.md) — Explicit v1 scope boundaries and current implementation limits.

## Document Lifecycle

- [plan.md](../plan.md) tracks the vertical-slice roadmap and working hypotheses for upcoming slices.
- [decisions.md](decisions.md) records decisions only after they are proven in code (or for hard cross-slice invariants).

## Workspace Architecture Overview

VeldMap is a strict 3-tier acyclic workspace with zero horizontal dependencies among middle crates:

```text
                       veldmap-core (pure leaf; #![forbid(unsafe_code)])
                      /              |               \
                     v               v                v
          veldmap-raster      veldmap-storage      veldmap-net
      (GeoTIFF/PNG/JPEG,    (SQLite WAL cache,   (tokio, reqwest,
      QOI, TwoLanePool)     atomic JSON, I/O)     rustls + ring)
                      \              |               /
                       v             v              v
                     veldmap-app (eframe/egui + wgpu)
```

### Runtime Tile & Frame Dataflow (`veldmap-app`)

1. **UI / Input (`app.rs`, `ui/*`):** Action-only views emit typed UI actions (`ToolbarAction`, `BasemapUiAction`, `LayerUiAction`, `CrsPickerAction`, `MeasureAction`) into `SceneState` / `VeldmapApp`.
2. **Scene Mediator (`scene.rs`):** Coordinates `Camera`, `BasemapManager`, `OverlayManager`, `VectorLayerStack`, `GpuTileCache`, and `PendingUploads`, computing proportional VRAM appetite caps and an immutable `DrawSnapshot` (`SnapshotKey`).
3. **Background Work:**
   - Local raster opens/reloads (`LocalFingerprintCache` → `LocalFileSource` → `TiffSource`/`PngSource`/`JpegSource` + sidecars → `Frame`) and GeoJSON vector loads/tessellations run off the UI thread on `OverlayIoWorker`.
   - Tile decodes and cursor probes run on `TwoLaneDecodePool` (`fast` for pointwise tiles/probes, `bulk` for sequential passes), backed by `SharedChunkCache` and `TileDiskCache` (SQLite WAL).
   - Completed `DecodedTile`s enter enqueue-bounded `PendingUploads` and upload into `GpuTileCache` on the next frame.
4. **Two-Pass Offscreen Renderer (`render/*`):** Pass 1 renders the opaque WGS84 quadtree body with `Depth32Float` Reverse-Z (`TRANSIENT_ATTACHMENT`, `StoreOp::Discard`); Pass 2 renders draped raster warp meshes (`render/overlay.rs`), draped GeoJSON & measurement vectors (`render/vector.rs`), and the procedural graticule depth-less (`LoadOp::Load`) before 1:1 sRGB compositing into `egui`.
5. **Deterministic Shutdown:** Cancel `CancelToken` / close `OverlayIoWorker` & `DiskCacheWorker` channels → close `PendingUploads` → join `TwoLaneDecodePool` (`fast` then `bulk`) → join `OverlayIoWorker` → shut down `tokio::Runtime` → join `DiskCacheWorker`.
