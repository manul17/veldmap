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
       QOI/Lz4Raw, Pool)    atomic JSON, I/O)     rustls + ring)
                      \              |               /
                       v             v              v
                     veldmap-app (eframe/egui + wgpu)
```

### Runtime Tile & Frame Dataflow (`veldmap-app`)

1. **UI / Input (`app/*`, `ui/*`):** Action-only views (top toolbar, collapsible `Layers` / `Cloud / STAC` workbench, tabbed Settings modal) emit typed UI actions (`ToolbarAction`, `SettingsAction`, `LayerUiAction`, `StacUiAction`, `CrsPickerAction`, `MeasureAction`) into `SceneState` / `VeldmapApp` with zero `veldmap-net` or `veldmap-raster` imports in `ui/`.
2. **Scene Mediator (`scene.rs`):** Coordinates `Camera`, shared `Arc<NetContext>`, `BasemapManager`, `OverlayManager`, `VectorManager`, `StacSession`, `GpuTileCache`, and `PendingUploads`, computing proportional VRAM appetite caps and an immutable `DrawSnapshot` (`SnapshotKey`).
3. **Background Work:**
   - Local raster opens/reloads (`LocalFingerprintCache` → `LocalFileSource` → `SinglePassIfdReader` / `TiffSource` / `PngSource` / `JpegSource` + sidecars → `Frame`) run off the UI thread on the overlay I/O `SingleThreadWorker` (`spawn_overlay_io_worker`); GeoJSON vector loads/tessellations run independently on the vector I/O `SingleThreadWorker` (`spawn_vector_io_worker`).
   - Remote COG opens (`HttpRangeSource` 1-RTT head probe on `NetContext` → `SinglePassIfdReader` IFD parsing on dedicated `veldmap-remote-io`), remote `band_stats`, cache-only remote cursor probes, and `StacClient` searches stay strictly isolated from local SSD decodes.
   - Local tile decodes and remote pinned-prefetch CPU decodes run on `TwoLaneDecodePool` (`fast` for pointwise tiles/probes, `bulk` for sequential passes), backed by `SharedChunkCache` and `TileDiskCache` (SQLite WAL), with pointwise QOI/LZ4 encoding and batched SQLite commits offloaded to `veldmap-tile-write` *after* GPU upload delivery.
   - Completed `DecodedTile`s enter enqueue-bounded `PendingUploads` and upload into `GpuTileCache` on the next frame.
4. **Two-Pass Offscreen Renderer (`render/*`):** Pass 1 renders the opaque WGS84 quadtree body with `Depth32Float` Reverse-Z (`TRANSIENT_ATTACHMENT`, `StoreOp::Discard`); Pass 2 segments around multi-asset `BandMath` layers in painter's order (lazy shared `R32Float` + `R8Uint` `SCL` scratch pool with co-grid MRT & scissored compositing in `render/scratch.rs`), alongside draped direct raster warp meshes (`render/overlay.rs`), draped GeoJSON / measurement / STAC AOI & HiDPI ribbon/hatch footprint vectors (`render/vector.rs`), and the procedural graticule depth-less (`LoadOp::Load`) before 1:1 sRGB compositing into `egui`.
5. **Deterministic Shutdown:** Persist final `session.json` snapshot → Phase 1 cancels overlay/vector/basemap/`StacSession`/`decode_shutdown` tokens, calls `NetContext::signal_shutdown()`, and closes overlay I/O, vector I/O, `veldmap-tile-write`, & `DiskCacheWorker` channels → Phase 2 closes `PendingUploads` → Phase 3 joins `TwoLaneDecodePool` (`fast` then `bulk`), joins `veldmap-remote-io`, overlay I/O, `veldmap-tile-write`, and vector I/O workers, shuts down `LiveTilePipeline` → shuts down `NetContext` (`tokio::Runtime`) → joins `DiskCacheWorker`.
