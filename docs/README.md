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

1. **UI / Input (`app/*`, `ui/*`):** Action-only views (top toolbar, collapsible `Layers` / `Cloud / STAC` workbench, tabbed Settings modal) emit typed UI actions (`ToolbarAction`, `BasemapUiAction`, `SettingsAction`, `LayerUiAction`, `StacUiAction`, `CrsPickerAction`, `MeasureAction`) into `SceneState` / `VeldmapApp` with zero `veldmap-net`, `veldmap-raster`, or `veldmap-storage` imports in `ui/`.
2. **Scene Mediator (`scene.rs`):** Coordinates `Camera`, shared `Arc<NetContext>`, `BasemapManager`, `OverlayManager`, `VectorManager`, `StacSession`, `GpuTileCache`, and `PendingUploads`, computing proportional VRAM appetite caps, `ScratchDemand`-adjusted target VRAM budgets, and an immutable `DrawSnapshot` (`SnapshotKey`).
3. **Background Work:**
   - Local raster opens/reloads (`LocalFingerprintCache` → `LocalFileSource` → `SinglePassIfdReader` / `TiffSource` / `PngSource` / `JpegSource` + sidecars → `Frame`) run off the UI thread on the overlay I/O `SingleThreadWorker` (`spawn_overlay_io_worker`); GeoJSON vector loads/tessellations run independently on the vector I/O `SingleThreadWorker` (`spawn_vector_io_worker`).
   - Remote COG opens (`HttpRangeSource` 1-RTT head probe on `NetContext` → `SinglePassIfdReader` IFD parsing on dedicated `veldmap-remote-io`), remote `band_stats`, cache-only remote cursor probes, and `StacClient` searches stay strictly isolated from local SSD decodes.
   - Local tile decodes and remote pinned-prefetch CPU decodes run on `TwoLaneDecodePool` (`fast` for pointwise tiles/probes, `bulk` for sequential passes), backed by `SharedChunkCache` and `TileDiskCache` (SQLite WAL), with pointwise QOI/LZ4 encoding and batched SQLite commits offloaded to `veldmap-tile-write` *after* GPU upload delivery.
   - Completed `DecodedTile`s enter enqueue-bounded `PendingUploads` and upload into `GpuTileCache` on the next frame.
4. **Two-Pass Offscreen Renderer (`render/*` via `egui_wgpu::CallbackResources`):**
   ```text
   DrawSnapshot (immutable Arc) ──► GlobePaintCallback::prepare (app/viewport_bridge.rs)
     ├─ Pass 1 (Opaque Body): Rgba8UnormSrgb (Clear) + Depth32Float Reverse-Z (TRANSIENT, Discard)
     ├─ Pass 2 (Segmented, depth_stencil: None, LoadOp::Load, painter's order):
     │    ├─ BandMath layer ─► ScratchTargetPool (1..=4 R32Float NaN-fill + MRT/per-band + R8Uint SCL)
     │    │                 ─► Scissored fullscreen BandMath composite into color target
     │    ├─ Direct overlays ─► Batched Display / Scientific (R32Float) / Categorical (R8/R16Uint)
     │    └─ Tail            ─► Draped GeoJSON / Measure / STAC ribbon-hatch vectors + fwidth() graticule
     └─ GlobePaintCallback::paint ─► 1:1 FilterMode::Nearest sRGB composite into egui swapchain
   ```
5. **Deterministic Shutdown:** Persist final `session.json` snapshot → Phase 1 cancels overlay/vector/basemap/`StacSession`/`decode_shutdown` tokens, calls `NetContext::signal_shutdown()`, and closes overlay I/O, vector I/O, `veldmap-tile-write`, & `DiskCacheWorker` channels → Phase 2 closes `PendingUploads` → Phase 3 joins `TwoLaneDecodePool` (`fast` then `bulk`), joins `veldmap-remote-io`, overlay I/O, `veldmap-tile-write`, and vector I/O workers, shuts down `LiveTilePipeline` → shuts down `NetContext` (`tokio::Runtime`) → joins `DiskCacheWorker`.

### `veldmap-app` Module Map

| Module | Responsibility |
|---|---|
| `app/` | `eframe::App` shell (`VeldmapApp`), keyboard/mouse input, non-blocking `rfd` dialogs, `egui_wgpu::CallbackResources` bridge, `session.json` sync & restore tracker |
| `basemap/` | Bundled NASA Blue Marble & live HTTP basemap policy, tile-carrier selection, and decode/disk adapters |
| `colormap.rs` | Linear-light scientific colormaps (`Grayscale`, `Viridis`, `RdYlGn`, `BrBG`, etc.) & categorical palettes (`SCL`, source LUT, hash) |
| `gpu_tiles/` | Bounded `GpuTileCache` and enqueue-bounded `PendingUploads` coarsest-first uploader |
| `layers/` | `OverlayManager`, `LayerStack`, `RasterAsset`, multi-asset `BandMath`, background I/O/decode/tile-write workers, and CPU `CursorProber` |
| `measure.rs` | Interactive Karney WGS84 geodesic distance/area measurement session & GeoJSON export |
| `render/` | Offscreen `GlobeRenderer`, `OverlayRenderer`, `VectorRenderer`, `ScratchTargetPool`, `DynamicUniformArena`, `TileBindGroupCache`, and `DrawSnapshot` |
| `scene.rs` | `SceneState` mediator owning caches, decode pool, shared `NetContext`, managers, VRAM accounting, and deterministic shutdown |
| `stac/` | `StacSession` async catalog search, AOI state, origin-scoped auth resolution, and read-only UI DTOs |
| `storage/` | `settings.json` / `session.json` schemas and async `DiskCacheWorker` |
| `ui/` | Presentation-only `egui` views emitting typed actions (zero `veldmap-net`/`veldmap-raster`/`veldmap-storage` imports) |
| `vectors.rs` | Retained GeoJSON vector layers and off-UI-thread spherical tessellation |
| `worker.rs` | Lightweight single-thread background worker primitive |
