# Building & Running VeldMap

## Prerequisites

- **Rust toolchain:** Rust `1.95` or newer (Rust 2024 edition). The repository pins the exact toolchain and components (`rustfmt`, `clippy`) in [rust-toolchain.toml](../rust-toolchain.toml).
- **C compiler (`cc`):** Required by self-contained bundled native crates (`libsqlite3-sys` for the SQLite tile cache, `ring` for TLS, and `zstd-sys` via `tiff`'s `zstd` feature—no external C system libraries or CMake required): `build-essential` on Debian/Ubuntu, Xcode Command Line Tools on macOS, or MSVC Build Tools on Windows.
- **Graphics driver:** A Vulkan, DirectX 12, or Metal driver (`wgpu::Backends::PRIMARY` minus `BROWSER_WEBGPU`). OpenGL/GLES and WebGPU are intentionally unsupported.
- **Linux system libraries:** X11/Wayland development headers are needed to compile `eframe`/`winit`:

```bash
sudo apt-get install -y \
  build-essential libxkbcommon-dev libwayland-dev libx11-dev \
  libxcursor-dev libxi-dev libxrandr-dev pkg-config
```

For headless GPU tests on Linux CI or machines without a hardware GPU, install `mesa-vulkan-drivers` (Lavapipe software Vulkan). Native OS file dialogs on Linux use `xdg-desktop-portal` over D-Bus (with `zenity` fallback); minimal window managers without a portal backend can still open files via drag-and-drop, the Layers panel `Path…` prompt, or CLI path arguments.

## Build, Run & Verify

```bash
# Run the desktop application
cargo run -p veldmap-app

# Run formatting, lints, rustdoc, and full test suite (matching CI)
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps --document-private-items

# Verify headless non-UI crates in isolation
cargo test --locked -p veldmap-core --no-default-features
cargo test --locked -p veldmap-raster
cargo test --locked -p veldmap-storage
cargo test --locked -p veldmap-net

# Optional local hygiene checks (if installed)
cargo deny check
lychee --offline --no-progress docs/ assets/README.md README.md plan.md CONTRIBUTING.md SECURITY.md THIRD_PARTY_NOTICES.md
```

## Runtime Environment Variables

| Variable | Effect |
|---|---|
| `RUST_LOG` | Configures `tracing-subscriber` log filtering (default: `info,veldmap=debug,veldmap_app=debug,veldmap_core=debug,veldmap_raster=debug,veldmap_net=debug,veldmap_storage=debug,wgpu_core=warn,wgpu_hal=warn`). |
| `WGPU_BACKEND` | Selects `wgpu` backend(s): `primary`, `vulkan` (`vk`), `dx12` (`d3d12`), or `metal` (`mtl`). OpenGL/GLES/WebGPU values are ignored with a warning. |
| `WGPU_POWER_PREF` | Selects adapter power preference: `high` (default), `low`, or `none`. |
| `VELDMAP_REQUIRE_GPU_TEST` | Set to `1` in headless test runs to fail if no `Backends::PRIMARY` adapter is found. |
| `COPERNICUS_ACCESS_KEY` / `COPERNICUS_ACCESS_SECRET` | *Planned (Slice 4B):* optional CDSE S3 credentials (template in [.env.example](../.env.example); not yet read). |

## Dev Build Ergonomics

The workspace enables `opt-level = 2` for external dependencies in dev builds (`[profile.dev.package."*"]` in [Cargo.toml](../Cargo.toml)), keeping interactive frame rates and raster/codec performance responsive while preserving fast incremental rebuilds of VeldMap crates.
