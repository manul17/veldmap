# Building & Running VeldMap

## Prerequisites

- **Rust toolchain:** Rust `1.95` or newer (Rust 2024 edition). The repository pins the exact toolchain and components (`rustfmt`, `clippy`) in [rust-toolchain.toml](../rust-toolchain.toml).
- **Graphics driver:** A Vulkan, DirectX 12, or Metal driver (`wgpu::Backends::PRIMARY` minus `BROWSER_WEBGPU`). OpenGL/GLES and WebGPU are intentionally unsupported.
- **Linux system libraries:** X11/Wayland development headers are needed to compile `eframe`/`winit`:

```bash
sudo apt-get install -y \
  libxkbcommon-dev libwayland-dev libx11-dev \
  libxcursor-dev libxi-dev libxrandr-dev pkg-config
```

For headless GPU tests on Linux CI or machines without a hardware GPU, install `mesa-vulkan-drivers` (Lavapipe software Vulkan).

## Build, Run & Verify

```bash
# Run the desktop application
cargo run -p veldmap-app

# Run formatting, lints, and tests
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# Check pure leaf domain build
cargo check -p veldmap-core --no-default-features
```

## Runtime Environment Variables

| Variable | Effect |
|---|---|
| `RUST_LOG` | Configures `tracing-subscriber` log filtering (default: `info,veldmap=debug,veldmap_app=debug,veldmap_core=debug,wgpu_core=warn,wgpu_hal=warn`). |
| `WGPU_BACKEND` | Selects `wgpu` backend(s): `primary`, `vulkan` (`vk`), `dx12` (`d3d12`), or `metal` (`mtl`). OpenGL/GLES/WebGPU values are ignored with a warning. |
| `WGPU_POWER_PREF` | Selects adapter power preference: `high` (default), `low`, or `none`. |
| `VELDMAP_REQUIRE_GPU_TEST` | Set to `1` in headless test runs to fail if no `Backends::PRIMARY` adapter is found. |
| `COPERNICUS_ACCESS_KEY` / `COPERNICUS_ACCESS_SECRET` | *Planned (Slice 4B):* optional CDSE S3 credentials (template in [.env.example](../.env.example); not yet read in Slice 1). |

## Dev Build Ergonomics

The workspace enables `opt-level = 2` for external dependencies in dev builds (`[profile.dev.package."*"]` in [Cargo.toml](../Cargo.toml)), keeping interactive frame rates and raster/codec performance responsive while preserving fast incremental rebuilds of VeldMap crates.
