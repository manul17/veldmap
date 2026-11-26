# Contributing to VeldMap

Thank you for contributing to VeldMap. See [docs/decisions.md](docs/decisions.md) for settled architectural tradeoffs and [plan.md](plan.md) for the vertical-slice roadmap.

## Toolchain & Verification

The workspace uses Rust 2024 with MSRV `1.95` pinned in `rust-toolchain.toml`. Before opening a pull request, run the full build, lint, and test commands listed in [docs/building.md](docs/building.md) (plus `cargo deny check` and offline `lychee` checks when installed locally).

## Code Hygiene Rules

1. **English only:** All code, comments, UI copy, docs, commit messages, and tests must be in English. CI rejects Cyrillic characters across the entire repository.
2. **No casual panics:** `unwrap_used`, `expect_used`, `allow_attributes`, `allow_attributes_without_reason`, `unimplemented`, `todo`, `dbg_macro`, `large_futures`, and `undocumented_unsafe_blocks` are enforced via workspace lints (`clippy.toml` exempts `#[test]` functions and `#[cfg(test)]` modules from `unwrap`/`expect`; non-`#[test]` helper functions inside `tests/integration/` harnesses still require explicit `#![expect(..., reason = "INVARIANT: ...")]`).
3. **Pure `veldmap-core`:** Never add `egui`, `eframe`, `wgpu`, `tokio`, `reqwest`, `image`, `tiff`, `rusqlite`, `memmap2`, OpenSSL, or OS-probing dependencies to `veldmap-core`. Keep default builds free of C system-library setup requirements unless explicitly justified in [docs/decisions.md](docs/decisions.md).
4. **Test placement & `test-support` seams:** Keep short single-module unit tests (`<= ~100 LOC`) inline in `#[cfg(test)] mod tests`, place larger single-module unit suites in sibling `#[cfg(test)] #[path = "*_tests.rs"]` files, and group cross-module, synthetic-file, mock-HTTP, and GPU/shader suites under `tests/integration/main.rs`. Gate test-only crate exports behind the `test-support` feature (enabled only via `[dev-dependencies]`), verify production builds without `test-support` using `cargo clippy --locked --workspace --lib --bins -- -D warnings`, and set `VELDMAP_REQUIRE_GPU_TEST=1` locally when validating headless GPU smoke tests.
5. **Keep docs lean & truthful:** Document implemented module details in Rust `//!` docstrings (`cargo doc`), and record major "why not X" tradeoffs in [docs/decisions.md](docs/decisions.md) once proven in code.
