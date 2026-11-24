# Contributing to VeldMap

Thank you for contributing to VeldMap. See [docs/decisions.md](docs/decisions.md) for settled architectural tradeoffs and [plan.md](plan.md) for the vertical-slice roadmap.

## Toolchain & Verification

The workspace uses Rust 2024 with MSRV `1.95` pinned in `rust-toolchain.toml`. Before opening a pull request, run the full build, lint, and test commands listed in [docs/building.md](docs/building.md) (plus `cargo deny check` and offline `lychee` checks when installed locally).

## Code Hygiene Rules

1. **English only:** All code, comments, UI copy, docs, commit messages, and tests outside the temporary `old/` reference tree must be in English. CI rejects Cyrillic characters outside `old/`.
2. **No casual panics:** `unwrap_used`, `expect_used`, `allow_attributes`, `allow_attributes_without_reason`, `unimplemented`, `todo`, `dbg_macro`, `large_futures`, and `undocumented_unsafe_blocks` are enforced via workspace lints (`clippy.toml` exempts `#[test]` functions and `#[cfg(test)]` modules from `unwrap`/`expect`, while integration-test harnesses in `tests/integration/main.rs` still require explicit `#![expect(..., reason = "INVARIANT: ...")]`).
3. **Pure `veldmap-core`:** Never add `egui`, `eframe`, `wgpu`, `memmap2`, or OpenSSL dependencies to `veldmap-core`. Keep default builds free of C system-library setup requirements unless explicitly justified in [docs/decisions.md](docs/decisions.md).
4. **No links into `old/`:** Documentation under `docs/` must never link into `old/`.
5. **Keep docs lean & truthful:** Document implemented module details in Rust `//!` docstrings (`cargo doc`), and record major "why not X" tradeoffs in [docs/decisions.md](docs/decisions.md) once proven in code.
