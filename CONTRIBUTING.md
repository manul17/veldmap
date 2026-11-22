# Contributing to VeldMap

Thank you for contributing to VeldMap. Every change follows the architectural guardrails in [plan.md](plan.md) and the ADRs in [docs/decisions/](docs/decisions/README.md).

## Toolchain & Verification

The workspace uses Rust 2024 with MSRV `1.95` pinned in `rust-toolchain.toml`. Before opening a pull request, run the full build, lint, and test commands listed in [docs/building.md](docs/building.md) (plus `cargo deny check` and offline `lychee` checks when installed locally).

## Code Hygiene Rules

1. **English only:** All code, comments, UI copy, docs, commit messages, and tests outside the temporary `old/` reference tree must be in English. CI rejects Cyrillic characters outside `old/`.
2. **No casual panics:** `unwrap_used`, `expect_used`, `allow_attributes`, `allow_attributes_without_reason`, `unimplemented`, `todo`, `dbg_macro`, `large_futures`, and `undocumented_unsafe_blocks` are enforced via workspace lints (`clippy.toml` exempts `#[test]` functions and `#[cfg(test)]` modules from `unwrap`/`expect`, while integration-test helpers in `tests/*.rs` still require explicit `#[expect(..., reason = "INVARIANT: ...")]`).
3. **Pure `veldmap-core`:** Never add `egui`, `eframe`, `wgpu`, `memmap2`, GDAL/PROJ C dependencies, or OpenSSL dependencies to `veldmap-core`. Respect the internal acyclic module DAG documented in [ADR 0001](docs/decisions/0001-native-workspace-and-threading.md).
4. **No links into `old/`:** Documentation under `docs/` must never link into `old/`.
5. **Focused ADRs:** Architectural changes land alongside an ADR under `docs/decisions/`.
