# Contributing to VeldMap

Thank you for contributing to VeldMap. Every change follows the architectural guardrails in [plan.md](plan.md) and the ADRs in [docs/decisions/](docs/decisions/README.md).

## Toolchain & Verification

The workspace uses Rust 2024 with MSRV `1.95` pinned in `rust-toolchain.toml`. Before opening a pull request, run:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

When `cargo-deny` and `lychee` are installed locally, also run:

```bash
cargo deny check
lychee --offline docs/ README.md CONTRIBUTING.md SECURITY.md THIRD_PARTY_NOTICES.md
```

## Code Hygiene Rules

1. **English only:** All code, comments, UI copy, docs, commit messages, and tests outside the temporary `old/` reference tree must be in English. CI rejects Cyrillic characters outside `old/`.
2. **No casual panics:** `unwrap_used`, `expect_used`, `unimplemented`, `todo`, `dbg_macro`, and `undocumented_unsafe_blocks` are enforced via workspace lints (tests are exempted from `unwrap`/`expect` via `clippy.toml`). When a lint exception is genuinely required, use `#[expect(lint_name, reason = "INVARIANT: ...")]`.
3. **Pure `veldmap-core`:** Never add `egui`, `eframe`, `wgpu`, `memmap2`, GDAL/PROJ C dependencies, or OpenSSL dependencies to `veldmap-core`. Respect the internal acyclic module DAG documented in [ADR 0001](docs/decisions/0001-native-workspace-and-threading.md).
4. **No links into `old/`:** Documentation under `docs/` must never link into `old/`.
5. **Single-decision ADRs:** Architectural changes land alongside an empirical ADR under `docs/decisions/`.
