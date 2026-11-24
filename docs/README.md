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
