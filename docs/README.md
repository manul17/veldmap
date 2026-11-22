# VeldMap Documentation

Authoritative architecture notes, build instructions, glossary, current limitations, and Architectural Decision Records (ADRs).

## Index

- [Building & Running](building.md) — Rust toolchain, platform dependencies, runtime environment variables, and verification commands.
- [3D WGS84 Globe Architecture](architecture/globe.md) — Coordinates, projections, MGRS, nadir camera, double-single RTE Reverse-Z rendering, and GIS HUD.
- [Architectural Decision Records](decisions/README.md) — Focused ADRs recording design contracts alongside each vertical slice.
- [Glossary](glossary.md) — Canonical English domain vocabulary shared across code, UI, and documentation.
- [Known Limitations](limitations.md) — Explicit scope boundaries and current implementation limits.

## Document Lifecycle

- [plan.md](../plan.md) tracks the executable vertical-slice roadmap and exit criteria.
- Once a slice lands and its ADR is accepted under `docs/decisions/`, that ADR is authoritative for the contracts it defines.
- Documentation under `docs/` never links into the temporary `old/` reference directory.
