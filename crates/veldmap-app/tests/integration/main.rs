//! Grouped integration tests for `veldmap-app`: CPU `naga` shader codegen
//! (SPIR-V/HLSL/MSL) and headless offscreen `wgpu` smoke & accuracy tests.
//!
//! Set `VELDMAP_REQUIRE_GPU_TEST=1` (enabled on Linux Lavapipe and Windows WARP
//! CI) to fail loudly if no `Backends::PRIMARY` adapter is available.

#![expect(
    clippy::expect_used,
    reason = "INVARIANT: integration test helpers panic on headless GPU harness failure"
)]

mod common;
mod compositing;
mod overlays;
mod precision_and_seams;
mod shader_codegen;
mod uploads_and_cache;
