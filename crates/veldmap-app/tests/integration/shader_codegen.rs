//! CPU-only `naga` WGSL validation and SPIR-V / HLSL / MSL codegen tests
//! (pinned to `wgpu` 30's `naga` 30 major version).

use naga::back::{hlsl, msl, spv};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use veldmap_app::render::{COMPOSITE_WGSL, GLOBE_WGSL};

fn validate_and_compile_all_backends(name: &str, wgsl_source: &str) {
    let module = naga::front::wgsl::parse_str(wgsl_source).unwrap_or_else(|err| {
        panic!(
            "{name}: WGSL parse failed:\n{}",
            err.emit_to_string(wgsl_source)
        )
    });

    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|err| panic!("{name}: naga validation failed:\n{err:?}"));

    // 1. SPIR-V (Vulkan)
    let spv_opts = spv::Options {
        lang_version: (1, 3),
        flags: spv::WriterFlags::ADJUST_COORDINATE_SPACE | spv::WriterFlags::CLAMP_FRAG_DEPTH,
        ..Default::default()
    };
    let spv_words = spv::write_vec(&module, &info, &spv_opts, None)
        .unwrap_or_else(|err| panic!("{name}: SPIR-V codegen failed: {err:?}"));
    assert!(
        spv_words.len() > 16,
        "{name}: SPIR-V output unexpectedly tiny ({} words)",
        spv_words.len()
    );

    // 2. HLSL (DirectX 12, Shader Model 5.1)
    let hlsl_opts = hlsl::Options {
        shader_model: hlsl::ShaderModel::V5_1,
        ..Default::default()
    };
    let hlsl_pipeline_opts = hlsl::PipelineOptions::default();
    let mut hlsl_out = String::new();
    let mut hlsl_writer = hlsl::Writer::new(&mut hlsl_out, &hlsl_opts, &hlsl_pipeline_opts);
    hlsl_writer
        .write(&module, &info, None)
        .unwrap_or_else(|err| panic!("{name}: HLSL codegen failed: {err:?}"));
    assert!(
        !hlsl_out.is_empty(),
        "{name}: HLSL codegen produced empty output"
    );

    // 3. MSL (Metal 2.1)
    let msl_opts = msl::Options {
        lang_version: (2, 1),
        ..Default::default()
    };
    let mut msl_out = String::new();
    let mut msl_writer = msl::Writer::new(&mut msl_out);
    msl_writer
        .write(&module, &info, &msl_opts, &msl::PipelineOptions::default())
        .unwrap_or_else(|err| panic!("{name}: MSL codegen failed: {err:?}"));
    assert!(
        !msl_out.is_empty(),
        "{name}: MSL codegen produced empty output"
    );
}

#[test]
fn globe_and_composite_shaders_compile_to_spv_hlsl_msl() {
    validate_and_compile_all_backends("globe.wgsl", GLOBE_WGSL);
    validate_and_compile_all_backends("composite.wgsl", COMPOSITE_WGSL);
}
