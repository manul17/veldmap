//! CPU-only `naga` WGSL validation, uniform struct layout parity checks, and
//! SPIR-V / HLSL / MSL codegen tests (pinned to `wgpu` 30's `naga` 30).

use naga::back::{hlsl, msl, spv};
use naga::proc::Layouter;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use veldmap_app::render::{COMPOSITE_WGSL, GLOBE_WGSL, OVERLAY_WGSL, VECTOR_WGSL};

fn parse_and_validate(name: &str, wgsl_source: &str) -> (naga::Module, naga::valid::ModuleInfo) {
    let module = naga::front::wgsl::parse_str(wgsl_source).unwrap_or_else(|err| {
        panic!(
            "{name}: WGSL parse failed:\n{}",
            err.emit_to_string(wgsl_source)
        )
    });
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|err| panic!("{name}: naga validation failed:\n{err:?}"));
    (module, info)
}

fn assert_struct_layout(
    module: &naga::Module,
    struct_name: &str,
    expected_size: u32,
    expected_members: &[(&str, u32)],
) {
    let mut layouter = Layouter::default();
    layouter
        .update(module.to_ctx())
        .expect("naga layouter update");
    let (handle, ty) = module
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some(struct_name))
        .unwrap_or_else(|| panic!("WGSL struct `{struct_name}` not found"));
    assert_eq!(
        layouter[handle].size, expected_size,
        "WGSL struct `{struct_name}` size mismatch"
    );
    let naga::TypeInner::Struct { members, .. } = &ty.inner else {
        panic!("`{struct_name}` is not a WGSL struct");
    };
    for &(field, expected_offset) in expected_members {
        let member = members
            .iter()
            .find(|m| m.name.as_deref() == Some(field))
            .unwrap_or_else(|| panic!("field `{struct_name}.{field}` not found"));
        assert_eq!(
            member.offset, expected_offset,
            "field `{struct_name}.{field}` offset mismatch"
        );
    }
}

fn validate_and_compile_all_backends(name: &str, wgsl_source: &str) -> naga::Module {
    let (module, info) = parse_and_validate(name, wgsl_source);

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
    module
}

#[test]
fn globe_overlay_and_composite_shaders_compile_and_match_rust_uniform_layouts() {
    let globe = validate_and_compile_all_backends("globe.wgsl", GLOBE_WGSL);
    let overlay = validate_and_compile_all_backends("overlay.wgsl", OVERLAY_WGSL);
    let vector = validate_and_compile_all_backends("vector.wgsl", VECTOR_WGSL);
    let _composite = validate_and_compile_all_backends("composite.wgsl", COMPOSITE_WGSL);

    let camera_members = &[
        ("view_proj", 0),
        ("grid_step_deg", 64),
        ("grid_width_px", 68),
        ("polar_r_cap", 72),
    ];
    assert_struct_layout(&globe, "CameraUniform", 80, camera_members);
    assert_struct_layout(&overlay, "CameraUniform", 80, camera_members);
    assert_struct_layout(&vector, "CameraUniform", 80, camera_members);
    assert_struct_layout(
        &globe,
        "TileUniform",
        48,
        &[
            ("offset_eye", 0),
            ("tile_kind", 12),
            ("uv_rect", 16),
            ("grid_anchor", 32),
        ],
    );
    assert_struct_layout(
        &overlay,
        "OverlayPatchUniform",
        32,
        &[("offset_eye", 0), ("opacity", 12), ("uv_rect", 16)],
    );
    assert_struct_layout(
        &vector,
        "VectorUniform",
        32,
        &[("offset_eye", 0), ("point_radius_m", 12), ("color", 16)],
    );
}
