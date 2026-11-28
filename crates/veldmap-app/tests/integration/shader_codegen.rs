//! CPU-only `naga` WGSL validation, uniform struct layout parity checks, and
//! SPIR-V / HLSL / MSL codegen tests (pinned to `wgpu` 30's `naga` 30).

use std::collections::HashSet;

use naga::back::{hlsl, msl, spv};
use naga::proc::Layouter;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use veldmap_app::render::{
    BAND_MATH_COMPOSITE_WGSL, BAND_MATH_MASK_WGSL, BAND_MATH_REDUCE_WGSL,
    BAND_MATH_SCRATCH_MRT_WGSL, BAND_MATH_SCRATCH_WGSL, COMPOSITE_WGSL, GLOBE_WGSL,
    OVERLAY_CATEGORICAL_WGSL, OVERLAY_SCIENTIFIC_WGSL, OVERLAY_WGSL, SCRATCH_NAN_FILL_WGSL,
    UNIFORM_LAYOUT_SPECS, UniformLayoutSpec, VECTOR_WGSL,
};

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

fn layout_spec(wgsl_name: &str) -> &'static UniformLayoutSpec {
    UNIFORM_LAYOUT_SPECS
        .iter()
        .find(|spec| spec.wgsl_name == wgsl_name)
        .unwrap_or_else(|| panic!("missing Rust UniformLayoutSpec for `{wgsl_name}`"))
}

fn assert_struct_layout(module: &naga::Module, spec: &UniformLayoutSpec) {
    let struct_name = spec.wgsl_name;
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
        layouter[handle].size as usize, spec.size,
        "WGSL struct `{struct_name}` size mismatch"
    );
    let naga::TypeInner::Struct { members, .. } = &ty.inner else {
        panic!("`{struct_name}` is not a WGSL struct");
    };
    for &(field, expected_offset) in spec.fields {
        let member = members
            .iter()
            .find(|m| m.name.as_deref() == Some(field))
            .unwrap_or_else(|| panic!("field `{struct_name}.{field}` not found"));
        assert_eq!(
            member.offset as usize, expected_offset,
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
fn all_wgsl_shaders_compile_and_match_rust_uniform_layouts() {
    let globe = validate_and_compile_all_backends("globe.wgsl", GLOBE_WGSL);
    let overlay = validate_and_compile_all_backends("overlay.wgsl", OVERLAY_WGSL);
    let overlay_sci =
        validate_and_compile_all_backends("overlay_scientific.wgsl", OVERLAY_SCIENTIFIC_WGSL);
    let overlay_cat =
        validate_and_compile_all_backends("overlay_categorical.wgsl", OVERLAY_CATEGORICAL_WGSL);
    let vector = validate_and_compile_all_backends("vector.wgsl", VECTOR_WGSL);
    let _composite = validate_and_compile_all_backends("composite.wgsl", COMPOSITE_WGSL);
    let scratch_clear =
        validate_and_compile_all_backends("scratch_nan_fill.wgsl", SCRATCH_NAN_FILL_WGSL);
    let bm_scratch =
        validate_and_compile_all_backends("band_math_scratch.wgsl", BAND_MATH_SCRATCH_WGSL);
    let bm_scratch_mrt =
        validate_and_compile_all_backends("band_math_scratch_mrt.wgsl", BAND_MATH_SCRATCH_MRT_WGSL);
    let bm_mask = validate_and_compile_all_backends("band_math_mask.wgsl", BAND_MATH_MASK_WGSL);
    let bm_reduce =
        validate_and_compile_all_backends("band_math_reduce.wgsl", BAND_MATH_REDUCE_WGSL);
    let bm_composite =
        validate_and_compile_all_backends("band_math_composite.wgsl", BAND_MATH_COMPOSITE_WGSL);

    let shader_uniforms: &[(&naga::Module, &[&str])] = &[
        (&globe, &["CameraUniform", "TileUniform"]),
        (&overlay, &["CameraUniform", "OverlayPatchUniform"]),
        (
            &overlay_sci,
            &["CameraUniform", "ScientificPatchUniform", "PaletteUniform"],
        ),
        (
            &overlay_cat,
            &["CameraUniform", "CategoricalPatchUniform", "PaletteUniform"],
        ),
        (&vector, &["CameraUniform", "VectorUniform"]),
        (&scratch_clear, &["ScratchClearUniform"]),
        (&bm_scratch, &["CameraUniform", "ScratchSinglePatchUniform"]),
        (
            &bm_scratch_mrt,
            &["CameraUniform", "ScratchMrtPatchUniform"],
        ),
        (&bm_mask, &["CameraUniform", "ScratchSinglePatchUniform"]),
        (&bm_reduce, &["BandMathReduceUniform"]),
        (
            &bm_composite,
            &["BandMathCompositeUniform", "PaletteUniform"],
        ),
    ];

    let mut verified = HashSet::new();
    for &(module, struct_names) in shader_uniforms {
        for &struct_name in struct_names {
            assert_struct_layout(module, layout_spec(struct_name));
            verified.insert(struct_name);
        }
    }

    for spec in UNIFORM_LAYOUT_SPECS {
        assert!(
            verified.contains(spec.wgsl_name),
            "UniformLayoutSpec `{}` was not verified against any WGSL shader",
            spec.wgsl_name
        );
    }
}
