// Single-band `R32Float` physical-value warp-mesh rasterizer writing into a
// viewport-sized `R32Float` scratch target (`blend: None`).
//
// Evaluates 8-tap nodata-aware trilinear `textureLoad` (mip LOD computed in
// uniform control flow before horizon `discard`), scales into canonical
// `val * scale + offset` physical units, and discards nodata/non-finite
// fragments so uncovered pixels keep their quiet-NaN sentinel while valid
// `0.0` reflectance is preserved intact.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    grid_step_deg: f32,
    grid_width_px: f32,
    polar_r_cap: f32,
    _pad0: f32,
};

struct ScratchSinglePatchUniform {
    offset_eye: vec3<f32>,
    scale: f32,
    uv_rect: vec4<f32>,
    offset: f32,
    mip_count: u32,
    has_nodata: u32,
    nodata_value: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> patch_u: ScratchSinglePatchUniform;
@group(2) @binding(0) var tile_tex: texture_2d<f32>;

struct VertexIn {
    @location(0) rel_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let world_rel = patch_u.offset_eye + in.rel_pos;
    var out: VertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_rel, 1.0);
    out.world_rel = world_rel;
    out.normal = in.normal;
    out.uv = patch_u.uv_rect.xy + in.uv * patch_u.uv_rect.zw;
    return out;
}

@fragment
fn fs_band_single(in: VertexOut) -> @location(0) f32 {
    let duv_dx = dpdx(in.uv);
    let duv_dy = dpdy(in.uv);

    if dot(in.normal, -in.world_rel) <= 0.0 {
        discard;
    }

    let sample = sample_trilinear_r32f(tile_tex, in.uv, duv_dx, duv_dy, patch_u.mip_count);
    if sample.y < 0.5 {
        discard;
    }

    let physical = fma(sample.x, patch_u.scale, patch_u.offset);
    if !is_finite_f32(physical) {
        discard;
    }
    return physical;
}
