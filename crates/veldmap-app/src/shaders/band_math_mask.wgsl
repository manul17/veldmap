// Nearest-neighbor `R8Uint` QA / Sentinel-2 `SCL` warp-mesh rasterizer writing
// into a dedicated viewport-sized `R8Uint` scratch target (`blend: None`,
// cleared to `255u` uncovered sentinel).

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
@group(2) @binding(0) var mask_tex: texture_2d<u32>;

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
fn fs_mask_u8(in: VertexOut) -> @location(0) u32 {
    if dot(in.normal, -in.world_rel) <= 0.0 {
        discard;
    }

    let dims = vec2<i32>(textureDimensions(mask_tex, 0));
    let max_coord = max(dims - vec2<i32>(1, 1), vec2<i32>(0, 0));
    let coord = clamp(
        vec2<i32>(floor(clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(dims))),
        vec2<i32>(0, 0),
        max_coord,
    );

    let class_id = textureLoad(mask_tex, coord, 0).r;
    if patch_u.has_nodata != 0u && class_id == patch_u.nodata_value {
        discard;
    }

    // Reserve `255u` strictly as the uncovered scratch sentinel.
    return min(class_id, 254u);
}
