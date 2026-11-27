// Multi-Render-Target (MRT, 2..=4 `R32Float` attachments) warp-mesh rasterizer
// for co-grid Band Math bands sharing one `Frame` and patch mesh list.
//
// Packs all 4 input band textures into `@group(2)` (`@binding(0..3)`) to stay
// strictly within `max_bind_groups == 4` and `max_color_attachments == 4`.
// Emits runtime quiet-NaN per invalid channel so one band's nodata never
// discards sibling bands written in the same MRT pass.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    grid_step_deg: f32,
    grid_width_px: f32,
    polar_r_cap: f32,
    _pad0: f32,
};

struct ScratchMrtPatchUniform {
    offset_eye: vec3<f32>,
    nan_bits: u32,
    uv_rects: array<vec4<f32>, 4>,
    scale_offsets: array<vec4<f32>, 4>,
    mip_counts: vec4<u32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> patch_u: ScratchMrtPatchUniform;
@group(2) @binding(0) var tex0: texture_2d<f32>;
@group(2) @binding(1) var tex1: texture_2d<f32>;
@group(2) @binding(2) var tex2: texture_2d<f32>;
@group(2) @binding(3) var tex3: texture_2d<f32>;

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
    out.uv = in.uv;
    return out;
}

fn eval_band_phys(
    tex: texture_2d<f32>,
    base_uv: vec2<f32>,
    base_duv_dx: vec2<f32>,
    base_duv_dy: vec2<f32>,
    uv_rect: vec4<f32>,
    scale_offset: vec4<f32>,
    mip_count: u32,
    qnan: f32,
) -> f32 {
    let uv = uv_rect.xy + base_uv * uv_rect.zw;
    let sample = sample_trilinear_r32f(
        tex,
        uv,
        base_duv_dx * uv_rect.zw,
        base_duv_dy * uv_rect.zw,
        mip_count,
    );
    if sample.y < 0.5 {
        return qnan;
    }
    let physical = fma(sample.x, scale_offset.x, scale_offset.y);
    return select(qnan, physical, is_finite_f32(physical));
}

struct MrtOut2 {
    @location(0) b0: f32,
    @location(1) b1: f32,
};

@fragment
fn fs_band_mrt2(in: VertexOut) -> MrtOut2 {
    let duv_dx = dpdx(in.uv);
    let duv_dy = dpdy(in.uv);
    if dot(in.normal, -in.world_rel) <= 0.0 {
        discard;
    }
    let qnan = bitcast<f32>(patch_u.nan_bits);
    let v0 = eval_band_phys(tex0, in.uv, duv_dx, duv_dy, patch_u.uv_rects[0], patch_u.scale_offsets[0], patch_u.mip_counts.x, qnan);
    let v1 = eval_band_phys(tex1, in.uv, duv_dx, duv_dy, patch_u.uv_rects[1], patch_u.scale_offsets[1], patch_u.mip_counts.y, qnan);
    return MrtOut2(v0, v1);
}

struct MrtOut3 {
    @location(0) b0: f32,
    @location(1) b1: f32,
    @location(2) b2: f32,
};

@fragment
fn fs_band_mrt3(in: VertexOut) -> MrtOut3 {
    let duv_dx = dpdx(in.uv);
    let duv_dy = dpdy(in.uv);
    if dot(in.normal, -in.world_rel) <= 0.0 {
        discard;
    }
    let qnan = bitcast<f32>(patch_u.nan_bits);
    let v0 = eval_band_phys(tex0, in.uv, duv_dx, duv_dy, patch_u.uv_rects[0], patch_u.scale_offsets[0], patch_u.mip_counts.x, qnan);
    let v1 = eval_band_phys(tex1, in.uv, duv_dx, duv_dy, patch_u.uv_rects[1], patch_u.scale_offsets[1], patch_u.mip_counts.y, qnan);
    let v2 = eval_band_phys(tex2, in.uv, duv_dx, duv_dy, patch_u.uv_rects[2], patch_u.scale_offsets[2], patch_u.mip_counts.z, qnan);
    return MrtOut3(v0, v1, v2);
}

struct MrtOut4 {
    @location(0) b0: f32,
    @location(1) b1: f32,
    @location(2) b2: f32,
    @location(3) b3: f32,
};

@fragment
fn fs_band_mrt4(in: VertexOut) -> MrtOut4 {
    let duv_dx = dpdx(in.uv);
    let duv_dy = dpdy(in.uv);
    if dot(in.normal, -in.world_rel) <= 0.0 {
        discard;
    }
    let qnan = bitcast<f32>(patch_u.nan_bits);
    let v0 = eval_band_phys(tex0, in.uv, duv_dx, duv_dy, patch_u.uv_rects[0], patch_u.scale_offsets[0], patch_u.mip_counts.x, qnan);
    let v1 = eval_band_phys(tex1, in.uv, duv_dx, duv_dy, patch_u.uv_rects[1], patch_u.scale_offsets[1], patch_u.mip_counts.y, qnan);
    let v2 = eval_band_phys(tex2, in.uv, duv_dx, duv_dy, patch_u.uv_rects[2], patch_u.scale_offsets[2], patch_u.mip_counts.z, qnan);
    let v3 = eval_band_phys(tex3, in.uv, duv_dx, duv_dy, patch_u.uv_rects[3], patch_u.scale_offsets[3], patch_u.mip_counts.w, qnan);
    return MrtOut4(v0, v1, v2, v3);
}
