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

fn is_finite_f32(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7F800000u) != 0x7F800000u;
}

fn sample_bilinear_tex(tex: texture_2d<f32>, uv: vec2<f32>, mip: i32) -> vec2<f32> {
    let dims = vec2<i32>(textureDimensions(tex, mip));
    let max_coord = max(dims - vec2<i32>(1, 1), vec2<i32>(0, 0));
    let fcoord = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(dims) - vec2<f32>(0.5);
    let i0 = vec2<i32>(floor(fcoord));
    let frac = fcoord - vec2<f32>(i0);

    let c00 = clamp(i0, vec2<i32>(0, 0), max_coord);
    let c10 = clamp(i0 + vec2<i32>(1, 0), vec2<i32>(0, 0), max_coord);
    let c01 = clamp(i0 + vec2<i32>(0, 1), vec2<i32>(0, 0), max_coord);
    let c11 = clamp(i0 + vec2<i32>(1, 1), vec2<i32>(0, 0), max_coord);

    let v00 = textureLoad(tex, c00, mip).r;
    let v10 = textureLoad(tex, c10, mip).r;
    let v01 = textureLoad(tex, c01, mip).r;
    let v11 = textureLoad(tex, c11, mip).r;

    let w00 = select(0.0, (1.0 - frac.x) * (1.0 - frac.y), is_finite_f32(v00));
    let w10 = select(0.0, frac.x * (1.0 - frac.y), is_finite_f32(v10));
    let w01 = select(0.0, (1.0 - frac.x) * frac.y, is_finite_f32(v01));
    let w11 = select(0.0, frac.x * frac.y, is_finite_f32(v11));

    let w_sum = w00 + w10 + w01 + w11;
    if w_sum <= 1e-6 {
        return vec2<f32>(0.0, 0.0);
    }

    let safe00 = select(0.0, v00, w00 > 0.0);
    let safe10 = select(0.0, v10, w10 > 0.0);
    let safe01 = select(0.0, v01, w01 > 0.0);
    let safe11 = select(0.0, v11, w11 > 0.0);
    let sum = safe00 * w00 + safe10 * w10 + safe01 * w01 + safe11 * w11;
    return vec2<f32>(sum / w_sum, w_sum);
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
    let dims0 = vec2<f32>(textureDimensions(tex, 0));
    let dx = base_duv_dx * uv_rect.zw * dims0;
    let dy = base_duv_dy * uv_rect.zw * dims0;
    let rho2 = max(dot(dx, dx), dot(dy, dy));
    let max_mip = max(i32(mip_count) - 1, 0);
    let lod = clamp(0.5 * log2(max(rho2, 1.0)), 0.0, f32(max_mip));

    let mip0 = i32(floor(lod));
    let mip1 = min(mip0 + 1, max_mip);
    let t_mip = lod - f32(mip0);

    let s0 = sample_bilinear_tex(tex, uv, mip0);
    var raw_val = s0.x;
    var valid_cov = s0.y;
    if mip1 != mip0 && t_mip > 1e-4 {
        let s1 = sample_bilinear_tex(tex, uv, mip1);
        let w0 = s0.y * (1.0 - t_mip);
        let w1 = s1.y * t_mip;
        let wt = w0 + w1;
        valid_cov = wt;
        raw_val = select(0.0, (s0.x * w0 + s1.x * w1) / max(wt, 1e-6), wt > 1e-6);
    }

    if valid_cov < 0.5 {
        return qnan;
    }
    let physical = fma(raw_val, scale_offset.x, scale_offset.y);
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
