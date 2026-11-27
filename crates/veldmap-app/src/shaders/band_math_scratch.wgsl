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

fn is_finite_f32(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7F800000u) != 0x7F800000u;
}

fn sample_bilinear_nodata(uv: vec2<f32>, mip: i32) -> vec2<f32> {
    let dims = vec2<i32>(textureDimensions(tile_tex, mip));
    let max_coord = max(dims - vec2<i32>(1, 1), vec2<i32>(0, 0));
    let fcoord = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(dims) - vec2<f32>(0.5);
    let i0 = vec2<i32>(floor(fcoord));
    let frac = fcoord - vec2<f32>(i0);

    let c00 = clamp(i0, vec2<i32>(0, 0), max_coord);
    let c10 = clamp(i0 + vec2<i32>(1, 0), vec2<i32>(0, 0), max_coord);
    let c01 = clamp(i0 + vec2<i32>(0, 1), vec2<i32>(0, 0), max_coord);
    let c11 = clamp(i0 + vec2<i32>(1, 1), vec2<i32>(0, 0), max_coord);

    let v00 = textureLoad(tile_tex, c00, mip).r;
    let v10 = textureLoad(tile_tex, c10, mip).r;
    let v01 = textureLoad(tile_tex, c01, mip).r;
    let v11 = textureLoad(tile_tex, c11, mip).r;

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

@fragment
fn fs_band_single(in: VertexOut) -> @location(0) f32 {
    let dims0 = vec2<f32>(textureDimensions(tile_tex, 0));
    let dx = dpdx(in.uv) * dims0;
    let dy = dpdy(in.uv) * dims0;
    let rho2 = max(dot(dx, dx), dot(dy, dy));
    let max_mip = max(i32(patch_u.mip_count) - 1, 0);
    let lod = clamp(0.5 * log2(max(rho2, 1.0)), 0.0, f32(max_mip));

    if dot(in.normal, -in.world_rel) <= 0.0 {
        discard;
    }

    let mip0 = i32(floor(lod));
    let mip1 = min(mip0 + 1, max_mip);
    let t_mip = lod - f32(mip0);

    let s0 = sample_bilinear_nodata(in.uv, mip0);
    var raw_val = s0.x;
    var valid_cov = s0.y;
    if mip1 != mip0 && t_mip > 1e-4 {
        let s1 = sample_bilinear_nodata(in.uv, mip1);
        let w0 = s0.y * (1.0 - t_mip);
        let w1 = s1.y * t_mip;
        let wt = w0 + w1;
        valid_cov = wt;
        raw_val = select(0.0, (s0.x * w0 + s1.x * w1) / max(wt, 1e-6), wt > 1e-6);
    }

    if valid_cov < 0.5 {
        discard;
    }

    let physical = fma(raw_val, patch_u.scale, patch_u.offset);
    if !is_finite_f32(physical) {
        discard;
    }
    return physical;
}
