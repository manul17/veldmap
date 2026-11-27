// Shared Metal fast-math safe finiteness check and 8-tap nodata-aware
// trilinear `R32Float` sampling helper (requiring zero `FLOAT32_FILTERABLE`
// feature flags and handling NPOT clipped edge tiles via per-mip
// `textureDimensions`).
//
// Callers MUST evaluate `dpdx`/`dpdy` in uniform control flow strictly before
// any `discard` and pass `(duv_dx, duv_dy)` into `sample_trilinear_r32f`.

fn is_finite_f32(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7F800000u) != 0x7F800000u;
}

fn sample_bilinear_r32f(tex: texture_2d<f32>, uv: vec2<f32>, mip: i32) -> vec2<f32> {
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

// Returns `vec2(raw_value, valid_coverage)` using precomputed uniform-flow
// `(duv_dx, duv_dy)` derivatives in sample-UV space.
fn sample_trilinear_r32f(
    tex: texture_2d<f32>,
    uv: vec2<f32>,
    duv_dx: vec2<f32>,
    duv_dy: vec2<f32>,
    mip_count: u32,
) -> vec2<f32> {
    let dims0 = vec2<f32>(textureDimensions(tex, 0));
    let dx = duv_dx * dims0;
    let dy = duv_dy * dims0;
    let rho2 = max(dot(dx, dx), dot(dy, dy));
    let max_mip = max(i32(mip_count) - 1, 0);
    let lod = clamp(0.5 * log2(max(rho2, 1.0)), 0.0, f32(max_mip));

    let mip0 = i32(floor(lod));
    let mip1 = min(mip0 + 1, max_mip);
    let t_mip = lod - f32(mip0);

    let s0 = sample_bilinear_r32f(tex, uv, mip0);
    var raw_val = s0.x;
    var valid_cov = s0.y;
    if mip1 != mip0 && t_mip > 1e-4 {
        let s1 = sample_bilinear_r32f(tex, uv, mip1);
        let w0 = s0.y * (1.0 - t_mip);
        let w1 = s1.y * t_mip;
        let wt = w0 + w1;
        valid_cov = wt;
        raw_val = select(0.0, (s0.x * w0 + s1.x * w1) / max(wt, 1e-6), wt > 1e-6);
    }
    return vec2<f32>(raw_val, valid_cov);
}
