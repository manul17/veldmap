// Fullscreen Epoch A scalar reduction pass for dual-epoch temporal diffing.
//
// Reads ONLY `scratch[0..N]` (up to 3 `R32Float` bands) plus optional `R8Uint`
// `SCL` mask from a non-overlapping read-only bind group `@group(0)` and writes
// the reduced scalar value (or quiet-NaN when masked, uncovered, or singular)
// into `scratch[N]` (`R32Float`, `blend: None`) with zero read/write texture
// hazards under `wgpu` and `downlevel_defaults()`.

struct BandMathReduceUniform {
    /// 0 = (A-B)/(A+B), 1 = A-B, 2 = A/B, 4 = single-band passthrough (A).
    mode: u32,
    scl_mask_enabled: u32,
    scl_mask_bits: u32,
    nan_bits: u32,
};

@group(0) @binding(0) var band0_tex: texture_2d<f32>;
@group(0) @binding(1) var band1_tex: texture_2d<f32>;
@group(0) @binding(2) var band2_tex: texture_2d<f32>;
@group(0) @binding(3) var mask_tex: texture_2d<u32>;

@group(1) @binding(0) var<uniform> params: BandMathReduceUniform;

@fragment
fn fs_reduce(@builtin(position) frag_pos: vec4<f32>) -> @location(0) f32 {
    let qnan = bitcast<f32>(params.nan_bits);
    let px = vec2<i32>(i32(frag_pos.x), i32(frag_pos.y));

    if params.scl_mask_enabled != 0u {
        let mdims = vec2<i32>(textureDimensions(mask_tex, 0));
        if px.x >= 0 && px.y >= 0 && px.x < mdims.x && px.y < mdims.y {
            let scl_class = textureLoad(mask_tex, px, 0).r;
            if scl_class < 32u && ((params.scl_mask_bits & (1u << scl_class)) != 0u) {
                return qnan;
            }
        }
    }

    let a = textureLoad(band0_tex, px, 0).r;
    if !is_finite_f32(a) {
        return qnan;
    }

    // Mode 4: single-band scientific/SAR passthrough (`N = 1`, checked before
    // loading `band1_tex` so the 1x1 dummy texture is never read out of bounds).
    if params.mode == 4u {
        return a;
    }

    let b = textureLoad(band1_tex, px, 0).r;
    if !is_finite_f32(b) {
        return qnan;
    }

    let res = eval_scalar_formula(params.mode, a, b);
    if !res.valid {
        return qnan;
    }
    return res.value;
}
