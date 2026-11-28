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

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn is_finite_f32(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7F800000u) != 0x7F800000u;
}

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

    // Mode 4: single-band scientific/SAR passthrough (`N = 1`).
    if params.mode == 4u {
        return a;
    }

    let b = textureLoad(band1_tex, px, 0).r;
    if !is_finite_f32(b) {
        return qnan;
    }

    var val: f32 = qnan;
    if params.mode == 0u {
        let denom = a + b;
        if abs(denom) >= 1e-6 {
            val = (a - b) / denom;
        }
    } else if params.mode == 1u {
        val = a - b;
    } else if params.mode == 2u {
        if abs(b) >= 1e-6 {
            val = a / b;
        }
    } else {
        // Reserve `band2_tex` binding for future 3-band scalar indices.
        let c = textureLoad(band2_tex, px, 0).r;
        if is_finite_f32(c) {
            val = a + b - c;
        }
    }

    if !is_finite_f32(val) {
        return qnan;
    }
    return val;
}
