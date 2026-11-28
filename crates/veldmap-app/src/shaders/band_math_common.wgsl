// Shared fullscreen vertex shader and scalar Band Math helpers included by
// `band_math_reduce.wgsl` and `band_math_composite.wgsl` so single-epoch,
// dual-epoch reduction, and final compositing evaluate identical formulas.

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn is_finite_f32(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7F800000u) != 0x7F800000u;
}

struct ScalarEval {
    value: f32,
    valid: bool,
};

/// Evaluate a 2-band scalar formula (`0 = (A-B)/(A+B)`, `1 = A-B`, `2 = A/B`).
/// Unsupported modes return `valid = false` (caller discards or writes qNaN).
fn eval_scalar_formula(mode: u32, a: f32, b: f32) -> ScalarEval {
    if mode == 0u {
        let denom = a + b;
        if abs(denom) < 1e-6 {
            return ScalarEval(0.0, false);
        }
        let v = (a - b) / denom;
        return ScalarEval(v, is_finite_f32(v));
    } else if mode == 1u {
        let v = a - b;
        return ScalarEval(v, is_finite_f32(v));
    } else if mode == 2u {
        if abs(b) < 1e-6 {
            return ScalarEval(0.0, false);
        }
        let v = a / b;
        return ScalarEval(v, is_finite_f32(v));
    }
    return ScalarEval(0.0, false);
}
