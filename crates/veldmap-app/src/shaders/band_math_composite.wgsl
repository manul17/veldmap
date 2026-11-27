// Fullscreen screen-space physical Band Math + `SCL` cloud/shadow masking +
// colormap / linear-light RGB composite onto the `Rgba8UnormSrgb` globe buffer.
//
// Reads physical `R32Float` band values from the shared viewport scratch pool
// (quiet-NaN = nodata/uncovered; valid `0.0` preserved intact) and discrete
// `R8Uint` `SCL` classes (`255u` = uncovered sentinel).

struct BandMathCompositeUniform {
    opacity: f32,
    mode: u32,
    stretch_min: f32,
    stretch_inv_range: f32,
    rgb_min: vec4<f32>,
    rgb_inv_range: vec4<f32>,
    invert_colormap: u32,
    scl_mask_enabled: u32,
    scl_mask_bits: u32,
    _pad0: u32,
};

struct PaletteUniform {
    entries: array<vec4<f32>, 256>,
};

@group(0) @binding(0) var band0_tex: texture_2d<f32>;
@group(0) @binding(1) var band1_tex: texture_2d<f32>;
@group(0) @binding(2) var band2_tex: texture_2d<f32>;
@group(0) @binding(3) var band3_tex: texture_2d<f32>;
@group(0) @binding(4) var mask_tex: texture_2d<u32>;

@group(1) @binding(0) var<uniform> params: BandMathCompositeUniform;
@group(2) @binding(0) var<uniform> palette: PaletteUniform;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn is_finite_f32(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7F800000u) != 0x7F800000u;
}

fn srgb_to_linear_channel(c: f32) -> f32 {
    let s = clamp(c, 0.0, 1.0);
    return select(s / 12.92, pow((s + 0.055) / 1.055, 2.4), s > 0.04045);
}

@fragment
fn fs_composite(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(i32(frag_pos.x), i32(frag_pos.y));

    if params.scl_mask_enabled != 0u {
        let mdims = vec2<i32>(textureDimensions(mask_tex, 0));
        if px.x >= 0 && px.y >= 0 && px.x < mdims.x && px.y < mdims.y {
            let scl_class = textureLoad(mask_tex, px, 0).r;
            if scl_class < 32u && ((params.scl_mask_bits & (1u << scl_class)) != 0u) {
                discard;
            }
        }
    }

    // Mode 3: 3-band physical RGB composite (True Color, False Color NIR, SWIR, custom RGB).
    if params.mode == 3u {
        let r = textureLoad(band0_tex, px, 0).r;
        let g = textureLoad(band1_tex, px, 0).r;
        let b = textureLoad(band2_tex, px, 0).r;
        if !is_finite_f32(r) || !is_finite_f32(g) || !is_finite_f32(b) {
            discard;
        }
        var sr = clamp((r - params.rgb_min.x) * params.rgb_inv_range.x, 0.0, 1.0);
        var sg = clamp((g - params.rgb_min.y) * params.rgb_inv_range.y, 0.0, 1.0);
        var sb = clamp((b - params.rgb_min.z) * params.rgb_inv_range.z, 0.0, 1.0);
        let inv_gamma = params.rgb_min.w;
        if abs(inv_gamma - 1.0) > 1e-4 {
            sr = select(0.0, pow(max(sr, 1e-12), inv_gamma), sr > 0.0);
            sg = select(0.0, pow(max(sg, 1e-12), inv_gamma), sg > 0.0);
            sb = select(0.0, pow(max(sb, 1e-12), inv_gamma), sb > 0.0);
        }
        let alpha = clamp(params.opacity, 0.0, 1.0);
        if alpha <= 0.001 {
            discard;
        }
        return vec4<f32>(
            srgb_to_linear_channel(sr),
            srgb_to_linear_channel(sg),
            srgb_to_linear_channel(sb),
            alpha,
        );
    }

    // Scalar modes: 0 = (A-B)/(A+B), 1 = A-B, 2 = A/B.
    let a = textureLoad(band0_tex, px, 0).r;
    let b = textureLoad(band1_tex, px, 0).r;
    if !is_finite_f32(a) || !is_finite_f32(b) {
        discard;
    }

    var val: f32 = 0.0;
    if params.mode == 0u {
        let denom = a + b;
        if abs(denom) < 1e-6 {
            discard;
        }
        val = (a - b) / denom;
    } else if params.mode == 1u {
        val = a - b;
    } else {
        if abs(b) < 1e-6 {
            discard;
        }
        val = a / b;
    }

    if !is_finite_f32(val) {
        discard;
    }

    var t = clamp((val - params.stretch_min) * params.stretch_inv_range, 0.0, 1.0);
    if params.invert_colormap != 0u {
        t = 1.0 - t;
    }

    let scaled = t * 255.0;
    let idx0 = min(u32(floor(scaled)), 255u);
    let idx1 = min(idx0 + 1u, 255u);
    let frac = scaled - f32(idx0);
    let lut_color = mix(palette.entries[idx0], palette.entries[idx1], frac);
    let alpha = lut_color.a * params.opacity;
    if alpha <= 0.001 {
        discard;
    }
    return vec4<f32>(lut_color.rgb, alpha);
}
