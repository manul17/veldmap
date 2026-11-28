// Fullscreen screen-space physical Band Math, dual-epoch scalar diffing, and
// 0%-bleed-through dual-epoch RGB cross-fade onto the `Rgba8UnormSrgb` globe
// buffer (or Epoch A `rgba_scratch[0]` bake target).
//
// Reads physical `R32Float` band values from `@group(0)` (quiet-NaN =
// nodata/uncovered; valid `0.0` preserved intact), discrete `R8Uint` `SCL`
// classes (`255u` = uncovered sentinel), and optional `Rgba8UnormSrgb` scratch
// slots from `@group(3)` (bound to 1x1 dummies whenever not reading `rgba_scratch`
// to guarantee zero `wgpu` read/write hazards).

struct BandMathCompositeUniform {
    opacity: f32,
    /// 0 = (A-B)/(A+B), 1 = A-B, 2 = A/B, 3 = 3-band RGB, 4 = 1-band passthrough,
    /// 5 = BandMath RGB cross-fade (rgba0 vs live band0..2 + mask),
    /// 6 = Direct Display RGB cross-fade (rgba0 vs rgba1).
    mode: u32,
    stretch_min: f32,
    stretch_inv_range: f32,
    /// `.xyz` = per-channel RGB min, `.w` = `inv_gamma`.
    rgb_min: vec4<f32>,
    /// `.xyz` = per-channel RGB inv range, `.w` = cross-fade `blend_t` in `[0, 1]`.
    rgb_inv_range: vec4<f32>,
    invert_colormap: u32,
    scl_mask_enabled: u32,
    scl_mask_bits: u32,
    /// When `1..=3`, subtract reduced Epoch A scalar stored in `band{diff_slot}_tex`.
    diff_slot: u32,
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

@group(3) @binding(0) var rgba0_tex: texture_2d<f32>;
@group(3) @binding(1) var rgba1_tex: texture_2d<f32>;

fn srgb_to_linear_channel(c: f32) -> f32 {
    let s = clamp(c, 0.0, 1.0);
    return select(s / 12.92, pow((s + 0.055) / 1.055, 2.4), s > 0.04045);
}

fn is_scl_masked(px: vec2<i32>) -> bool {
    if params.scl_mask_enabled == 0u {
        return false;
    }
    let mdims = vec2<i32>(textureDimensions(mask_tex, 0));
    if px.x >= 0 && px.y >= 0 && px.x < mdims.x && px.y < mdims.y {
        let scl_class = textureLoad(mask_tex, px, 0).r;
        if scl_class < 32u && ((params.scl_mask_bits & (1u << scl_class)) != 0u) {
            return true;
        }
    }
    return false;
}

fn stretch_band_rgb_linear(px: vec2<i32>) -> vec4<f32> {
    if is_scl_masked(px) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let r = textureLoad(band0_tex, px, 0).r;
    let g = textureLoad(band1_tex, px, 0).r;
    let b = textureLoad(band2_tex, px, 0).r;
    if !is_finite_f32(r) || !is_finite_f32(g) || !is_finite_f32(b) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
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
    return vec4<f32>(
        srgb_to_linear_channel(sr),
        srgb_to_linear_channel(sg),
        srgb_to_linear_channel(sb),
        1.0,
    );
}

fn load_rgba_safe(tex: texture_2d<f32>, px: vec2<i32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(tex, 0));
    if px.x < 0 || px.y < 0 || px.x >= dims.x || px.y >= dims.y {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    return textureLoad(tex, px, 0);
}

/// Premultiplied-safe cross-fade between two straight-alpha linear colors:
/// - Both opaque (`a0 = 1, a1 = 1`): `mix(rgb0, rgb1, t)` with `alpha = 1.0`
///   (caller multiplies outer `opacity`, yielding 0% basemap bleed-through).
/// - Single-valid pixel: graceful fade without darkening or double-premultiplication.
fn cross_fade_straight(c0: vec4<f32>, c1: vec4<f32>, blend_t: f32) -> vec4<f32> {
    let t = clamp(blend_t, 0.0, 1.0);
    let p0 = vec4<f32>(c0.rgb * c0.a, c0.a);
    let p1 = vec4<f32>(c1.rgb * c1.a, c1.a);
    let out_p = mix(p0, p1, t);
    if out_p.a <= 0.001 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    return vec4<f32>(out_p.rgb / out_p.a, out_p.a);
}

fn load_diff_epoch_a(px: vec2<i32>, slot: u32) -> f32 {
    if slot == 1u {
        return textureLoad(band1_tex, px, 0).r;
    } else if slot == 2u {
        return textureLoad(band2_tex, px, 0).r;
    }
    return textureLoad(band3_tex, px, 0).r;
}

@fragment
fn fs_composite(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(i32(frag_pos.x), i32(frag_pos.y));
    let opacity = clamp(params.opacity, 0.0, 1.0);
    if opacity <= 0.001 {
        discard;
    }

    // Mode 6: Direct Display dual-epoch RGB cross-fade (`rgba0_tex` vs `rgba1_tex`).
    if params.mode == 6u {
        let c0 = load_rgba_safe(rgba0_tex, px);
        let c1 = load_rgba_safe(rgba1_tex, px);
        let blended = cross_fade_straight(c0, c1, params.rgb_inv_range.w);
        let alpha = blended.a * opacity;
        if alpha <= 0.001 {
            discard;
        }
        return vec4<f32>(blended.rgb, alpha);
    }

    // Mode 5: BandMath RGB dual-epoch cross-fade (`rgba0_tex` vs live Epoch B `band0..2 + mask`).
    if params.mode == 5u {
        let c0 = load_rgba_safe(rgba0_tex, px);
        let c1 = stretch_band_rgb_linear(px);
        let blended = cross_fade_straight(c0, c1, params.rgb_inv_range.w);
        let alpha = blended.a * opacity;
        if alpha <= 0.001 {
            discard;
        }
        return vec4<f32>(blended.rgb, alpha);
    }

    // Mode 3: 3-band physical RGB composite (also used with `blend: None` to bake Epoch A).
    if params.mode == 3u {
        let c = stretch_band_rgb_linear(px);
        let alpha = c.a * opacity;
        if alpha <= 0.001 {
            discard;
        }
        return vec4<f32>(c.rgb, alpha);
    }

    if is_scl_masked(px) {
        discard;
    }

    // Scalar modes: 0 = (A-B)/(A+B), 1 = A-B, 2 = A/B, 4 = 1-band passthrough (A).
    let a = textureLoad(band0_tex, px, 0).r;
    if !is_finite_f32(a) {
        discard;
    }

    var val: f32 = 0.0;
    if params.mode == 4u {
        val = a;
    } else {
        let b = textureLoad(band1_tex, px, 0).r;
        if !is_finite_f32(b) {
            discard;
        }
        let res = eval_scalar_formula(params.mode, a, b);
        if !res.valid {
            discard;
        }
        val = res.value;
    }

    // Optional dual-epoch scalar diff: `formula(B, mask_B) - reduced_A`.
    if params.diff_slot != 0u {
        let epoch_a = load_diff_epoch_a(px, params.diff_slot);
        if !is_finite_f32(epoch_a) {
            discard;
        }
        val = val - epoch_a;
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
    let alpha = lut_color.a * opacity;
    if alpha <= 0.001 {
        discard;
    }
    return vec4<f32>(lut_color.rgb, alpha);
}
