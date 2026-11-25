// Depth-less post-body scientific (`R32Float`) raster overlay warp-mesh shader:
// - Per-cell CPU f64 RTC vertex transforms (`offset_eye + rel_pos`)
// - Uniform-flow `dpdx`/`dpdy` mip LOD selection strictly BEFORE any `discard`
// - Manual 8-tap nodata-aware trilinear `textureLoad(tile_tex, coord, mip)`
//   requiring zero `FLOAT32_FILTERABLE` feature flags and handling NPOT clipped
//   edge tiles via per-mip `textureDimensions(tile_tex, mip)`
// - Metal fast-math safe NaN/±Inf bitmask check via `bitcast<u32>` exponent bits
//   strictly BEFORE any scale/offset/stretch math
// - Canonical physical reflectance/value scaling `fma(raw, scale, offset)`
//   plus linear or `log10` dB (`10 * log10`) stretch into a 256-entry linear LUT.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    grid_step_deg: f32,
    grid_width_px: f32,
    polar_r_cap: f32,
    _pad0: f32,
};

struct ScientificPatchUniform {
    /// `(patch_origin_ecef_m - camera_eye_ecef_m) as f32` computed in f64 on CPU.
    offset_eye: vec3<f32>,
    /// Layer opacity multiplier in `[0, 1]`.
    opacity: f32,
    /// Carrier UV sub-rect `(u_offset, v_offset, u_scale, v_scale)`.
    uv_rect: vec4<f32>,
    /// Canonical physical affine scale (`val * scale + offset`).
    scale: f32,
    /// Canonical physical affine offset (`val * scale + offset`).
    offset: f32,
    /// Stretch lower bound (physical units, or dB when `stretch_mode == 1u`).
    stretch_min: f32,
    /// Reciprocal stretch span `1.0 / (stretch_max - stretch_min)`.
    stretch_inv_range: f32,
    /// `0u` = linear stretch, `1u` = `10 * log10(max(physical, 1e-10))` dB stretch.
    stretch_mode: u32,
    /// Non-zero to reverse colormap direction.
    invert_colormap: u32,
    /// Number of uploaded mip levels (`>= 1`) on `tile_tex`.
    mip_count: u32,
    _pad0: u32,
};

struct PaletteUniform {
    /// 256-entry linear-light RGBA colormap stops.
    entries: array<vec4<f32>, 256>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> tile: ScientificPatchUniform;
@group(2) @binding(0) var tile_tex: texture_2d<f32>;
@group(3) @binding(0) var<uniform> palette: PaletteUniform;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    /// Eye-relative ECEF position (`P - E`, metres).
    @location(0) world: vec3<f32>,
    /// Outward WGS84 geodetic surface normal.
    @location(1) normal: vec3<f32>,
    /// Cell-local texture coordinates in `[0, 1]^2`.
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(
    @location(0) rel_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VsOut {
    var out: VsOut;
    let rel = tile.offset_eye + rel_pos;
    out.clip = camera.view_proj * vec4<f32>(rel, 1.0);
    out.world = rel;
    out.normal = normal;
    out.uv = uv;
    return out;
}

/// Metal fast-math safe finiteness check inspecting raw IEEE-754 exponent bits
/// (`0x7F80_0000`) before any floating-point comparison or arithmetic.
fn is_finite_f32(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7F800000u) != 0x7F800000u;
}

/// 4-tap nodata-aware bilinear filter on mip level `mip` returning
/// `vec2(normalized_value, valid_weight_sum)`.
fn sample_bilinear_nodata(uv: vec2<f32>, mip: i32) -> vec2<f32> {
    let dims = textureDimensions(tile_tex, mip);
    let max_coord = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);
    let fpos = uv * vec2<f32>(dims) - vec2<f32>(0.5, 0.5);
    let p0 = vec2<i32>(floor(fpos));
    let frac = fpos - vec2<f32>(p0);

    let x0 = clamp(p0.x, 0, max_coord.x);
    let y0 = clamp(p0.y, 0, max_coord.y);
    let x1 = clamp(p0.x + 1, 0, max_coord.x);
    let y1 = clamp(p0.y + 1, 0, max_coord.y);

    let v00 = textureLoad(tile_tex, vec2<i32>(x0, y0), mip).r;
    let v10 = textureLoad(tile_tex, vec2<i32>(x1, y0), mip).r;
    let v01 = textureLoad(tile_tex, vec2<i32>(x0, y1), mip).r;
    let v11 = textureLoad(tile_tex, vec2<i32>(x1, y1), mip).r;

    let w00 = (1.0 - frac.x) * (1.0 - frac.y);
    let w10 = frac.x * (1.0 - frac.y);
    let w01 = (1.0 - frac.x) * frac.y;
    let w11 = frac.x * frac.y;

    var sum = 0.0;
    var w_sum = 0.0;
    if is_finite_f32(v00) {
        sum += v00 * w00;
        w_sum += w00;
    }
    if is_finite_f32(v10) {
        sum += v10 * w10;
        w_sum += w10;
    }
    if is_finite_f32(v01) {
        sum += v01 * w01;
        w_sum += w01;
    }
    if is_finite_f32(v11) {
        sum += v11 * w11;
        w_sum += w11;
    }

    if w_sum > 0.0 {
        return vec2<f32>(sum / w_sum, w_sum);
    }
    return vec2<f32>(0.0, 0.0);
}

@fragment
fn fs_scientific(in: VsOut) -> @location(0) vec4<f32> {
    // Evaluate derivatives in uniform control flow strictly BEFORE horizon discard.
    let sample_uv = tile.uv_rect.xy + in.uv * tile.uv_rect.zw;
    let dims0 = vec2<f32>(textureDimensions(tile_tex, 0));
    let texel_pos = sample_uv * dims0;
    let dx = dpdx(texel_pos);
    let dy = dpdy(texel_pos);
    let rho_sq = max(dot(dx, dx), dot(dy, dy));
    let max_mip_u = max(tile.mip_count, 1u) - 1u;
    let lod = clamp(0.5 * log2(max(rho_sq, 1e-12)), 0.0, f32(max_mip_u));

    if dot(in.normal, -in.world) <= 0.0 {
        discard;
    }

    let m0 = u32(floor(lod));
    let m1 = min(m0 + 1u, max_mip_u);
    let mfrac = lod - f32(m0);

    let s0 = sample_bilinear_nodata(sample_uv, i32(m0));
    var raw_val = s0.x;
    var valid_cov = s0.y;
    if m1 != m0 {
        let s1 = sample_bilinear_nodata(sample_uv, i32(m1));
        valid_cov = mix(s0.y, s1.y, mfrac);
        if s0.y > 0.0 && s1.y > 0.0 {
            raw_val = mix(s0.x, s1.x, mfrac);
        } else if s1.y > 0.0 {
            raw_val = s1.x;
        }
    }

    // Crisp nodata cutoff at exact texel boundary (`< 0.5` coverage discards).
    if valid_cov < 0.5 {
        discard;
    }

    let physical = fma(raw_val, tile.scale, tile.offset);
    var mapped = physical;
    if tile.stretch_mode == 1u {
        // SAR dB stretch: `10 * log10(x) = 10 * log2(x) / log2(10)`.
        let safe_pos = max(physical, 1e-10);
        mapped = log2(safe_pos) * 3.010299956639812;
    }

    var t = clamp((mapped - tile.stretch_min) * tile.stretch_inv_range, 0.0, 1.0);
    if tile.invert_colormap != 0u {
        t = 1.0 - t;
    }

    let scaled = t * 255.0;
    let i0 = min(u32(floor(scaled)), 255u);
    let i1 = min(i0 + 1u, 255u);
    let f = scaled - f32(i0);
    let lut_color = mix(palette.entries[i0], palette.entries[i1], f);

    let alpha = lut_color.a * tile.opacity;
    if alpha <= 0.001 {
        discard;
    }
    return vec4<f32>(lut_color.rgb, alpha);
}
