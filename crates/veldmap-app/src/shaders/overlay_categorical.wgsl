// Depth-less post-body categorical (`R8Uint` / `R16Uint`) raster overlay shader:
// - Shares `texture_2d<u32>` across 8-bit and 16-bit discrete class / QA bands
// - Exact nearest-neighbor `textureLoad(tile_tex, coord, 0).r` (never blending class IDs)
// - Sag-safe ellipsoidal horizon facing clip (`dot(in.normal, -in.world) <= 0.0`)
// - Exact nodata class discard and 256-entry linear-light RGBA palette lookup.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    grid_step_deg: f32,
    grid_width_px: f32,
    polar_r_cap: f32,
    _pad0: f32,
};

struct CategoricalPatchUniform {
    /// `(patch_origin_ecef_m - camera_eye_ecef_m) as f32` computed in f64 on CPU.
    offset_eye: vec3<f32>,
    /// Layer opacity multiplier in `[0, 1]`.
    opacity: f32,
    /// Carrier UV sub-rect `(u_offset, v_offset, u_scale, v_scale)`.
    uv_rect: vec4<f32>,
    /// Non-zero when `nodata_value` should be discarded as transparent.
    has_nodata: u32,
    /// Declared discrete nodata class value.
    nodata_value: u32,
    _pad0: u32,
    _pad1: u32,
};

struct PaletteUniform {
    /// 256-entry linear-light RGBA class palette.
    entries: array<vec4<f32>, 256>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> tile: CategoricalPatchUniform;
@group(2) @binding(0) var tile_tex: texture_2d<u32>;
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

@fragment
fn fs_categorical(in: VsOut) -> @location(0) vec4<f32> {
    if dot(in.normal, -in.world) <= 0.0 {
        discard;
    }

    let sample_uv = tile.uv_rect.xy + in.uv * tile.uv_rect.zw;
    let dims = textureDimensions(tile_tex, 0);
    let max_coord = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);
    let coord = clamp(
        vec2<i32>(floor(sample_uv * vec2<f32>(dims))),
        vec2<i32>(0, 0),
        max_coord,
    );
    let class_id = textureLoad(tile_tex, coord, 0).r;

    if tile.has_nodata != 0u && class_id == tile.nodata_value {
        discard;
    }

    var idx = class_id;
    if class_id >= 256u {
        // Deterministic non-zero slot hash for R16Uint class/QA values >= 256.
        idx = ((class_id * 2654435761u) >> 24u) % 255u + 1u;
    }
    let color = palette.entries[idx];
    let alpha = color.a * tile.opacity;
    if alpha <= 0.001 {
        discard;
    }
    return vec4<f32>(color.rgb, alpha);
}
