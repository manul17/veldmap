// Depth-less post-body WGS84 raster overlay warp-mesh shader:
// - Per-cell CPU f64 RTC vertex transforms (`offset_eye + rel_pos`)
// - Sag-safe ellipsoidal horizon facing clip (`dot(in.normal, -in.world) <= 0.0`)
//   matching `globe.wgsl::fs_grid`, guaranteeing zero holes beneath the nadir
//   camera at 2 m altitude even on coarse/global cells while clipping back-hemisphere
//   limb fragments
// - Uniform-flow `textureSample` strictly before any fragment `discard`
// - Alpha-blended sRGB display overlays over the opaque globe body.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    grid_step_deg: f32,
    grid_width_px: f32,
    polar_r_cap: f32,
    _pad0: f32,
};

struct OverlayPatchUniform {
    /// `(patch_origin_ecef_m - camera_eye_ecef_m) as f32` computed in f64 on CPU.
    offset_eye: vec3<f32>,
    /// Layer opacity multiplier in `[0, 1]`.
    opacity: f32,
    /// Carrier UV sub-rect `(u_offset, v_offset, u_scale, v_scale)`.
    uv_rect: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> tile: OverlayPatchUniform;
@group(2) @binding(0) var tile_tex: texture_2d<f32>;
@group(2) @binding(1) var tile_sampler: sampler;

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
fn fs_overlay(in: VsOut) -> @location(0) vec4<f32> {
    // Evaluate texture sampling in uniform control flow strictly before any
    // horizon or transparency `discard`.
    let sample_uv = tile.uv_rect.xy + in.uv * tile.uv_rect.zw;
    let color = textureSample(tile_tex, tile_sampler, sample_uv);

    if dot(in.normal, -in.world) <= 0.0 {
        discard;
    }

    let alpha = color.a * tile.opacity;
    if alpha <= 0.001 {
        discard;
    }
    return vec4<f32>(color.rgb, alpha);
}
