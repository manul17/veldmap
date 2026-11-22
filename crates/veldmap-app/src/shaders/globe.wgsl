// Precision Nadir WGS84 Quadtree Globe shader:
// - Per-tile CPU f64 RTC vertex transforms (zero GPU double-single reassociation)
// - Untextured WGS84 ellipsoid body (`fs_body_ellipsoid`)
// - sRGB textured planetary basemap (`fs_body_textured`)
// - Post-overlay procedural `fwidth()` graticule (`fs_grid`) reusing the exact
//   same quadtree vertex/index buffers, with unconditional uniform-flow
//   `fwidth()` before horizon `discard` and smooth polar meridian fade.

struct CameraUniform {
    /// Rotation-only Reverse-Z infinite perspective view-projection matrix
    /// (operating on eye-relative ECEF coordinates in metres).
    view_proj: mat4x4<f32>,
    /// Adaptive graticule spacing in degrees (evenly divides 90 deg).
    grid_step_deg: f32,
    /// Procedural graticule half-width in physical pixels (HiDPI-scaled).
    grid_width_px: f32,
    /// Polar cap stereographic disk scale `tan((90 - 85.05112878) / 2)`.
    polar_r_cap: f32,
    _pad0: f32,
};

struct TileUniform {
    /// `(tile_origin_ecef_m - camera_eye_ecef_m) as f32` computed in f64 on CPU.
    offset_eye: vec3<f32>,
    /// `0.0` = LatLon (`Geodetic`/`WebMercator`),
    /// `±1.0` = North/South PolarCap near/touching the pole,
    /// `±2.0` = North/South PolarCap strictly clear of the pole (enables
    /// cancellation-free `f64`-wrapped relative longitude phase at 2 m zoom).
    tile_kind: f32,
    /// Ancestor UV sub-rect: `(u_offset, v_offset, u_scale, v_scale)`.
    uv_rect: vec4<f32>,
    /// Graticule anchor in `f32`:
    /// - LatLon: `(lon_phase_deg, lat_phase_deg, anchor_lat_deg, 0.0)` where
    ///   `*_phase_deg = rem_euclid(anchor_deg, grid_step_deg)` in `f64` on CPU.
    /// - PolarCap: `(anchor_dx, anchor_dy, colat_phase_deg, lon_phase_deg)`.
    grid_anchor: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> tile: TileUniform;
@group(2) @binding(0) var tile_tex: texture_2d<f32>;
@group(2) @binding(1) var tile_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    /// Position relative to camera eye (`P - E`, metres).
    @location(0) world: vec3<f32>,
    /// Outward WGS84 ellipsoidal normal `N_ellipsoid(P)`.
    @location(1) normal: vec3<f32>,
    /// Tile-local texture coordinates in `[0, 1]^2`.
    @location(2) uv: vec2<f32>,
    /// Tile-relative graticule coordinates (`(dlon, dlat)` or `(dx, dy)`).
    @location(3) grid_rel: vec2<f32>,
};

@vertex
fn vs_main(
    @location(0) rel_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) grid_rel: vec2<f32>,
) -> VsOut {
    var out: VsOut;
    let rel = tile.offset_eye + rel_pos;
    out.clip = camera.view_proj * vec4<f32>(rel, 1.0);
    out.world = rel;
    out.normal = normal;
    out.uv = uv;
    out.grid_rel = grid_rel;
    return out;
}

/// Ellipsoidal horizon facing cosine: `dot(N_ellipsoid(P), normalize(E - P))`.
fn horizon_facing(in: VsOut) -> f32 {
    let to_eye = normalize(-in.world);
    return dot(normalize(in.normal), to_eye);
}

// Fixed directional sun in ECEF coordinates (used only in Ellipsoid-only mode).
const LIGHT: vec3<f32> = vec3<f32>(0.42, 0.60, 0.68);

// Exact IEC 61966-2-1 linear RGB palette constants.
const LIT_LINEAR: vec3<f32> = vec3<f32>(0.964686, 0.938686, 0.879623);     // #FBF8F1
const SHADE_LINEAR: vec3<f32> = vec3<f32>(0.651406, 0.584078, 0.423268);   // #D3C9AE
const ACCENT_LINEAR: vec3<f32> = vec3<f32>(0.068478, 0.194618, 0.049707);  // #4A7A3F
const GRID_LINEAR: vec3<f32> = vec3<f32>(0.391572, 0.485150, 0.327778);    // #A8B99B

@fragment
fn fs_body_ellipsoid(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let lambert = clamp(dot(n, normalize(LIGHT)), 0.0, 1.0);
    let body = mix(SHADE_LINEAR, LIT_LINEAR, 0.45 + 0.55 * lambert);

    let facing = clamp(horizon_facing(in), 0.0, 1.0);
    let inv_facing = 1.0 - facing;
    let rim = inv_facing * inv_facing * inv_facing;
    return vec4<f32>(mix(body, ACCENT_LINEAR, rim * 0.85), 1.0);
}

@fragment
fn fs_body_textured(in: VsOut) -> @location(0) vec4<f32> {
    let sample_uv = tile.uv_rect.xy + in.uv * tile.uv_rect.zw;
    let color = textureSample(tile_tex, tile_sampler, sample_uv);
    return vec4<f32>(color.rgb, 1.0);
}

fn draped_line_alpha(facing: f32) -> f32 {
    return smoothstep(0.0, 0.30, facing);
}

@fragment
fn fs_grid(in: VsOut) -> @location(0) vec4<f32> {
    let step = max(camera.grid_step_deg, 1e-6);

    // 1. LatLon tile coordinates: CPU f64-wrapped phase + f32 tile-relative delta.
    let ll_lon = (tile.grid_anchor.x + in.grid_rel.x) / step;
    let ll_lat = (tile.grid_anchor.y + in.grid_rel.y) / step;

    // 2. PolarCap tile coordinates: cancellation-free deltas relative to tile
    //    center `a = tile.grid_anchor.xy` + CPU f64-wrapped phase in `zw`.
    let a = tile.grid_anchor.xy;
    let d = in.grid_rel.xy;
    let ps = a + d;
    let rho = length(ps);
    let rho0 = length(a);
    let r_cap = camera.polar_r_cap;
    let drho = (2.0 * dot(a, d) + dot(d, d)) / max(rho + rho0, 1e-12);
    let d_colat_deg = degrees(2.0 * atan((drho * r_cap) / (1.0 + rho * rho0 * r_cap * r_cap)));
    let pc_lat = (tile.grid_anchor.z + d_colat_deg) / step;

    // Guard `atan2(0, 0)` at exact poles to prevent NaN propagation on DX12/HLSL.
    let is_north_cap = tile.tile_kind > 0.0;
    let sy = select(ps.y, -ps.y, is_north_cap);
    let safe_x = select(ps.x, 1.0, rho < 1e-12);
    let safe_sy = select(sy, 0.0, rho < 1e-12);
    let pc_abs_lon_a = degrees(atan2(safe_sy, safe_x)) / step;
    let pc_abs_lon_b = degrees(atan2(-safe_sy, -safe_x)) / step;

    // Signed longitude delta from `a` to `a + d` for PolarCap tiles strictly
    // clear of the pole (`abs(tile_kind) > 1.5`), avoiding f32 ULP collapse at 2 m.
    let a_sy = select(a.y, -a.y, is_north_cap);
    let d_sy = select(d.y, -d.y, is_north_cap);
    let rel_num = a.x * d_sy - a_sy * d.x;
    let rel_den = max(dot(a, a) + dot(a, d), 1e-12);
    let d_lon_deg = degrees(atan2(rel_num, rel_den));
    let pc_rel_lon = (tile.grid_anchor.w + d_lon_deg) / step;

    let use_pc_rel_lon = abs(tile.tile_kind) > 1.5;
    let pc_lon_a = select(pc_abs_lon_a, pc_rel_lon, use_pc_rel_lon);
    let pc_lon_b = select(pc_abs_lon_b, pc_rel_lon, use_pc_rel_lon);

    let is_polar = tile.tile_kind != 0.0;
    let lat_cycles = select(ll_lat, pc_lat, is_polar);
    let lon_a = select(ll_lon, pc_lon_a, is_polar);
    let lon_b = select(ll_lon, pc_lon_b, is_polar);

    // Compute ALL derivatives unconditionally in uniform control flow BEFORE
    // any horizon discard so `naga` validation succeeds across SPV/HLSL/MSL.
    let dw_lat = fwidth(lat_cycles);
    let dw_lon = min(fwidth(lon_a), fwidth(lon_b));

    let facing = horizon_facing(in);
    if facing <= 0.0 {
        discard;
    }

    let half_w = max(camera.grid_width_px, 0.5);
    let dist_lat = abs(fract(lat_cycles + 0.5) - 0.5) / max(dw_lat, 1e-6);
    let dist_lon = abs(fract(lon_a + 0.5) - 0.5) / max(dw_lon, 1e-6);

    let line_lat = 1.0 - smoothstep(half_w - 0.5, half_w + 0.5, dist_lat);
    let line_lon_raw = 1.0 - smoothstep(half_w - 0.5, half_w + 0.5, dist_lon);

    // Smooth polar meridian fade (`edge0 < edge1` strictly per WGSL/Metal/HLSL spec):
    // cleanly suppresses meridian convergence and moire near ±90 deg while
    // keeping fine zoomed-in meridians sharp.
    let colat_deg = degrees(2.0 * atan(rho * r_cap));
    let abs_lat = select(
        abs(tile.grid_anchor.z + in.grid_rel.y),
        90.0 - colat_deg,
        is_polar,
    );
    let pole_fade = 1.0 - smoothstep(86.0, 89.95, abs_lat);
    let density_fade = 1.0 - smoothstep(0.12, 0.45, dw_lon);
    let line_lon = line_lon_raw * pole_fade * density_fade;

    let line = max(line_lat, line_lon);
    let alpha = line * draped_line_alpha(facing);
    if alpha <= 0.001 {
        discard;
    }
    return vec4<f32>(GRID_LINEAR, alpha);
}
