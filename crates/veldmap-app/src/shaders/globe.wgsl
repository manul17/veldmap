// Precision Nadir WGS84 Globe shader: octant-RTC ellipsoid body + depth-less
// draped graticule, outlines, screen-space ribbons, and pending-scene hatch.
//
// All palette constants are exact IEC 61966-2-1 linear RGB values rendered into
// an Rgba8UnormSrgb target (no manual encode_srgb hacks).

struct CameraUniform {
    /// Rotation-only Reverse-Z infinite perspective view-projection matrix.
    view_proj: mat4x4<f32>,
    /// Offscreen render target size in physical pixels.
    viewport_px: vec2<f32>,
    /// Display scale factor (`pixels_per_point`) for HiDPI-scaled screen widths.
    pixels_per_point: f32,
    _pad: f32,
};

struct ChunkUniform {
    /// High half of `(chunk_origin_ecef - camera_eye_ecef)` computed in f64 on CPU.
    origin_minus_eye_hi: vec3<f32>,
    _pad0: f32,
    /// Low half of `(chunk_origin_ecef - camera_eye_ecef)` computed in f64 on CPU.
    origin_minus_eye_lo: vec3<f32>,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> chunk: ChunkUniform;

/// Compute vertex position relative to camera eye with exact high-half
/// cancellation (Sterbenz's lemma) plus low-half sub-millimetre remainder.
/// Parentheses are load-bearing: never add high + low before combining pairs.
fn relative_to_eye(pos_hi: vec3<f32>, pos_lo: vec3<f32>) -> vec3<f32> {
    return (pos_hi + chunk.origin_minus_eye_hi) + (pos_lo + chunk.origin_minus_eye_lo);
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    /// Position relative to camera eye (`P - E`).
    @location(0) world: vec3<f32>,
    /// Outward WGS84 ellipsoidal normal `N_ellipsoid(P)`.
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) pos_hi: vec3<f32>,
    @location(1) pos_lo: vec3<f32>,
    @location(2) normal: vec3<f32>,
) -> VsOut {
    var out: VsOut;
    let rel = relative_to_eye(pos_hi, pos_lo);
    out.clip = camera.view_proj * vec4<f32>(rel, 1.0);
    out.world = rel;
    out.normal = normal;
    return out;
}

/// Ellipsoidal horizon facing cosine: `dot(N_ellipsoid(P), normalize(E - P))`.
/// Positive on the visible front hemisphere of the convex WGS84 ellipsoid,
/// negative behind the horizon.
fn horizon_facing(in: VsOut) -> f32 {
    let to_eye = normalize(-in.world);
    return dot(normalize(in.normal), to_eye);
}

// Fixed directional sun in ECEF coordinates.
const LIGHT: vec3<f32> = vec3<f32>(0.42, 0.60, 0.68);

// Exact IEC 61966-2-1 linear RGB palette constants.
const LIT_LINEAR: vec3<f32> = vec3<f32>(0.964686, 0.938686, 0.879623);     // #FBF8F1
const SHADE_LINEAR: vec3<f32> = vec3<f32>(0.651406, 0.584078, 0.423268);   // #D3C9AE
const ACCENT_LINEAR: vec3<f32> = vec3<f32>(0.068478, 0.194618, 0.049707);  // #4A7A3F
const GRID_LINEAR: vec3<f32> = vec3<f32>(0.391572, 0.485150, 0.327778);    // #A8B99B
const OUTLINE_LINEAR: vec3<f32> = vec3<f32>(0.107023, 0.093059, 0.064803); // #5C5648

@fragment
fn fs_body(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let lambert = clamp(dot(n, normalize(LIGHT)), 0.0, 1.0);
    let body = mix(SHADE_LINEAR, LIT_LINEAR, 0.45 + 0.55 * lambert);

    let facing = clamp(horizon_facing(in), 0.0, 1.0);
    let rim = pow(1.0 - facing, 3.0);
    return vec4<f32>(mix(body, ACCENT_LINEAR, rim * 0.85), 1.0);
}

fn draped_line_alpha(facing: f32) -> f32 {
    return smoothstep(0.0, 0.30, facing);
}

@fragment
fn fs_grid(in: VsOut) -> @location(0) vec4<f32> {
    let facing = horizon_facing(in);
    if facing <= 0.0 {
        discard;
    }
    return vec4<f32>(GRID_LINEAR, draped_line_alpha(facing));
}

@fragment
fn fs_outline(in: VsOut) -> @location(0) vec4<f32> {
    let facing = horizon_facing(in);
    if facing <= 0.0 {
        discard;
    }
    return vec4<f32>(OUTLINE_LINEAR, draped_line_alpha(facing));
}

@fragment
fn fs_outline_dim(in: VsOut) -> @location(0) vec4<f32> {
    let facing = horizon_facing(in);
    if facing <= 0.0 {
        discard;
    }
    return vec4<f32>(OUTLINE_LINEAR, draped_line_alpha(facing) * 0.35);
}

// --- Screen-space widened selected outline ribbon (HiDPI-scaled) ---

/// Base ribbon width in logical points (scaled by `camera.pixels_per_point`).
const RIBBON_PX: f32 = 4.0;
const MITER_LIMIT: f32 = 3.0;

fn to_screen(clip: vec4<f32>) -> vec2<f32> {
    return clip.xy / clip.w * camera.viewport_px * 0.5;
}

fn along_2d(tail: vec2<f32>, head: vec2<f32>) -> vec2<f32> {
    let delta = head - tail;
    let len = length(delta);
    if len < 1e-5 {
        return vec2<f32>(1.0, 0.0);
    }
    return delta / len;
}

@vertex
fn vs_ribbon(
    @location(0) pos_hi: vec3<f32>,
    @location(1) pos_lo: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) prev: vec3<f32>,
    @location(4) next: vec3<f32>,
    @location(5) side: f32,
) -> VsOut {
    var out: VsOut;
    let at_eye = relative_to_eye(pos_hi, pos_lo);
    out.world = at_eye;
    out.normal = normal;

    let clip = camera.view_proj * vec4<f32>(at_eye, 1.0);
    let clip_prev = camera.view_proj * vec4<f32>(at_eye + prev, 1.0);
    let clip_next = camera.view_proj * vec4<f32>(at_eye + next, 1.0);
    out.clip = clip;
    if clip.w <= 0.0 || clip_prev.w <= 0.0 || clip_next.w <= 0.0 {
        return out;
    }

    let here = to_screen(clip);
    let back = along_2d(to_screen(clip_prev), here);
    let ahead = along_2d(here, to_screen(clip_next));

    let n_back = vec2<f32>(-back.y, back.x);
    let n_ahead = vec2<f32>(-ahead.y, ahead.x);
    var miter = n_back + n_ahead;
    let len = length(miter);
    if len < 1e-5 {
        miter = n_ahead;
    } else {
        miter = miter / len;
    }
    let stretch = 1.0 / max(dot(miter, n_ahead), 1.0 / MITER_LIMIT);
    let width_px = RIBBON_PX * max(camera.pixels_per_point, 1.0);
    let offset = miter * side * width_px * 0.5 * stretch;

    out.clip = vec4<f32>(
        clip.xy + offset / (camera.viewport_px * 0.5) * clip.w,
        clip.z,
        clip.w,
    );
    return out;
}

@fragment
fn fs_ribbon(in: VsOut) -> @location(0) vec4<f32> {
    let facing = horizon_facing(in);
    if facing <= 0.0 {
        discard;
    }
    return vec4<f32>(ACCENT_LINEAR, draped_line_alpha(facing));
}

// --- Pending-scene diagonal hatch (HiDPI-scaled) ---

/// Base hatching period in logical points (scaled by `camera.pixels_per_point`).
const HATCH_PX: f32 = 12.0;

@fragment
fn fs_hatch(in: VsOut) -> @location(0) vec4<f32> {
    let facing = horizon_facing(in);
    if facing <= 0.0 {
        discard;
    }
    let step_px = HATCH_PX * max(camera.pixels_per_point, 1.0);
    if fract((in.clip.x + in.clip.y) / step_px) > 0.5 {
        discard;
    }
    return vec4<f32>(OUTLINE_LINEAR, draped_line_alpha(facing) * 0.5);
}
