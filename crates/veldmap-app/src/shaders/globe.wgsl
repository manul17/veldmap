// Precision Nadir WGS84 Globe shader: double-single RTE ellipsoid body and
// depth-less draped graticule with exact ellipsoidal-normal horizon discard.
//
// All palette constants are exact IEC 61966-2-1 linear RGB values rendered into
// an Rgba8UnormSrgb target (no manual encode_srgb hacks).

struct CameraUniform {
    /// Rotation-only Reverse-Z infinite perspective view-projection matrix.
    view_proj: mat4x4<f32>,
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

/// Compute vertex position relative to camera eye with high-half cancellation
/// (Sterbenz's lemma) plus low-half sub-millimetre remainder.
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
