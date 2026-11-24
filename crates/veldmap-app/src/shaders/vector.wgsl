// Depth-less post-body draped vector geometry shader (GeoJSON polygons/lines
// and interactive geodesic measurement features) with per-cluster CPU `f64`
// RTC offsets and smooth outward-normal horizon clipping.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    grid_step_deg: f32,
    grid_width_px: f32,
    polar_r_cap: f32,
    _pad0: f32,
};

struct VectorUniform {
    offset_eye: vec3<f32>,
    _pad0: f32,
    color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> vec_u: VectorUniform;

struct VertexIn {
    @location(0) rel_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let rel = vec_u.offset_eye + in.rel_pos;
    out.clip_pos = camera.view_proj * vec4<f32>(rel, 1.0);
    out.world = rel;
    out.normal = in.normal;
    return out;
}

@fragment
fn fs_vector(in: VertexOut) -> @location(0) vec4<f32> {
    if dot(in.normal, -in.world) <= 0.0 {
        discard;
    }
    if vec_u.color.a <= 0.001 {
        discard;
    }
    return vec_u.color;
}
