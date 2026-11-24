// Depth-less post-body draped vector geometry shader (GeoJSON polygons,
// polylines, instanced point markers, and interactive geodesic measurement
// features) with per-cluster CPU `f64` RTC offsets and smooth outward-normal
// horizon clipping.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    grid_step_deg: f32,
    grid_width_px: f32,
    polar_r_cap: f32,
    _pad0: f32,
};

struct VectorUniform {
    offset_eye: vec3<f32>,
    point_radius_m: f32,
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

@vertex
fn vs_point(@builtin(vertex_index) vid: u32, in: VertexIn) -> VertexOut {
    let n = normalize(in.normal);
    let ref_axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.9);
    let ex = normalize(cross(ref_axis, n));
    let ey = normalize(cross(n, ex));
    var corner: vec3<f32>;
    switch vid {
        case 0u, 3u: { corner = ex; }
        case 1u: { corner = ey; }
        case 2u, 4u: { corner = -ex; }
        default: { corner = -ey; }
    }
    let center_rel = vec_u.offset_eye + in.rel_pos;
    let pos_rel = center_rel + corner * vec_u.point_radius_m;
    var out: VertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(pos_rel, 1.0);
    // Horizon test uses the marker's true surface center so tangent corners
    // never leak past the ellipsoid silhouette at orbital zoom.
    out.world = center_rel;
    out.normal = n;
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
