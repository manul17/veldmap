// Depth-less post-body draped vector geometry shader (GeoJSON polygons,
// HiDPI-scaled screen-space ribbon polylines/outlines, diagonal hatch fills,
// instanced point markers, STAC footprints/AOI, and interactive geodesic
// measurement features) with per-cluster CPU `f64` RTC offsets and smooth
// outward-normal horizon clipping.

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
    viewport_inv_size: vec2<f32>,
    stroke_half_width_px: f32,
    hatch_spacing_px: f32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> vec_u: VectorUniform;

struct VertexIn {
    @location(0) rel_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct RibbonIn {
    @location(0) p0: vec3<f32>,
    @location(1) n0: vec3<f32>,
    @location(2) p1: vec3<f32>,
    @location(3) n1: vec3<f32>,
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
fn vs_ribbon(@builtin(vertex_index) vid: u32, in: RibbonIn) -> VertexOut {
    var out: VertexOut;
    var rel0 = vec_u.offset_eye + in.p0;
    var rel1 = vec_u.offset_eye + in.p1;
    var n0 = in.n0;
    var n1 = in.n1;
    var c0 = camera.view_proj * vec4<f32>(rel0, 1.0);
    var c1 = camera.view_proj * vec4<f32>(rel1, 1.0);

    let min_w = 1e-4;
    if c0.w <= min_w && c1.w <= min_w {
        out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        out.world = rel0;
        out.normal = n0;
        return out;
    }
    if c0.w <= min_w {
        let t = (min_w - c0.w) / max(c1.w - c0.w, 1e-6);
        c0 = mix(c0, c1, t);
        rel0 = mix(rel0, rel1, t);
        n0 = normalize(mix(n0, n1, t));
    } else if c1.w <= min_w {
        let t = (min_w - c1.w) / max(c0.w - c1.w, 1e-6);
        c1 = mix(c1, c0, t);
        rel1 = mix(rel1, rel0, t);
        n1 = normalize(mix(n1, n0, t));
    }

    let ndc0 = c0.xy / c0.w;
    let ndc1 = c1.xy / c1.w;
    let inv_sz = max(vec_u.viewport_inv_size, vec2<f32>(1e-6, 1e-6));
    let px_dir = (ndc1 - ndc0) / inv_sz;
    let px_len = length(px_dir);
    let dir_px = select(vec2<f32>(1.0, 0.0), px_dir / max(px_len, 1e-6), px_len > 1e-5);
    let perp_px = vec2<f32>(-dir_px.y, dir_px.x);
    let offset_ndc = perp_px * max(vec_u.stroke_half_width_px, 0.5) * inv_sz;

    var is_end1 = false;
    var side = -1.0;
    switch vid {
        case 0u: { is_end1 = false; side = -1.0; }
        case 1u: { is_end1 = true; side = -1.0; }
        case 2u: { is_end1 = true; side = 1.0; }
        case 3u: { is_end1 = false; side = -1.0; }
        case 4u: { is_end1 = true; side = 1.0; }
        default: { is_end1 = false; side = 1.0; }
    }

    let base_clip = select(c0, c1, is_end1);
    out.clip_pos = vec4<f32>(
        base_clip.xy + offset_ndc * (side * base_clip.w),
        base_clip.z,
        base_clip.w,
    );
    out.world = select(rel0, rel1, is_end1);
    out.normal = select(n0, n1, is_end1);
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
    if vec_u.hatch_spacing_px > 0.5 {
        let diag = in.clip_pos.x + in.clip_pos.y;
        let phase = fract(diag / vec_u.hatch_spacing_px);
        let stripe = select(0.28, 1.0, phase < 0.24);
        return vec4<f32>(vec_u.color.rgb, vec_u.color.a * stripe);
    }
    return vec_u.color;
}
