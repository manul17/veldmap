struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOut {
    let x = f32((idx << 1u) & 2u);
    let y = f32(idx & 2u);
    var out: VertexOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let top = vec3<f32>(0.02, 0.04, 0.09);
    let bottom = vec3<f32>(0.05, 0.09, 0.16);
    let color = mix(top, bottom, clamp(in.uv.y, 0.0, 1.0));
    return vec4<f32>(color, 1.0);
}
