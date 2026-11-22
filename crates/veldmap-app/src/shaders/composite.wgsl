// 1:1 nearest-neighbor composite shader from the offscreen Rgba8UnormSrgb
// globe target into egui's render pass.
//
// The bound texture view matches the swapchain color space via `view_formats`:
//   - `Rgba8UnormSrgb` view when the target expects linear output (sRGB or float targets)
//   - `Rgba8Unorm` view for non-sRGB 8-bit targets
// This preserves exact sRGB pixel values with zero manual gamma encoding.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    let x = f32((idx << 1u) & 2u);
    let y = f32(idx & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@group(0) @binding(0) var offscreen_tex: texture_2d<f32>;
@group(0) @binding(1) var offscreen_sampler: sampler;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(offscreen_tex, offscreen_sampler, in.uv);
}
