// Fullscreen quiet-NaN scratch target initializer for 1..=4 `R32Float` targets.
//
// Reads IEEE-754 quiet NaN (`0x7FC0_0000u`) from a runtime uniform rather than
// a WGSL const-expression (which `naga` rejects at compile time).

struct ScratchClearUniform {
    nan_bits: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> clear_u: ScratchClearUniform;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_clear_1() -> @location(0) f32 {
    return bitcast<f32>(clear_u.nan_bits);
}

struct ClearOut2 {
    @location(0) b0: f32,
    @location(1) b1: f32,
};

@fragment
fn fs_clear_2() -> ClearOut2 {
    let q = bitcast<f32>(clear_u.nan_bits);
    return ClearOut2(q, q);
}

struct ClearOut3 {
    @location(0) b0: f32,
    @location(1) b1: f32,
    @location(2) b2: f32,
};

@fragment
fn fs_clear_3() -> ClearOut3 {
    let q = bitcast<f32>(clear_u.nan_bits);
    return ClearOut3(q, q, q);
}

struct ClearOut4 {
    @location(0) b0: f32,
    @location(1) b1: f32,
    @location(2) b2: f32,
    @location(3) b3: f32,
};

@fragment
fn fs_clear_4() -> ClearOut4 {
    let q = bitcast<f32>(clear_u.nan_bits);
    return ClearOut4(q, q, q, q);
}
