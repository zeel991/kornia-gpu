// GPU kernel: gray_from_rgb
//
// BT.601 luminance:  gray = 0.299*R + 0.587*G + 0.114*B
// Input:  array<f32> interleaved RGB, layout [H, W, 3]
// Output: array<f32> single channel,  layout [H, W, 1]

struct Uniforms {
    h:    u32,
    w:    u32,
    _p0:  u32,
    _p1:  u32,
}

@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform>             u   : Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= u.w || row >= u.h { return; }

    let base = (row * u.w + col) * 3u;
    let r = src[base];
    let g = src[base + 1u];
    let b = src[base + 2u];
    dst[row * u.w + col] = 0.299 * r + 0.587 * g + 0.114 * b;
}
