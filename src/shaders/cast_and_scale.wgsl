// GPU kernel: cast_and_scale
//
// Multiplies every pixel value by `scale`. One thread per (col, row).
// 2D dispatch avoids the 65535-per-dimension limit of 1D dispatch at 4K.

struct Uniforms {
    scale:    f32,
    width:    u32,
    height:   u32,
    channels: u32,
}

@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform>             u   : Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= u.width || row >= u.height { return; }

    let row_offset = row * u.width * u.channels;
    for (var c = 0u; c < u.channels; c++) {
        let idx = row_offset + col * u.channels + c;
        dst[idx] = src[idx] * u.scale;
    }
}
