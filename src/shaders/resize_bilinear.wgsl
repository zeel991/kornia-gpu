// GPU kernel: resize_bilinear
//
// Bilinear resize with half-pixel alignment (align_corners = false):
//   src_x = (col + 0.5) * src_w / dst_w - 0.5
//   src_y = (row + 0.5) * src_h / dst_h - 0.5
// Border handling: replicate (clamp to [0, dim-1]).

struct Uniforms {
    src_h:    u32,
    src_w:    u32,
    dst_h:    u32,
    dst_w:    u32,
    channels: u32,
    _p0:      u32,
    _p1:      u32,
    _p2:      u32,
}

@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform>             u   : Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= u.dst_w || row >= u.dst_h { return; }

    let src_x = (f32(col) + 0.5) * f32(u.src_w) / f32(u.dst_w) - 0.5;
    let src_y = (f32(row) + 0.5) * f32(u.src_h) / f32(u.dst_h) - 0.5;

    let x0f = floor(src_x);
    let y0f = floor(src_y);
    let fx  = src_x - x0f;
    let fy  = src_y - y0f;

    let sw = i32(u.src_w);
    let sh = i32(u.src_h);

    let cx0 = u32(clamp(i32(x0f),     0, sw - 1));
    let cx1 = u32(clamp(i32(x0f) + 1, 0, sw - 1));
    let cy0 = u32(clamp(i32(y0f),     0, sh - 1));
    let cy1 = u32(clamp(i32(y0f) + 1, 0, sh - 1));

    let dst_base = (row * u.dst_w + col) * u.channels;

    for (var c = 0u; c < u.channels; c++) {
        let p00 = src[(cy0 * u.src_w + cx0) * u.channels + c];
        let p10 = src[(cy0 * u.src_w + cx1) * u.channels + c];
        let p01 = src[(cy1 * u.src_w + cx0) * u.channels + c];
        let p11 = src[(cy1 * u.src_w + cx1) * u.channels + c];

        dst[dst_base + c] = (1.0 - fx) * (1.0 - fy) * p00
                          +        fx  * (1.0 - fy) * p10
                          + (1.0 - fx) *        fy  * p01
                          +        fx  *        fy  * p11;
    }
}
