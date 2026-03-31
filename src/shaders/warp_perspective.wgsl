// GPU kernel: warp_perspective
//
// Applies an inverse homography + perspective divide, then bilinear-interpolates.
// One thread per output pixel. Border pixels (src coord out of range) are zeroed.
//
// h_inv_r0/r1/r2 are the rows of the inverse homography matrix (9 values, packed
// as 3 vec4s with the .w component unused).

struct Uniforms {
    h_inv_r0: vec4<f32>,  // [h[0], h[1], h[2], 0]
    h_inv_r1: vec4<f32>,  // [h[3], h[4], h[5], 0]
    h_inv_r2: vec4<f32>,  // [h[6], h[7], h[8], 0]
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

    let uf = f32(col);
    let vf = f32(row);

    // Inverse homography + perspective divide
    let w  = u.h_inv_r2.x * uf + u.h_inv_r2.y * vf + u.h_inv_r2.z;
    let sx = (u.h_inv_r0.x * uf + u.h_inv_r0.y * vf + u.h_inv_r0.z) / w;
    let sy = (u.h_inv_r1.x * uf + u.h_inv_r1.y * vf + u.h_inv_r1.z) / w;

    let x0f = floor(sx);
    let y0f = floor(sy);
    let fx  = sx - x0f;
    let fy  = sy - y0f;

    let ix0 = i32(x0f);
    let iy0 = i32(y0f);

    let in_bounds = ix0 >= 0 && iy0 >= 0
                 && ix0 + 1 < i32(u.src_w) && iy0 + 1 < i32(u.src_h);

    let dst_base = (row * u.dst_w + col) * u.channels;

    for (var c = 0u; c < u.channels; c++) {
        var val = 0.0f;
        if in_bounds {
            let x0   = u32(ix0);
            let y0   = u32(iy0);
            let base = (y0 * u.src_w + x0) * u.channels + c;
            let p00  = src[base];
            let p10  = src[base + u.channels];
            let p01  = src[base + u.src_w * u.channels];
            let p11  = src[base + u.src_w * u.channels + u.channels];
            val = (1.0 - fx) * (1.0 - fy) * p00
                +        fx  * (1.0 - fy) * p10
                + (1.0 - fx) *        fy  * p01
                +        fx  *        fy  * p11;
        }
        dst[dst_base + c] = val;
    }
}
