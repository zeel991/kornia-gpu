// warp_perspective: perspective warp with bilinear interpolation.
// One thread per output pixel. h_inv is the 9-element inverse homography (row-major).
//
// Key optimizations vs the CubeCL wgpu version:
//   - __ldg() for read-only texture cache access on source pixels
//   - Bounds check once for all channels (not per-channel)
//   - Flat indexing: (row * W + col) * C + c, 1 mul-add per read
extern "C" __global__ void warp_perspective(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const float* __restrict__ h_inv,
    unsigned int src_h,
    unsigned int src_w,
    unsigned int dst_h,
    unsigned int dst_w,
    unsigned int channels
) {
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= dst_w || v >= dst_h) return;

    float uf = (float)u;
    float vf = (float)v;

    // Inverse homography + perspective divide
    float w  = h_inv[6] * uf + h_inv[7] * vf + h_inv[8];
    float sx = (h_inv[0] * uf + h_inv[1] * vf + h_inv[2]) / w;
    float sy = (h_inv[3] * uf + h_inv[4] * vf + h_inv[5]) / w;

    // Bilinear coords
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    // Bounds check ONCE for all channels
    bool in_bounds = (x0 >= 0 && y0 >= 0 &&
                      x0 + 1 < (int)src_w &&
                      y0 + 1 < (int)src_h);

    unsigned int dst_base = (v * dst_w + u) * channels;

    for (unsigned int c = 0; c < channels; ++c) {
        float val = 0.0f;
        if (in_bounds) {
            // Flat indexing + __ldg() for L1 texture cache
            unsigned int base = ((unsigned int)y0 * src_w + (unsigned int)x0) * channels + c;
            float p00 = __ldg(&src[base]);
            float p10 = __ldg(&src[base + channels]);
            float p01 = __ldg(&src[base + src_w * channels]);
            float p11 = __ldg(&src[base + src_w * channels + channels]);

            val = (1.0f - fx) * (1.0f - fy) * p00
                + fx           * (1.0f - fy) * p10
                + (1.0f - fx) * fy           * p01
                + fx           * fy           * p11;
        }
        dst[dst_base + c] = val;
    }
}
