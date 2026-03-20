// gray_from_rgb: BT.601 luminance weights.
// Matches kornia_imgproc::color::gray_from_rgb exactly.
extern "C" __global__ void gray_from_rgb(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int height,
    unsigned int width
) {
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;

    unsigned int src_base = (v * width + u) * 3;
    float r = __ldg(&src[src_base]);
    float g = __ldg(&src[src_base + 1]);
    float b = __ldg(&src[src_base + 2]);

    dst[v * width + u] = 0.299f * r + 0.587f * g + 0.114f * b;
}
