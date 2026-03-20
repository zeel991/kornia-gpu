// cast_and_scale: multiply every f32 element by scale.
// One thread per element. 2D launch: gridDim.x * blockDim.x covers width,
// gridDim.y * blockDim.y covers height.
extern "C" __global__ void cast_and_scale(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scale,
    unsigned int width,
    unsigned int channels,
    unsigned int height
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    unsigned int base = (row * width + col) * channels;
    for (unsigned int c = 0; c < channels; ++c) {
        output[base + c] = input[base + c] * scale;
    }
}
