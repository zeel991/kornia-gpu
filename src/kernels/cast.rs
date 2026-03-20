//! GPU kernel: cast_and_scale
//!
//! Multiplies every pixel value by scale. One thread per element
//!
//! This is the simplest possible GPU kernel - it validates the full
//! GpuAllocator → CubeCL → result stack and sets the pattern for more
//! complex kernels (warp_perspective, gray_from_rgb, gaussian_blur)
//!
//! CPU equivalent: Image::cast_and_scale in kornia-image/src/image.rs
//! (uses rayon par_iter; this uses one GPU thread per element)

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use crate::error::GpuError;
use crate::image::GpuImage;

// CubeCL kernel

/// Per-element multiply kernel. 2D dispatch: one unit per (row, col) pair.
/// Using 2D avoids the wgpu CubeCount limit of 65535 per dimension,
/// which a 1D dispatch hits at ~16M elements (e.g. 4K RGB).
#[cube(launch)]
fn cast_and_scale_kernel(
    input: &Tensor<f32>,
    output: &mut Tensor<f32>,
    scale: f32,
    width: u32,
    channels: u32,
) {
    let col = ABSOLUTE_POS_X;
    let row = ABSOLUTE_POS_Y;

    if col >= width || row >= input.shape(0) as u32 {
        terminate!();
    }

    let row_offset = row * width * channels;
    let mut c: u32 = 0;
    loop {
        if c >= channels { break; }
        let idx = (row_offset + col * channels + c) as usize;
        output[idx] = input[idx] * scale;
        c += 1u32;
    }
}

// Public launch function

/// Multiply every pixel value by scale, writing results to a new GpuImage.
///
/// Mirrors Image::cast_and_scale but runs on the GPU.
///
/// # Arguments
///
/// * src   - Input GPU image.
/// * scale - Scale factor (e.g. 1.0 / 255.0 to normalise u8→f32 range)
///
/// # Returns
///
/// A new GpuImage<f32, C> with the same dimensions as src
pub fn cast_and_scale<const C: usize>(
    src: &GpuImage<f32, C>,
    scale: f32,
) -> Result<GpuImage<f32, C>, GpuError> {
    let alloc = src.alloc();
    let dst = GpuImage::<f32, C>::empty(src.height(), src.width(), alloc);

    let tile = 16u32;
    let cube_dim = CubeDim { x: tile, y: tile, z: 1 };
    let cube_count = CubeCount::Static(
        (src.width() as u32 + tile - 1) / tile,
        (src.height() as u32 + tile - 1) / tile,
        1,
    );

    let _ = cast_and_scale_kernel::launch::<WgpuRuntime>(
        &alloc.client,
        cube_count,
        cube_dim,
        src.mem.as_tensor_arg(&src.shape(), &src.strides(), 1),
        dst.mem.as_tensor_arg(&dst.shape(), &dst.strides(), 1),
        ScalarArg::new(scale),
        ScalarArg::new(src.width() as u32),
        ScalarArg::new(C as u32),
    );

    Ok(dst)
}

/// Write cast_and_scale result into an existing GPU buffer (zero allocation)
///
/// Use with GpuImagePool for persistent VRAM reuse across frames
pub fn cast_and_scale_into<const C: usize>(
    src: &GpuImage<f32, C>,
    dst: &GpuImage<f32, C>,
    scale: f32,
) -> Result<(), crate::error::GpuError> {
    use crate::error::GpuError;
    if dst.height() != src.height() || dst.width() != src.width() {
        return Err(GpuError::BufferSizeMismatch(
            src.height(), src.width(), dst.height(), dst.width(),
        ));
    }
    let alloc = src.alloc();
    let tile = 16u32;
    let cube_dim = CubeDim { x: tile, y: tile, z: 1 };
    let cube_count = CubeCount::Static(
        (src.width() as u32 + tile - 1) / tile,
        (src.height() as u32 + tile - 1) / tile,
        1,
    );
    let _ = cast_and_scale_kernel::launch::<WgpuRuntime>(
        &alloc.client, cube_count, cube_dim,
        src.mem.as_tensor_arg(&src.shape(), &src.strides(), 1),
        dst.mem.as_tensor_arg(&dst.shape(), &dst.strides(), 1),
        ScalarArg::new(scale),
        ScalarArg::new(src.width() as u32),
        ScalarArg::new(C as u32),
    );
    Ok(())
}