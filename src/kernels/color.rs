//! GPU kernel: gray_from_rgb
//!
//! Converts an RGB image to grayscale using the standard luminance weights.
//!
//! CPU equivalent: kornia_imgproc::color::gray_from_rgb
//! (uses rayon par_iter over rows).
//!
//! Weights match the CPU implementation:
//!   gray = 0.299 * R + 0.587 * G + 0.114 * B

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use crate::error::GpuError;
use crate::image::GpuImage;

// CubeCL kernel

/// One thread per output pixel. Reads 3 contiguous f32 values (R, G, B),
/// writes 1 f32 (gray) at the same spatial location.
#[cube(launch)]
fn gray_from_rgb_kernel(
    src: &Tensor<f32>,  // [H, W, 3]
    dst: &mut Tensor<f32>, // [H, W, 1]
    h: u32,
    w: u32,
) {
    let u = ABSOLUTE_POS_X;
    let v = ABSOLUTE_POS_Y;

    if u >= w || v >= h {
        terminate!();
    }

    // Flat src index — 3 channels, interleaved
    let src_base = (v * w + u) * 3u32;
    let r = src[src_base as usize];
    let g = src[(src_base + 1u32) as usize];
    let b = src[(src_base + 2u32) as usize];

    // BT.601 luminance weights — matches kornia_imgproc::color::gray_from_rgb
    let gray = 0.299_f32 * r + 0.587_f32 * g + 0.114_f32 * b;

    let dst_idx = (v * w + u) as usize;
    dst[dst_idx] = gray;
}

/// Convert an RGB GPU image to grayscale.
///
/// Mirrors kornia_imgproc::color::gray_from_rgb but runs on the GPU.
///
/// # Arguments
///
/// * src - Input GPU image with 3 channels (RGB, f32).
///
/// # Returns
///
/// A new GpuImage<f32, 1> (single-channel grayscale).
pub fn gray_from_rgb(src: &GpuImage<f32, 3>) -> Result<GpuImage<f32, 1>, GpuError> {
    let alloc = src.alloc();
    let (h, w) = (src.height(), src.width());
    let dst = GpuImage::<f32, 1>::empty(h, w, alloc);

    let tile = 16u32;
    let cube_dim = CubeDim { x: tile, y: tile, z: 1 };
    let cube_count = CubeCount::Static(
        (w as u32 + tile - 1) / tile,
        (h as u32 + tile - 1) / tile,
        1,
    );

    let _ = gray_from_rgb_kernel::launch::<WgpuRuntime>(
        &alloc.client,
        cube_count,
        cube_dim,
        src.mem.as_tensor_arg(&src.shape(), &src.strides(), 1),
        dst.mem.as_tensor_arg(&dst.shape(), &dst.strides(), 1),
        ScalarArg::new(h as u32),
        ScalarArg::new(w as u32),
    );

    Ok(dst)
}
