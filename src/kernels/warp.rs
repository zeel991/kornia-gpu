//! GPU kernel: warp_perspective
//!
//! # CPU equivalent : kornia_imgproc::warp::warp_perspective
//!
//! # GPU improvements over CPU path
//!
//! The CPU path calls get_iter_offset_unchecked which computes idx * stride for
//! all 3 tensor dimensions per read:
//!   - 4 reads × 3 channels × 3 dims = 36 mul-adds for address arithmetic alone per pixel
//!   - 3 per-channel boundary branches
//!   - ~H rayon work units (one per row)
//!
//! This kernel uses flat indexing (row * W + col) * C + c:
//!   - 1 mul-add per read
//!   - 1 bounds check for all channels
//!   - H × W independent GPU threads
//!
//! # Homography convention
//!
//! Matches kornia-imgproc exactly: m is the forward homography (src → dst).
//! We invert it once on CPU before uploading — same as inverse_perspective_matrix
//! in perspective.rs.

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use crate::error::GpuError;
use crate::image::GpuImage;

// Homography helpers (CPU-side, called once before kernel launch)

fn determinant3x3(m: &[f32; 9]) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

fn adjugate3x3(m: &[f32; 9]) -> [f32; 9] {
    [
        m[4] * m[8] - m[5] * m[7],
        m[2] * m[7] - m[1] * m[8],
        m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8],
        m[0] * m[8] - m[2] * m[6],
        m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6],
        m[1] * m[6] - m[0] * m[7],
        m[0] * m[4] - m[1] * m[3],
    ]
}

/// Invert a 3×3 perspective matrix. Matches inverse_perspective_matrix in perspective.rs.
fn invert_perspective(m: &[f32; 9]) -> Result<[f32; 9], GpuError> {
    let det = determinant3x3(m);
    if det == 0.0 {
        return Err(GpuError::SingularHomography);
    }
    let adj = adjugate3x3(m);
    let inv_det = 1.0 / det;
    let mut inv = [0.0f32; 9];
    for i in 0..9 {
        inv[i] = adj[i] * inv_det;
    }
    Ok(inv)
}

// CubeCL kernel

/// Perspective warp kernel. One thread per output pixel (u, v).
///
/// Each thread:
///   1. Applies the inverse homography + perspective divide to get (src_x, src_y)
///   2. Bilinear-interpolates the source image at that position
///   3. Writes the result to dst
///
/// Uses flat indexing (row * W + col) * C + c : 1 mul-add per read.
/// Bounds check is done once for all channels (not per-channel like the CPU path).
#[cube(launch)]
fn warp_perspective_kernel(
    src: &Tensor<f32>,
    dst: &mut Tensor<f32>,
    h_inv: &Tensor<f32>, // 9-element row-major inverse homography
    src_h: u32,
    src_w: u32,
    dst_h: u32,
    dst_w: u32,
    channels: u32,
) {
    let u = ABSOLUTE_POS_X;
    let v = ABSOLUTE_POS_Y;

    if u >= dst_w || v >= dst_h {
        terminate!();
    }

    let uf = f32::cast_from(u);
    let vf = f32::cast_from(v);

    // Inverse homography + perspective divide
    let w  = h_inv[6usize] * uf + h_inv[7usize] * vf + h_inv[8usize];
    let sx = (h_inv[0usize] * uf + h_inv[1usize] * vf + h_inv[2usize]) / w;
    let sy = (h_inv[3usize] * uf + h_inv[4usize] * vf + h_inv[5usize]) / w;

    // Bilinear interpolation coordinates
    let x0f = f32::floor(sx);
    let y0f = f32::floor(sy);
    let fx = sx - x0f;
    let fy = sy - y0f;

    let ix0 = i32::cast_from(x0f);
    let iy0 = i32::cast_from(y0f);

    // Bounds check ONCE for all channels — eliminates per-channel branches of CPU path
    let in_bounds = ix0 >= 0
        && iy0 >= 0
        && ix0 + 1 < i32::cast_from(src_w)
        && iy0 + 1 < i32::cast_from(src_h);

    let dst_base = (v * dst_w + u) * channels;

    let mut c: u32 = 0;
    loop {
        if c >= channels { break; }

        let val = if in_bounds {
            // Flat indexing: 1 mul-add per read (vs 36 in CPU get_iter_offset_unchecked)
            let x0 = u32::cast_from(ix0);
            let y0 = u32::cast_from(iy0);
            let base = (y0 * src_w + x0) * channels + c;

            let p00 = src[base as usize];
            let p10 = src[(base + channels) as usize];
            let p01 = src[(base + src_w * channels) as usize];
            let p11 = src[(base + src_w * channels + channels) as usize];

            (1.0 - fx) * (1.0 - fy) * p00
                + fx         * (1.0 - fy) * p10
                + (1.0 - fx) * fy         * p01
                + fx         * fy         * p11
        } else {
            0.0_f32.into()
        };

        dst[(dst_base + c) as usize] = val;
        c += 1u32;
    }
}
/// Apply a perspective transformation to a GPU image with bilinear interpolation.
///
/// Mirrors kornia_imgproc::warp::warp_perspective but runs entirely on the GPU.
/// m is the forward homography (src → dst), same convention as the CPU function.
///
/// # Arguments
///
/// * src      - Input GPU image.
/// * dst_size - Output dimensions (height, width).
/// * m        - 3×3 perspective matrix src → dst (row-major, 9 elements).
///
/// # Returns
///
/// A new GpuImage<f32, C> with shape [dst_height, dst_width, C].

pub fn warp_perspective<const C: usize>(
    src: &GpuImage<f32, C>,
    dst_size: (usize, usize), // (height, width)
    m: &[f32; 9],
) -> Result<GpuImage<f32, C>, GpuError> {
    let (dst_h, dst_w) = dst_size;
    let alloc = src.alloc();

    // Invert homography on CPU once — same as inverse_perspective_matrix in perspective.rs
    let h_inv = invert_perspective(m)?;

    // Upload inverse homography as a flat 1D tensor [9]
    let h_inv_bytes = cubecl::bytes::Bytes::from_elems(h_inv.to_vec());
    let h_inv_handle = alloc.client.create(h_inv_bytes);
    let h_inv_shape = [9usize];
    let h_inv_strides = [1usize];

    let dst = GpuImage::<f32, C>::empty(dst_h, dst_w, alloc);

    // 2D tile launch: one thread per output pixel
    let tile = 16u32;
    let cube_dim = CubeDim { x: tile, y: tile, z: 1 };
    let cube_count = CubeCount::Static(
        (dst_w as u32 + tile - 1) / tile,
        (dst_h as u32 + tile - 1) / tile,
        1,
    );

    let _ = warp_perspective_kernel::launch::<WgpuRuntime>(
        &alloc.client,
        cube_count,
        cube_dim,
        src.mem.as_tensor_arg(&src.shape(), &src.strides(), 1),
        dst.mem.as_tensor_arg(&dst.shape(), &dst.strides(), 1),
        unsafe {
            TensorArg::from_raw_parts::<f32>(&h_inv_handle, &h_inv_strides, &h_inv_shape, 1)
        },
        ScalarArg::new(src.height() as u32),
        ScalarArg::new(src.width() as u32),
        ScalarArg::new(dst_h as u32),
        ScalarArg::new(dst_w as u32),
        ScalarArg::new(C as u32),
    );

    Ok(dst)
}

/// Write warp_perspective result into an existing GPU buffer (zero allocation).
///
/// Use with GpuImagePool for persistent VRAM reuse across frames.
pub fn warp_perspective_into<const C: usize>(
    src: &GpuImage<f32, C>,
    dst: &GpuImage<f32, C>,
    m: &[f32; 9],
) -> Result<(), crate::error::GpuError> {
    let (dst_h, dst_w) = (dst.height(), dst.width());
    let alloc = src.alloc();
    let h_inv = invert_perspective(m)?;
    let h_inv_bytes = cubecl::bytes::Bytes::from_elems(h_inv.to_vec());
    let h_inv_handle = alloc.client.create(h_inv_bytes);
    let h_inv_shape = [9usize];
    let h_inv_strides = [1usize];

    let tile = 16u32;
    let cube_dim = CubeDim { x: tile, y: tile, z: 1 };
    let cube_count = CubeCount::Static(
        (dst_w as u32 + tile - 1) / tile,
        (dst_h as u32 + tile - 1) / tile,
        1,
    );

    let _ = warp_perspective_kernel::launch::<WgpuRuntime>(
        &alloc.client, cube_count, cube_dim,
        src.mem.as_tensor_arg(&src.shape(), &src.strides(), 1),
        dst.mem.as_tensor_arg(&dst.shape(), &dst.strides(), 1),
        unsafe {
            TensorArg::from_raw_parts::<f32>(&h_inv_handle, &h_inv_strides, &h_inv_shape, 1)
        },
        ScalarArg::new(src.height() as u32),
        ScalarArg::new(src.width() as u32),
        ScalarArg::new(dst_h as u32),
        ScalarArg::new(dst_w as u32),
        ScalarArg::new(C as u32),
    );
    Ok(())
}