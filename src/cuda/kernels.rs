//! CUDA kernel launchers via cudarc 0.19 PTX loading.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::cuda::image::CudaImage;
use crate::error::GpuError;

#[cfg(cuda_kernels_compiled)]
static CAST_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cast_and_scale.ptx"));
#[cfg(cuda_kernels_compiled)]
static WARP_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/warp_perspective.ptx"));
#[cfg(cuda_kernels_compiled)]
static GRAY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/gray_from_rgb.ptx"));

#[cfg(not(cuda_kernels_compiled))]
static CAST_PTX: &str = "";
#[cfg(not(cuda_kernels_compiled))]
static WARP_PTX: &str = "";
#[cfg(not(cuda_kernels_compiled))]
static GRAY_PTX: &str = "";

fn tile_config(w: usize, h: usize) -> LaunchConfig {
    let tile = 16u32;
    LaunchConfig {
        grid_dim: ((w as u32).div_ceil(tile), (h as u32).div_ceil(tile), 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    }
}

fn invert(m: &[f32; 9]) -> Result<[f32; 9], GpuError> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-10 {
        return Err(GpuError::SingularHomography);
    }
    let d = 1.0 / det;
    Ok([
        (m[4] * m[8] - m[5] * m[7]) * d,
        (m[2] * m[7] - m[1] * m[8]) * d,
        (m[1] * m[5] - m[2] * m[4]) * d,
        (m[5] * m[6] - m[3] * m[8]) * d,
        (m[0] * m[8] - m[2] * m[6]) * d,
        (m[2] * m[3] - m[0] * m[5]) * d,
        (m[3] * m[7] - m[4] * m[6]) * d,
        (m[1] * m[6] - m[0] * m[7]) * d,
        (m[0] * m[4] - m[1] * m[3]) * d,
    ])
}

pub fn cast_and_scale<const C: usize>(
    src: &CudaImage<C>,
    scale: f32,
) -> Result<CudaImage<C>, GpuError> {
    let alloc = src.alloc();
    let module = alloc
        .ctx
        .load_module(CAST_PTX.into())
        .map_err(|e| GpuError::CudaError(e.to_string()))?;
    let f = module
        .load_function("cast_and_scale")
        .map_err(|e| GpuError::CudaError(e.to_string()))?;

    let dst = CudaImage::<C>::empty(src.height(), src.width(), alloc)?;
    let cfg = tile_config(src.width(), src.height());

    let w = src.width() as u32;
    let c = C as u32;
    let h = src.height() as u32;

    let mut builder = alloc.stream.launch_builder(&f);
    builder.arg(&src.slice);
    builder.arg(&dst.slice);
    builder.arg(&scale);
    builder.arg(&w);
    builder.arg(&c);
    builder.arg(&h);
    unsafe { builder.launch(cfg) }.map_err(|e| GpuError::CudaError(e.to_string()))?;
    Ok(dst)
}

pub fn warp_perspective<const C: usize>(
    src: &CudaImage<C>,
    dst_size: (usize, usize),
    m: &[f32; 9],
) -> Result<CudaImage<C>, GpuError> {
    let (dst_h, dst_w) = dst_size;
    let alloc = src.alloc();
    let h_inv = invert(m)?;

    let module = alloc
        .ctx
        .load_module(WARP_PTX.into())
        .map_err(|e| GpuError::CudaError(e.to_string()))?;
    let f = module
        .load_function("warp_perspective")
        .map_err(|e| GpuError::CudaError(e.to_string()))?;

    let h_inv_dev = alloc
        .stream
        .clone_htod(&h_inv)
        .map_err(|e| GpuError::CudaError(e.to_string()))?;
    let dst = CudaImage::<C>::empty(dst_h, dst_w, alloc)?;
    let cfg = tile_config(dst_w, dst_h);

    let src_h = src.height() as u32;
    let src_w = src.width() as u32;
    let dh = dst_h as u32;
    let dw = dst_w as u32;
    let ch = C as u32;

    let mut builder = alloc.stream.launch_builder(&f);
    builder.arg(&src.slice);
    builder.arg(&dst.slice);
    builder.arg(&h_inv_dev);
    builder.arg(&src_h);
    builder.arg(&src_w);
    builder.arg(&dh);
    builder.arg(&dw);
    builder.arg(&ch);
    unsafe { builder.launch(cfg) }.map_err(|e| GpuError::CudaError(e.to_string()))?;
    Ok(dst)
}

pub fn gray_from_rgb(src: &CudaImage<3>) -> Result<CudaImage<1>, GpuError> {
    let alloc = src.alloc();
    let module = alloc
        .ctx
        .load_module(GRAY_PTX.into())
        .map_err(|e| GpuError::CudaError(e.to_string()))?;
    let f = module
        .load_function("gray_from_rgb")
        .map_err(|e| GpuError::CudaError(e.to_string()))?;

    let dst = CudaImage::<1>::empty(src.height(), src.width(), alloc)?;
    let cfg = tile_config(src.width(), src.height());

    let h = src.height() as u32;
    let w = src.width() as u32;

    let mut builder = alloc.stream.launch_builder(&f);
    builder.arg(&src.slice);
    builder.arg(&dst.slice);
    builder.arg(&h);
    builder.arg(&w);
    unsafe { builder.launch(cfg) }.map_err(|e| GpuError::CudaError(e.to_string()))?;
    Ok(dst)
}
