#![cfg(feature = "cuda")]
//! CUDA backend for kornia-gpu.
//!
//! Uses cudarc for safe device memory management and PTX kernel loading.
//! Kernels are compiled from .cu files at build time via build.rs.
//!
//! Same public API as the wgpu backend - swap CudaAllocator for GpuAllocator.

pub mod allocator;
pub mod image;
pub mod kernels;
pub mod pool;

pub use allocator::CudaAllocator;
pub use image::{CudaImage, CudaImageExt};
pub use pool::CudaImagePool;
