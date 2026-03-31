//! Hardware-aware GPU backend dispatch.
//!
//! Backend::auto() selects the best available backend:
//! --features cuda + NVIDIA GPU present → CUDA (13× faster kernel)
//! Everything else → wgpu/Vulkan (cross-platform, any GPU)
//!
//! Usage:
//!
//! // Automatically picks the best backend - no flags needed at call site
//! let backend = Backend::auto()?;
//! println!("Using: {}", backend.name());
//!
//! let gpu_img = backend.upload(&cpu_img)?;
//! let warped  = backend.warp_perspective(&gpu_img, (h, w), &H)?;
//! let result  = backend.download(&warped)?;
//!
//! Feature flags:
//!
//! Default build (cross-platform, no NVIDIA requirement):
//! toml
//! kornia-gpu = { path = "..." }
//!
//!
//! With CUDA support (requires nvcc + NVIDIA GPU at runtime):
//! toml
//! kornia-gpu = { path = "...", features = ["cuda"] }
//!

use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;

use crate::allocator::GpuAllocator;
use crate::error::GpuError;
use crate::image::{GpuImage, ImageExt};

#[cfg(feature = "cuda")]
use crate::cuda::allocator::CudaAllocator;
#[cfg(feature = "cuda")]
use crate::cuda::image::{CudaImage, CudaImageExt};

// AnyGpuImage - backend-erased image handle

pub enum AnyGpuImage<const C: usize> {
    Wgpu(GpuImage<f32, C>),
    #[cfg(feature = "cuda")]
    Cuda(CudaImage<C>),
}

impl<const C: usize> AnyGpuImage<C> {
    pub fn height(&self) -> usize {
        match self {
            AnyGpuImage::Wgpu(img) => img.height(),
            #[cfg(feature = "cuda")]
            AnyGpuImage::Cuda(img) => img.height(),
        }
    }
    pub fn width(&self) -> usize {
        match self {
            AnyGpuImage::Wgpu(img) => img.width(),
            #[cfg(feature = "cuda")]
            AnyGpuImage::Cuda(img) => img.width(),
        }
    }
    pub fn size(&self) -> ImageSize {
        ImageSize {
            width: self.width(),
            height: self.height(),
        }
    }
}

// Backend

pub enum Backend {
    Wgpu(GpuAllocator),
    #[cfg(feature = "cuda")]
    Cuda(CudaAllocator),
}

impl Backend {
    /// Auto-select the best available backend.
    ///
    /// With --features cuda: tries CUDA first, falls back to wgpu if no
    /// NVIDIA GPU is present or driver is not installed.
    /// Without --features cuda: always uses wgpu (cross-platform default).
    pub fn auto() -> Result<Self, GpuError> {
        #[cfg(feature = "cuda")]
        {
            match CudaAllocator::new() {
                Ok(alloc) => {
                    eprintln!("[kornia-gpu] backend: CUDA (NVIDIA GPU detected)");
                    return Ok(Backend::Cuda(alloc));
                }
                Err(e) => {
                    eprintln!(
                        "[kornia-gpu] backend: wgpu/Vulkan (CUDA unavailable: {})",
                        e
                    );
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        eprintln!("[kornia-gpu] backend: wgpu/Vulkan");
        Ok(Backend::Wgpu(GpuAllocator::new()))
    }

    /// Force wgpu/Vulkan backend (cross-platform, works without CUDA).
    pub fn wgpu() -> Self {
        Backend::Wgpu(GpuAllocator::new())
    }

    /// Force CUDA backend.
    /// Only available with --features cuda. Errors if no NVIDIA GPU present.
    #[cfg(feature = "cuda")]
    pub fn cuda() -> Result<Self, GpuError> {
        Ok(Backend::Cuda(CudaAllocator::new()?))
    }

    pub fn name(&self) -> &'static str {
        match self {
            Backend::Wgpu(_) => "wgpu/Vulkan",
            #[cfg(feature = "cuda")]
            Backend::Cuda(_) => "CUDA",
        }
    }

    // -----------------------------------------------------------------------
    // Transfer
    // -----------------------------------------------------------------------

    pub fn upload<const C: usize>(
        &self,
        img: &Image<f32, C, CpuAllocator>,
    ) -> Result<AnyGpuImage<C>, GpuError> {
        match self {
            Backend::Wgpu(alloc) => Ok(AnyGpuImage::Wgpu(img.to_gpu(alloc)?)),
            #[cfg(feature = "cuda")]
            Backend::Cuda(alloc) => Ok(AnyGpuImage::Cuda(img.to_cuda(alloc)?)),
        }
    }

    pub fn download<const C: usize>(
        &self,
        img: &AnyGpuImage<C>,
    ) -> Result<Image<f32, C, CpuAllocator>, GpuError> {
        match img {
            AnyGpuImage::Wgpu(img) => img.to_cpu(),
            #[cfg(feature = "cuda")]
            AnyGpuImage::Cuda(img) => img.to_cpu(),
        }
    }

    // -----------------------------------------------------------------------
    // Kernels
    // -----------------------------------------------------------------------

    pub fn warp_perspective<const C: usize>(
        &self,
        src: &AnyGpuImage<C>,
        dst_size: (usize, usize),
        m: &[f32; 9],
    ) -> Result<AnyGpuImage<C>, GpuError> {
        match src {
            AnyGpuImage::Wgpu(img) => Ok(AnyGpuImage::Wgpu(crate::kernels::warp_perspective(
                img, dst_size, m,
            )?)),
            #[cfg(feature = "cuda")]
            AnyGpuImage::Cuda(img) => Ok(AnyGpuImage::Cuda(
                crate::cuda::kernels::warp_perspective(img, dst_size, m)?,
            )),
        }
    }

    pub fn cast_and_scale<const C: usize>(
        &self,
        src: &AnyGpuImage<C>,
        scale: f32,
    ) -> Result<AnyGpuImage<C>, GpuError> {
        match src {
            AnyGpuImage::Wgpu(img) => Ok(AnyGpuImage::Wgpu(crate::kernels::cast_and_scale(
                img, scale,
            )?)),
            #[cfg(feature = "cuda")]
            AnyGpuImage::Cuda(img) => Ok(AnyGpuImage::Cuda(crate::cuda::kernels::cast_and_scale(
                img, scale,
            )?)),
        }
    }

    pub fn gray_from_rgb(&self, src: &AnyGpuImage<3>) -> Result<AnyGpuImage<1>, GpuError> {
        match src {
            AnyGpuImage::Wgpu(img) => Ok(AnyGpuImage::Wgpu(crate::kernels::gray_from_rgb(img)?)),
            #[cfg(feature = "cuda")]
            AnyGpuImage::Cuda(img) => {
                Ok(AnyGpuImage::Cuda(crate::cuda::kernels::gray_from_rgb(img)?))
            }
        }
    }
}
