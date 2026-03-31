//! Zero-copy GPU pipeline for chaining image processing operations.
//!
//! The key insight: GPU transfer overhead is fixed (~0.4ms for 1080p),
//! but each additional GPU kernel costs ~0.1-0.3ms. By chaining kernels
//! with data staying in VRAM between stages, we pay the transfer cost once
//! regardless of how many operations are in the pipeline.

use crate::allocator::GpuAllocator;
use crate::error::GpuError;
use crate::image::{GpuImage, ImageExt};
use crate::kernels;

use kornia_image::Image;
use kornia_tensor::CpuAllocator;

/// Entry point for building a GPU pipeline.
pub struct GpuPipeline<'a> {
    alloc: &'a GpuAllocator,
}

impl<'a> GpuPipeline<'a> {
    pub fn new(alloc: &'a GpuAllocator) -> Self {
        Self { alloc }
    }

    /// Upload a CPU image to the GPU, starting the pipeline.
    pub fn upload<T: bytemuck::Pod + Send + Sync, const C: usize>(
        &self,
        img: &Image<T, C, CpuAllocator>,
    ) -> Result<Stage<'a, T, C>, GpuError> {
        let gpu_img = img.to_gpu(self.alloc)?;
        Ok(Stage {
            image: gpu_img,
            alloc: self.alloc,
        })
    }
}

/// An in-flight GPU image. Each method returns the next stage - data stays in VRAM.
pub struct Stage<'a, T: bytemuck::Pod, const C: usize> {
    pub image: GpuImage<T, C>,
    alloc: &'a GpuAllocator,
}

impl<'a> Stage<'a, f32, 3> {
    /// Multiply every pixel by scale. Stays in VRAM.
    pub fn cast_and_scale(self, scale: f32) -> Result<Stage<'a, f32, 3>, GpuError> {
        let out = kernels::cast_and_scale(&self.image, scale)?;
        Ok(Stage {
            image: out,
            alloc: self.alloc,
        })
    }

    /// Apply perspective warp. Stays in VRAM.
    pub fn warp_perspective(
        self,
        dst_size: (usize, usize),
        m: &[f32; 9],
    ) -> Result<Stage<'a, f32, 3>, GpuError> {
        let out = kernels::warp_perspective(&self.image, dst_size, m)?;
        Ok(Stage {
            image: out,
            alloc: self.alloc,
        })
    }

    /// Convert RGB → grayscale. Stays in VRAM.
    pub fn gray_from_rgb(self) -> Result<Stage<'a, f32, 1>, GpuError> {
        let out = kernels::gray_from_rgb(&self.image)?;
        Ok(Stage {
            image: out,
            alloc: self.alloc,
        })
    }

    /// Download to CPU, ending the pipeline.
    pub fn download(self) -> Result<Image<f32, 3, CpuAllocator>, GpuError> {
        self.image.to_cpu()
    }
}

impl<'a> Stage<'a, f32, 1> {
    /// Download to CPU, ending the pipeline.
    pub fn download(self) -> Result<Image<f32, 1, CpuAllocator>, GpuError> {
        self.image.to_cpu()
    }
}
