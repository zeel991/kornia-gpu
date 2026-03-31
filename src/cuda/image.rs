#![cfg(feature = "cuda")]
//! CudaImage using cudarc 0.19 CudaSlice<f32>.

use cudarc::driver::CudaSlice;
use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;

use crate::cuda::allocator::CudaAllocator;
use crate::error::GpuError;

pub struct CudaImage<const C: usize> {
    pub(crate) slice: CudaSlice<f32>,
    pub(crate) height: usize,
    pub(crate) width: usize,
    pub(crate) alloc: CudaAllocator,
}

impl<const C: usize> CudaImage<C> {
    pub fn empty(height: usize, width: usize, alloc: &CudaAllocator) -> Result<Self, GpuError> {
        let n = height * width * C;
        let slice = alloc
            .stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| GpuError::CudaError(e.to_string()))?;
        Ok(Self {
            slice,
            height,
            width,
            alloc: alloc.clone(),
        })
    }

    pub fn height(&self) -> usize {
        self.height
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn numel(&self) -> usize {
        self.height * self.width * C
    }
    pub fn size(&self) -> ImageSize {
        ImageSize {
            width: self.width,
            height: self.height,
        }
    }
    pub fn alloc(&self) -> &CudaAllocator {
        &self.alloc
    }

    pub fn to_cpu(&self) -> Result<Image<f32, C, CpuAllocator>, GpuError> {
        let data = self
            .alloc
            .stream
            .clone_dtoh(&self.slice)
            .map_err(|e| GpuError::CudaError(e.to_string()))?;
        Image::new(self.size(), data, CpuAllocator).map_err(GpuError::ImageError)
    }
}

pub trait CudaImageExt<const C: usize> {
    fn to_cuda(&self, alloc: &CudaAllocator) -> Result<CudaImage<C>, GpuError>;
}

impl<const C: usize> CudaImageExt<C> for Image<f32, C, CpuAllocator> {
    fn to_cuda(&self, alloc: &CudaAllocator) -> Result<CudaImage<C>, GpuError> {
        let slice = alloc
            .stream
            .clone_htod(self.as_slice())
            .map_err(|e| GpuError::CudaError(e.to_string()))?;
        Ok(CudaImage {
            slice,
            height: self.height(),
            width: self.width(),
            alloc: alloc.clone(),
        })
    }
}
