//! GPU-backed image type and CPU ↔ GPU transfer API.
//!
//! GpuImage<T, C> mirrors kornia_image::Image<T, C, CpuAllocator> in shape
//! and channel conventions but stores its data in VRAM via a GpuMemory<T>.
//!
//! Layout: row-major, channels interleaved - pixel (row, col) channel c lives at
//!   data[(row  width + col)  C + c]
//!
//! This is the same layout as kornia_image::Image (shape [H, W, C], strides
//! [W*C, C, 1]), so CPU and GPU buffers can be compared element-for-element.

use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;

use crate::allocator::{GpuAllocator, GpuMemory};
use crate::error::GpuError;

/// GPU-backed image with compile-time channel count.
///
/// GpuImage<T, C> is the GPU equivalent of Image<T, C, CpuAllocator>.
/// It owns a GpuMemory<T> which holds the wgpu buffer handle in VRAM.
///
/// Shape:
///
/// [height, width, C] - matches kornia_image::Image layout exactly.
/// Strides: [width * C, C, 1].
pub struct GpuImage<T, const C: usize> {
    pub(crate) mem: GpuMemory<T>,
    pub(crate) height: usize,
    pub(crate) width: usize,
}

impl<T: bytemuck::Pod + Send + Sync, const C: usize> GpuImage<T, C> {
    /// Allocate a zero-initialised GPU image with given dimensions.
    pub fn empty(height: usize, width: usize, alloc: &GpuAllocator) -> Self {
        let n = height * width * C;
        let size = (n * std::mem::size_of::<T>()) as u64;
        let buffer = alloc.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            mem: GpuMemory {
                buffer: std::sync::Arc::new(buffer),
                len: n,
                alloc: alloc.clone(),
                _marker: std::marker::PhantomData,
            },
            height,
            width,
        }
    }

    pub fn size(&self) -> ImageSize {
        ImageSize {
            width: self.width,
            height: self.height,
        }
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn num_channels(&self) -> usize {
        C
    }
    pub fn numel(&self) -> usize {
        self.height * self.width * C
    }
    pub fn row_stride(&self) -> usize {
        self.width * C
    }
    pub fn shape(&self) -> [usize; 3] {
        [self.height, self.width, C]
    }
    pub fn strides(&self) -> [usize; 3] {
        [self.width * C, C, 1]
    }
    pub fn alloc(&self) -> &GpuAllocator {
        &self.mem.alloc
    }
}

// ── Transfer API ──────────────────────────────────────────────────────────────

/// Extension trait adding to_gpu() to kornia_image::Image.
pub trait ImageExt<T: bytemuck::Pod + Send + Sync, const C: usize> {
    fn to_gpu(&self, alloc: &GpuAllocator) -> Result<GpuImage<T, C>, GpuError>;
}

impl<T: bytemuck::Pod + Send + Sync, const C: usize> ImageExt<T, C> for Image<T, C, CpuAllocator> {
    fn to_gpu(&self, alloc: &GpuAllocator) -> Result<GpuImage<T, C>, GpuError> {
        let mem = GpuMemory::upload(self.as_slice(), alloc);
        Ok(GpuImage {
            mem,
            height: self.height(),
            width: self.width(),
        })
    }
}

impl<T: bytemuck::Pod + Send + Sync, const C: usize> GpuImage<T, C> {
    /// Download this image back to the CPU.
    ///
    /// Blocks the calling thread until the GPU finishes and data is copied
    /// back over PCIe.
    pub fn to_cpu(&self) -> Result<Image<T, C, CpuAllocator>, GpuError> {
        let data = self.mem.download();
        Image::new(self.size(), data, CpuAllocator).map_err(GpuError::ImageError)
    }
}
