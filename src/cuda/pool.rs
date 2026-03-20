#![cfg(feature = "cuda")]
//! CudaImagePool: persistent CUDA device memory, zero per-frame allocation.

use std::sync::{Arc, Mutex};
use crate::cuda::allocator::CudaAllocator;
use crate::cuda::image::CudaImage;
use crate::error::GpuError;

pub struct CudaImagePool<const C: usize> {
    available: Arc<Mutex<Vec<CudaImage<C>>>>,
}

impl<const C: usize> CudaImagePool<C> {
    pub fn new(capacity: usize, height: usize, width: usize, alloc: &CudaAllocator)
        -> Result<Self, GpuError>
    {
        let mut buffers = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffers.push(CudaImage::<C>::empty(height, width, alloc)?);
        }
        Ok(Self { available: Arc::new(Mutex::new(buffers)) })
    }

    pub fn acquire(&self) -> Result<CudaImage<C>, GpuError> {
        self.available.lock().unwrap().pop().ok_or(GpuError::PoolExhausted)
    }

    pub fn release(&self, buf: CudaImage<C>) {
        self.available.lock().unwrap().push(buf);
    }

    pub fn available_count(&self) -> usize {
        self.available.lock().unwrap().len()
    }
}
