//! Persistent VRAM buffer pool for reusable GPU image allocations.

use std::sync::{Arc, Mutex};

use crate::allocator::GpuAllocator;
use crate::error::GpuError;
use crate::image::GpuImage;

/// A pool of pre-allocated GPU image buffers.
///
/// All buffers in the pool share the same dimensions and channel count.
/// This is appropriate for a fixed-resolution pipeline (e.g. all frames
/// are 1920×1080 RGB).
pub struct GpuImagePool<T: bytemuck::Pod + Send + Sync, const C: usize> {
    /// Available (not currently in use) buffers.
    available: Arc<Mutex<Vec<GpuImage<T, C>>>>,
}

impl<T: bytemuck::Pod + Send + Sync, const C: usize> GpuImagePool<T, C> {
    /// Create a pool with capacity pre-allocated buffers of size height × width.
    ///
    /// All VRAM allocations happen here, at startup - not per-frame.
    pub fn new(
        capacity: usize,
        height: usize,
        width: usize,
        alloc: &GpuAllocator,
    ) -> Result<Self, GpuError> {
        let mut buffers = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffers.push(GpuImage::<T, C>::empty(height, width, alloc));
        }
        Ok(Self {
            available: Arc::new(Mutex::new(buffers)),
        })
    }

    /// Acquire a buffer from the pool.
    ///
    /// Returns Err(GpuError::PoolExhausted) if no buffers are available.
    /// The caller must release() the buffer when done or it is permanently removed.
    pub fn acquire(&self) -> Result<GpuImage<T, C>, GpuError> {
        self.available
            .lock()
            .unwrap()
            .pop()
            .ok_or(GpuError::PoolExhausted)
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&self, buf: GpuImage<T, C>) {
        self.available.lock().unwrap().push(buf);
    }

    /// Number of buffers currently available (not acquired).
    pub fn available_count(&self) -> usize {
        self.available.lock().unwrap().len()
    }
}