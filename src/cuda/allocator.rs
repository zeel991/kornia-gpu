#![cfg(feature = "cuda")]
//! CudaAllocator: wraps cudarc 0.19 CudaContext + CudaStream.
//!
//! Uses cudarc's dynamic-loading feature - libcuda.so is loaded at runtime
//! via dlopen(), so this compiles and runs on non-NVIDIA machines without error.
//! CudaAllocator::new() returns Err if no NVIDIA GPU is present.

use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaStream};
use kornia_tensor::allocator::{TensorAllocator, TensorAllocatorError};
use crate::error::GpuError;

#[derive(Clone)]
pub struct CudaAllocator {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
}

impl CudaAllocator {
    /// Try to initialise CUDA on device 0.
    ///
    /// Returns Err if:
    /// - No NVIDIA GPU is present
    /// - CUDA driver is not installed
    /// - libcuda.so cannot be found (non-NVIDIA machine)
    ///
    /// This is safe to call on any machine - cudarc uses dynamic loading
    /// so the binary links fine without CUDA present.
    pub fn new() -> Result<Self, GpuError> {
        let ctx = CudaContext::new(0)
            .map_err(|e| GpuError::CudaError(e.to_string()))?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream })
    }

    /// Returns true if a CUDA-capable NVIDIA GPU is available on this machine.
    pub fn is_available() -> bool {
        CudaContext::new(0).is_ok()
    }
}

impl TensorAllocator for CudaAllocator {
    fn alloc(&self, layout: std::alloc::Layout) -> Result<*mut u8, TensorAllocatorError> {
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() { Err(TensorAllocatorError::NullPointer) } else { Ok(ptr) }
    }
    fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        if !ptr.is_null() { unsafe { std::alloc::dealloc(ptr, layout) } }
    }
}
