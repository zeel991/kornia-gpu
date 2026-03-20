//! GPU allocator for kornia-rs.
//!
//! # Design
//!
//! kornia-tensor's TensorAllocator trait returns *mut u8 — a host pointer:
//!
//! pub trait TensorAllocator: Clone {
//!     fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError>;
//!     fn dealloc(&self, ptr: *mut u8, layout: Layout);
//! }
//!
//! GPU memory is not host-addressable, so GpuAllocator cannot implement this
//! trait to actually allocate GPU memory through it. Instead, GpuAllocator
//! implements TensorAllocator by delegating to the system allocator — this
//! allows Image<T, C, GpuAllocator> to exist as a type, while actual GPU
//! memory is managed separately through CubeCL handles stored in GpuMemory.
//!
//! The to_gpu() / to_cpu() transfer functions are the only points where
//! data moves between CPU and GPU. No operation implicitly copies data.
//!
//! # Future integration path
//!
//! When kornia-tensor adds an associated storage type to TensorAllocator,
//! GpuAllocator can hold a CubeCL handle directly in TensorStorage,
//! eliminating the need for the separate GpuMemory wrapper.

use std::alloc::Layout;
use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use kornia_tensor::allocator::{TensorAllocator, TensorAllocatorError};

/// GPU device handle wrapping a CubeCL ComputeClient.
///
/// Clone is cheap — ComputeClient is Arc-based internally.
///
#[derive(Clone)]
pub struct GpuAllocator {
    pub(crate) client: ComputeClient<WgpuRuntime>,
}

impl GpuAllocator {
    /// Create a GpuAllocator on the default wgpu device.
    ///
    /// Uses Vulkan on Linux, Metal on macOS, DX12 on Windows
    /// No CUDA installation required
    pub fn new() -> Self {
        let device: <WgpuRuntime as Runtime>::Device = Default::default();
        let client = WgpuRuntime::client(&device);
        Self { client }
    }

    /// Returns a reference to the underlying CubeCL compute client.
    pub fn client(&self) -> &ComputeClient<WgpuRuntime> {
        &self.client
    }
}

impl Default for GpuAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Implement TensorAllocator so Image<T, C, GpuAllocator> is a valid type.
///
/// This delegates to the system allocator - the CPU-side tensor that carries
/// GpuAllocator acts as a type-level marker. The actual GPU buffer is managed
/// by GpuMemory<T> and accessed via the transfer API (to_gpu / to_cpu)
impl TensorAllocator for GpuAllocator {
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            Err(TensorAllocatorError::NullPointer)
        } else {
            Ok(ptr)
        }
    }

    fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !ptr.is_null() {
            unsafe { std::alloc::dealloc(ptr, layout) }
        }
    }
}

/// Owned GPU buffer: a CubeCL handle + element count.
///
/// This is the actual GPU-side storage. It is separate from TensorStorage
/// because TensorStorage holds a NonNull<T> (host pointer) which cannot
/// represent device memory.
///
/// Created by to_gpu(), consumed by kernels, released on drop
pub struct GpuMemory<T> {
    pub(crate) handle: cubecl::server::Handle,
    pub(crate) len: usize, // element count
    pub(crate) alloc: GpuAllocator,
    pub(crate) _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod + Send + Sync> GpuMemory<T> {
    /// Upload a host slice to VRAM. Returns a GpuMemory<T>.
    pub fn upload(data: &[T], alloc: &GpuAllocator) -> Self {
        let bytes = cubecl::bytes::Bytes::from_elems(data.to_vec());
        let handle = alloc.client.create(bytes);
        Self {
            handle,
            len: data.len(),
            alloc: alloc.clone(),
            _marker: PhantomData,
        }
    }

    /// Download VRAM contents back to a Vec<T>.
    pub fn download(&self) -> Vec<T> {
        let results = self.alloc.client.read(vec![self.handle.clone()]);
        let raw: &[u8] = results[0].as_ref();
        bytemuck::cast_slice(raw).to_vec()
    }

    /// Number of elements stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Build a TensorArg for passing to a CubeCL kernel.
    ///
    /// vectorization - 1 for scalar Tensor<f32>, 4 for Tensor<Line<f32>>.
    pub(crate) fn as_tensor_arg<'a>(
        &'a self,
        shape: &'a [usize],
        strides: &'a [usize],
        vectorization: u8,
    ) -> TensorArg<'a, WgpuRuntime> {
        unsafe {
            TensorArg::from_raw_parts::<f32>(
                &self.handle,
                strides,
                shape,
                vectorization as usize,
            )
        }
    }
}