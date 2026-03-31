use std::alloc::Layout;
use std::marker::PhantomData;
use std::sync::Arc;

use kornia_tensor::allocator::{TensorAllocator, TensorAllocatorError};
use wgpu::util::DeviceExt;

// ── Pre-compiled pipelines ────────────────────────────────────────────────────

/// All four compute pipelines compiled once at GpuAllocator::new().
pub struct GpuPipelines {
    /// Shared bind group layout used by every kernel:
    ///   binding 0 - storage read-only  (input)
    ///   binding 1 - storage read-write (output)
    ///   binding 2 - uniform            (kernel params)
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub cast_and_scale: wgpu::ComputePipeline,
    pub gray_from_rgb: wgpu::ComputePipeline,
    pub warp_perspective: wgpu::ComputePipeline,
    pub resize_bilinear: wgpu::ComputePipeline,
}

impl GpuPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kornia-gpu bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let cast_and_scale = Self::compile(
            device,
            &pipeline_layout,
            include_str!("shaders/cast_and_scale.wgsl"),
            "cast_and_scale",
        );
        let gray_from_rgb = Self::compile(
            device,
            &pipeline_layout,
            include_str!("shaders/gray_from_rgb.wgsl"),
            "gray_from_rgb",
        );
        let warp_perspective = Self::compile(
            device,
            &pipeline_layout,
            include_str!("shaders/warp_perspective.wgsl"),
            "warp_perspective",
        );
        let resize_bilinear = Self::compile(
            device,
            &pipeline_layout,
            include_str!("shaders/resize_bilinear.wgsl"),
            "resize_bilinear",
        );

        Self {
            bind_group_layout: bgl,
            cast_and_scale,
            gray_from_rgb,
            warp_perspective,
            resize_bilinear,
        }
    }

    fn compile(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        source: &str,
        label: &str,
    ) -> wgpu::ComputePipeline {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module: &module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }
}

// ── GpuAllocator ─────────────────────────────────────────────────────────────

/// wgpu-backed allocator implementing [kornia_tensor::TensorAllocator].
///
/// Holds the wgpu device, queue, and pre-compiled compute pipelines.
/// Clone is cheap - all fields are Arc-wrapped.
///
/// Intended upstream path: kornia-tensor gains a GpuAllocator alongside
/// the existing CpuAllocator, making Image<T, C, GpuAllocator> a valid type.
#[derive(Clone)]
pub struct GpuAllocator {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) pipelines: Arc<GpuPipelines>,
}

impl GpuAllocator {
    /// Initialise wgpu, select the best available GPU, compile all pipelines.
    ///
    /// Uses Vulkan on Linux, Metal on macOS, DX12 on Windows.
    /// No CUDA installation required.
    pub fn new() -> Self {
        pollster::block_on(Self::init_async())
    }

    async fn init_async() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("[kornia-gpu] No suitable GPU adapter found");

        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .await
            .expect("[kornia-gpu] Failed to create wgpu device");

        let pipelines = Arc::new(GpuPipelines::new(&device));

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipelines,
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Shared wgpu device handle.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Shared wgpu queue handle.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Pre-compiled compute pipelines (all 4 kernels).
    pub fn pipelines(&self) -> &Arc<GpuPipelines> {
        &self.pipelines
    }
}

impl Default for GpuAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Implement TensorAllocator so Image<T, C, GpuAllocator> is a valid type.
///
/// Delegates to the system allocator.  The CPU-side tensor is only a shape
/// carrier; actual GPU data lives in GpuMemory<T>.
impl TensorAllocator for GpuAllocator {
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            Err(TensorAllocatorError::NullPointer)
        } else {
            Ok(ptr)
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !ptr.is_null() {
            unsafe { std::alloc::dealloc(ptr, layout) }
        }
    }
}

// ── GpuMemory ─────────────────────────────────────────────────────────────────

/// Owned VRAM buffer.
///
/// Created by upload() / GpuImage::empty().
/// Released when dropped (wgpu handles the lifetime).
pub struct GpuMemory<T> {
    pub(crate) buffer: Arc<wgpu::Buffer>,
    pub(crate) len: usize, // element count (not bytes)
    pub(crate) alloc: GpuAllocator,
    pub(crate) _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod + Send + Sync> GpuMemory<T> {
    /// Upload a host slice to VRAM.
    pub fn upload(data: &[T], alloc: &GpuAllocator) -> Self {
        let buffer = alloc
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        Self {
            buffer: Arc::new(buffer),
            len: data.len(),
            alloc: alloc.clone(),
            _marker: PhantomData,
        }
    }

    /// Download VRAM contents to a Vec<T>.
    ///
    /// Creates a temporary staging buffer, copies, maps, reads back, then
    /// unmaps.  Blocks the calling thread until the GPU transfer completes.
    pub fn download(&self) -> Vec<T> {
        let byte_len = (self.len * std::mem::size_of::<T>()) as u64;

        let staging = self.alloc.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .alloc
            .device
            .create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, byte_len);
        self.alloc.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.alloc.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("map_async channel closed")
            .expect("buffer map failed");

        let view = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, T>(&view).to_vec();
        drop(view);
        staging.unmap();
        result
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
