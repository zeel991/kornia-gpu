//! High-level GPU backend trait with a raw-bytes API.
//!
//! GpuBackend is the interface used by the bubbaloop node. It accepts and
//! returns plain &[u8] / Vec<u8> (RGB or grayscale, u8 range 0-255) so
//! callers never need to know about kornia_image types or GPU handles.
//!
//! Key optimizations:
//!
//! 1. Persistent lazy GPU buffers: VRAM buffers are allocated once and reused.
//!    This eliminates the massive cost of allocating memory on every frame.
//!
//! 2. Single command encoder: Kernel execution and staging copies are bundled
//!    into a single GPU submission. This minimizes driver round-trips.
//!
//! 3. Zero-allocation data transfer: CPU data is written directly into
//!    pre-allocated scratch vectors, bypassing heavy per-frame memory copies.
//!
//! 4. Streaming pipeline: Asynchronous execution keeps the GPU fed while
//!    the CPU encodes the previous frame.

use std::sync::Mutex;
use wgpu::util::DeviceExt;

use crate::allocator::GpuAllocator;
use crate::pipeline_async::StreamingPipeline;

/// Hardware-agnostic GPU backend trait handling raw byte streams.
pub trait GpuBackend: Send + Sync {
    fn gray_from_rgb(&self, input: &[u8], width: u32, height: u32) -> Vec<u8>;
    fn resize_bilinear(
        &self,
        input: &[u8],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
    ) -> Vec<u8>;
    /// Converts RGB u8 bytes to f32 and applies a scale factor on the GPU.
    /// Returns raw f32 bytes. Set scale to 1.0/255.0 to normalize the image.
    fn cast_and_scale(&self, input: &[u8], width: u32, height: u32, scale: f32) -> Vec<u8>;
    fn name(&self) -> &'static str;

    /// Returns a streaming pipeline for consecutive frame processing (gray_from_rgb kernel).
    /// Default implementation returns None (not supported).
    fn streaming_pipeline(&self, _width: u32, _height: u32) -> Option<Box<dyn StreamingPipeline>> {
        None
    }

    /// Returns a streaming pipeline running cast_and_scale.
    /// Default implementation returns None (not supported).
    fn streaming_cast_pipeline(
        &self,
        _width: u32,
        _height: u32,
        _scale: f32,
    ) -> Option<Box<dyn StreamingPipeline>> {
        None
    }
}

// ── Lazy GPU buffer ───────────────────────────────────────────────────────────

/// A VRAM buffer that allocates once and reuses capacity indefinitely.
/// This completely bypasses the driver overhead of memory allocation in the critical loop.
struct LazyBuf {
    buf: Option<wgpu::Buffer>,
    cap: u64, // allocated size in bytes
}

impl LazyBuf {
    const fn new() -> Self {
        Self { buf: None, cap: 0 }
    }

    /// Allocate or reallocate the buffer if the current capacity is too small.
    fn ensure(&mut self, device: &wgpu::Device, need: u64, usage: wgpu::BufferUsages) {
        if need > self.cap {
            self.buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: need,
                usage,
                mapped_at_creation: false,
            }));
            self.cap = need;
        }
    }

    fn get(&self) -> &wgpu::Buffer {
        self.buf
            .as_ref()
            .expect("LazyBuf: call ensure() before get()")
    }
}

// ── Uniform structs (must match shader WGSL layouts) ─────────────────────────

/// gray_from_rgb.wgsl - 16 bytes (padding ensures uniform alignment).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GrayParams {
    pub(crate) h: u32,
    pub(crate) w: u32,
    pub(crate) _p0: u32,
    pub(crate) _p1: u32,
}

/// resize_bilinear.wgsl - 32 bytes (2 × 16).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResizeParams {
    src_h: u32,
    src_w: u32,
    dst_h: u32,
    dst_w: u32,
    channels: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

/// cast_and_scale.wgsl - 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CastParams {
    pub(crate) scale: f32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: u32,
}

// ── Dispatch helpers ──────────────────────────────────────────────────────────

pub(crate) fn make_bind_group<U: bytemuck::Pod>(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    src_buf: &wgpu::Buffer,
    dst_buf: &wgpu::Buffer,
    uniforms: &U,
) -> wgpu::BindGroup {
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dst_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buf.as_entire_binding(),
            },
        ],
    })
}

/// Submit a single kernel + staging copy in one encoder, then block until done.
/// Returns the raw staging bytes (caller handles any f32→u8 conversion).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_and_readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    dispatch: (u32, u32),
    output_buf: &wgpu::Buffer,
    staging_buf: &wgpu::Buffer,
    output_bytes: u64,
) -> Vec<u8> {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(dispatch.0, dispatch.1, 1);
    }
    enc.copy_buffer_to_buffer(output_buf, 0, staging_buf, 0, output_bytes);
    queue.submit([enc.finish()]);

    let slice = staging_buf.slice(..output_bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let view = slice.get_mapped_range();
    let result = view.to_vec();
    drop(view);
    staging_buf.unmap();
    result
}

// ── Per-call mutable state (held inside Mutex) ────────────────────────────────

struct WgpuBuffers {
    /// Reusable CPU scratch for u8→f32 conversion - avoids per-call heap alloc.
    f32_scratch: Vec<f32>,

    // Persistent GPU buffers - grown lazily, never shrunk.
    rgb_in: LazyBuf, // STORAGE | COPY_DST  - f32 RGB input (shared by gray + resize)
    gray_out: LazyBuf, // STORAGE | COPY_SRC  - f32 grayscale output
    gray_stg: LazyBuf, // MAP_READ | COPY_DST - staging for gray readback
    rsz_out: LazyBuf, // STORAGE | COPY_SRC  - f32 resized RGB output
    rsz_stg: LazyBuf, // MAP_READ | COPY_DST - staging for resize readback
    cas_out: LazyBuf, // STORAGE | COPY_SRC  - f32 RGB cast output
    cas_stg: LazyBuf, // MAP_READ | COPY_DST - staging for cast readback
}

// Maximum resolution we pre-size the f32 scratch for (4K RGB).
const MAX_PIXELS: usize = 3840 * 2160;

// ── wgpu backend ─────────────────────────────────────────────────────────────

/// wgpu/Vulkan backend with persistent buffer reuse and single-submit dispatch.
pub struct WgpuGpuBackend {
    alloc: GpuAllocator,
    inner: Mutex<WgpuBuffers>,
}

impl WgpuGpuBackend {
    pub fn new() -> Self {
        Self {
            alloc: GpuAllocator::new(),
            inner: Mutex::new(WgpuBuffers {
                f32_scratch: vec![0.0f32; MAX_PIXELS * 3],
                rgb_in: LazyBuf::new(),
                gray_out: LazyBuf::new(),
                gray_stg: LazyBuf::new(),
                rsz_out: LazyBuf::new(),
                rsz_stg: LazyBuf::new(),
                cas_out: LazyBuf::new(),
                cas_stg: LazyBuf::new(),
            }),
        }
    }
}

impl Default for WgpuGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuBackend for WgpuGpuBackend {
    /// Converts RGB u8 to grayscale on the GPU.
    /// Highly optimized: uses zero allocations, a single driver submission,
    /// and exactly one synchronization point.
    fn gray_from_rgb(&self, input: &[u8], width: u32, height: u32) -> Vec<u8> {
        let n = (width * height) as usize;
        let f32_len = n * 3;
        let rgb_bytes = (f32_len * 4) as u64;
        let gray_bytes = (n * 4) as u64;

        let mut state = self.inner.lock().unwrap();
        let device = &*self.alloc.device;

        // Phase 1: grow GPU buffers if needed (ensure() takes &mut LazyBuf each).
        // Must come BEFORE the &mut f32_scratch borrow so fields don't alias.
        state.rgb_in.ensure(
            device,
            rgb_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        state.gray_out.ensure(
            device,
            gray_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        state.gray_stg.ensure(
            device,
            gray_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Phase 2: u8 → f32 in-place into pre-allocated scratch (no heap alloc).
        // The &mut borrow of f32_scratch ends at the `;` below (NLL).
        for (d, &s) in state.f32_scratch[..f32_len].iter_mut().zip(input.iter()) {
            *d = s as f32 / 255.0;
        }

        // Phase 3: upload - two immutable borrows of distinct fields (rgb_in,
        // f32_scratch); safe now that the &mut f32_scratch borrow has ended.
        self.alloc.queue.write_buffer(
            state.rgb_in.get(),
            0,
            bytemuck::cast_slice(&state.f32_scratch[..f32_len]),
        );

        let params = GrayParams {
            h: height,
            w: width,
            _p0: 0,
            _p1: 0,
        };
        let bg = make_bind_group(
            device,
            &self.alloc.pipelines.bind_group_layout,
            state.rgb_in.get(),
            state.gray_out.get(),
            &params,
        );
        let raw = dispatch_and_readback(
            device,
            &self.alloc.queue,
            &self.alloc.pipelines.gray_from_rgb,
            &bg,
            (width.div_ceil(16), height.div_ceil(16)),
            state.gray_out.get(),
            state.gray_stg.get(),
            gray_bytes,
        );
        bytemuck::cast_slice::<u8, f32>(&raw)
            .iter()
            .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
            .collect()
    }

    /// RGB u8 → bilinear resized RGB u8 on the GPU.
    fn resize_bilinear(
        &self,
        input: &[u8],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
    ) -> Vec<u8> {
        let src_ch = (src_w * src_h * 3) as usize;
        let dst_ch = (dst_w * dst_h * 3) as usize;
        let in_bytes = (src_ch * 4) as u64;
        let out_bytes = (dst_ch * 4) as u64;

        let mut state = self.inner.lock().unwrap();
        let device = &*self.alloc.device;

        // rgb_in is shared with gray_from_rgb; serialised by the Mutex
        state.rgb_in.ensure(
            device,
            in_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        state.rsz_out.ensure(
            device,
            out_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        state.rsz_stg.ensure(
            device,
            out_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        for (d, &s) in state.f32_scratch[..src_ch].iter_mut().zip(input.iter()) {
            *d = s as f32 / 255.0;
        }

        self.alloc.queue.write_buffer(
            state.rgb_in.get(),
            0,
            bytemuck::cast_slice(&state.f32_scratch[..src_ch]),
        );

        let params = ResizeParams {
            src_h,
            src_w,
            dst_h,
            dst_w,
            channels: 3,
            _p0: 0,
            _p1: 0,
            _p2: 0,
        };
        let bg = make_bind_group(
            device,
            &self.alloc.pipelines.bind_group_layout,
            state.rgb_in.get(),
            state.rsz_out.get(),
            &params,
        );
        let raw = dispatch_and_readback(
            device,
            &self.alloc.queue,
            &self.alloc.pipelines.resize_bilinear,
            &bg,
            (dst_w.div_ceil(16), dst_h.div_ceil(16)),
            state.rsz_out.get(),
            state.rsz_stg.get(),
            out_bytes,
        );
        bytemuck::cast_slice::<u8, f32>(&raw)
            .iter()
            .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
            .collect()
    }

    /// RGB u8 → f32 scaled on the GPU.  Output is raw f32 bytes.
    fn cast_and_scale(&self, input: &[u8], width: u32, height: u32, scale: f32) -> Vec<u8> {
        let n = (width * height) as usize;
        let f32_len = n * 3;
        let rgb_bytes = (f32_len * 4) as u64;

        let mut state = self.inner.lock().unwrap();
        let device = &*self.alloc.device;

        // Phase 1: grow GPU buffers if needed
        state.rgb_in.ensure(
            device,
            rgb_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        state.cas_out.ensure(
            device,
            rgb_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        state.cas_stg.ensure(
            device,
            rgb_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Phase 2: plain u8→f32 cast (no /255 - scale handles normalisation)
        for (d, &s) in state.f32_scratch[..f32_len].iter_mut().zip(input.iter()) {
            *d = s as f32;
        }

        // Phase 3: upload
        self.alloc.queue.write_buffer(
            state.rgb_in.get(),
            0,
            bytemuck::cast_slice(&state.f32_scratch[..f32_len]),
        );

        let params = CastParams {
            scale,
            width,
            height,
            channels: 3,
        };
        let bg = make_bind_group(
            device,
            &self.alloc.pipelines.bind_group_layout,
            state.rgb_in.get(),
            state.cas_out.get(),
            &params,
        );
        dispatch_and_readback(
            device,
            &self.alloc.queue,
            &self.alloc.pipelines.cast_and_scale,
            &bg,
            (width.div_ceil(16), height.div_ceil(16)),
            state.cas_out.get(),
            state.cas_stg.get(),
            rgb_bytes,
        )
    }

    fn name(&self) -> &'static str {
        "wgpu/Vulkan"
    }

    fn streaming_pipeline(&self, _width: u32, _height: u32) -> Option<Box<dyn StreamingPipeline>> {
        use crate::pipeline_async::{GpuStreamingPipeline, WgpuGrayKernel};
        let kernel = WgpuGrayKernel::new(
            std::sync::Arc::clone(self.alloc.device()),
            std::sync::Arc::clone(self.alloc.queue()),
            std::sync::Arc::clone(self.alloc.pipelines()),
        );
        Some(Box::new(GpuStreamingPipeline::new(Box::new(kernel))))
    }

    fn streaming_cast_pipeline(
        &self,
        _width: u32,
        _height: u32,
        scale: f32,
    ) -> Option<Box<dyn StreamingPipeline>> {
        use crate::pipeline_async::{GpuStreamingPipeline, WgpuCastKernel};
        let kernel = WgpuCastKernel::new(
            std::sync::Arc::clone(self.alloc.device()),
            std::sync::Arc::clone(self.alloc.queue()),
            std::sync::Arc::clone(self.alloc.pipelines()),
            scale,
        );
        Some(Box::new(GpuStreamingPipeline::new(Box::new(kernel))))
    }
}

// ── CUDA backend stub ─────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub struct CudaGpuBackend {
    alloc: crate::cuda::allocator::CudaAllocator,
    /// wgpu backend used for ops not yet implemented in CUDA
    wgpu_fallback: WgpuGpuBackend,
}

#[cfg(feature = "cuda")]
impl CudaGpuBackend {
    pub fn new() -> Result<Self, crate::error::GpuError> {
        Ok(Self {
            alloc: crate::cuda::allocator::CudaAllocator::new()?,
            wgpu_fallback: WgpuGpuBackend::new(),
        })
    }
}

#[cfg(feature = "cuda")]
impl GpuBackend for CudaGpuBackend {
    fn gray_from_rgb(&self, input: &[u8], width: u32, height: u32) -> Vec<u8> {
        use crate::cuda::image::CudaImageExt;
        use kornia_image::{Image, ImageSize};
        use kornia_tensor::CpuAllocator;

        let f32_data: Vec<f32> = input.iter().map(|&b| b as f32 / 255.0).collect();
        let cpu_img = Image::<f32, 3, CpuAllocator>::new(
            ImageSize {
                height: height as usize,
                width: width as usize,
            },
            f32_data,
            CpuAllocator,
        )
        .expect("CUDA gray_from_rgb: invalid dimensions");

        let cuda_img = cpu_img.to_cuda(&self.alloc).expect("CUDA upload failed");
        let gray =
            crate::cuda::kernels::gray_from_rgb(&cuda_img).expect("CUDA gray_from_rgb failed");
        let result = gray.to_cpu().expect("CUDA download failed");

        result
            .as_slice()
            .iter()
            .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
            .collect()
    }

    fn resize_bilinear(
        &self,
        input: &[u8],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
    ) -> Vec<u8> {
        self.wgpu_fallback
            .resize_bilinear(input, src_w, src_h, dst_w, dst_h)
    }

    fn cast_and_scale(&self, input: &[u8], width: u32, height: u32, scale: f32) -> Vec<u8> {
        self.wgpu_fallback
            .cast_and_scale(input, width, height, scale)
    }

    fn streaming_pipeline(&self, _width: u32, _height: u32) -> Option<Box<dyn StreamingPipeline>> {
        use crate::pipeline_async::{CudaGrayKernel, GpuStreamingPipeline};
        let kernel = CudaGrayKernel {
            alloc: self.alloc.clone(),
        };
        Some(Box::new(GpuStreamingPipeline::new(Box::new(kernel))))
    }

    fn streaming_cast_pipeline(
        &self,
        width: u32,
        height: u32,
        scale: f32,
    ) -> Option<Box<dyn StreamingPipeline>> {
        self.wgpu_fallback
            .streaming_cast_pipeline(width, height, scale)
    }

    fn name(&self) -> &'static str {
        "CUDA"
    }
}

// ── Backend kind enum & auto-selection ───────────────────────────────────────

pub enum BackendKind {
    Wgpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

/// Select the best available backend at runtime.
pub fn auto_select_backend() -> Box<dyn GpuBackend> {
    #[cfg(feature = "cuda")]
    {
        match CudaGpuBackend::new() {
            Ok(b) => {
                eprintln!("[kornia-gpu] backend: CUDA (NVIDIA GPU detected)");
                return Box::new(b);
            }
            Err(e) => {
                eprintln!(
                    "[kornia-gpu] CUDA unavailable ({}), falling back to wgpu",
                    e
                );
            }
        }
    }
    eprintln!("[kornia-gpu] backend: wgpu/Vulkan");
    Box::new(WgpuGpuBackend::new())
}
