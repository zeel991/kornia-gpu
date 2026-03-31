//! Streaming GPU pipeline with pluggable kernel dispatch.
//!
//! Architecture:
//!
//! GpuStreamingPipeline owns a Box<dyn StreamKernel> and a 1-frame result
//! buffer (pending).  On each push_frame call it runs the kernel synchronously
//! and returns the previous frame's result - giving 1-frame latency without
//! requiring wgpu-specific double-buffer bookkeeping.
//!
//! The StreamKernel trait is the extension point.  Adding a new backend
//! (CUDA, Metal, CPU) means implementing one function: process.
//!
//! Upstream path:
//!
//! In kornia-rs, StreamKernel would be generic over TensorAllocator:
//!
//!     trait StreamKernel<A: TensorAllocator>: Send {
//!         fn process(&self, input: &Image<u8,3,CpuAllocator>, alloc: &A) -> Image<u8,1,A>;
//!     }
//!
//! This Box<dyn> version maps directly to that pattern.

use std::sync::{Arc, Mutex};

use crate::allocator::GpuPipelines;
use crate::gpu_backend::{dispatch_and_readback, make_bind_group, CastParams, GrayParams};

// ── StreamKernel trait ────────────────────────────────────────────────────────

/// A GPU kernel that can be plugged into [GpuStreamingPipeline].
///
/// Implementors handle upload, dispatch, and readback for one frame.
/// The pipeline manages the 1-frame latency buffering; kernels only process.
pub trait StreamKernel: Send {
    /// Process one frame.  Input is raw RGB u8 bytes; output format depends on
    /// the kernel (grayscale u8, raw f32 bytes, etc.).
    fn process(&self, input: &[u8], width: u32, height: u32) -> Vec<u8>;
}

// ── StreamingPipeline trait ───────────────────────────────────────────────────

/// A stateful pipeline for processing a stream of frames with minimal latency.
pub trait StreamingPipeline: Send {
    /// Push input (raw RGB u8 bytes) into the pipeline.
    ///
    /// Returns None on the very first call (pipeline is filling).
    /// Returns Some(frame) on every subsequent call - the result is the
    /// previous frame (1-frame latency).
    fn push_frame(&mut self, input: &[u8], width: u32, height: u32) -> Option<Vec<u8>>;

    /// Flush the pipeline and return the last buffered frame, if any.
    fn flush(&mut self) -> Option<Vec<u8>>;
}

// ── Lazy GPU buffer ───────────────────────────────────────────────────────────

struct LazyBuf {
    buf: Option<wgpu::Buffer>,
    cap: u64,
}

impl LazyBuf {
    const fn new() -> Self {
        Self { buf: None, cap: 0 }
    }

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

// ── WgpuGrayKernel ────────────────────────────────────────────────────────────

struct WgpuGrayInner {
    f32_scratch: Vec<f32>,
    rgb_in: LazyBuf,
    gray_out: LazyBuf,
    gray_stg: LazyBuf,
}

/// wgpu gray_from_rgb kernel with persistent lazy buffers.
pub struct WgpuGrayKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: Arc<GpuPipelines>,
    inner: Mutex<WgpuGrayInner>,
}

impl WgpuGrayKernel {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipelines: Arc<GpuPipelines>,
    ) -> Self {
        Self {
            device,
            queue,
            pipelines,
            inner: Mutex::new(WgpuGrayInner {
                f32_scratch: Vec::new(),
                rgb_in: LazyBuf::new(),
                gray_out: LazyBuf::new(),
                gray_stg: LazyBuf::new(),
            }),
        }
    }
}

impl StreamKernel for WgpuGrayKernel {
    fn process(&self, input: &[u8], width: u32, height: u32) -> Vec<u8> {
        let n = (width * height) as usize;
        let f32_len = n * 3;
        let rgb_bytes = (f32_len * 4) as u64;
        let gray_bytes = (n * 4) as u64;

        let mut st = self.inner.lock().unwrap();
        let device = &*self.device;

        st.rgb_in.ensure(
            device,
            rgb_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        st.gray_out.ensure(
            device,
            gray_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        st.gray_stg.ensure(
            device,
            gray_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        if st.f32_scratch.len() < f32_len {
            st.f32_scratch.resize(f32_len, 0.0);
        }
        for (d, &s) in st.f32_scratch[..f32_len].iter_mut().zip(input.iter()) {
            *d = s as f32 / 255.0;
        }
        self.queue.write_buffer(
            st.rgb_in.get(),
            0,
            bytemuck::cast_slice(&st.f32_scratch[..f32_len]),
        );

        let params = GrayParams {
            h: height,
            w: width,
            _p0: 0,
            _p1: 0,
        };
        let bg = make_bind_group(
            device,
            &self.pipelines.bind_group_layout,
            st.rgb_in.get(),
            st.gray_out.get(),
            &params,
        );
        let raw = dispatch_and_readback(
            device,
            &self.queue,
            &self.pipelines.gray_from_rgb,
            &bg,
            (width.div_ceil(16), height.div_ceil(16)),
            st.gray_out.get(),
            st.gray_stg.get(),
            gray_bytes,
        );
        bytemuck::cast_slice::<u8, f32>(&raw)
            .iter()
            .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
            .collect()
    }
}

// ── WgpuCastKernel ────────────────────────────────────────────────────────────

struct WgpuCastInner {
    f32_scratch: Vec<f32>,
    rgb_in: LazyBuf,
    cas_out: LazyBuf,
    cas_stg: LazyBuf,
}

/// wgpu cast_and_scale kernel with persistent lazy buffers.
pub struct WgpuCastKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: Arc<GpuPipelines>,
    scale: f32,
    inner: Mutex<WgpuCastInner>,
}

impl WgpuCastKernel {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipelines: Arc<GpuPipelines>,
        scale: f32,
    ) -> Self {
        Self {
            device,
            queue,
            pipelines,
            scale,
            inner: Mutex::new(WgpuCastInner {
                f32_scratch: Vec::new(),
                rgb_in: LazyBuf::new(),
                cas_out: LazyBuf::new(),
                cas_stg: LazyBuf::new(),
            }),
        }
    }
}

impl StreamKernel for WgpuCastKernel {
    fn process(&self, input: &[u8], width: u32, height: u32) -> Vec<u8> {
        let n = (width * height) as usize;
        let f32_len = n * 3;
        let rgb_bytes = (f32_len * 4) as u64;

        let mut st = self.inner.lock().unwrap();
        let device = &*self.device;

        st.rgb_in.ensure(
            device,
            rgb_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        st.cas_out.ensure(
            device,
            rgb_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        st.cas_stg.ensure(
            device,
            rgb_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        if st.f32_scratch.len() < f32_len {
            st.f32_scratch.resize(f32_len, 0.0);
        }
        for (d, &s) in st.f32_scratch[..f32_len].iter_mut().zip(input.iter()) {
            *d = s as f32;
        }
        self.queue.write_buffer(
            st.rgb_in.get(),
            0,
            bytemuck::cast_slice(&st.f32_scratch[..f32_len]),
        );

        let params = CastParams {
            scale: self.scale,
            width,
            height,
            channels: 3,
        };
        let bg = make_bind_group(
            device,
            &self.pipelines.bind_group_layout,
            st.rgb_in.get(),
            st.cas_out.get(),
            &params,
        );
        dispatch_and_readback(
            device,
            &self.queue,
            &self.pipelines.cast_and_scale,
            &bg,
            (width.div_ceil(16), height.div_ceil(16)),
            st.cas_out.get(),
            st.cas_stg.get(),
            rgb_bytes,
        )
    }
}

// ── CudaGrayKernel ────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub struct CudaGrayKernel {
    pub alloc: crate::cuda::allocator::CudaAllocator,
}

#[cfg(feature = "cuda")]
impl StreamKernel for CudaGrayKernel {
    fn process(&self, input: &[u8], width: u32, height: u32) -> Vec<u8> {
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
        .expect("CudaGrayKernel: invalid dimensions");

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
}

// ── GpuStreamingPipeline ──────────────────────────────────────────────────────

/// Streaming pipeline with pluggable kernel dispatch and 1-frame latency.
///
/// Runs the kernel synchronously on each push_frame call and holds the
/// previous result in pending.  Returns None on the first call.
///
/// Tradeoff vs. prior SubmissionIndex-based design:
///
/// The previous implementation used wgpu SubmissionIndex to wait only for
/// the previous frame, enabling true PCIe overlap between upload N+1 and
/// kernel N.  This simpler design calls the kernel synchronously, which is
/// correct for both CUDA (already synchronous) and wgpu.  For wgpu, the
/// persistent LazyBuf buffers inside WgpuGrayKernel still avoid per-frame
/// VRAM allocation; only the explicit overlap is lost.
pub struct GpuStreamingPipeline {
    kernel: Box<dyn StreamKernel>,
    pending: Option<Vec<u8>>,
}

impl GpuStreamingPipeline {
    pub fn new(kernel: Box<dyn StreamKernel>) -> Self {
        Self {
            kernel,
            pending: None,
        }
    }
}

impl StreamingPipeline for GpuStreamingPipeline {
    fn push_frame(&mut self, input: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
        let current = self.kernel.process(input, width, height);
        self.pending.replace(current)
    }

    fn flush(&mut self) -> Option<Vec<u8>> {
        self.pending.take()
    }
}
