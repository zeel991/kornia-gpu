//! Half-pixel alignment (align_corners = false), matching torchvision default.

use crate::error::GpuError;
use crate::image::GpuImage;
use wgpu::util::DeviceExt;

// ── Uniform layout (must match resize_bilinear.wgsl) ─────────────────────────

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct ResizeUniforms {
    src_h: u32,
    src_w: u32,
    dst_h: u32,
    dst_w: u32,
    channels: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32, // total = 32 bytes (2 × 16)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Resize a GPU image to (dst_h, dst_w) using bilinear interpolation.
pub fn resize_bilinear<const C: usize>(
    src: &GpuImage<f32, C>,
    dst_h: usize,
    dst_w: usize,
) -> Result<GpuImage<f32, C>, GpuError> {
    let alloc = src.alloc();
    let dst = GpuImage::<f32, C>::empty(dst_h, dst_w, alloc);
    let uniforms = ResizeUniforms {
        src_h: src.height() as u32,
        src_w: src.width() as u32,
        dst_h: dst_h as u32,
        dst_w: dst_w as u32,
        channels: C as u32,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    let tile = 16u32;
    let device = alloc.device();
    let queue = alloc.queue();
    let pipelines = alloc.pipelines();
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipelines.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src.mem.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dst.mem.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipelines.resize_bilinear);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(
            (dst_w as u32).div_ceil(tile),
            (dst_h as u32).div_ceil(tile),
            1,
        );
    }
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);
    Ok(dst)
}
