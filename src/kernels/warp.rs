//! Homography convention matches kornia-imgproc: m is the forward matrix
//! (src → dst).  We invert it once on the CPU before uploading to the GPU.

use crate::error::GpuError;
use crate::image::GpuImage;
use wgpu::util::DeviceExt;

// ── CPU homography helpers ────────────────────────────────────────────────────

fn determinant3x3(m: &[f32; 9]) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

fn adjugate3x3(m: &[f32; 9]) -> [f32; 9] {
    [
        m[4] * m[8] - m[5] * m[7],
        m[2] * m[7] - m[1] * m[8],
        m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8],
        m[0] * m[8] - m[2] * m[6],
        m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6],
        m[1] * m[6] - m[0] * m[7],
        m[0] * m[4] - m[1] * m[3],
    ]
}

fn invert_perspective(m: &[f32; 9]) -> Result<[f32; 9], GpuError> {
    let det = determinant3x3(m);
    if det == 0.0 {
        return Err(GpuError::SingularHomography);
    }
    let adj = adjugate3x3(m);
    let inv_det = 1.0 / det;
    let mut inv = [0.0f32; 9];
    for i in 0..9 {
        inv[i] = adj[i] * inv_det;
    }
    Ok(inv)
}

// ── Uniform layout (must match warp_perspective.wgsl) ────────────────────────
//
// h_inv rows packed as [f32; 4] (last element unused) to match vec4<f32>
// alignment in the WGSL uniform block.

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct WarpUniforms {
    h_inv_r0: [f32; 4], // h_inv[0..=2], 0.0
    h_inv_r1: [f32; 4], // h_inv[3..=5], 0.0
    h_inv_r2: [f32; 4], // h_inv[6..=8], 0.0
    src_h: u32,
    src_w: u32,
    dst_h: u32,
    dst_w: u32,
    channels: u32,
    _p: [u32; 3], // total = 80 bytes (5 × 16)
}

// ── Launch helper ─────────────────────────────────────────────────────────────

fn run_warp<const C: usize>(src: &GpuImage<f32, C>, dst: &GpuImage<f32, C>, h_inv: &[f32; 9]) {
    let alloc = src.alloc();
    let (dst_h, dst_w) = (dst.height() as u32, dst.width() as u32);
    let uniforms = WarpUniforms {
        h_inv_r0: [h_inv[0], h_inv[1], h_inv[2], 0.0],
        h_inv_r1: [h_inv[3], h_inv[4], h_inv[5], 0.0],
        h_inv_r2: [h_inv[6], h_inv[7], h_inv[8], 0.0],
        src_h: src.height() as u32,
        src_w: src.width() as u32,
        dst_h,
        dst_w,
        channels: C as u32,
        _p: [0; 3],
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
        pass.set_pipeline(&pipelines.warp_perspective);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(dst_w.div_ceil(tile), dst_h.div_ceil(tile), 1);
    }
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Apply a perspective transformation with bilinear interpolation.
///
/// m is the forward homography (src → dst), same convention as kornia-imgproc.
pub fn warp_perspective<const C: usize>(
    src: &GpuImage<f32, C>,
    dst_size: (usize, usize),
    m: &[f32; 9],
) -> Result<GpuImage<f32, C>, GpuError> {
    let (dst_h, dst_w) = dst_size;
    let h_inv = invert_perspective(m)?;
    let dst = GpuImage::<f32, C>::empty(dst_h, dst_w, src.alloc());
    run_warp(src, &dst, &h_inv);
    Ok(dst)
}

/// Write warp_perspective result into an existing GPU buffer (zero allocation).
pub fn warp_perspective_into<const C: usize>(
    src: &GpuImage<f32, C>,
    dst: &GpuImage<f32, C>,
    m: &[f32; 9],
) -> Result<(), GpuError> {
    let h_inv = invert_perspective(m)?;
    run_warp(src, dst, &h_inv);
    Ok(())
}
