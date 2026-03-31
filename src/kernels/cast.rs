use crate::error::GpuError;
use crate::image::GpuImage;
use wgpu::util::DeviceExt;

// ── Uniform layout (must match cast_and_scale.wgsl) ──────────────────────────

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct CastUniforms {
    scale: f32,
    width: u32,
    height: u32,
    channels: u32,
}

// ── Launch helpers ────────────────────────────────────────────────────────────

fn run_cast<const C: usize>(src: &GpuImage<f32, C>, dst: &GpuImage<f32, C>, scale: f32) {
    let alloc = src.alloc();
    let device = alloc.device();
    let queue = alloc.queue();
    let pipelines = alloc.pipelines();
    let uniforms = CastUniforms {
        scale,
        width: src.width() as u32,
        height: src.height() as u32,
        channels: C as u32,
    };
    let tile = 16u32;
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
        pass.set_pipeline(&pipelines.cast_and_scale);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(
            (src.width() as u32).div_ceil(tile),
            (src.height() as u32).div_ceil(tile),
            1,
        );
    }
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Multiply every pixel value by scale, writing results to a new GpuImage.
pub fn cast_and_scale<const C: usize>(
    src: &GpuImage<f32, C>,
    scale: f32,
) -> Result<GpuImage<f32, C>, GpuError> {
    let dst = GpuImage::<f32, C>::empty(src.height(), src.width(), src.alloc());
    run_cast(src, &dst, scale);
    Ok(dst)
}

/// Write cast_and_scale result into an existing GPU buffer (zero allocation).
pub fn cast_and_scale_into<const C: usize>(
    src: &GpuImage<f32, C>,
    dst: &GpuImage<f32, C>,
    scale: f32,
) -> Result<(), GpuError> {
    if dst.height() != src.height() || dst.width() != src.width() {
        return Err(GpuError::BufferSizeMismatch(
            src.height(),
            src.width(),
            dst.height(),
            dst.width(),
        ));
    }
    run_cast(src, dst, scale);
    Ok(())
}
