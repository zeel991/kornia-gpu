//! Auto-dispatch BEV example: uses CUDA on NVIDIA, wgpu everywhere else.
//!
//! This is the intended user-facing API for kornia-gpu.
//! No backend selection needed - Backend::auto() handles it.
//!
//! Usage:
//!   cargo run --release --example bev_auto -- frame.jpg
//!   cargo run --release --example bev_auto --features cuda -- frame.jpg

use kornia_gpu::Backend;
use kornia_image::Image;
use kornia_tensor::CpuAllocator;

const BEV_HOMOGRAPHY: [f32; 9] = [1.2, 0.05, -120.0, -0.03, 1.15, -90.0, 0.0001, 0.0002, 1.0];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input_path = args.get(1).map(String::as_str).unwrap_or("frame.jpg");

    // Load image
    let cpu_u8 = kornia_io::functional::read_image_any_rgb8(input_path)?;
    let (h, w) = (cpu_u8.height(), cpu_u8.width());
    let cpu_f32: Image<f32, 3, CpuAllocator> = cpu_u8.clone().cast_and_scale::<f32>(1.0 / 255.0)?;

    // Auto-select backend - CUDA on NVIDIA, wgpu elsewhere
    let backend = Backend::auto()?;
    println!("Backend: {}", backend.name());
    println!("Input:   {}x{} RGB", w, h);

    // Upload → warp → download (same API regardless of backend)
    let gpu_img = backend.upload(&cpu_f32)?;

    let t0 = std::time::Instant::now();
    let warped = backend.warp_perspective(&gpu_img, (h, w), &BEV_HOMOGRAPHY)?;
    let result = backend.download(&warped)?;
    let elapsed = t0.elapsed();

    println!("warp_perspective: {:.3} ms", elapsed.as_secs_f64() * 1000.0);

    // Save output
    let raw: &[u8] = bytemuck::cast_slice(result.as_slice());
    std::fs::write("output_auto.bin", raw)?;
    println!("Output saved: output_auto.bin");
    println!(
        "Compare: python3 crates/kornia-gpu/tools/opencv_compare.py {} output_auto.bin {} {}",
        input_path, w, h
    );

    Ok(())
}
