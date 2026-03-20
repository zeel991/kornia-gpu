//! CUDA backend benchmark and accuracy test.
//!
//! Runs warp_perspective on a real image using the CUDA backend,
//! compares timing against the wgpu backend.
//!
//! Usage:
//!   cargo run --release --example bev_cuda --features cuda -- frame.jpg

use kornia_gpu::cuda::{CudaAllocator, CudaImageExt, kernels as cuda_kernels};
use kornia_gpu::{GpuAllocator, ImageExt, kernels as wgpu_kernels};
use kornia_image::Image;
use kornia_tensor::CpuAllocator;

const BEV_HOMOGRAPHY: [f32; 9] = [
     1.2,    0.05, -120.0,
    -0.03,   1.15,  -90.0,
     0.0001, 0.0002,  1.0,
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input_path = args.get(1).map(String::as_str).unwrap_or("frame.jpg");

    let cpu_u8 = kornia_io::functional::read_image_any_rgb8(input_path)?;
    let (h, w) = (cpu_u8.height(), cpu_u8.width());
    println!("Input: {}x{} RGB", w, h);

    let cpu_f32: Image<f32, 3, CpuAllocator> = cpu_u8.clone().cast_and_scale::<f32>(1.0 / 255.0)?;

    // -----------------------------------------------------------------------
    // wgpu baseline
    // -----------------------------------------------------------------------
    let gpu = GpuAllocator::new();
    let gpu_src = cpu_f32.to_gpu(&gpu)?;

    // warmup
    for _ in 0..3 {
        let _ = wgpu_kernels::warp_perspective(&gpu_src, (h, w), &BEV_HOMOGRAPHY)?;
    }

    let n = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let _ = wgpu_kernels::warp_perspective(&gpu_src, (h, w), &BEV_HOMOGRAPHY)?;
    }
    let wgpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

    // -----------------------------------------------------------------------
    // CUDA
    // -----------------------------------------------------------------------
    let cuda = CudaAllocator::new()?;
    let cuda_src = cpu_f32.to_cuda(&cuda)?;

    // warmup
    for _ in 0..3 {
        let _ = cuda_kernels::warp_perspective(&cuda_src, (h, w), &BEV_HOMOGRAPHY)?;
    }

    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let _ = cuda_kernels::warp_perspective(&cuda_src, (h, w), &BEV_HOMOGRAPHY)?;
    }
    let cuda_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

    // -----------------------------------------------------------------------
    // Results
    // -----------------------------------------------------------------------
    println!("\n=== warp_perspective @ {}x{} ({} iters) ===", w, h, n);
    println!("wgpu (Vulkan):  {:.3} ms", wgpu_ms);
    println!("CUDA:           {:.3} ms", cuda_ms);
    println!("CUDA speedup:   {:.2}x", wgpu_ms / cuda_ms);

    // Accuracy check: compare CUDA output vs wgpu output
    let wgpu_result = wgpu_kernels::warp_perspective(&gpu_src, (h, w), &BEV_HOMOGRAPHY)?
        .to_cpu()?;
    let cuda_result = cuda_kernels::warp_perspective(&cuda_src, (h, w), &BEV_HOMOGRAPHY)?
        .to_cpu()?;

    let diffs: Vec<f32> = wgpu_result.as_slice().iter()
        .zip(cuda_result.as_slice().iter())
        .map(|(a, b)| (a - b).abs())
        .collect();

    let mean_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let max_diff = diffs.iter().cloned().fold(0.0f32, f32::max);
    let match_1 = diffs.iter().filter(|&&d| d < 1.0/255.0).count() as f32 / diffs.len() as f32;

    println!("\n=== CUDA vs wgpu accuracy ===");
    println!("Mean abs diff:  {:.6}", mean_diff);
    println!("Max abs diff:   {:.6}", max_diff);
    println!("Match <1/255:   {:.2}%", match_1 * 100.0);

    // Save CUDA output for visual inspection
    let raw: &[u8] = bytemuck::cast_slice(cuda_result.as_slice());
    std::fs::write("output_cuda.bin", raw)?;
    println!("\nCUDA output saved: output_cuda.bin");
    println!("Compare: python3 crates/kornia-gpu/tools/opencv_compare.py {} output_cuda.bin {} {}", input_path, w, h);

    Ok(())
}
