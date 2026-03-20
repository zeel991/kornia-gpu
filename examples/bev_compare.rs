//! BEV warp comparison: Rust GPU output vs OpenCV reference
//!
//! Usage:
//!   cargo run --release --example bev_compare -- input.jpg
//!
//! Then compare with OpenCV:
//!   python3 crates/kornia-gpu/tools/opencv_compare.py input.jpg output_gpu.bin 1280 720

use kornia_gpu::{kernels, GpuAllocator, ImageExt};
use kornia_image::Image;
use kornia_tensor::CpuAllocator;

const BEV_HOMOGRAPHY: [f32; 9] = [
     1.2,    0.05, -120.0,
    -0.03,   1.15,  -90.0,
     0.0001, 0.0002,  1.0,
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input_path  = args.get(1).map(String::as_str).unwrap_or("input.jpg");
    let output_path = args.get(2).map(String::as_str).unwrap_or("output_gpu.bin");

    let cpu_u8 = kornia_io::functional::read_image_any_rgb8(input_path)
        .map_err(|e| format!("Failed to read {}: {}", input_path, e))?;

    let (h, w) = (cpu_u8.height(), cpu_u8.width());
    println!("Input:  {}  ({}x{} RGB)", input_path, w, h);

    let cpu_f32: Image<f32, 3, CpuAllocator> = cpu_u8.clone().cast_and_scale::<f32>(1.0 / 255.0)?;

    let gpu = GpuAllocator::new();
    let gpu_src = cpu_f32.to_gpu(&gpu)?;

    let t0 = std::time::Instant::now();
    let gpu_warped = kernels::warp_perspective(&gpu_src, (h, w), &BEV_HOMOGRAPHY)?;
    let result = gpu_warped.to_cpu()?;
    let elapsed = t0.elapsed();
    println!("GPU warp: {:.2} ms", elapsed.as_secs_f64() * 1000.0);

    let raw_bytes: &[u8] = bytemuck::cast_slice(result.as_slice());
    std::fs::write(output_path, raw_bytes)?;
    println!("Output:  {}  ({} bytes, f32 RGB)", output_path, raw_bytes.len());
    println!("Width: {}, Height: {}", w, h);
    println!();
    println!("Compare with OpenCV:");
    println!("  python3 crates/kornia-gpu/tools/opencv_compare.py {} {} {} {}", input_path, output_path, w, h);

    Ok(())
}