//! Minimal BEV node skeleton demonstrating the GPU pipeline architecture
//!
//! Shows correct async/blocking separation for use in a Bubbaloop Zenoh node
//! The full integration uses bubbaloop-node-sdk (not yet a workspace dep)

use std::sync::Arc;
use kornia_gpu::{
    kernels, pool::GpuImagePool, GpuAllocator, ImageExt,
};
use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;

struct BevConfig {
    input_topic:  String,
    output_topic: String,
    height: usize,
    width:  usize,
    homography: [f32; 9],
}

impl BevConfig {
    fn from_env() -> Self {
        Self {
            input_topic:  std::env::var("BEV_INPUT_TOPIC")
                .unwrap_or_else(|_| "camera/frames".to_string()),
            output_topic: std::env::var("BEV_OUTPUT_TOPIC")
                .unwrap_or_else(|_| "camera/bev".to_string()),
            height: std::env::var("CAM_HEIGHT")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(720),
            width: std::env::var("CAM_WIDTH")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(1280),
            homography: [
                 1.2,    0.05, -120.0,
                -0.03,   1.15,  -90.0,
                 0.0001, 0.0002,  1.0,
            ],
        }
    }
}

/// GPU pipeline state, initialised once, reused every frame
struct BevPipeline {
    gpu:       GpuAllocator,
    warp_pool: GpuImagePool<f32, 3>,
    homography: [f32; 9],
    height: usize,
    width:  usize,
}

impl BevPipeline {
    fn new(cfg: &BevConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let gpu = GpuAllocator::new();
        // Allocate 2 warp buffers at startup — zero per-frame VRAM allocation
        let warp_pool = GpuImagePool::new(2, cfg.height, cfg.width, &gpu)?;
        println!("[bev_node] GPU ready — {}×{} pool allocated", cfg.width, cfg.height);
        Ok(Self { gpu, warp_pool, homography: cfg.homography, height: cfg.height, width: cfg.width })
    }

    /// Process one frame. Must be called from a blocking thread (not the Tokio executor)
    ///
    /// Input: raw RGB f32 bytes (height × width × 3 × 4 bytes)
    fn process_frame(&self, raw: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let n = self.height * self.width * 3;
        if raw.len() != n * 4 {
            return Err(format!("frame size mismatch: expected {} bytes, got {}", n * 4, raw.len()).into());
        }

        let f32_data: &[f32] = bytemuck::cast_slice(raw);
        let cpu_img = Image::<f32, 3, CpuAllocator>::new(
            ImageSize { height: self.height, width: self.width },
            f32_data.to_vec(),
            CpuAllocator,
        )?;

        let t0 = std::time::Instant::now();
        let gpu_src = cpu_img.to_gpu(&self.gpu)?;
        let t1 = std::time::Instant::now();

        // Acquire pre-allocated buffer, no GPU allocation here
        let warp_buf = self.warp_pool.acquire()?;
        kernels::warp_perspective_into(&gpu_src, &warp_buf, &self.homography)?;
        let t2 = std::time::Instant::now();

        let result = warp_buf.to_cpu()?;
        let t3 = std::time::Instant::now();

        // Return buffer for next frame
        self.warp_pool.release(warp_buf);

        println!("[bev_node] upload={:.1}ms  kernel={:.1}ms  download={:.1}ms",
            (t1-t0).as_secs_f64()*1000.0,
            (t2-t1).as_secs_f64()*1000.0,
            (t3-t2).as_secs_f64()*1000.0,
        );

        Ok(bytemuck::cast_slice(result.as_slice()).to_vec())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cfg = BevConfig::from_env();
    println!("[bev_node] input={} output={} size={}×{}",
        cfg.input_topic, cfg.output_topic, cfg.width, cfg.height);

    // GPU init on a std thread (not async), GpuAllocator::new() may block
    let pipeline = Arc::new(BevPipeline::new(&cfg)?);

    // Demonstrate the pipeline with a synthetic frame (all zeros)
    // In production this comes from a Zenoh subscriber on cfg.input_topic
    let synthetic = vec![0u8; cfg.height * cfg.width * 3 * 4];
    let p = Arc::clone(&pipeline);
    let result = std::thread::spawn(move || p.process_frame(&synthetic))
        .join()
        .unwrap()?;

    println!("[bev_node] frame processed: {} output bytes", result.len());
    println!("[bev_node] ready for Zenoh integration via bubbaloop-node-sdk");

    // NOTE: The full Bubbaloop integration replaces this with:
    //   1. zenoh::open() session
    //   2. session.declare_subscriber(cfg.input_topic) → decode → process_frame
    //      (process_frame called via std::thread::spawn to avoid blocking async)
    //   3. session.declare_publisher(cfg.output_topic) → publish result bytes
    //   4. Wrap in bubbaloop_node_sdk::run_node::<BevNode>() for health/schema/lifecycle

    Ok(())
}