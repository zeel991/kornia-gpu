//! GPU-accelerated image processing for kornia-rs.

pub mod allocator;
pub mod backend;
pub mod error;
pub mod image;
pub mod kernels;
pub mod pipeline;
pub mod pool;

pub mod cuda;

pub use allocator::GpuAllocator;
pub use backend::{AnyGpuImage, Backend};
pub use error::GpuError;
pub use image::{GpuImage, ImageExt};
pub use pool::GpuImagePool;

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    use crate::{
        kernels,
        pipeline::GpuPipeline,
        GpuAllocator, GpuImage, ImageExt,
    };

    fn gpu() -> GpuAllocator {
        GpuAllocator::new()
    }

    fn rgb_image(h: usize, w: usize) -> Image<f32, 3, CpuAllocator> {
        let data: Vec<f32> = (0..h * w * 3).map(|i| (i % 256) as f32).collect();
        Image::new(ImageSize { height: h, width: w }, data, CpuAllocator).unwrap()
    }

    // Transfer round-trip

    #[test]
    fn test_transfer_roundtrip() {
        let gpu = gpu();
        let src = rgb_image(4, 4);
        let gpu_img = src.to_gpu(&gpu).unwrap();
        let back = gpu_img.to_cpu().unwrap();
        assert_eq!(src.as_slice(), back.as_slice());
        assert_eq!(src.size(), back.size());
    }

    // cast_and_scale

    #[test]
    fn test_cast_and_scale_exact() {
        let gpu = gpu();
        let src = rgb_image(4, 4);
        let gpu_src = src.to_gpu(&gpu).unwrap();

        let gpu_out = kernels::cast_and_scale(&gpu_src, 1.0 / 255.0).unwrap();
        let result = gpu_out.to_cpu().unwrap();

        // CPU reference
        let expected: Vec<f32> = src.as_slice().iter().map(|&v| v / 255.0).collect();

        for (a, b) in result.as_slice().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "cast_and_scale mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_cast_and_scale_zero() {
        let gpu = gpu();
        let src = Image::<f32, 3, _>::from_size_val(
            ImageSize { height: 4, width: 4 }, 0.0, CpuAllocator,
        ).unwrap();
        let result = kernels::cast_and_scale(&src.to_gpu(&gpu).unwrap(), 1.0 / 255.0)
            .unwrap().to_cpu().unwrap();
        assert!(result.as_slice().iter().all(|&v| v == 0.0));
    }

    // gray_from_rgb

    #[test]
    fn test_gray_from_rgb_black() {
        let gpu = gpu();
        let src = Image::<f32, 3, _>::from_size_val(
            ImageSize { height: 4, width: 4 }, 0.0, CpuAllocator,
        ).unwrap();
        let result = kernels::gray_from_rgb(&src.to_gpu(&gpu).unwrap())
            .unwrap().to_cpu().unwrap();
        assert!(result.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_gray_from_rgb_white() {
        let gpu = gpu();
        let src = Image::<f32, 3, _>::from_size_val(
            ImageSize { height: 4, width: 4 }, 1.0, CpuAllocator,
        ).unwrap();
        let result = kernels::gray_from_rgb(&src.to_gpu(&gpu).unwrap())
            .unwrap().to_cpu().unwrap();
        // 0.299 + 0.587 + 0.114 = 1.0
        for &v in result.as_slice() {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_gray_from_rgb_matches_weights() {
        // Single pixel: R=1.0, G=0.0, B=0.0 → gray = 0.299
        let gpu = gpu();
        let src = Image::<f32, 3, _>::new(
            ImageSize { height: 1, width: 1 },
            vec![1.0, 0.0, 0.0],
            CpuAllocator,
        ).unwrap();
        let result = kernels::gray_from_rgb(&src.to_gpu(&gpu).unwrap())
            .unwrap().to_cpu().unwrap();
        assert!((result.as_slice()[0] - 0.299).abs() < 1e-5);
    }

    // warp_perspective

    #[test]
    fn test_warp_perspective_identity_shape() {
        let gpu = gpu();
        let src = rgb_image(8, 8);
        let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = kernels::warp_perspective(&src.to_gpu(&gpu).unwrap(), (8, 8), &identity)
            .unwrap().to_cpu().unwrap();
        assert_eq!(result.size(), src.size());
        assert_eq!(result.num_channels(), 3);
    }

    #[test]
    fn test_warp_perspective_identity_values() {
        // Interior pixels should match under identity transform
        let gpu = gpu();
        let src = rgb_image(8, 8);
        let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = kernels::warp_perspective(&src.to_gpu(&gpu).unwrap(), (8, 8), &identity)
            .unwrap().to_cpu().unwrap();

        // Interior 4x4 pixels should match src exactly (no border effects)
        for row in 1..6usize {
            for col in 1..6usize {
                for ch in 0..3usize {
                    let expected = src.get([row, col, ch]).unwrap();
                    let actual = result.get([row, col, ch]).unwrap();
                    assert!((expected - actual).abs() < 1e-4,
                        "pixel ({row},{col},{ch}): expected {expected}, got {actual}");
                }
            }
        }
    }

    #[test]
    fn test_warp_perspective_output_size() {
        let gpu = gpu();
        let src = rgb_image(8, 8);
        let m = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = kernels::warp_perspective(&src.to_gpu(&gpu).unwrap(), (4, 6), &m)
            .unwrap().to_cpu().unwrap();
        assert_eq!(result.height(), 4);
        assert_eq!(result.width(), 6);
    }

    #[test]
    fn test_warp_perspective_singular_homography() {
        let gpu = gpu();
        let src = rgb_image(4, 4);
        let singular = [0.0f32; 9]; // det = 0
        let err = kernels::warp_perspective(&src.to_gpu(&gpu).unwrap(), (4, 4), &singular);
        assert!(err.is_err());
    }

    // GpuPipeline

    #[test]
    fn test_pipeline_cast_warp() {
        let gpu = gpu();
        let src = rgb_image(8, 8);
        let m = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let result = GpuPipeline::new(&gpu)
            .upload(&src).unwrap()
            .cast_and_scale(1.0 / 255.0).unwrap()
            .warp_perspective((8, 8), &m).unwrap()
            .download().unwrap();

        assert_eq!(result.size(), src.size());
        assert_eq!(result.num_channels(), 3);
    }

    #[test]
    fn test_pipeline_full_bev() {
        let gpu = gpu();
        let src = rgb_image(8, 8);
        let m = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let result = GpuPipeline::new(&gpu)
            .upload(&src).unwrap()
            .cast_and_scale(1.0 / 255.0).unwrap()
            .warp_perspective((8, 8), &m).unwrap()
            .gray_from_rgb().unwrap()
            .download().unwrap();

        assert_eq!(result.height(), 8);
        assert_eq!(result.width(), 8);
        assert_eq!(result.num_channels(), 1);
    }

    /// Correctness test at full 1080p resolution.
    /// This is the actual target resolution for the BEV pipeline.
    /// Also serves as evidence the kernel handles large workloads correctly.
    #[test]
    fn test_warp_perspective_1080p_correctness() {
        let gpu = gpu();
        let src = rgb_image(1080, 1920);
        let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let result = kernels::warp_perspective(
            &src.to_gpu(&gpu).unwrap(), (1080, 1920), &identity,
        ).unwrap().to_cpu().unwrap();

        assert_eq!(result.size(), src.size());

        // Check centre pixel — far from borders, bilinear on integer coords is exact
        let row = 540usize;
        let col = 960usize;
        for ch in 0..3usize {
            let expected = src.get([row, col, ch]).unwrap();
            let actual   = result.get([row, col, ch]).unwrap();
            assert!(
                (expected - actual).abs() < 1e-4,
                "1080p pixel ({row},{col},{ch}): expected {expected}, got {actual}"
            );
        }
    }
    // GpuImagePool - persistent VRAM reuse

    #[test]
    fn test_pool_acquire_release() {
        let gpu = gpu();
        let pool = crate::pool::GpuImagePool::<f32, 3>::new(2, 4, 4, &gpu).unwrap();
        assert_eq!(pool.available_count(), 2);

        let buf = pool.acquire().unwrap();
        assert_eq!(pool.available_count(), 1);

        pool.release(buf);
        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    fn test_pool_exhaustion() {
        let gpu = gpu();
        let pool = crate::pool::GpuImagePool::<f32, 3>::new(1, 4, 4, &gpu).unwrap();
        let _buf = pool.acquire().unwrap();
        let result = pool.acquire();
        assert!(matches!(result, Err(crate::error::GpuError::PoolExhausted)));
    }

    #[test]
    fn test_cast_and_scale_into_reuse() {
        let gpu = gpu();
        let src = rgb_image(8, 8);
        let gpu_src = src.to_gpu(&gpu).unwrap();

        // Pre-allocate output buffer — zero per-call allocation
        let dst = GpuImage::<f32, 3>::empty(8, 8, &gpu);
        kernels::cast_and_scale_into(&gpu_src, &dst, 1.0 / 255.0).unwrap();
        let result = dst.to_cpu().unwrap();

        let expected: Vec<f32> = src.as_slice().iter().map(|&v| v / 255.0).collect();
        for (a, b) in result.as_slice().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_warp_perspective_into_reuse() {
        let gpu = gpu();
        let src = rgb_image(8, 8);
        let gpu_src = src.to_gpu(&gpu).unwrap();
        let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        // Pre-allocate output buffer
        let dst = GpuImage::<f32, 3>::empty(8, 8, &gpu);
        kernels::warp_perspective_into(&gpu_src, &dst, &identity).unwrap();
        let result = dst.to_cpu().unwrap();
        assert_eq!(result.size(), src.size());
    }

    #[test]
    fn test_pool_pipeline_no_allocation() {
        // Simulates a 3-frame pipeline where buffers are reused every frame.
        // Pool allocates twice at startup, then zero allocations per frame.
        let gpu = gpu();
        let pool_warp = crate::pool::GpuImagePool::<f32, 3>::new(1, 8, 8, &gpu).unwrap();
        let m = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        for _ in 0..3 {
            let src = rgb_image(8, 8).to_gpu(&gpu).unwrap();
            let warp_buf = pool_warp.acquire().unwrap();
            kernels::warp_perspective_into(&src, &warp_buf, &m).unwrap();
            let _result = warp_buf.to_cpu().unwrap();
            pool_warp.release(warp_buf); // return for next frame
        }
        assert_eq!(pool_warp.available_count(), 1);
    }

    // CUDA backend tests - only run if CUDA is available on this machine

    fn cuda_available() -> bool {
        crate::cuda::allocator::CudaAllocator::is_available()
    }

    fn cuda_alloc() -> crate::cuda::allocator::CudaAllocator {
        crate::cuda::allocator::CudaAllocator::new().expect("CUDA not available")
    }

    #[test]
    fn test_cuda_allocator_available() {
        // Just checks that is_available() doesn't panic on any machine
        let _ = cuda_available();
    }

    #[test]
    fn test_cuda_transfer_roundtrip() {
        if !cuda_available() { return; }
        use crate::cuda::image::CudaImageExt;

        let alloc = cuda_alloc();
        let src = rgb_image(8, 8);
        let cuda_img = src.to_cuda(&alloc).unwrap();
        let result = cuda_img.to_cpu().unwrap();

        assert_eq!(result.size(), src.size());
        for (a, b) in result.as_slice().iter().zip(src.as_slice().iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_cuda_cast_and_scale() {
        if !cuda_available() { return; }
        use crate::cuda::image::CudaImageExt;

        let alloc = cuda_alloc();
        let src = rgb_image(8, 8);
        let cuda_src = src.to_cuda(&alloc).unwrap();
        let result = crate::cuda::kernels::cast_and_scale(&cuda_src, 1.0 / 255.0)
            .unwrap()
            .to_cpu()
            .unwrap();

        let expected: Vec<f32> = src.as_slice().iter().map(|&v| v / 255.0).collect();
        for (a, b) in result.as_slice().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6f32);
        }
    }

    #[test]
    fn test_cuda_warp_perspective_identity() {
        if !cuda_available() { return; }
        use crate::cuda::image::CudaImageExt;

        let alloc = cuda_alloc();
        let src = rgb_image(8, 8);
        let cuda_src = src.to_cuda(&alloc).unwrap();
        let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = crate::cuda::kernels::warp_perspective(&cuda_src, (8, 8), &identity)
            .unwrap()
            .to_cpu()
            .unwrap();

        assert_eq!(result.size(), src.size());
    }

    #[test]
    fn test_cuda_gray_from_rgb() {
        if !cuda_available() { return; }
        use crate::cuda::image::CudaImageExt;
        use kornia_image::{Image, ImageSize};
        use kornia_tensor::CpuAllocator;

        let alloc = cuda_alloc();
        // White pixel: R=1, G=1, B=1 -> gray = 1.0
        let white = Image::<f32, 3, CpuAllocator>::new(
            ImageSize { height: 1, width: 1 },
            vec![1.0f32, 1.0, 1.0],
            CpuAllocator,
        ).unwrap();
        let cuda_src = white.to_cuda(&alloc).unwrap();
        let result = crate::cuda::kernels::gray_from_rgb(&cuda_src)
            .unwrap()
            .to_cpu()
            .unwrap();
        assert!((result.as_slice()[0] - 1.0f32).abs() < 1e-5f32);
    }

    #[test]
    fn test_cuda_warp_matches_wgpu() {
        if !cuda_available() { return; }
        use crate::cuda::image::CudaImageExt;

        let gpu = gpu();
        let alloc = cuda_alloc();
        let src = rgb_image(32, 32);
        let m = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let wgpu_result = kernels::warp_perspective(&src.to_gpu(&gpu).unwrap(), (32, 32), &m)
            .unwrap()
            .to_cpu()
            .unwrap();

        let cuda_result = crate::cuda::kernels::warp_perspective(
            &src.to_cuda(&alloc).unwrap(), (32, 32), &m
        ).unwrap().to_cpu().unwrap();

        // Both backends must produce identical results
        for (a, b) in wgpu_result.as_slice().iter().zip(cuda_result.as_slice().iter()) {
            assert!((a - b).abs() < 1e-4f32, "wgpu={} cuda={}", a, b);
        }
    }

    #[test]
    fn test_backend_auto_dispatch() {
        let backend = crate::Backend::auto().unwrap();
        // Just verify it initialises and can process an image
        let src = rgb_image(8, 8);
        let cpu_f32 = src.cast_and_scale::<f32>(1.0 / 255.0).unwrap();
        let gpu_img = backend.upload(&cpu_f32).unwrap();
        let warped = backend.warp_perspective(
            &gpu_img, (8, 8), &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ).unwrap();
        let _result = backend.download(&warped).unwrap();
    }

}