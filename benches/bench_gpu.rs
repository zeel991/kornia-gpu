//! Benchmarks: GPU vs CPU for warp_perspective and the full BEV pipeline,
//! as well as streaming backend throughput.
//!
//! Run with:
//!   cargo bench -p kornia-gpu --bench bench_gpu
//!
//! Two timing modes per GPU operation:
//!   gpu_compute  - kernel only (data already in VRAM, result stays in VRAM)
//!   gpu_e2e      - upload + kernel + download (true wall-clock cost)
//!
//! The compute time shows raw kernel speedup.
//! The e2e time shows real-world speedup when data must cross the PCIe bus.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_gpu::{kernels, pipeline::GpuPipeline, GpuAllocator, ImageExt};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_tensor::CpuAllocator;

const SIZES: &[(usize, usize)] = &[(720, 1280), (1080, 1920), (2160, 3840)];

fn make_cpu_image(h: usize, w: usize) -> Image<f32, 3, CpuAllocator> {
    let data: Vec<f32> = (0..h * w * 3).map(|i| (i % 256) as f32 / 255.0).collect();
    Image::new(
        ImageSize {
            height: h,
            width: w,
        },
        data,
        CpuAllocator,
    )
    .unwrap()
}

fn bev_homography() -> [f32; 9] {
    [1.2, 0.1, -100.0, -0.05, 1.1, -80.0, 0.0001, 0.0002, 1.0]
}

// warp_perspective

fn bench_warp_perspective(c: &mut Criterion) {
    let gpu = GpuAllocator::new();
    let m = bev_homography();
    let mut group = c.benchmark_group("warp_perspective");

    for &(h, w) in SIZES {
        let label = format!("{}x{}", w, h);
        let cpu_img = make_cpu_image(h, w);
        let gpu_img = cpu_img.to_gpu(&gpu).unwrap();

        // Warmup
        for _ in 0..20 {
            let _ = kernels::warp_perspective(&gpu_img, (h, w), &m).unwrap();
        }

        // CPU (kornia-imgproc rayon reference)
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            let mut dst = Image::<f32, 3, _>::from_size_val(
                ImageSize {
                    height: h,
                    width: w,
                },
                0.0,
                CpuAllocator,
            )
            .unwrap();
            b.iter(|| {
                kornia_imgproc::warp::warp_perspective(
                    &cpu_img,
                    &mut dst,
                    &m,
                    InterpolationMode::Bilinear,
                )
                .unwrap()
            })
        });

        // GPU compute only
        group.bench_with_input(BenchmarkId::new("gpu_compute", &label), &(), |b, _| {
            b.iter(|| kernels::warp_perspective(&gpu_img, (h, w), &m).unwrap())
        });

        // GPU end-to-end
        group.bench_with_input(BenchmarkId::new("gpu_e2e", &label), &(), |b, _| {
            b.iter(|| {
                kernels::warp_perspective(&cpu_img.to_gpu(&gpu).unwrap(), (h, w), &m)
                    .unwrap()
                    .to_cpu()
                    .unwrap()
            })
        });
    }
    group.finish();
}

// Full BEV pipeline

fn bench_bev_pipeline(c: &mut Criterion) {
    let gpu = GpuAllocator::new();
    let m = bev_homography();
    let mut group = c.benchmark_group("bev_pipeline");

    for &(h, w) in SIZES {
        let label = format!("{}x{}", w, h);
        let cpu_img = make_cpu_image(h, w);

        // CPU: cast → warp → gray (3 separate rayon passes)
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter(|| {
                let scaled = cpu_img.clone().cast_and_scale::<f32>(1.0 / 255.0).unwrap();
                let mut warped = Image::<f32, 3, _>::from_size_val(
                    ImageSize {
                        height: h,
                        width: w,
                    },
                    0.0,
                    CpuAllocator,
                )
                .unwrap();
                kornia_imgproc::warp::warp_perspective(
                    &scaled,
                    &mut warped,
                    &m,
                    InterpolationMode::Bilinear,
                )
                .unwrap();
                let mut gray = Image::<f32, 1, _>::from_size_val(
                    ImageSize {
                        height: h,
                        width: w,
                    },
                    0.0,
                    CpuAllocator,
                )
                .unwrap();
                kornia_imgproc::color::gray_from_rgb(&warped, &mut gray).unwrap();
                gray
            })
        });

        // GPU pipeline: upload once → 3 kernels → download once
        group.bench_with_input(BenchmarkId::new("gpu_e2e", &label), &(), |b, _| {
            b.iter(|| {
                GpuPipeline::new(&gpu)
                    .upload(&cpu_img)
                    .unwrap()
                    .cast_and_scale(1.0 / 255.0)
                    .unwrap()
                    .warp_perspective((h, w), &m)
                    .unwrap()
                    .gray_from_rgb()
                    .unwrap()
                    .download()
                    .unwrap()
            })
        });
    }
    group.finish();
}

// ── Streaming pipeline throughput ─────────────────────────────────────────────

/// Compare throughput (frames/sec) for 30 frames:
///   sequential - WgpuGpuBackend::gray_from_rgb called one frame at a time
///   pipelined  - AsyncGpuPipeline with double-buffered upload/kernel overlap
fn bench_streaming_pipeline(c: &mut Criterion) {
    use kornia_gpu::gpu_backend::GpuBackend;
    use kornia_gpu::WgpuGpuBackend;

    const FRAMES: usize = 30;
    let (h, w) = (1080usize, 1920usize);

    let backend = WgpuGpuBackend::new();
    let cpu_img = make_cpu_image(h, w);
    let rgb_u8: Vec<u8> = cpu_img
        .as_slice()
        .iter()
        .map(|&v| (v * 255.0) as u8)
        .collect();

    let mut group = c.benchmark_group("streaming_pipeline");
    group.throughput(criterion::Throughput::Elements(FRAMES as u64));
    group.sample_size(20); // fewer samples - each iteration is 30 frames

    // Warmup to ensure lazy buffers are allocated before timing starts
    for _ in 0..5 {
        let _ = backend.gray_from_rgb(&rgb_u8, w as u32, h as u32);
    }

    // Sequential: N frames × (upload + kernel + download), fully sequential
    group.bench_function("sequential_30frames", |b| {
        b.iter(|| {
            for _ in 0..FRAMES {
                let _ = backend.gray_from_rgb(&rgb_u8, w as u32, h as u32);
            }
        })
    });

    // Pipelined: AsyncGpuPipeline overlaps frame N+1 upload with frame N kernel
    group.bench_function("pipelined_30frames", |b| {
        b.iter(|| {
            let mut pipeline = backend
                .streaming_pipeline(w as u32, h as u32)
                .expect("streaming_pipeline not supported");
            let mut count = 0usize;
            for _ in 0..FRAMES {
                if pipeline.push_frame(&rgb_u8, w as u32, h as u32).is_some() {
                    count += 1;
                }
            }
            // Drain the last buffered frame
            if pipeline.flush().is_some() {
                count += 1;
            }
            assert_eq!(count, FRAMES); // FRAMES-1 from push + 1 from flush
        })
    });

    group.finish();
}

// ── Streaming cast_and_scale throughput ──────────────────────────────────────

/// Compare throughput (frames/sec) for 30 frames:
///   sequential - WgpuGpuBackend::cast_and_scale called one frame at a time
///   pipelined  - AsyncGpuPipeline with CastAndScale op (double-buffered)
fn bench_streaming_cast_and_scale(c: &mut Criterion) {
    use kornia_gpu::gpu_backend::GpuBackend;
    use kornia_gpu::WgpuGpuBackend;

    const FRAMES: usize = 30;
    let (h, w) = (1080usize, 1920usize);

    let backend = WgpuGpuBackend::new();
    let input: Vec<u8> = (0..h * w * 3).map(|i| (i % 256) as u8).collect();

    let mut group = c.benchmark_group("streaming_cast");
    group.throughput(criterion::Throughput::Elements(FRAMES as u64));
    group.sample_size(20);

    // Warmup
    for _ in 0..5 {
        let _ = backend.cast_and_scale(&input, w as u32, h as u32, 1.0 / 255.0);
    }

    // Sequential: N frames × (upload + kernel + download), fully sequential
    group.bench_function("sequential_30frames", |b| {
        b.iter(|| {
            for _ in 0..FRAMES {
                let _ = backend.cast_and_scale(&input, w as u32, h as u32, 1.0 / 255.0);
            }
        })
    });

    // Pipelined: AsyncGpuPipeline overlaps frame N+1 upload with frame N kernel
    group.bench_function("pipelined_30frames", |b| {
        b.iter(|| {
            let mut pipeline = backend
                .streaming_cast_pipeline(w as u32, h as u32, 1.0 / 255.0)
                .expect("streaming_cast_pipeline not supported");
            let mut count = 0usize;
            for _ in 0..FRAMES {
                if pipeline.push_frame(&input, w as u32, h as u32).is_some() {
                    count += 1;
                }
            }
            if pipeline.flush().is_some() {
                count += 1;
            }
            assert_eq!(count, FRAMES);
        })
    });

    group.finish();
}

// CUDA benchmarks (--features cuda)

#[cfg(feature = "cuda")]
fn bench_warp_perspective_cuda(c: &mut Criterion) {
    use kornia_gpu::cuda::allocator::CudaAllocator;
    use kornia_gpu::cuda::image::CudaImageExt;
    use kornia_gpu::cuda::kernels;

    let cuda = CudaAllocator::new().expect("CUDA not available");
    let m = bev_homography();
    let mut group = c.benchmark_group("warp_perspective_cuda");
    group.sample_size(50);

    for &(h, w) in SIZES {
        let label = format!("{}x{}", w, h);
        let cpu_img = make_cpu_image(h, w);
        let cuda_img = cpu_img.to_cuda(&cuda).unwrap();

        // kernel only - data stays in VRAM
        group.bench_with_input(BenchmarkId::new("cuda_compute", &label), &(), |b, _| {
            b.iter(|| kernels::warp_perspective(&cuda_img, (h, w), &m).unwrap())
        });

        // e2e - upload + kernel + download
        group.bench_with_input(BenchmarkId::new("cuda_e2e", &label), &(), |b, _| {
            b.iter(|| {
                let img = cpu_img.to_cuda(&cuda).unwrap();
                let out = kernels::warp_perspective(&img, (h, w), &m).unwrap();
                out.to_cpu().unwrap()
            })
        });
    }
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_bev_pipeline_cuda(c: &mut Criterion) {
    use kornia_gpu::cuda::allocator::CudaAllocator;
    use kornia_gpu::cuda::image::CudaImageExt;
    use kornia_gpu::cuda::kernels;

    let cuda = CudaAllocator::new().expect("CUDA not available");
    let m = bev_homography();
    let mut group = c.benchmark_group("bev_pipeline_cuda");
    group.sample_size(50);

    for &(h, w) in SIZES {
        let label = format!("{}x{}", w, h);
        let cpu_img = make_cpu_image(h, w);

        // Full BEV pipeline: upload → cast → warp → gray → download
        group.bench_with_input(BenchmarkId::new("cuda_e2e", &label), &(), |b, _| {
            b.iter(|| {
                let img = cpu_img.to_cuda(&cuda).unwrap();
                let cast = kernels::cast_and_scale(&img, 1.0 / 255.0).unwrap();
                let warp = kernels::warp_perspective(&cast, (h, w), &m).unwrap();
                let gray = kernels::gray_from_rgb(&warp).unwrap();
                gray.to_cpu().unwrap()
            })
        });
    }
    group.finish();
}

#[cfg(feature = "cuda")]
criterion_group!(
    cuda_benches,
    bench_warp_perspective_cuda,
    bench_bev_pipeline_cuda,
);

criterion_group!(
    benches,
    bench_warp_perspective,
    bench_bev_pipeline,
    bench_streaming_pipeline,
    bench_streaming_cast_and_scale,
);

#[cfg(feature = "cuda")]
criterion_main!(benches, cuda_benches);

#[cfg(not(feature = "cuda"))]
criterion_main!(benches);
