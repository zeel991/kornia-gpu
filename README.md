# GSoC 2026 Proposal - GPU-Accelerated BEV Camera Application on Bubbaloop

**Applicant:** Zeel Darji ([@0xZeeast](https://github.com/0xZeeast))  
**Organization:** [kornia](https://github.com/kornia/kornia-rs)  
**Mentor:** Edgar Riba  
**Pre-application implementation:** [`feat/kornia-gpu-cuda`](https://github.com/0xZeeast/kornia-rs/tree/feat/kornia-gpu-cuda)

---

## Application Choice: Bird's-Eye View Transform

The chosen application is a real-time Bird's-Eye View (BEV) camera pipeline for a self-driving scenario. A forward-facing camera stream is transformed to a top-down road view using `warp_perspective` - a perspective warp with a calibrated IPM (Inverse Perspective Mapping) homography.

This choice is motivated by:
- `warp_perspective` is compute-intensive and GPU-appropriate - every output pixel requires 4 source pixel reads and bilinear interpolation
- BEV is directly useful for Bubbaloop's robotics/self-driving use cases
- The operation has a well-defined OpenCV reference for accuracy validation
- It runs at every frame, making per-frame overhead elimination (via `GpuImagePool`) measurable and meaningful

**GPU technology chosen: CubeCL (wgpu backend) + raw CUDA via cudarc, with runtime hardware detection.**

### Application pipeline

```
Camera (live input)
  │
  ▼
Zenoh Subscriber
  │
  ▼
JPEG/PNG decode → RGB8  (CPU, dedicated thread)
  │
  ▼
cast_and_scale u8→f32   (GPU kernel)
  │
  ▼
warp_perspective        (GPU kernel, IPM homography)
  │
  ▼
gray_from_rgb           (GPU kernel, optional)
  │
  ▼
Zenoh Publisher → BEV topic
```

All three GPU kernels execute in VRAM without intermediate CPU downloads. `GpuImagePool` pre-allocates output buffers at startup - zero GPU allocation per frame in steady state. GPU work runs on a dedicated thread to avoid stalling the async Zenoh executor.

The BEV node will be implemented as a Bubbaloop node following the existing node conventions - subscribing to a camera frame topic, running the GPU pipeline on a dedicated thread, and publishing the warped output on a BEV topic. Topic names and frame dimensions will be configurable via environment variables. The exact message format and topic schema will be aligned with the Bubbaloop SDK during Phase 3.

---

## Pre-Application Work

A working `kornia-gpu` crate has been built and integrated into the kornia-rs workspace. This is the GPU backend that will be contributed to kornia-rs as the primary GSoC deliverable.

Full source: [`feat/kornia-gpu-cuda`](https://github.com/0xZeeast/kornia-rs/tree/feat/kornia-gpu-cuda)

### What is already built

**kornia-gpu crate (workspace-integrated):**
- `GpuAllocator` implementing kornia's `TensorAllocator`
- `GpuImage<T,C>` with same `[H,W,C]` layout as `kornia_image::Image`
- `GpuImagePool` - pre-allocated VRAM, zero per-frame allocation
- `GpuPipeline` - kernel chaining without intermediate CPU downloads
- `Backend::auto()` - runtime hardware detection, CUDA on NVIDIA, wgpu everywhere else
- `AnyGpuImage<C>` - backend-erased image handle, same API regardless of backend

**Kernels (both wgpu and CUDA):**
- `cast_and_scale` + `cast_and_scale_into`
- `warp_perspective` + `warp_perspective_into`
- `gray_from_rgb` (BT.601 weights)

**Tests:** 18 passing (wgpu), 25 passing (--features cuda), including `test_cuda_warp_matches_wgpu` which verifies both backends produce identical output.

---

## Design Decisions

### CubeCL for wgpu - chosen for kernel portability

CubeCL 0.9 lets us write one kernel in Rust using `#[cube(launch)]` that targets Vulkan, Metal, and DX12 - no platform-specific shader code. Tested on Vulkan (NVIDIA). Metal and DX12 are CubeCL's stated targets but are not yet verified in this implementation.

At 1080p+, kernel dispatch overhead begins to dominate. This is a known CubeCL limitation - persistent command buffers would address it and are a planned Phase 2 deliverable. For Bubbaloop's primary deployment target (NVIDIA Jetson), the CUDA backend eliminates this entirely.

### Raw CUDA via cudarc - not CubeCL's CUDA backend

CubeCL has a CUDA backend. The first attempt was to switch `WgpuRuntime` to `CudaRuntime`. This failed: the `Runtime` generic parameter propagates through every type - `GpuMemory<T, R>`, `TensorArg<R>`, `ComputeClient<R>` - requiring changes to every public API surface.

The alternative: write kernels in CUDA C, compile to PTX via nvcc at build time, load via cudarc 0.19. Three `.cu` files, one `build.rs`, ~200 lines total. The CUDA kernels use `__ldg()` for texture cache reads - a hardware optimization CubeCL's compute-only model cannot expose.

On maintenance surface: both backends share identical kernel logic and the same public API. The CUDA `.cu` files mirror the CubeCL `#[cube(launch)]` kernels operation-for-operation. Adding a new operation means writing it once in CubeCL and once in CUDA C - the difference is a few hundred lines, and correctness is enforced by `test_cuda_warp_matches_wgpu` style parity tests for every kernel.

### `cuda` as opt-in feature - not the default

The first implementation had `cuda` in `[features] default`. The reasoning: cudarc uses dynamic loading (`libcuda.so` opened at runtime), so the binary compiles on any machine and falls back to wgpu if no NVIDIA GPU is present.

This was reverted. A default dependency means `cudarc` compiles on every machine and `nvcc` is invoked at build time - contradicting kornia-rs's minimal dependency philosophy. Users on AMD or Apple Silicon should not have CUDA in their dependency tree.

**Current design:** `cuda = ["dep:cudarc"]`, opt-in. `Backend::auto()` still tries CUDA first when the feature is enabled, falls back to wgpu silently.

### `Backend::auto()` with `AnyGpuImage<C>` - runtime dispatch over compile-time generics

The generic parameter approach (`fn warp<B: Backend>`) propagates `B` through every type and every caller. The runtime enum approach matches on the backend once per kernel call - negligible overhead compared to kernel time, and keeps the public API clean.

```rust
let backend = Backend::auto()?;
// NVIDIA + --features cuda → CUDA
// Everything else          → wgpu/Vulkan

let gpu_img = backend.upload(&cpu_img)?;
let warped  = backend.warp_perspective(&gpu_img, (h, w), &H)?;
let result  = backend.download(&warped)?;
```

### `GpuImagePool` with explicit `acquire()`/`release()`

The alternative is automatic buffer return on drop via `Arc<PooledBufferGuard>`. This is ergonomic but if the caller holds the image past a cancel point in async code, the buffer stays out of the pool until drop runs - causing `PoolExhausted` in a bounded pool that is hard to debug.

Explicit `acquire()`/`release()` makes pool lifetime visible and auditable at the call site.

---

## Benchmarks

`warp_perspective` - RTX 3050 Laptop, Ubuntu 24.04, CUDA 12.0.  
Criterion: 100 samples wgpu, 50 samples CUDA.

### Kernel-only (data in VRAM, no PCIe transfer)

| Resolution | CPU rayon | wgpu | CUDA | CUDA vs CPU |
|---|---|---|---|---|
| 720p  (1280×720)  | 9.34 ms  | 3.38 ms   | 0.316 ms | **29.6×** |
| 1080p (1920×1080) | 17.30 ms | 24.88 ms  | 0.678 ms | **25.5×** |
| 4K    (3840×2160) | 56.70 ms | 103.49 ms | 2.59 ms  | **21.9×** |

### End-to-end (upload + kernel + download)

| Resolution | wgpu | CUDA |
|---|---|---|
| 720p  | 10.69 ms | 2.91 ms |
| 1080p | 74.48 ms | 6.63 ms |
| 4K    | 375.07 ms | 83.73 ms |

CUDA e2e at 1080p: 6.63ms - within the 33ms budget for 30fps on Bubbaloop.

### Accuracy vs OpenCV `INTER_LINEAR`

| Metric | Value |
|---|---|
| Mean abs pixel diff | 0.417 |
| Max abs pixel diff | 192 (border pixel) |
| Match ≤1 intensity | **99.25%** |
| Match ≤2 intensity | 99.84% |
| CUDA vs wgpu parity | **100%** (max diff 0.000035) |

---

## GSoC Deliverables

The four required outcomes and how this project addresses each:

**1. Working Bubbaloop application with live camera input on edge GPU hardware**

A full Zenoh node: camera subscribe → decode → GPU pipeline (cast → warp → gray) → publish. `GpuImagePool` for zero per-frame allocation. GPU work on dedicated thread off async executor. Primary validation target: A100 cloud instance (sm_80). Jetson Orin (sm_87) if hardware access is available - both are within `compute_75` PTX coverage per NVIDIA specifications, though Jetson validation is not yet confirmed.

**2. GPU implementations contributed to kornia-rs**

`kornia-gpu` becomes a workspace crate in kornia-rs implementing both CubeCL (wgpu) and CUDA backends. `kornia-imgproc` gains a `gpu` feature flag. The following operations get GPU dispatch when a `Backend` is provided - CPU path completely untouched, zero breaking changes:

```rust
// Proposed API - exact signature to be agreed with maintainers during Phase 1

// wgpu backend - tested on Vulkan/NVIDIA
let backend = Backend::wgpu();
imgproc::warp::warp_perspective(&src, &mut dst, &transform, mode, Some(&backend))?;

// CUDA backend - NVIDIA only, --features cuda
let backend = Backend::cuda()?;
imgproc::warp::warp_perspective(&src, &mut dst, &transform, mode, Some(&backend))?;

// Auto-detect - picks CUDA on NVIDIA, wgpu everywhere else
let backend = Backend::auto()?;
imgproc::warp::warp_perspective(&src, &mut dst, &transform, mode, Some(&backend))?;

// No backend - CPU path, unchanged behaviour
imgproc::warp::warp_perspective(&src, &mut dst, &transform, mode, None)?;
```

Both backends implement the same kernels:
- `warp_perspective` - perspective transform with bilinear interpolation
- `gray_from_rgb` - BT.601 RGB to grayscale
- `cast_and_scale` - type cast + scale (u8→f32 for upload, f32→u8 for download)

wgpu kernel source: `#[cube(launch)]` Rust kernels via CubeCL 0.9.
CUDA kernel source: `.cu` files compiled to PTX via nvcc, loaded at runtime via cudarc 0.19.
Both produce identical output - verified by `test_cuda_warp_matches_wgpu` (max diff 0.000035).

**3. Benchmarks comparing GPU vs CPU**

Already built - criterion suite at 720p/1080p/4K for all three kernels. Will be integrated into kornia-rs CI.

**4. Video demo on target hardware**

Recorded at the end of Phase 4 on target GPU hardware, showing live BEV transform with GPU vs CPU latency comparison.

### 12-Week Plan

| Weeks | Phase | Deliverables | Exit Criteria |
|---|---|---|---|
| 1–3 | kornia-rs integration | `gpu` feature in kornia-imgproc. `warp_perspective`, `gray_from_rgb`, `cast_and_scale` dispatch to both CubeCL (wgpu) and CUDA backends. CPU path untouched. | Draft PR open in kornia/kornia-rs. Tests passing for both backends. Example in `examples/`. |
| 4–6 | Performance hardening | Persistent wgpu command buffers (reduces 1080p+ dispatch overhead). Additional kernels: `resize`, `normalize`. CUDA `compute_75` PTX validated on available hardware. | Criterion benchmarks at 720p/1080p/4K for all kernels. wgpu dispatch overhead measurably reduced vs baseline. |
| 7–9 | Bubbaloop BEV node | Full Zenoh node: live camera → GPU pipeline → publish. Both backends exercised. `GpuImagePool` zero per-frame allocation. Stage-level latency logging (decode/h2d/kernel/d2h/encode). | Working node on development machine with measured end-to-end latency. |
| 10–12 | Edge validation + video demo | Deploy on GPU-equipped edge hardware. Validate CUDA and wgpu paths. CI benchmark integration. Video demo showing GPU vs CPU latency comparison. | Reproducible benchmark on target hardware. Video demo published. Final PR links consolidated. |

---

## Stretch Goals

If the core deliverables are completed ahead of schedule:

**Additional kernels:**
- `remap` - general pixel remapping, enables multi-camera undistortion and LUT-based transforms
- `normalize` - per-channel mean/std normalization for ML preprocessing

**CubeCL optimizations:**
- Persistent command buffers - eliminate Vulkan dispatch overhead at 1080p+, bringing wgpu kernel time in line with CUDA
- Shared memory tiling for `warp_perspective` - reduces global memory reads for the bilinear interpolation scatter pattern

**Multi-camera BEV:**
- Extend the pipeline to stitch two camera streams into a single top-down surround view - a natural extension of the single-camera BEV that demonstrates the library's composability

**Texture sampler for wgpu:**
- Use wgpu texture bindings for `warp_perspective` instead of storage buffers - delegates bilinear interpolation to hardware texture units (TMUs), removing the manual interpolation in the shader

---

## Risks and Mitigations

**PCIe transfer dominates at 4K.** Mitigation: `GpuImagePool` and `_into` variants keep intermediate results in VRAM. Only final output crosses PCIe.

**Edge hardware.** Primary validation: cloud GPU instance (A100, sm_80) - access confirmed. I am in the process of confirming Jetson Orin access via the IIIT Nagpur robotics lab before the submission deadline. Both targets are covered by the same `compute_75` PTX binary - no recompilation needed regardless of which hardware is used.

---

## Prior Work in kornia-rs

**[PR #658](https://github.com/kornia/kornia-rs/pull/658) - Euclidean Distance Transform (merged)**  
Implemented the Felzenszwalb & Huttenlocher O(N) EDT algorithm for binary images, replacing a previous O(N²) implementation. Used the parabolic lower envelope approach. Updated function signatures to use `ImageAllocator` and `CpuAllocator`. Fixed the vanilla implementation by removing ndarray dependency and correcting loop bounds.

**[PR #767](https://github.com/kornia/kornia-rs/pull/767) - Suzuki-Abe contour detection (under review)**  
Contour detection for `kornia-imgproc`. CComp hierarchy indexing fix, `WorkBuffers` struct for buffer reuse across frames, structured `ContoursError` enum, benchmarked `PARALLEL_THRESHOLD` with data, full example at `examples/find_contours.rs`.

---

## About Me

I am a final-year B.Tech CSE student at IIIT Nagpur, graduating mid-2026. I have no internship or employment commitments during the GSoC period - I am fully available from day one of the coding period through the final evaluation. I have been writing Rust for two years, starting with Substrate runtime pallet test cases and progressing to systems programming, GPU compute, and blockchain infrastructure.

**Relevant experience:**
- **Rust:** CubeCL kernels, cudarc, Substrate pallets, kornia-rs contributions
- **GPU:** CUDA C kernel authoring, PTX compilation, wgpu/Vulkan via CubeCL, bilinear interpolation implementation
- **Computer vision:** OpenCV (Python/C++), kornia-rs, image processing pipelines
- **Systems:** Memory allocator design, zero-copy buffer pooling, async-safe GPU dispatch

**Open source track record:**
- Ranked 2nd at C4GT 2024 among 500+ participants
- kornia-rs: [find_contours PR #767](https://github.com/kornia/kornia-rs/pull/767) - contribution to kornia-imgproc (under review)
- Multiple solo hackathon wins in Web3/DeFi (Best DeFi Product at Asset Hub Goa, Avalanche Mumbai, Monad Blitz Nagpur)

**Availability:** Full-time June–September 2026. No internship or employment commitments during this period. Cloud GPU access confirmed for edge validation. IST (UTC+5:30). Weekly progress updates on Discord and GitHub.

**GitHub:** [@0xZeeast](https://github.com/0xZeeast)  
**Discord:** @0xZeeast

---

## AI Tooling Disclosure

In compliance with kornia-rs AI policy: Claude was used as a coding collaborator - API design, debugging CubeCL type errors, cudarc API reference, and documentation. Every line has been executed, tested, and benchmarked on my machine. I can explain every decision and reproduce every benchmark. Full responsibility for correctness, safety, and licensing.