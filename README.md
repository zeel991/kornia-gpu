# GSoC 2026 Proposal - GPU-Accelerated BEV Camera Application on Bubbaloop

**Applicant:** Zeel Darji ([@0xZeeast](https://github.com/zeel991))  
**Organization:** [kornia](https://github.com/kornia/kornia-rs)  
**Mentor:** Edgar Riba  
**Demo repo:** [github.com/zeel991/kornia-gpu](https://github.com/zeel991/kornia-gpu)  
**Prior work:** [PR #658](https://github.com/kornia/kornia-rs/pull/658) merged · [PR #767](https://github.com/kornia/kornia-rs/pull/767) under review

---

## The Problem

kornia-rs today is hardcoded to `CpuAllocator`. Every op in `kornia-imgproc` takes
`Image<T, C, CpuAllocator>` - GPU dispatch is architecturally impossible without
introducing a second allocator path. The fix is straightforward because `Image<T, C, A>`
is already allocator-generic: adding `GpuAllocator` as a `TensorAllocator` impl and
making ops generic over `A` is all that is needed. No existing CPU code changes.

This project builds that path, validates it with a working Bubbaloop BEV node, and
contributes it upstream to kornia-rs.

---

## Application: Bird's-Eye View Camera Pipeline

The chosen application is a real-time Bird's-Eye View (BEV) transform for a robotics
camera stream. A forward-facing camera feed is warped to a top-down road view using
`warp_perspective` with a calibrated IPM homography - directly useful for
Bubbaloop's self-driving use cases.

`warp_perspective` is the right anchor op for this project: every output pixel
requires 4 source pixel reads and bilinear interpolation, making it genuinely
compute-bound and GPU-appropriate. It also has a well-defined OpenCV reference for
accuracy validation.

### Pipeline

```
Camera frame (live)
        │
Zenoh Subscriber
        │
JPEG decode → RGB u8  (CPU)
        │
cast_and_scale u8→f32  (GPU)
        │
warp_perspective       (GPU, IPM homography)
        │
gray_from_rgb          (GPU, optional)
        │
GpuStreamingPipeline   (double-buffered async, overlaps upload/kernel)
        │
JPEG encode (turbojpeg, ~1ms)
        │
Zenoh Publisher → BEV topic
```

All three GPU kernels execute in VRAM without intermediate CPU downloads. Persistent
`LazyBuf` buffers eliminate per-frame VRAM allocation. GPU work runs on a dedicated
thread to avoid stalling the async Zenoh executor.

**This pipeline is already working.** The `bubbaloop-node` crate in this repo runs it
live: webcam → GPU kernels → JPEG → Zenoh, with per-frame timing printed to stdout.
Release mode on RTX 3050 Mobile: **5–8ms kernel time at 720p, ~1ms JPEG encode.**

---

## Pre-Application Implementation

### What is already built

**`GpuAllocator`** - implements `kornia_tensor::TensorAllocator`. Holds a wgpu
device/queue and four pre-compiled compute pipelines (compiled once at startup, zero
JIT per frame). Arc-wrapped, cheap to clone. Designed to live in `kornia-tensor`
alongside `CpuAllocator`.

**Four kernels, both backends:**
- `gray_from_rgb` - BT.601 RGB→grayscale
- `cast_and_scale` - type cast + scale (u8→f32 normalization)
- `warp_perspective` - perspective transform with bilinear interpolation
- `resize_bilinear` - half-pixel aligned, border-replicate clamping

Each kernel has an allocating variant and a write-into `_into` variant matching
kornia-imgproc's existing op signature convention.

**`GpuPipeline`** - chains multiple ops in VRAM without intermediate CPU downloads.
`cast → warp → gray` at 1080p: 17.3ms wgpu, 5.2ms CUDA (vs 48.3ms CPU for 3 passes).

**`GpuStreamingPipeline`** - double-buffered async pipeline. Uses a `StreamKernel`
trait so wgpu and CUDA kernels slot in without duplicating pipeline logic. Improves
sustained throughput from 101 fps to 144 fps at 1080p.

**`GpuImagePool`** - pre-allocated VRAM buffers, explicit `acquire()`/`release()`.
Zero per-frame GPU allocation in steady state.

**CUDA backend** - kernels written in CUDA C, compiled to PTX at build time via nvcc,
loaded at runtime via cudarc. `gray_from_rgb` and `warp_perspective` implemented.
`Backend::auto()` selects CUDA on NVIDIA hardware, falls back to wgpu silently.

**Tests:** 19 passing (wgpu) + CUDA parity tests including `test_cuda_warp_matches_wgpu`
(max diff 0.000035 - both backends produce identical output).

---

## Benchmarks

**Hardware:** RTX 3050 Mobile, Ubuntu 24.04, CUDA 12.x, Vulkan  
**Framework:** Criterion, 100 samples (wgpu), 50 samples (CUDA)

### The PCIe gap

Raw kernel speedups look impressive in isolation. The real challenge is PCIe transfer.
At 1080p with `gray_from_rgb`:

```
CPU baseline:           43 ms
GPU kernel-only:         0.55 ms   (78× faster - data already in VRAM)
GPU E2E, naive:         29 ms      (alloc new buffers + upload + kernel + download)
GPU E2E, persistent:    10.5 ms    (LazyBuf reuse - 4.1× faster than CPU)
GPU E2E, zero-copy:     ~1–2 ms    (Phase 2 target - MAPPABLE_PRIMARY_BUFFER)
```

Persistent buffer reuse alone cut E2E from 29ms to 10.5ms. The remaining gap
is physical PCIe bandwidth - zero-copy closes it.

### Full BEV pipeline: cast → warp → gray

Chaining ops in VRAM means paying the PCIe cost once instead of three times:

| Resolution | CPU (3 passes) | wgpu E2E | CUDA E2E | wgpu speedup | CUDA speedup |
|------------|----------------|----------|----------|--------------|--------------|
| 720p       | 12.6 ms        | 9.0 ms   | 2.2 ms   | 1.4×         | **5.8×**     |
| 1080p      | 48.3 ms        | 17.3 ms  | 5.2 ms   | 2.8×         | **9.3×**     |
| 4K         | 211.1 ms       | 69.7 ms  | 20.0 ms  | 3.0×         | **10.5×**    |

CUDA E2E at 1080p: 5.2ms - well within the 33ms budget for 30fps on Bubbaloop.
On Jetson Orin (unified memory), PCIe transfer cost is zero and E2E approaches
kernel-only time.

### warp_perspective

| Resolution | CPU     | wgpu E2E | CUDA E2E |
|------------|---------|----------|----------|
| 720p       | 5.8 ms  | 19.5 ms  | 2.7 ms   |
| 1080p      | 12.5 ms | 35.9 ms  | 6.1 ms   |
| 4K         | 47.3 ms | 154.8 ms | 78.1 ms  |

wgpu E2E is slower than CPU in isolation because warp_perspective is
memory-bandwidth-bound and this measures one PCIe round-trip per op.
In the BEV pipeline sharing one transfer, wgpu reaches 2.8× at 1080p.

### CubeCL → raw WGSL

Early prototype used CubeCL. Replaced with handwritten WGSL after JIT dispatch
overhead dominated at 1080p+:

| Resolution | CubeCL E2E | Raw WGSL E2E |
|------------|------------|--------------|
| 720p       | 9.7 ms     | 18.9 ms      |
| 1080p      | 56.7 ms    | 35.1 ms      |
| 4K         | 309.0 ms   | 164.6 ms     |

`GpuPipelines` compiles all four WGSL shaders once at `GpuAllocator::new()` -
zero JIT overhead per frame at any resolution.

### Streaming pipeline throughput (gray_from_rgb, 30 frames, 1080p)

| Mode                  | Throughput |
|-----------------------|------------|
| Sequential            | 101 fps    |
| Async double-buffered | **144 fps** (+42%) |

On desktop PCIe the gain is modest because transfer dominates kernel time.
On Jetson Orin the gain is larger - async pipelining keeps the GPU fed
without CPU stalls between frames.

### Accuracy vs OpenCV `INTER_LINEAR`

| Metric              | Value      |
|---------------------|------------|
| Mean abs pixel diff | 0.417      |
| Match ≤1 intensity  | **99.25%** |
| CUDA vs wgpu parity | **100%** (max diff 0.000035) |

---

## Design Decisions

### Why raw WGSL over CubeCL

CubeCL's `#[cube(launch)]` macro compiles Rust to Vulkan/Metal/DX12. The appeal is
portability. The problem: JIT compilation per dispatch dominates at 1080p+ (56.7ms
vs 35.1ms at 1080p). `GpuPipelines` compiles all four WGSL shaders once at startup,
achieving stable dispatch overhead at every resolution.

### Why raw cudarc over CubeCL's CUDA backend

Switching `WgpuRuntime` to `CudaRuntime` propagates a runtime generic through every
type - `GpuMemory<T, R>`, `TensorArg<R>`, `ComputeClient<R>` - touching every public
API surface. The alternative: write kernels in CUDA C, compile to PTX at build time,
load via cudarc. Three `.cu` files, one `build.rs`. CUDA kernels use `__ldg()` for
texture cache reads - a hardware optimization CubeCL's compute-only model cannot expose.

### Why `cuda` is opt-in

cudarc uses dynamic loading so the binary compiles anywhere. But `nvcc` runs at build
time, contradicting kornia-rs's minimal dependency philosophy. Users on AMD or Apple
Silicon should not have CUDA in their dependency tree.

### Why `StreamKernel` trait in `GpuStreamingPipeline`

The double-buffered scheduling logic is backend-agnostic. `StreamKernel` separates
scheduling from kernel implementation - `WgpuGrayKernel` and `CudaGrayKernel` slot in
without duplicating pipeline code. In kornia upstream this maps directly to a generic
`StreamKernel<A: TensorAllocator>` - the architecture is already correct for upstream.

---

## Upstream Integration

kornia's `Image<T, C, A>` is already allocator-generic. The integration requires
two changes and breaks nothing:

**1. Add `GpuAllocator` to `kornia-tensor`**

Already implements `TensorAllocator` - drop-in alongside `CpuAllocator`.

**2. Make `kornia-imgproc` ops allocator-generic**

```rust
// Today
pub fn gray_from_rgb(
    src: &Image<f32, 3, CpuAllocator>,
    dst: &mut Image<f32, 1, CpuAllocator>,
) -> Result<()>

// After
pub fn gray_from_rgb<A: TensorAllocator>(
    src: &Image<f32, 3, A>,
    dst: &mut Image<f32, 1, A>,
) -> Result<()>
```

GPU dispatch slots in behind `#[cfg(feature = "gpu")]`. All existing CPU tests pass
unmodified. See [UPSTREAM.md](./UPSTREAM.md) for full crate layout and migration steps.

---

## 12-Week Plan

| Weeks | Phase | Deliverables | Exit Criteria |
|-------|-------|--------------|---------------|
| 1–3 | kornia-rs integration | `GpuAllocator` in kornia-tensor. `gpu` feature in kornia-imgproc. `warp_perspective`, `gray_from_rgb`, `cast_and_scale` dispatch to wgpu + CUDA. CPU path untouched. | Draft PR open. All existing tests passing. Example in `examples/`. |
| 4–6 | Performance hardening | Zero-copy buffer path (`MAPPABLE_PRIMARY_BUFFER`). Additional kernels: `resize`, `normalize`. Criterion benchmarks in kornia-rs CI. | E2E at 1080p within 2× of kernel-only on wgpu. |
| 7–9 | Bubbaloop BEV node | Full Zenoh node: live camera → cast → warp → gray → publish. `GpuImagePool` zero per-frame allocation. Stage-level latency logging. | Working node at 30fps. Measured end-to-end latency. |
| 10–12 | Edge validation + demo | Deploy on edge GPU hardware. Validate both backends. Video demo with GPU vs CPU comparison. Final PRs. | Reproducible benchmark on target hardware. Demo published. |

---

## Stretch Goals

**Additional kernels:** `remap` (LUT-based undistortion), `normalize` (ML preprocessing).

**Texture sampler for wgpu:** Use wgpu texture bindings for `warp_perspective` instead
of storage buffers - delegates bilinear interpolation to hardware TMUs.

**Multi-camera BEV:** Stitch two camera streams into a surround top-down view -
demonstrates the library's composability.

---

## Risks and Mitigations

**PCIe transfer dominates at 4K.**
`GpuPipeline` and `_into` variants keep intermediate results in VRAM.
Phase 2 zero-copy path eliminates the transfer entirely.

**Edge hardware availability.**
Primary validation: cloud A100 (sm_80) - access confirmed.
Jetson Orin via IIIT Nagpur robotics lab being confirmed.
Both covered by `compute_75` PTX - no recompilation needed.

---

## Prior Work in kornia-rs

**[PR #658](https://github.com/kornia/kornia-rs/pull/658) - Euclidean Distance Transform (merged)**
O(N) Felzenszwalb & Huttenlocher EDT replacing a prior O(N²) implementation.
Updated to `ImageAllocator`/`CpuAllocator` signatures.

**[PR #767](https://github.com/kornia/kornia-rs/pull/767) - Suzuki-Abe contour detection (under review)**
CComp hierarchy indexing fix, `WorkBuffers` for buffer reuse across frames,
`PARALLEL_THRESHOLD` raised to 1080p with benchmark evidence,
full example at `examples/find_contours.rs`.

---

## About Me

Final-year B.Tech CSE student at IIIT Nagpur, graduating mid-2026. Fully available
from day one of the coding period - no internship or employment commitments during GSoC.

Two years of Rust: Substrate runtime pallets → kornia-rs contributions → GPU compute
with wgpu, cudarc, and CubeCL. Ranked 2nd at C4GT 2024 among 500+ participants.

**GitHub:** [@0xZeeast](https://github.com/zeel991) · **Discord:** zeel#2929  
**Timezone:** IST (UTC+5:30) · Weekly progress updates on Discord and GitHub

---

## AI Tooling Disclosure

In compliance with kornia-rs AI policy: this project is the result of my own
implementation and design. AI tools were used selectively for research, optimization
ideas, and debugging. Every line has been manually verified, tested, and benchmarked.
I take full responsibility for correctness, safety, and architectural decisions.