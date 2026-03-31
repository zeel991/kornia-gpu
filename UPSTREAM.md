# Upstream Integration Plan

This prototype is structured to merge into kornia-rs with minimal friction.
The architecture maps directly onto kornia's existing abstractions - no new
patterns are introduced, only a second allocator path alongside `CpuAllocator`.

---

## The One Structural Change Required

kornia's `Image<T, C, A>` is already allocator-generic. The only thing blocking
GPU dispatch today is that every op in `kornia-imgproc` hardcodes `CpuAllocator`:

```rust
// Today - CpuAllocator hardcoded, GPU impossible
pub fn gray_from_rgb(
    src: &Image<f32, 3, CpuAllocator>,
    dst: &mut Image<f32, 1, CpuAllocator>,
) -> Result<()>

// After - allocator-generic, GPU slots in naturally
pub fn gray_from_rgb<A: TensorAllocator>(
    src: &Image<f32, 3, A>,
    dst: &mut Image<f32, 1, A>,
) -> Result<()>
```

All existing CPU tests pass unmodified - `CpuAllocator` is still the default.
GPU is additive, gated behind `#[cfg(feature = "gpu")]`.

---

## Target Crate Layout

```
kornia-rs/
├── crates/
│   ├── kornia-tensor/
│   │   └── src/allocator.rs       ← GpuAllocator added here alongside CpuAllocator
│   ├── kornia-imgproc/
│   │   └── src/                   ← ops made generic over A: TensorAllocator
│   │       └── Cargo.toml         ← [features] gpu = ["kornia-gpu"]
│   └── kornia-gpu/                ← this crate (new)
│       ├── Cargo.toml
│       └── src/
│           ├── allocator.rs       # GpuAllocator: TensorAllocator impl
│           ├── image.rs           # GpuImage<T,C> = Image<T,C,GpuAllocator>
│           ├── error.rs           # GpuError
│           ├── pool.rs            # GpuImagePool - pre-allocated VRAM, zero per-frame alloc
│           ├── pipeline.rs        # GpuPipeline - chained ops, single upload/download
│           ├── pipeline_async.rs  # GpuStreamingPipeline - double-buffered streaming
│           ├── gpu_backend.rs     # GpuBackend trait, WgpuGpuBackend, StreamKernel trait
│           ├── backend.rs         # Backend enum, AnyGpuImage, auto()
│           ├── kernels/
│           │   ├── mod.rs
│           │   ├── color.rs       # gray_from_rgb, gray_from_rgb_into
│           │   ├── cast.rs        # cast_and_scale, cast_and_scale_into
│           │   ├── warp.rs        # warp_perspective, warp_perspective_into
│           │   └── resize.rs      # resize_bilinear
│           ├── shaders/
│           │   ├── gray_from_rgb.wgsl
│           │   ├── cast_and_scale.wgsl
│           │   ├── warp_perspective.wgsl
│           │   └── resize_bilinear.wgsl
│           └── cuda/
│               ├── allocator.rs   # CudaAllocator
│               ├── image.rs       # CudaImageExt
│               └── kernels.rs     # gray_from_rgb, warp_perspective via cudarc
```

---

## Migration Steps

### Step 1 - Add `GpuAllocator` to `kornia-tensor`

`GpuAllocator` already implements `TensorAllocator`. It delegates host-side
allocation to the system allocator - the CPU-side `Image` is a shape carrier,
actual GPU data lives in `GpuMemory<T>` (a wgpu storage buffer).

```rust
// kornia-tensor/src/allocator.rs - add alongside CpuAllocator
impl TensorAllocator for GpuAllocator {
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        // delegates to system allocator - CPU side is shape-only
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() { Err(TensorAllocatorError::NullPointer) } else { Ok(ptr) }
    }
    fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !ptr.is_null() { unsafe { std::alloc::dealloc(ptr, layout) } }
    }
}
```

No changes to `CpuAllocator`, `Tensor`, or `Image`. Drop-in.

### Step 2 - Make `kornia-imgproc` ops allocator-generic

This is a mechanical change - add `<A: TensorAllocator>` to each op signature.
The implementation body is unchanged; only the type bounds widen.

```rust
// kornia-imgproc/src/color/gray.rs
pub fn gray_from_rgb<A: TensorAllocator>(
    src: &Image<f32, 3, A>,
    dst: &mut Image<f32, 1, A>,
) -> Result<(), ImageError> {
    // existing CPU implementation unchanged
}
```

### Step 3 - Add GPU dispatch behind feature flag

```toml
# kornia-imgproc/Cargo.toml
[features]
gpu = ["dep:kornia-gpu"]
```

```rust
// kornia-imgproc/src/color/gray.rs
pub fn gray_from_rgb<A: TensorAllocator>(
    src: &Image<f32, 3, A>,
    dst: &mut Image<f32, 1, A>,
) -> Result<(), ImageError> {
    #[cfg(feature = "gpu")]
    if let Some(gpu_result) = kornia_gpu::try_dispatch_gray_from_rgb(src, dst) {
        return gpu_result;
    }
    // CPU fallback - unchanged
    kornia_imgproc_cpu::gray_from_rgb(src, dst)
}
```

---

## What Does NOT Change

- `Image<T, C, A>` type - already allocator-generic, no changes needed
- All existing CPU tests - pass unmodified, `CpuAllocator` path unchanged
- Public `kornia-imgproc` API - GPU is purely additive
- `kornia-tensor` public API - `GpuAllocator` is an addition, not a replacement

---

## Key Abstractions and Their Upstream Roles

### `GpuAllocator`
Lives in `kornia-tensor`. The wgpu device, queue, and pre-compiled pipelines
are Arc-wrapped and shared across all `GpuImage` instances derived from the
same allocator. One `GpuAllocator::new()` call at application startup; all
downstream GPU ops share it cheaply via clone.

### `GpuImage<T, C>` = `Image<T, C, GpuAllocator>`
Same `[H, W, C]` layout as `kornia_image::Image`. `ImageExt` trait adds
`to_gpu(&alloc)` and `to_cpu()` for explicit transfers. No implicit copies.

### `GpuPipeline`
Chains multiple ops in VRAM - `cast → warp → gray` at 1080p costs one PCIe
upload and one download regardless of chain length. This is the primary
mechanism for amortizing transfer overhead in the BEV node.

### `GpuStreamingPipeline` + `StreamKernel` trait
Double-buffered async pipeline for frame streaming. `StreamKernel` separates
scheduling from kernel dispatch - in upstream kornia this becomes:

```rust
pub trait StreamKernel<A: TensorAllocator>: Send {
    fn process(
        &self,
        input: &Image<u8, 3, CpuAllocator>,
        alloc: &A,
    ) -> Image<u8, 1, A>;
}
```

The `Box<dyn StreamKernel>` in this prototype maps to the generic parameter
in the upstream version - same architecture, different dispatch mechanism.

### `GpuImagePool`
Pre-allocates VRAM buffers at startup. Explicit `acquire()`/`release()` makes
pool lifetime visible at the call site - avoids silent pool exhaustion in async
contexts where `Drop`-based return is unpredictable.

### Kernel `_into` variants
Every kernel has both an allocating variant and a write-into variant:

```rust
// Allocating - convenience, used in benchmarks and pipelines
pub fn gray_from_rgb(src: &GpuImage<f32, 3>) -> Result<GpuImage<f32, 1>, GpuError>

// Write-into - zero allocation, matches kornia-imgproc op convention
pub fn gray_from_rgb_into(
    src: &GpuImage<f32, 3>,
    dst: &GpuImage<f32, 1>,
) -> Result<(), GpuError>
```

The `_into` variants are the upstream-facing API. The allocating variants are
convenience wrappers used internally by `GpuPipeline`.

---

## CUDA Backend

The CUDA backend (`--features cuda`) is opt-in and does not affect the wgpu
path. `Backend::auto()` selects CUDA on NVIDIA hardware and falls back to wgpu
silently. In upstream kornia, CUDA would remain opt-in:

```toml
# kornia-gpu/Cargo.toml
[features]
cuda = ["dep:cudarc"]
```

CUDA kernels are written in `.cu`, compiled to PTX at build time via nvcc,
loaded at runtime via cudarc. This avoids the `nvcc` build-time dependency for
users who don't have CUDA installed.

---

## Files to Touch in kornia-rs

| File | Change |
|------|--------|
| `crates/kornia-tensor/src/allocator.rs` | Add `GpuAllocator` impl of `TensorAllocator` |
| `crates/kornia-tensor/Cargo.toml` | Add `wgpu`, `pollster`, `bytemuck` behind `gpu` feature |
| `crates/kornia-imgproc/src/color/gray.rs` | Generalize over `A: TensorAllocator` |
| `crates/kornia-imgproc/src/warp/perspective.rs` | Generalize over `A: TensorAllocator` |
| `crates/kornia-imgproc/src/imgproc.rs` | Add `cast_and_scale` generic variant |
| `crates/kornia-imgproc/Cargo.toml` | Add `gpu = ["dep:kornia-gpu"]` feature |
| `crates/kornia-gpu/` | New crate - this repo |
| `Cargo.toml` (workspace) | Add `kornia-gpu` to workspace members |