//! Error types for kornia-gpu.

use thiserror::Error;

/// Errors that can occur in GPU operations.
#[derive(Debug, Error)]
pub enum GpuError {
    /// Error from kornia-image operations.
    #[error("Image error: {0}")]
    ImageError(#[from] kornia_image::ImageError),

    /// Kernel launch failed.
    #[error("Kernel launch failed: {0}")]
    KernelLaunchError(String),

    /// Shape mismatch between src and dst images.
    #[error("Shape mismatch: src {0:?} != dst {1:?}")]
    ShapeMismatch(Vec<usize>, Vec<usize>),

    /// Singular homography matrix (determinant is zero).
    #[error("Singular homography matrix (det = 0)")]
    SingularHomography,

    /// CUDA runtime error.
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// All buffers in the pool are currently acquired.
    #[error("GPU image pool exhausted - all buffers in use")]
    PoolExhausted,

    /// Buffer dimensions don't match the operation's required output size.
    #[error("Buffer size mismatch: expected {0}x{1}, got {2}x{3}")]
    BufferSizeMismatch(usize, usize, usize, usize),
}
