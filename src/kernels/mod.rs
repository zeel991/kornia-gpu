//! GPU kernels for kornia-imgproc operations.
//!
//! Each module provides a GPU implementation of an existing CPU function
//! in kornia-imgproc. The CPU functions are the correctness reference.

pub mod cast;
pub mod color;
pub mod warp;

pub use cast::{cast_and_scale, cast_and_scale_into};
pub use color::gray_from_rgb;
pub use warp::{warp_perspective, warp_perspective_into};