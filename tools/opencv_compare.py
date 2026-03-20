#!/usr/bin/env python3
"""
Compare Rust GPU warp output against OpenCV reference.

Usage:
    python3 tools/opencv_compare.py input.jpg output_gpu.bin WIDTH HEIGHT

Applies the same homography as BEV_HOMOGRAPHY in examples/bev_compare.rs
using OpenCV, then computes pixel-level statistics.
"""

import sys
import numpy as np
import cv2

# Must match BEV_HOMOGRAPHY in examples/bev_compare.rs
H = np.array([
    [ 1.2,    0.05, -120.0],
    [-0.03,   1.15,  -90.0],
    [ 0.0001, 0.0002,  1.0],
], dtype=np.float32)

def main():
    if len(sys.argv) < 5:
        print("Usage: python3 tools/opencv_compare.py input.jpg output_gpu.bin WIDTH HEIGHT")
        sys.exit(1)

    input_path = sys.argv[1]
    gpu_bin    = sys.argv[2]
    w          = int(sys.argv[3])
    h          = int(sys.argv[4])
    opencv_out = gpu_bin.replace(".bin", "_opencv.png")

    src = cv2.imread(input_path)
    if src is None:
        print(f"Error: cannot read {input_path}")
        sys.exit(1)

    # OpenCV reference warp (INTER_LINEAR = bilinear, matches GPU kernel)
    ref = cv2.warpPerspective(src, H, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.imwrite(opencv_out, ref)
    print(f"OpenCV reference saved: {opencv_out}")

    # Load Rust GPU output (raw f32 RGB bytes)
    raw = np.frombuffer(open(gpu_bin, "rb").read(), dtype=np.float32)
    gpu_f32 = raw.reshape(h, w, 3)
    gpu_u8  = (gpu_f32 * 255.0).clip(0, 255).astype(np.uint8)
    # Convert RGB -> BGR for OpenCV comparison
    gpu_bgr = gpu_u8[:, :, ::-1]

    cv2.imwrite(gpu_bin.replace(".bin", "_gpu.png"), gpu_bgr)

    # Pixel diff statistics
    diff = np.abs(ref.astype(np.float32) - gpu_bgr.astype(np.float32))
    total = diff.size

    print(f"\nResolution: {w}x{h}")
    print(f"Mean abs pixel diff:  {diff.mean():.4f}")
    print(f"Max abs pixel diff:   {diff.max():.0f}")
    print(f"Match <=1 intensity:  {(diff <= 1).sum() / total * 100:.2f}%")
    print(f"Match <=2 intensity:  {(diff <= 2).sum() / total * 100:.2f}%")
    print(f"Match <=4 intensity:  {(diff <= 4).sum() / total * 100:.2f}%")
    print(f"\nGPU output image: {gpu_bin.replace('.bin', '_gpu.png')}")

if __name__ == "__main__":
    main()