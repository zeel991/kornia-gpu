//! Build script: compile CUDA kernels to PTX when nvcc is available.
//!
//! If nvcc is not found, CUDA kernel compilation is skipped and the
//! cuda module falls back gracefully - CudaAllocator::new() will fail
//! at runtime on non-NVIDIA machines anyway.
//!
//! CUDA toolkit path on Debian/Ubuntu: /usr/lib/cuda
//! Install with: sudo apt install nvidia-cuda-toolkit

fn main() {
    println!("cargo:rustc-check-cfg=cfg(cuda_kernels_compiled)");
    if try_compile_cuda_kernels() {
        println!("cargo:rustc-cfg=cuda_kernels_compiled");
    }
    println!("cargo:rerun-if-changed=build.rs");
}

fn try_compile_cuda_kernels() -> bool {
    use std::path::PathBuf;
    use std::process::Command;

    // Find nvcc - check common locations
    let nvcc_candidates = [
        "/usr/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/usr/lib/nvidia-cuda-toolkit/bin/nvcc",
    ];

    let nvcc = nvcc_candidates
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .copied();

    let nvcc = match nvcc {
        Some(p) => p,
        None => {
            println!("cargo:warning=nvcc not found - CUDA kernels not compiled. Install nvidia-cuda-toolkit for CUDA support.");
            return false;
        }
    };

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let kernels_dir = PathBuf::from("kernels/cuda");

    // Verify kernels directory exists
    if !kernels_dir.exists() {
        println!("cargo:warning=kernels/cuda/ not found - skipping CUDA kernel compilation.");
        return false;
    }

    // Detect SM version from nvidia-smi, default to sm_75 (Turing, widely compatible)
    let sm = detect_sm_version().unwrap_or("75".to_string());
    println!("cargo:warning=Compiling CUDA kernels for sm_{}", sm);

    let cuda_include = if std::path::Path::new("/usr/lib/cuda/include").exists() {
        "/usr/lib/cuda/include"
    } else {
        "/usr/local/cuda/include"
    };

    let kernels = ["cast_and_scale", "warp_perspective", "gray_from_rgb"];
    let mut all_ok = true;

    for kernel in &kernels {
        let src = kernels_dir.join(format!("{}.cu", kernel));
        let ptx = out_dir.join(format!("{}.ptx", kernel));

        if !src.exists() {
            println!(
                "cargo:warning=CUDA kernel source not found: {}",
                src.display()
            );
            all_ok = false;
            continue;
        }

        let arch = format!("arch=compute_{},code=sm_{}", sm, sm);
        let status = Command::new(nvcc)
            .args([
                "-ptx",
                "-O3",
                "--generate-code",
                &arch,
                "-I",
                cuda_include,
                "-I",
                "/usr/include",
                "-o",
                ptx.to_str().unwrap(),
                src.to_str().unwrap(),
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:rerun-if-changed=kernels/cuda/{}.cu", kernel);
            }
            Ok(s) => {
                println!("cargo:warning=nvcc failed for {}.cu (exit {})", kernel, s);
                all_ok = false;
            }
            Err(e) => {
                println!("cargo:warning=Failed to run nvcc: {}", e);
                all_ok = false;
            }
        }
    }

    all_ok
}

fn detect_sm_version() -> Option<String> {
    // Query nvidia-smi for compute capability
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;

    let cap = String::from_utf8(out.stdout).ok()?;
    let cap = cap.trim();
    // "8.6" → "86"
    let sm = cap.replace('.', "");
    if sm.chars().all(|c| c.is_ascii_digit()) && sm.len() == 2 {
        Some(sm)
    } else {
        None
    }
}
