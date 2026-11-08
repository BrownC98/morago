fn main() {
    link_clang_runtime_on_macos();
    delay_load_cuda_on_windows();
    tauri_build::build()
}

/// llama.cpp's Metal code uses `@available`, which needs
/// `___isPlatformVersionAtLeast` from the clang runtime (libclang_rt.osx.a).
/// rustc does not link that library, so the release link failed with
/// "Undefined symbols ... ___isPlatformVersionAtLeast" (CI run 42466893692).
fn link_clang_runtime_on_macos() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let out = std::process::Command::new("clang")
        .arg("--print-resource-dir")
        .output()
        .expect("clang is needed to build on macOS");
    let resource_dir = String::from_utf8(out.stdout).expect("clang output is UTF-8");
    let dir = std::path::Path::new(resource_dir.trim()).join("lib/darwin");
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
}

/// llama.cpp's CUDA backend imports these DLLs. With a static import Windows
/// refuses to start the exe on a PC without them (0xC0000135). Delay-loaded,
/// they load at the first CUDA call; `gpu::delayload` turns a missing DLL
/// into "no CUDA device", so the app runs on the CPU.
const CUDA_DLLS: [&str; 4] = [
    "nvcuda.dll",
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
];

fn delay_load_cuda_on_windows() {
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        return;
    }
    println!("cargo:rustc-link-lib=delayimp");
    for dll in CUDA_DLLS {
        println!("cargo:rustc-link-arg=/DELAYLOAD:{dll}");
    }
}
