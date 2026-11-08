fn main() {
    link_clang_runtime_on_macos();
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
