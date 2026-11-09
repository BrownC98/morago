// transcribe-cpp-sys, shared-ggml variant. transcribe.cpp is built against the ggml
// that llama-cpp-sys-2 already built and links (DEP_LLAMA_GGML_CMAKE_DIR), so the
// final binary has ONE ggml. No ggml sources and no ggml link lines here.
// src/transcribe_sys.rs is the pre-generated binding of the crate: no bindgen.
use cmake::Config;
use std::env;
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap();
    println!("cargo:rerun-if-changed=transcribe.cpp");
    println!("cargo:rerun-if-env-changed=DEP_LLAMA_GGML_CMAKE_DIR");

    let ggml_cmake = PathBuf::from(
        env::var("DEP_LLAMA_GGML_CMAKE_DIR")
            .expect("llama-cpp-sys-2 did not export ggml_cmake_dir (needs 0.1.161 or newer)"),
    );
    // <prefix>/lib/cmake -> <prefix>
    let ggml_prefix = ggml_cmake.parent().unwrap().parent().unwrap().to_path_buf();

    let mut config = Config::new("transcribe.cpp");
    config
        .profile("Release")
        .define("TRANSCRIBE_USE_SYSTEM_GGML", "ON")
        .define("CMAKE_PREFIX_PATH", &ggml_prefix)
        .define("TRANSCRIBE_INSTALL", "OFF")
        .define("TRANSCRIBE_BUILD_TESTS", "OFF")
        .define("TRANSCRIBE_BUILD_EXAMPLES", "OFF")
        .define("TRANSCRIBE_BUILD_TOOLS", "OFF")
        .define("TRANSCRIBE_BUILD_SHARED", "OFF")
        // No BLAS on Windows; on macOS the library links Accelerate itself.
        .define("TRANSCRIBE_USE_SYSTEM_BLAS", "OFF")
        // Without NEW, CMake ignores CMAKE_MSVC_RUNTIME_LIBRARY.
        .define("CMAKE_POLICY_DEFAULT_CMP0091", "NEW")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build_target("transcribe");
    if target.contains("msvc") {
        // The cmake crate drops /O2 from the Release flags of the Visual Studio
        // generator. /utf-8: the sources are UTF-8, a Korean Windows reads them as CP949.
        for flag in ["/O2", "/Ob2", "/DNDEBUG", "/utf-8"] {
            config.cflag(flag);
            config.cxxflag(flag);
        }
        // Same CRT as llama.cpp (LLAMA_STATIC_CRT) and the sherpa-onnx static libs (/MT).
        config.static_crt(env::var("LLAMA_STATIC_CRT").is_ok_and(|v| v == "1"));
        config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
    }
    let build = config.build().join("build").join("src");
    // Single-config generators (Ninja, Make) and Visual Studio (Release/).
    println!("cargo:rustc-link-search=native={}", build.display());
    println!("cargo:rustc-link-search=native={}", build.join("Release").display());
    println!("cargo:rustc-link-lib=static=transcribe");
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
