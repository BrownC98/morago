// whisper-rs-sys, shared-ggml variant. whisper.cpp is built against the ggml that
// llama-cpp-sys-2 already built and links (DEP_LLAMA_GGML_CMAKE_DIR), so the final
// binary has ONE ggml. No ggml sources and no ggml link lines here.
use cmake::Config;
use std::env;
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    if target.contains("windows") {
        println!("cargo:rustc-link-lib=advapi32");
    }
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=whisper.cpp");
    println!("cargo:rerun-if-env-changed=DEP_LLAMA_GGML_CMAKE_DIR");

    let ggml_cmake = PathBuf::from(
        env::var("DEP_LLAMA_GGML_CMAKE_DIR")
            .expect("llama-cpp-sys-2 did not export ggml_cmake_dir (needs 0.1.161 or newer)"),
    );
    // <prefix>/lib/cmake -> <prefix>
    let ggml_prefix = ggml_cmake.parent().unwrap().parent().unwrap().to_path_buf();
    let ggml_include = ggml_prefix.join("include");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let whisper_root = out.join("whisper.cpp");
    let copy = fs_extra::dir::CopyOptions::new().overwrite(true);
    fs_extra::dir::copy("./whisper.cpp", &out, &copy).unwrap();

    bindgen::Builder::default()
        .rust_edition(bindgen::RustEdition::Edition2021)
        .rust_target(bindgen::RustTarget::stable(88, 0).map_err(|v| v.to_string()).unwrap())
        .header("wrapper.h")
        .clang_arg("-I./whisper.cpp/")
        .clang_arg("-I./whisper.cpp/include")
        .clang_arg(format!("-I{}", ggml_include.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen")
        .write_to_file(out.join("bindings.rs"))
        .unwrap();

    if env::var("DOCS_RS").is_ok() {
        return;
    }

    let mut config = Config::new(&whisper_root);
    config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_ALL_WARNINGS", "OFF")
        .define("WHISPER_ALL_WARNINGS_3RD_PARTY", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .define("WHISPER_USE_SYSTEM_GGML", "ON")
        .define("CMAKE_PREFIX_PATH", &ggml_prefix)
        // Without NEW, CMake ignores CMAKE_MSVC_RUNTIME_LIBRARY.
        .define("CMAKE_POLICY_DEFAULT_CMP0091", "NEW")
        .define("CMAKE_BUILD_TYPE", "Release")
        .pic(true);
    if target.contains("msvc") {
        config.cxxflag("/utf-8");
        // Same CRT as llama.cpp (LLAMA_STATIC_CRT) and the sherpa-onnx static libs (/MT).
        config.static_crt(env::var("LLAMA_STATIC_CRT").is_ok_and(|v| v == "1"));
        config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
    }
    let destination = config.build();
    println!("cargo:rustc-link-search=native={}", destination.display());
    println!("cargo:rustc-link-search=native={}", destination.join("lib").display());
    println!("cargo:rustc-link-lib=static=whisper");
    println!("cargo:WHISPER_CPP_VERSION=1.9.5");
}
