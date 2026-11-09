# whisper-rs-sys (fork for morago)

A fork of [whisper-rs-sys](https://codeberg.org/tazz4843/whisper-rs) 0.15.0 (Unlicense, `LICENSE`). `src-tauri/Cargo.toml` uses it in place of the crates.io crate (`[patch.crates-io]`), under `whisper-rs` 0.16.0.

## Why

llama.cpp (`llama-cpp-sys-2`) and whisper.cpp each bundle their own copy of ggml. In one static binary, the linker then takes each ggml symbol from the first archive that has it, and the two copies come from different ggml commits. This fork builds whisper.cpp with `WHISPER_USE_SYSTEM_GGML=ON` against the ggml that `llama-cpp-sys-2` built. The binary then has one ggml, and the ggml backends (CPU, Metal, CUDA) that the `llama-cpp-2` features select serve Whisper too.

## Changes to the crate

- `whisper.cpp/`: whisper.cpp v1.9.5 sources without `ggml/`, `examples/`, `tests/` and the other folders that the library build does not read. The crate shipped v1.8.3; v1.9.5 is the version that was built and tested against the ggml of `llama-cpp-sys-2` 0.1.161.
- `whisper.cpp/src/whisper.cpp`: one patch (marked `morago patch`). With the language "auto", `whisper_full` ran the encoder twice on the first window: once for the language detection and once for the decoding. The patch keeps the encoder output of the detection when the window (mel offset, audio context) is the same. Default meeting language "detect" then costs one encoder run per segment, as with a set language.
- `build.rs`: new. It reads `DEP_LLAMA_GGML_CMAKE_DIR` (exported by `llama-cpp-sys-2`), gives its prefix to CMake as `CMAKE_PREFIX_PATH` and links only `whisper`. Bindgen reads the ggml headers from the same prefix. On MSVC it builds with the static CRT (`/MT`, policy CMP0091 NEW), as llama.cpp and sherpa-onnx.
- `wrapper.h`: `<ggml.h>` from the ggml include folder.
- `Cargo.toml`: depends on `llama-cpp-sys-2` (same version as `llama-cpp-2`). The GPU features stay only because `whisper-rs` forwards them; they do nothing.
- `src/lib.rs`: unchanged. The pre-generated `src/bindings.rs` is not used and is not here.

## Update

On a new `whisper-rs`, `whisper-rs-sys` or `llama-cpp-2` version:

1. Set the versions in `src-tauri/Cargo.toml` and here (`version`, `llama-cpp-sys-2`). The `llama-cpp-sys-2` version must be equal to the one `llama-cpp-2` uses.
2. Download the whisper.cpp release that the new `whisper-rs-sys` uses, or a newer one that builds against the ggml of `llama-cpp-sys-2`. Replace `whisper.cpp/` with its `CMakeLists.txt`, `LICENSE`, `bindings/CMakeLists.txt`, `bindings/javascript/package-tmpl.json`, `cmake/`, `include/` and `src/`.
3. Apply the `morago patch` again (`diff` the old folder against the old release), unless the release reuses the detection encoder output itself.
4. Set `WHISPER_CPP_VERSION` at the end of `build.rs`.
5. Compare `src/lib.rs` and the `[features]` list with the new crate.
6. Build and run the Whisper test on Windows and macOS (`speech::whisper`, see `.github/workflows/ci.yml` job `e2e`).
