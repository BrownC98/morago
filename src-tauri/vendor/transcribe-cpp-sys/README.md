# transcribe-cpp-sys (fork for morago)

A fork of [transcribe-cpp-sys](https://crates.io/crates/transcribe-cpp-sys) 0.4.1 (MIT, `LICENSE`), the native part of [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) at commit `4d7c1cce`. `src-tauri/Cargo.toml` uses it in place of the crates.io crate (`[patch.crates-io]`), under `transcribe-cpp` 0.4.1 from crates.io. morago uses it for Fun-ASR-MLT-Nano (GGUF, `speech/funasr.rs`).

## Why

transcribe.cpp bundles its own ggml. In one static binary with llama.cpp (`llama-cpp-sys-2`), the linker would take each ggml symbol from the first archive that has it. This fork builds transcribe.cpp against the ggml that `llama-cpp-sys-2` built, so the binary has one ggml, and the backends (CPU, Metal, CUDA) that the `llama-cpp-2` features select serve Fun-ASR too. The bundled ggml of 0.4.1 is ggml 0.26.0 (upstream `d7cb574`), the same as that of `llama-cpp-sys-2` 0.1.161.

## Changes to the crate

- `transcribe.cpp/`: `CMakeLists.txt`, `LICENSE`, `THIRD-PARTY-LICENSES.md`, `cmake/`, `include/` and `src/` of the crate, without `ggml/`.
- `transcribe.cpp/CMakeLists.txt`: patch A (marked `morago patch`). The option `TRANSCRIBE_USE_SYSTEM_GGML` takes ggml from `find_package(ggml)`.
- `transcribe.cpp/src/transcribe.cpp`: patch B (marked `morago patch`). The bundled ggml has a patch with `ggml_backend_set_reg_filter`; the ggml of llama.cpp has none. With the system ggml the filter is not installed: in one ggml it would also filter the backends of llama.cpp. The other two ggml patches of transcribe.cpp (`patches/ggml/0001` MSVC barrier, `0003` Metal mapped buffer) are absent: llama.cpp runs without them too.
- `transcribe.cpp/src/arch/funasr_nano/model.cpp`: patch C (marked `morago patch`). The single run checked the abort callback only before the frontend. It now also checks after the encoder, before the prefill and before each decode step, so that a final run of morago stops a partial run within one step (`speech/funasr.rs`).
- `build.rs`: new. It reads `DEP_LLAMA_GGML_CMAKE_DIR` (exported by `llama-cpp-sys-2`), gives its prefix to CMake as `CMAKE_PREFIX_PATH`, builds only the `transcribe` target in Release without BLAS, and links only `transcribe`. On MSVC it builds with the static CRT (`/MT`, policy CMP0091 NEW), `/O2` (the cmake crate drops it under the Visual Studio generator) and `/utf-8`.
- `Cargo.toml`: depends on `llama-cpp-sys-2` (same version as `llama-cpp-2`). The GPU features stay only because `transcribe-cpp` forwards them; they do nothing.
- `src/lib.rs`, `src/transcribe_sys.rs`: the crate's `bindings/rust/sys/src`, unchanged (pre-generated bindings, no bindgen).

## Update

Update only to a transcribe.cpp whose bundled ggml (`ggml/UPSTREAM` in the crate) is the ggml of the `llama-cpp-sys-2` version in use. On a new `transcribe-cpp` or `llama-cpp-2` version:

1. Set the versions in `src-tauri/Cargo.toml` and here (`version`, `llama-cpp-sys-2`). The `llama-cpp-sys-2` version must be equal to the one `llama-cpp-2` uses.
2. Download the `transcribe-cpp-sys` crate of the new version (`https://crates.io/api/v1/crates/transcribe-cpp-sys/<version>/download`). Replace the folders and files of `transcribe.cpp/` above with its own, and `src/` with its `bindings/rust/sys/src`.
3. Apply patches A, B and C again (`diff` the old folder against the old crate).
4. Compare the `[features]` list with the new crate.
5. Build and run the Fun-ASR test on Windows and macOS (`speech::funasr`, see `.github/workflows/ci.yml` job `e2e`).
