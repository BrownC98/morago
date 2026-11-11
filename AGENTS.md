# morago: agent notes

Desktop app (Windows, macOS) for meeting minutes with live translated subtitles.
Stack: Tauri 2 (Rust core) + React/TypeScript (Vite). Reason: native audio APIs (WASAPI loopback, Core Audio tap) and in-process native models (sherpa-onnx, llama.cpp, whisper.cpp) from Rust; small installer.

## Layout

- `src-tauri/src/types.rs`: shared data types (contract between modules).
- `src-tauri/src/audio/`: mic + system audio capture, resample to 16 kHz mono.
- `src-tauri/src/speech/`: VAD, ASR providers, speaker diarization. sherpa-onnx: SenseVoice, Qwen3-ASR, VAD, CAM++. Whisper: whisper.cpp (`whisper-rs`) on the ggml of llama.cpp through the fork `src-tauri/vendor/whisper-rs-sys`. Fun-ASR-MLT-Nano: transcribe.cpp (`transcribe-cpp`) on the same ggml through the fork `src-tauri/vendor/transcribe-cpp-sys`. Each fork's README says how to update it.
- `src-tauri/src/models.rs`: downloadable models (ASR: SenseVoice Small default, Qwen3-ASR 0.6B, Whisper ggml, Fun-ASR GGUF; Gemma 4) with URL lists and pinned SHA-256; bundled models (Silero VAD, CAM++) in `src-tauri/resources/models/` (Tauri resources), checked by `verify_bundled`.
- `src-tauri/src/translate/`: translation providers.
- `src-tauri/src/gpu.rs`: GPU choice for llama.cpp, whisper.cpp and transcribe.cpp (one ggml; Metal on macOS, CUDA on Windows), the downloadable CUDA DLL pack, delay-load failure hook. `engine::place_models` places both models of a session on one GPU.
- `src-tauri/src/settings.rs`, `secrets.rs`: settings JSON and API keys in the OS credential store.
- `src-tauri/src/store/`: session files and export.
- `src-tauri/src/engine.rs`: wires the pipeline and emits UI events.
- `src/`: UI. Approved mockup: `docs/mockups/app-screens.html`.

## Rules

- Never write an API key to settings, logs, session files, exports or error text. Use `secrets::Secret`.
- Change `types.rs` and module signatures only with the coordinator's approval.
- Windows link: sherpa-onnx prebuilt libs use the static CRT (/MT); `src-tauri/.cargo/config.toml` builds llama.cpp the same way. Do not delete it.
- Windows CUDA: the installer has no CUDA DLL. `build.rs` delay-loads them (`/DELAYLOAD`); `gpu.rs` has the delay-load hooks: they load the pack DLLs from `gpu::app_dir()` (the first load comes before `main`), else stubs. A new static import of a CUDA DLL stops the app on PCs without one; the test `cuda_dlls_are_delay_loaded` checks this. A CUDA out-of-memory error ends the process. By the decision of 2026-11-09 the app does not check free GPU memory: `gpu::place` puts the translation model, Whisper and Fun-ASR on the chosen GPU, and the user accepts the risk. Do not add a memory check without a new decision.
- macOS: minimum 14.2 (Core Audio tap API); system audio is gated at 14.6 at run time. `build.rs` links libclang_rt.osx for llama.cpp Metal.
- Linux is dev-only: onnxruntime + llama.cpp in one process abort on Linux (std::regex clash). Run model tests on Windows/macOS CI (job `e2e`).
- GitHub release downloads in CI must send the job token (unauthenticated requests got 504 on 2026-11-08); see `.github/actions/sherpa-libs`.

## Commands

- Setup (Linux/WSL dev): rustup (it installs the version and components of `rust-toolchain.toml`), `pnpm install`. Linux needs webkit2gtk-4.1, cmake, clang, libclang-dev.
- While editing: `cd src-tauri && memcap -m 4G cargo check` (UI: `pnpm build`). Do not run full local builds or test suites on the shared VM.
- Windows check on this PC, from WSL: `scripts/win-check.sh` runs the CI `check` steps (clippy `-D warnings`, tests) on the Windows host; `scripts/win-check.sh check` or other cargo arguments run that cargo command. It copies the work tree, also uncommitted files, with git to `C:\Users\<user>\morago-win-check` and builds there (measured 2026-11-08: 2-4 min after a one-file change, about 9 min on an empty target). Run it before you push Windows-specific code. CI stays the gate. Windows tools: rustup (stable MSVC), VS 2022 Build Tools (C++), CMake and LLVM from winget, CUDA Toolkit 12.9. For the CUDA build it calls vcvars64, uses Ninja and the target folder `C:\mct`. A cold CUDA build of llama.cpp takes about 30 min. If `cmd.exe` fails with `UtilAcceptVsock ... accept4 failed`, run with `WSL_INTEROP=/run/WSL/2_interop`.
- Rust version: `rust-toolchain.toml` pins it for CI and local builds. To bump it, set `channel` to the new stable and fix new clippy warnings; the first Windows `check` runs cold (15-28 min) until master saves its cache.
- WSL builds share compiled crates through sccache: `~/.cargo/config.toml` (user level, CI does not read it) has `[build] rustc-wrapper = "sccache"`.
- Completion evidence and gate: the GitHub Actions run of `.github/workflows/ci.yml` for the exact head commit passes the `check` job on Windows and macOS (fmt, clippy, tests). Push, then `scripts/gate/check.sh` finds and waits for that run. Save the run URL in your task `logs/`.
- Small changes outside app code (version bump, `*.md`, `docs/`, `scripts/`; not `src/`, `src-tauri/`, `.github/`) skip `loom ready` and review: push the branch, `scripts/gate/check.sh` passes, then merge into master and push. App code and CI changes go through `loom ready`.
- Installers (NSIS, dmg): the `bundle` job runs only on tags `v*` and manual runs: `gh workflow run ci.yml --ref <branch> -f job=bundle`. Only manual runs save the Windows build cache, and a tag run sees only the cache of master. Before a tag, if `gh cache list -k v0-rust-bundle-vs2022 -r refs/heads/master` shows none, run `gh workflow run ci.yml --ref master -f job=bundle` and wait for it (measured 2026-11-09: Windows job 76 min cold, CUDA build of llama.cpp; 9.5 min warm). Unused caches expire after 7 days.
- Release (with the updater): set the same version in `package.json`, `src-tauri/Cargo.toml` and `src-tauri/tauri.conf.json`, merge to master, push the tag `v<version>`. The `bundle` job signs the update files with the updater key and runs `scripts/update-test.sh` (an installed 0.0.1 build updates to the new files); the `release` job writes `latest.json` (`scripts/release-manifest.sh`) and uploads the installers, `morago.app.tar.gz` and `latest.json` to the GitHub release (it creates the release if none exists). Then write the release notes with `gh release edit`. Installed apps read `releases/latest/download/latest.json`: a draft or pre-release is not "latest". Updater key: public key in `tauri.conf.json` (`plugins.updater.pubkey`), private key and password in the GitHub secrets `TAURI_SIGNING_PRIVATE_KEY` and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`; the backup copy is in `~/.config/morago-updater/` on the dev PC. Without that key no installed app accepts a new update; keep it out of the repo and logs.
- End-to-end test with real models: manual run (`gh workflow run ci.yml --ref <branch> -f job=e2e`), job `e2e` (Windows, macOS). Only master saves its caches (Rust build, models); after a change of the model cache key, run it once on master.
- Run the app: `LLAMA_LIB_PROFILE=Release pnpm tauri dev`. On the Windows host of this PC, from WSL: `scripts/win-check.sh run` builds the work tree (target `C:\mcr`; the first CUDA build is long) and starts it.
- Manual audio device test: `docs/manual-audio-test.md`.
