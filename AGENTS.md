# morago: agent notes

Desktop app (Windows, macOS) for meeting minutes with live translated subtitles.
Stack: Tauri 2 (Rust core) + React/TypeScript (Vite). Reason: native audio APIs (WASAPI loopback, Core Audio tap) and in-process native models (sherpa-onnx, llama.cpp) from Rust; small installer.

## Layout

- `src-tauri/src/types.rs`: shared data types (contract between modules).
- `src-tauri/src/audio/`: mic + system audio capture, resample to 16 kHz mono.
- `src-tauri/src/speech/`: VAD, ASR providers, speaker diarization.
- `src-tauri/src/models.rs`: downloadable models (Whisper, Gemma 4) with URL lists and pinned SHA-256; bundled models (Silero VAD, CAM++) in `src-tauri/resources/models/` (Tauri resources), checked by `verify_bundled`.
- `src-tauri/src/translate/`: translation providers.
- `src-tauri/src/settings.rs`, `secrets.rs`: settings JSON and API keys in the OS credential store.
- `src-tauri/src/store/`: session files and export.
- `src-tauri/src/engine.rs`: wires the pipeline and emits UI events.
- `src/`: UI. Approved mockup: `docs/mockups/app-screens.html`.

## Rules

- Never write an API key to settings, logs, session files, exports or error text. Use `secrets::Secret`.
- Change `types.rs` and module signatures only with the coordinator's approval.
- Windows link: sherpa-onnx prebuilt libs use the static CRT (/MT); `src-tauri/.cargo/config.toml` builds llama.cpp the same way. Do not delete it.
- macOS: minimum 14.2 (Core Audio tap API); system audio is gated at 14.6 at run time. `build.rs` links libclang_rt.osx for llama.cpp Metal.
- Linux is dev-only: onnxruntime + llama.cpp in one process abort on Linux (std::regex clash). Run model tests on Windows/macOS CI (job `e2e`).
- GitHub release downloads in CI must send the job token (unauthenticated requests got 504 on 2026-11-08); see `.github/actions/sherpa-libs`.

## Commands

- Setup (Linux/WSL dev): `rustup default stable`, `rustup component add rustfmt clippy`, `pnpm install`. Linux needs webkit2gtk-4.1, cmake, clang, libclang-dev.
- While editing: `cd src-tauri && memcap -m 4G cargo check` (UI: `pnpm build`). Do not run full local builds or test suites on the shared VM.
- Completion evidence and gate: the GitHub Actions run of `.github/workflows/ci.yml` for the exact head commit passes the `check` job on Windows and macOS (fmt, clippy, tests). Push, then `scripts/gate/check.sh` finds and waits for that run. Save the run URL in your task `logs/`.
- Installers (NSIS, dmg): the `bundle` job runs only on tags `v*` and manual runs: `gh workflow run ci.yml --ref <branch>`.
- End-to-end test with real models: manual run (`gh workflow run ci.yml --ref <branch>`), job `e2e` (Windows, macOS).
- Run the app: `LLAMA_LIB_PROFILE=Release pnpm tauri dev`.
- Manual audio device test: `docs/manual-audio-test.md`.
