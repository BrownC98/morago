# morago: agent notes

Desktop app (Windows, macOS) for meeting minutes with live translated subtitles.
Stack: Tauri 2 (Rust core) + React/TypeScript (Vite). Reason: native audio APIs (WASAPI loopback, Core Audio tap) and in-process native models (sherpa-onnx, llama.cpp) from Rust; small installer.

## Layout

- `src-tauri/src/types.rs`: shared data types (contract between modules).
- `src-tauri/src/audio/`: mic + system audio capture, resample to 16 kHz mono.
- `src-tauri/src/speech/`: VAD, ASR providers, speaker diarization. `models.rs`: model catalog and download.
- `src-tauri/src/translate/`: translation providers.
- `src-tauri/src/settings.rs`, `secrets.rs`: settings JSON and API keys in the OS credential store.
- `src-tauri/src/store/`: session files and export.
- `src-tauri/src/engine.rs`: wires the pipeline and emits UI events.
- `src/`: UI. Approved mockup: `docs/mockups/app-screens.html`.

## Rules

- Never write an API key to settings, logs, session files, exports or error text. Use `secrets::Secret`.
- Change `types.rs` and module signatures only with the coordinator's approval.

## Commands

- Setup (Linux/WSL dev): `rustup default stable`, `rustup component add rustfmt clippy`, `pnpm install`. Linux needs webkit2gtk-4.1, cmake, clang, libclang-dev.
- While editing: `cd src-tauri && memcap -m 4G cargo check` (UI: `pnpm build`). Do not run full local builds or test suites on the shared VM.
- Completion evidence and gate: the GitHub Actions run of `.github/workflows/ci.yml` for the exact head commit passes the `check` job on Windows and macOS (fmt, clippy, tests). Push, then `scripts/gate/check.sh` finds and waits for that run. Save the run URL in your task `logs/`.
- Installers (NSIS, dmg): the `bundle` job runs only on tags `v*` and manual runs: `gh workflow run ci.yml --ref <branch>`.
- Run the app: `pnpm tauri dev`.
