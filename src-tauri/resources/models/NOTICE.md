# Models shipped with morago

| Folder | Model | Source | License |
|---|---|---|---|
| `silero-vad/model.onnx` | Silero VAD (voice activity detection) | https://github.com/snakers4/silero-vad, ONNX file from https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models (`silero_vad.onnx`) | MIT, Copyright (c) 2020-present Silero Team |
| `speaker-campplus/model.onnx` | CAM++ speaker embedding, 3D-Speaker (zh-en, 16 kHz, common advanced) | https://github.com/modelscope/3D-Speaker, ONNX file from https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models | Apache-2.0 |

SHA-256 values are in `src-tauri/src/models.rs` (`BUNDLED`); the app checks them before use.

Downloaded on request (not in the installer): Gemma 4 E2B / 12B instruct GGUF (Google DeepMind, Apache-2.0, https://ai.google.dev/gemma/docs/gemma_4_license), from Google's Hugging Face repositories, unchanged; Whisper (OpenAI, MIT) ONNX exports by the sherpa-onnx project.
