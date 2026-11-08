# Models shipped with morago

| Folder | Model | Source | License |
|---|---|---|---|
| `silero-vad/model.onnx` | Silero VAD (voice activity detection) | https://github.com/snakers4/silero-vad, ONNX file from https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models (`silero_vad.onnx`) | MIT, Copyright (c) 2020-present Silero Team |
| `speaker-campplus/model.onnx` | CAM++ speaker embedding, 3D-Speaker (zh-en, 16 kHz, common advanced) | https://github.com/modelscope/3D-Speaker, ONNX file from https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models | Apache-2.0 |

SHA-256 values are in `src-tauri/src/models.rs` (`BUNDLED`); the app checks them before use.
