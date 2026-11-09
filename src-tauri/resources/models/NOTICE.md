# Models shipped with morago

| Folder | Model | Source | License |
|---|---|---|---|
| `silero-vad/model.onnx` | Silero VAD (voice activity detection) | https://github.com/snakers4/silero-vad, ONNX file from https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models (`silero_vad.onnx`) | MIT, Copyright (c) 2020-present Silero Team |
| `speaker-campplus/model.onnx` | CAM++ speaker embedding, 3D-Speaker (zh-en, 16 kHz, common advanced) | https://github.com/modelscope/3D-Speaker, ONNX file from https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models | Apache-2.0 |

SHA-256 values are in `src-tauri/src/models.rs` (`BUNDLED`); the app checks them before use.

Downloaded on request (not in the installer): Gemma 4 E2B / 12B instruct GGUF (Google DeepMind, Apache-2.0, https://ai.google.dev/gemma/docs/gemma_4_license), from Google's Hugging Face repositories, unchanged; Whisper (OpenAI, MIT) ONNX exports by the sherpa-onnx project; SenseVoice Small (FunAudioLLM, Alibaba Group, FunASR Model License 1.1, https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE; model name and authors kept) int8 ONNX export by the sherpa-onnx project (Hugging Face `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`); Qwen3-ASR 0.6B (Qwen team, Alibaba Cloud, Apache-2.0, https://github.com/QwenLM/Qwen3-ASR) int8 ONNX export by Wasser1462 (https://github.com/Wasser1462/Qwen3-ASR-onnx), from Hugging Face `csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25`; Fun-ASR-MLT-Nano-2512 (FunAudioLLM, Alibaba Group / Tongyi Lab, Apache-2.0, https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512, commit `2c088f535108689beff7e11cdc3d792af7d56552`) int8 ONNX export (max_total_len 1024) with the Wasser1462 tool (https://github.com/Wasser1462/FunASR-nano-onnx), from Hugging Face `luigi000/fun-asr-mlt-nano-onnx-int8-1024`, commit `39334e8a32d91f46d47f80d96d749409c86a56b1`.
