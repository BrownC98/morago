# morago

회의 음성을 실시간으로 받아 적고, 외국어 발화를 고른 언어로 번역해 자막처럼 보여 주는 데스크톱 앱이다. Windows와 macOS에서 동작한다.

- 마이크와 시스템 소리(화상회의 상대의 목소리)를 함께 받는다.
- 발화마다 화자, 시작·끝 시각, 원문, 번역을 보이고 저장한다.
- 지난 회의를 다시 열고 Markdown, SRT 자막, JSON으로 내보낸다.
- 음성 인식과 번역은 이 컴퓨터의 로컬 모델이나 사용자 지정 서비스(OpenAI 호환 주소: OpenAI, Ollama, LM Studio, vLLM 등)를 쓴다.
- API 키는 Windows 자격 증명 관리자, macOS 키체인에만 저장한다.

## 소리 입력 방법과 운영체제 제한

| 운영체제 | 마이크 | 시스템 소리 | 제한 |
|---|---|---|---|
| Windows 10 이상 | WASAPI | WASAPI 루프백(기본 출력 장치) | 권한 창 없음 |
| macOS 14.2 이상 | Core Audio | Core Audio 프로세스 탭 | 시스템 소리는 macOS 14.6 이상. 처음 시작할 때 마이크 권한과 「시스템 오디오 녹음」 권한을 묻는다 |

## 모델

| 용도 | 로컬 | 사용자 지정 |
|---|---|---|
| 음성 인식 | sherpa-onnx SenseVoice Small(기본), Qwen3-ASR 0.6B, whisper.cpp Whisper(base, small, medium, turbo) + Silero VAD | `POST {주소}/audio/transcriptions` |
| 화자 구분 | sherpa-onnx CAM++ 화자 임베딩 + 온라인 군집 | – |
| 번역 | llama.cpp + Gemma 4 Instruct(E2B 기본, 12B) | `POST {주소}/chat/completions` |

Silero VAD와 CAM++(약 29 MB)는 설치 파일에 들어 있다(`src-tauri/resources/models`, 출처와 라이선스는 그 폴더의 `NOTICE.md`). 앱은 쓰기 전에 SHA-256을 확인한다. 저장소에 파일을 그대로 넣은 이유: MIT·Apache-2.0이라 재배포가 되고, 크기가 작으며, 개발·CI·설치 파일 빌드가 내려받기 없이 같은 파일을 쓴다.

음성 인식 모델(CPU에서 잰 값, 오류율은 한·일·중 CER %, 영 WER %, FLEURS 문장 20개씩, 2026-11-08):

| 모델 | 내려받기 | 메모리 | 한 / 일 / 중 / 영 |
|---|---|---|---|
| SenseVoice Small int8(기본, 2024-07-17 패키지) | 240 MB | 약 1 GB | 6.9 / 7.2 / 4.5 / 8.7 |
| Qwen3-ASR 0.6B int8 | 987 MB | 약 2.5 GB | 5.1 / 12.5 / 2.7 / 8.1 |
| Whisper small q8_0 | 264 MB | 약 0.6 GB | 7.5 / 12.7 / 10.6 / 11.2 |
| Whisper large-v3-turbo q5_0 | 574 MB | 약 1 GB | 7.4 / 4.8 / 5.3 / 9.3(문장 10개씩). CPU에서는 실시간보다 느리다 |

Whisper는 whisper.cpp(ggml 형식, `ggerganov/whisper.cpp`의 파일)로 돌린다. Whisper 두 줄은 2026-11-09 Windows 노트북(Core Ultra 7 255H, 4스레드)에서 회의 언어를 정하고 쟀다. 처리 시간/음성 길이는 small 0.35, turbo 2.0이다. 메모리는 whisper.cpp가 잡는 모델과 작업 버퍼의 합이다. Whisper base(q8_0, 82 MB), medium(q5_0, 539 MB)은 재지 않았다. 2026-11-09 이전 판은 Whisper를 sherpa-onnx로 돌렸다. 그 파일이 남아 있으면 설정 화면이 그 모델을 「이전 형식」으로 보이고 [삭제]로 지운다. 다시 내려받으면 새 파일로 바뀐다.

음성 인식 모델과 번역 모델은 설정 화면에서 처음 한 번 내려받는다. 저장 위치는 앱 데이터 폴더의 `models`이다. 파일마다 주소 목록을 차례로 시도하고 크기와 SHA-256을 확인한다.

## GPU 가속

로컬 번역 모델과 Whisper 음성 인식은 GPU가 있으면 GPU에서 돌린다. 설정 › GPU 가속에서 「자동」(처음 값), GPU 이름, 「끔」을 고른다. SenseVoice, Qwen3-ASR, 목소리 구간, 화자 구분은 CPU에서 돌린다. whisper.cpp는 llama.cpp의 ggml을 함께 쓰므로 GPU 코드, GPU 구성 요소, 장치 선택이 번역과 같다.

- macOS: Metal. 따로 내려받을 것이 없다.
- Windows: NVIDIA GPU(CUDA)만 쓴다. 다른 GPU(Intel Arc 등)는 목록에 없다. 설치 파일에는 CUDA DLL이 없다. 설정에서 「GPU 구성 요소」(NVIDIA CUDA 12.9.1 런타임과 cuBLAS, 내려받기 528 MB, 풀면 0.7 GB)를 한 번 내려받고 앱을 다시 시작하면 GPU를 쓴다. 저장 위치는 앱 데이터 폴더의 `gpu`이다. NVIDIA 드라이버 528.33 이상(RTX 50 시리즈는 570.65 이상), GPU는 RTX 20(Turing) 이상이 필요하다.
- Windows 설치 파일은 v0.1.0의 35 MB에서 181 MB로 커졌다. `morago.exe`가 GPU 4세대(RTX 20·30·40·50)의 llama.cpp CUDA 커널을 담기 때문이다(33 MB → 199 MB). 이 커널은 거의 압축되지 않는다. NVIDIA GPU가 없는 PC도 같은 설치 파일을 쓴다.
- 두 모델은 한 GPU의 메모리를 나눠 쓴다. 회의를 시작하기 전에 남은 GPU 메모리가 두 모델의 합(모델마다 파일 크기 + 30% + 512 MiB)을 담는지 본다. 번역 모델이 먼저 GPU를 받고, 둘이 함께 들어가지 않으면 Whisper를 CPU에서 돌린다.
- GPU 메모리가 모자라면 CPU에서 돌리고 설정에 그 까닭을 보인다. GPU로 모델을 불러오지 못하면 회의를 시작하지 않고 원인을 보인다. 회의 중에 GPU 오류가 나면 회의를 멈추고 저장한 뒤 오류를 보인다.

## 설치와 실행(개발)

준비물: Node.js 24, pnpm 11, Rust stable, cmake, clang.

- Windows: Visual Studio Build Tools(C++), WebView2(Windows 11에는 기본 포함), CUDA Toolkit 12.9(`CUDA_PATH`), Ninja. llama.cpp CUDA 빌드는 「x64 Native Tools」 명령 창(vcvars64)에서 `CMAKE_GENERATOR=Ninja`로 한다. Visual Studio 생성기는 긴 CUDA 경로에서 실패한다. GPU 아키텍처는 `src-tauri/.cargo/config.toml`의 `CMAKE_CUDA_ARCHITECTURES`이다. 처음 빌드는 오래 걸린다(16코어에서 약 30분). `scripts/win-check.sh`는 이 환경을 맞춘다.
- macOS: Xcode Command Line Tools(`xcode-select --install`).
- Linux(개발용): `webkit2gtk-4.1`, `cmake`, `clang`, `libclang-dev`.

```bash
pnpm install
LLAMA_LIB_PROFILE=Release pnpm tauri dev   # 개발 모드로 앱 실행(llama.cpp는 릴리스로: 디버그는 번역이 약 10배 느리다)
pnpm tauri build        # 설치 파일 만들기(src-tauri/target/release/bundle)
scripts/gate/check.sh   # 이 커밋의 Windows·macOS CI 결과를 기다려 확인
```

Linux는 대상이 아니다. Linux에서는 sherpa-onnx(onnxruntime)와 llama.cpp를 한 프로세스에 함께 넣으면 첫 음성 인식 호출에서 프로세스가 멈춘다(std::regex 충돌, 2026-11-08 확인). Windows·macOS에서는 CI 종단 시험이 통과한다.

종단 시험(실제 모델: VAD, SenseVoice Small, CAM++, Gemma 4 E2B, Whisper small)은 수동 CI 실행의 `e2e` 잡이 Windows·macOS에서 돌린다. 소리 장치 시험은 [docs/manual-audio-test.md](docs/manual-audio-test.md)의 순서로 실제 기기에서 한다.

GitHub Actions(`.github/workflows/ci.yml`)가 push마다 Windows와 macOS에서 검사(rustfmt, clippy, 테스트)를 돌린다. 설치 파일(Windows NSIS, macOS dmg)은 `v*` 태그와 수동 실행(`gh workflow run ci.yml --ref <브랜치> -f job=bundle`)에서만 만든다. 종단 시험은 `-f job=e2e`로 돌린다. 서명하지 않은 빌드이므로 macOS에서는 처음 열 때 Finder에서 오른쪽 클릭 → 열기를 쓴다.

## 쓰는 법

1. 설정 › 음성 인식, 번역에서 로컬 모델을 내려받거나 사용자 지정 서비스의 주소, 모델, API 키를 넣고 [연결 시험]을 누른다.
2. 회의 탭에서 번역할 언어를 고르고 [회의 시작]을 누른다.
3. [중지하고 저장]을 누르면 기록 탭에 저장된다. 기록에서 화자 이름을 바꾸고 파일로 내보낸다.

## 내보내기

기록 탭에서 Markdown(`.md`), SRT 자막(`.srt`), JSON(`.json`)으로 내보낸다. 셋 다 화자 이름, 시작·끝 시각, 원문, 번역을 담는다.

## 저장 형식

회의 하나는 `문서/morago/sessions/<id>/` 폴더이다.

- `meta.json`: 제목, 시작·끝 시각, 번역 언어, 모델 이름, 화자 이름.
- `utterances.jsonl`: 발화 한 줄에 하나(화자, 시각, 언어, 원문, 번역). 번역이 끝나면 같은 id의 줄을 덧붙인다. 앱이 갑자기 꺼져도 마지막 줄까지 남는다.

## 라이선스

앱에 든 모델(Silero VAD: MIT, CAM++: Apache-2.0)과 내려받는 모델(Gemma 4: Apache-2.0, SenseVoice: FunASR Model License, Qwen3-ASR: Apache-2.0, Whisper: MIT)의 출처는 `src-tauri/resources/models/NOTICE.md`와 앱의 설정 › 정보에 있다. whisper.cpp(MIT)는 `src-tauri/vendor/whisper-rs-sys`에 고친 판으로 들어 있다(그 폴더의 `README.md`).
