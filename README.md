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
| 음성 인식 | sherpa-onnx SenseVoice Small(기본), Qwen3-ASR 0.6B, Whisper(base, small, medium, turbo) + Silero VAD | `POST {주소}/audio/transcriptions` |
| 화자 구분 | sherpa-onnx CAM++ 화자 임베딩 + 온라인 군집 | – |
| 번역 | llama.cpp + Gemma 4 Instruct(E2B 기본, 12B) | `POST {주소}/chat/completions` |

Silero VAD와 CAM++(약 29 MB)는 설치 파일에 들어 있다(`src-tauri/resources/models`, 출처와 라이선스는 그 폴더의 `NOTICE.md`). 앱은 쓰기 전에 SHA-256을 확인한다. 저장소에 파일을 그대로 넣은 이유: MIT·Apache-2.0이라 재배포가 되고, 크기가 작으며, 개발·CI·설치 파일 빌드가 내려받기 없이 같은 파일을 쓴다.

음성 인식 모델(CPU에서 잰 값, 오류율은 한·일·중 CER %, 영 WER %, FLEURS 문장 20개씩, 2026-11-08):

| 모델 | 내려받기 | 메모리 | 한 / 일 / 중 / 영 |
|---|---|---|---|
| SenseVoice Small int8(기본, 2024-07-17 패키지) | 240 MB | 약 1 GB | 6.9 / 7.2 / 4.5 / 8.7 |
| Qwen3-ASR 0.6B int8 | 987 MB | 약 2.5 GB | 5.1 / 12.5 / 2.7 / 8.1 |
| Whisper small int8 | 375 MB | 약 2 GB | 6.4 / 14.4 / 12.1 / 10.3 |
| Whisper turbo int8 | 1036 MB | 3 GB 이상 | 6.8(한국어 10개) / – / – / –. CPU에서는 실시간보다 느릴 수 있다 |

Whisper base, medium은 재지 않았다.

음성 인식 모델과 번역 모델은 설정 화면에서 처음 한 번 내려받는다. 저장 위치는 앱 데이터 폴더의 `models`이다. 파일마다 주소 목록을 차례로 시도하고 크기와 SHA-256을 확인한다.

## 설치와 실행(개발)

준비물: Node.js 24, pnpm 11, Rust stable, cmake, clang.

- Windows: Visual Studio Build Tools(C++), WebView2(Windows 11에는 기본 포함).
- macOS: Xcode Command Line Tools(`xcode-select --install`).
- Linux(개발용): `webkit2gtk-4.1`, `cmake`, `clang`, `libclang-dev`.

```bash
pnpm install
LLAMA_LIB_PROFILE=Release pnpm tauri dev   # 개발 모드로 앱 실행(llama.cpp는 릴리스로: 디버그는 번역이 약 10배 느리다)
pnpm tauri build        # 설치 파일 만들기(src-tauri/target/release/bundle)
scripts/gate/check.sh   # 이 커밋의 Windows·macOS CI 결과를 기다려 확인
```

Linux는 대상이 아니다. Linux에서는 sherpa-onnx(onnxruntime)와 llama.cpp를 한 프로세스에 함께 넣으면 첫 음성 인식 호출에서 프로세스가 멈춘다(std::regex 충돌, 2026-11-08 확인). Windows·macOS에서는 CI 종단 시험이 통과한다.

종단 시험(실제 모델: VAD, SenseVoice Small, CAM++, Gemma 4 E2B)은 수동 CI 실행의 `e2e` 잡이 Windows·macOS에서 돌린다. 소리 장치 시험은 [docs/manual-audio-test.md](docs/manual-audio-test.md)의 순서로 실제 기기에서 한다.

GitHub Actions(`.github/workflows/ci.yml`)가 push마다 Windows와 macOS에서 검사(rustfmt, clippy, 테스트)를 돌린다. 설치 파일(Windows NSIS, macOS dmg)은 `v*` 태그와 수동 실행(`gh workflow run ci.yml --ref <브랜치>`)에서만 만든다. 서명하지 않은 빌드이므로 macOS에서는 처음 열 때 Finder에서 오른쪽 클릭 → 열기를 쓴다.

## 쓰는 법

1. 설정 › 음성 인식, 번역에서 로컬 모델을 내려받거나 사용자 지정 서비스의 주소, 모델, API 키를 넣고 [연결 시험]을 누른다.
2. 회의 탭에서 번역할 언어를 고르고 [회의 시작]을 누른다.
3. [중지하고 저장]을 누르면 기록 탭에 저장된다. 기록에서 화자 이름을 바꾸고 파일로 내보낸다.

## 내보내기

기록 탭에서 Markdown(`.md`), SRT 자막(`.srt`), JSON(`.json`), CSV(`.csv`)로 내보낸다. 넷 다 화자 이름, 시작·끝 시각, 원문, 번역을 담는다.

CSV는 발화 하나를 한 행에 쓴다. 열: 번호, 입력, 화자 ID, 화자 이름, 시작, 종료, 시작_ms, 종료_ms, 원문 언어, 원문, 번역 상태, 번역 언어, 번역문, 번역 오류. 파일은 BOM이 있는 UTF-8이라 Excel에서 한글이 바로 보인다. `= + - @`로 시작하는 칸은 앞에 `'`를 붙여 수식으로 실행되지 않게 한다.

## 저장 형식

회의 하나는 `문서/morago/sessions/<id>/` 폴더이다.

- `meta.json`: 제목, 시작·끝 시각, 번역 언어, 모델 이름, 화자 이름.
- `utterances.jsonl`: 발화 한 줄에 하나(화자, 시각, 언어, 원문, 번역). 번역이 끝나면 같은 id의 줄을 덧붙인다. 앱이 갑자기 꺼져도 마지막 줄까지 남는다.

## 라이선스

앱에 든 모델(Silero VAD: MIT, CAM++: Apache-2.0)과 내려받는 모델(Gemma 4: Apache-2.0, SenseVoice: FunASR Model License, Qwen3-ASR: Apache-2.0, Whisper: MIT)의 출처는 `src-tauri/resources/models/NOTICE.md`와 앱의 설정 › 정보에 있다.
