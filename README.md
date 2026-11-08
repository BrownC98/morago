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
| macOS | Core Audio | Core Audio 프로세스 탭 | 시스템 소리는 macOS 14.6 이상. 처음 시작할 때 마이크 권한과 「시스템 오디오 녹음」 권한을 묻는다 |

## 모델

| 용도 | 로컬 | 사용자 지정 |
|---|---|---|
| 음성 인식 | sherpa-onnx Whisper(base, small, medium) + Silero VAD | `POST {주소}/audio/transcriptions` |
| 화자 구분 | sherpa-onnx 화자 임베딩 + 온라인 군집 | – |
| 번역 | llama.cpp + Qwen2.5 Instruct(1.5B, 3B) | `POST {주소}/chat/completions` |

로컬 모델은 설정 화면에서 처음 한 번 내려받는다. 저장 위치는 앱 데이터 폴더의 `models`이다.

## 설치와 실행(개발)

준비물: Node.js 24, pnpm 11, Rust stable, cmake, clang.

- Windows: Visual Studio Build Tools(C++), WebView2(Windows 11에는 기본 포함).
- macOS: Xcode Command Line Tools(`xcode-select --install`).
- Linux(개발용): `webkit2gtk-4.1`, `cmake`, `clang`, `libclang-dev`.

```bash
pnpm install
pnpm tauri dev          # 개발 모드로 앱 실행
pnpm tauri build        # 설치 파일 만들기(src-tauri/target/release/bundle)
scripts/gate/check.sh   # 검사(프런트 빌드, rustfmt, clippy, 테스트)
```

GitHub Actions(`.github/workflows/ci.yml`)가 push마다 Windows(NSIS 설치 파일)와 macOS(dmg)를 빌드하고 결과물을 올린다. 서명하지 않은 빌드이므로 macOS에서는 처음 열 때 Finder에서 오른쪽 클릭 → 열기를 쓴다.

## 쓰는 법

1. 설정 › 음성 인식, 번역에서 로컬 모델을 내려받거나 사용자 지정 서비스의 주소, 모델, API 키를 넣고 [연결 시험]을 누른다.
2. 회의 탭에서 번역할 언어를 고르고 [회의 시작]을 누른다.
3. [중지하고 저장]을 누르면 기록 탭에 저장된다. 기록에서 화자 이름을 바꾸고 파일로 내보낸다.

## 저장 형식

회의 하나는 `문서/morago/sessions/<id>/` 폴더이다. 설정에서 위치를 바꿀 수 있다.

- `meta.json`: 제목, 시작·끝 시각, 번역 언어, 모델 이름, 화자 이름.
- `utterances.jsonl`: 발화 한 줄에 하나(화자, 시각, 언어, 원문, 번역). 번역이 끝나면 같은 id의 줄을 덧붙인다. 앱이 갑자기 꺼져도 마지막 줄까지 남는다.
