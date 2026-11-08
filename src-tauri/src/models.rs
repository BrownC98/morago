//! Local models. Every model is a folder `<dir>/<id>` with fixed file names.
//! - Downloads chosen by the user (ASR, translation LLM): `SPECS`, into the
//!   app data folder.
//! - Shipped in the installer (Silero VAD, CAM++ speaker embedding): `BUNDLED`,
//!   in the Tauri resource folder `models/` (human decision 2026-11-08: no
//!   download for them). Checked with `verify_bundled` before use.

use crate::types::DownloadProgress;
use anyhow::{Context, bail};
use serde::Serialize;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Asr,
    Translation,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    /// Stable id, for example "whisper-small", "gemma-4-e2b-it-q4".
    pub id: String,
    pub kind: ModelKind,
    /// Display name, for example "Whisper small".
    pub name: String,
    pub size_bytes: u64,
    /// Short user-facing note (Korean), for example "빠름, 정확도 보통".
    pub note: String,
}

/// Id (folder) of the bundled Silero VAD model.
pub const VAD_MODEL_ID: &str = "silero-vad";
/// Id (folder) of the bundled speaker-embedding model.
pub const SPEAKER_MODEL_ID: &str = "speaker-campplus";

/// Models in the installer: (id, file, SHA-256). Licenses and sources:
/// `src-tauri/resources/models/NOTICE.md`.
pub const BUNDLED: &[(&str, &str, &str)] = &[
    (
        VAD_MODEL_ID,
        VAD_FILE,
        "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
    ),
    (
        SPEAKER_MODEL_ID,
        SPEAKER_FILE,
        "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2",
    ),
];

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Checks the shipped models in `dir` (the resource folder `models/`).
pub fn verify_bundled(dir: &Path) -> anyhow::Result<()> {
    for (id, file, sha) in BUNDLED {
        let p = path(dir, id).join(file);
        let bytes = std::fs::read(&p).with_context(|| {
            format!(
                "앱에 들어 있는 모델이 없습니다: {}. 앱을 다시 설치하십시오.",
                p.display()
            )
        })?;
        if sha256_hex(&bytes) != *sha {
            bail!("앱에 들어 있는 모델이 손상되었습니다: {id}. 앱을 다시 설치하십시오.");
        }
    }
    Ok(())
}

/// File names inside a model folder. `speech` and `translate` read these.
pub const VAD_FILE: &str = "model.onnx";
pub const SPEAKER_FILE: &str = "model.onnx";
pub const ENCODER_FILE: &str = "encoder.int8.onnx";
pub const DECODER_FILE: &str = "decoder.int8.onnx";
pub const TOKENS_FILE: &str = "tokens.txt";
pub const SENSE_VOICE_FILE: &str = "model.int8.onnx";
pub const QWEN3_FRONTEND_FILE: &str = "conv_frontend.onnx";
/// Qwen3-ASR reads its tokenizer files from the model folder.
pub const QWEN3_TOKENIZER_FILES: [&str; 3] = ["vocab.json", "merges.txt", "tokenizer_config.json"];
/// Model ids by recognizer type; `speech::local` picks the config by id.
pub const SENSE_VOICE_ID: &str = "sense-voice-small";
pub const QWEN3_ASR_ID: &str = "qwen3-asr-0.6b";
pub const LLM_FILE: &str = "model.gguf";

/// One file of a model: where it comes from and its name in the folder.
/// `urls` are tried in order; each download must match `size` and `sha256`
/// (pinned 2026-11-08 from the Hugging Face LFS hash or the file itself), so
/// a mirror cannot change the model.
pub struct FileSpec<'a> {
    pub urls: &'a [&'a str],
    pub name: &'a str,
    pub size: u64,
    pub sha256: &'a str,
}

pub struct Spec<'a> {
    pub id: &'a str,
    pub kind: ModelKind,
    pub name: &'a str,
    pub note: &'a str,
    pub files: &'a [FileSpec<'a>],
}

macro_rules! hf {
    ($path:expr) => {
        &[
            concat!("https://huggingface.co/", $path),
            concat!("https://hf-mirror.com/", $path),
        ]
    };
}

macro_rules! whisper_files {
    ($m:literal, $enc:expr, $enc_sha:literal, $dec:expr, $dec_sha:literal) => {
        &[
            FileSpec {
                urls: hf!(concat!(
                    "csukuangfj/sherpa-onnx-whisper-",
                    $m,
                    "/resolve/main/",
                    $m,
                    "-encoder.int8.onnx"
                )),
                name: ENCODER_FILE,
                size: $enc,
                sha256: $enc_sha,
            },
            FileSpec {
                urls: hf!(concat!(
                    "csukuangfj/sherpa-onnx-whisper-",
                    $m,
                    "/resolve/main/",
                    $m,
                    "-decoder.int8.onnx"
                )),
                name: DECODER_FILE,
                size: $dec,
                sha256: $dec_sha,
            },
            FileSpec {
                urls: hf!(concat!(
                    "csukuangfj/sherpa-onnx-whisper-",
                    $m,
                    "/resolve/main/",
                    $m,
                    "-tokens.txt"
                )),
                name: TOKENS_FILE,
                size: 816_730,
                sha256: "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126",
            },
        ]
    };
}

// Error rates in the notes: CER % (ko, ja, zh) and WER % (en), FLEURS clips,
// CPU bake-off of 2026-11-08 (research-cpu-asr.md of the coordinator task).
const SPECS: &[Spec] = &[
    // SenseVoice-Small int8, FunASR Model License. Package 2024-07-17: the int8
    // file of 2025-09-09 breaks ko, ja and en.
    Spec {
        id: SENSE_VOICE_ID,
        kind: ModelKind::Asr,
        name: "SenseVoice Small",
        note: "가볍고 빠름, 메모리 약 1GB (기본). 오류율 % 한 6.9 · 일 7.2 · 중 4.5 · 영 8.7",
        files: &[
            FileSpec {
                urls: hf!(
                    "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx"
                ),
                name: SENSE_VOICE_FILE,
                size: 239_233_841,
                sha256: "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51",
            },
            FileSpec {
                urls: hf!(
                    "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt"
                ),
                name: TOKENS_FILE,
                size: 315_894,
                sha256: "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc",
            },
        ],
    },
    // Qwen3-ASR 0.6B int8 (Alibaba Qwen, Apache-2.0), sherpa-onnx export.
    Spec {
        id: QWEN3_ASR_ID,
        kind: ModelKind::Asr,
        name: "Qwen3-ASR 0.6B",
        note: "정확도 높음(한국어·중국어), 느림, 메모리 약 2.5GB. 오류율 % 한 5.1 · 일 12.5 · 중 2.7 · 영 8.1",
        files: &[
            FileSpec {
                urls: hf!(
                    "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/resolve/main/conv_frontend.onnx"
                ),
                name: QWEN3_FRONTEND_FILE,
                size: 44_148_281,
                sha256: "d22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e",
            },
            FileSpec {
                urls: hf!(
                    "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/resolve/main/encoder.int8.onnx"
                ),
                name: ENCODER_FILE,
                size: 182_491_662,
                sha256: "60748d3e6744a57c9c91e1b17424a6c2990567e8adceb0783940c03ed98fa9d9",
            },
            FileSpec {
                urls: hf!(
                    "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/resolve/main/decoder.int8.onnx"
                ),
                name: DECODER_FILE,
                size: 755_914_231,
                sha256: "4f6885be5959ae26af3089d38ee7972c5fafbeeb1cf8d5e76eab6d8b61ca5771",
            },
            FileSpec {
                urls: hf!(
                    "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/resolve/main/tokenizer/vocab.json"
                ),
                name: QWEN3_TOKENIZER_FILES[0],
                size: 2_776_833,
                sha256: "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
            },
            FileSpec {
                urls: hf!(
                    "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/resolve/main/tokenizer/merges.txt"
                ),
                name: QWEN3_TOKENIZER_FILES[1],
                size: 1_671_853,
                sha256: "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
            },
            FileSpec {
                urls: hf!(
                    "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/resolve/main/tokenizer/tokenizer_config.json"
                ),
                name: QWEN3_TOKENIZER_FILES[2],
                size: 12_487,
                sha256: "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c",
            },
        ],
    },
    Spec {
        id: "whisper-base",
        kind: ModelKind::Asr,
        name: "Whisper base",
        note: "빠름, 정확도 낮음",
        files: whisper_files!(
            "base",
            29_120_534,
            "0b8fb1304b6109976038efff5ace81720e00386f3ff6b54ee8c75291ca0a1e11",
            130_672_026,
            "9759d217388a01b3a4c7c15533201067b48ae819c4daafc8624e64b9409dc02d"
        ),
    },
    Spec {
        id: "whisper-small",
        kind: ModelKind::Asr,
        name: "Whisper small",
        note: "SenseVoice보다 느림, 메모리 약 2GB. 오류율 % 한 6.4 · 일 14.4 · 중 12.1 · 영 10.3",
        files: whisper_files!(
            "small",
            112_442_483,
            "4cbe7b22fa9026b843b60a68640c747de05bafb1a11b57edc0e66c232d9f33a9",
            262_226_114,
            "acad50b5c782696e91b55914cc5ab4f756f1532f76e22aa6fc615f39fb69a8ee"
        ),
    },
    Spec {
        id: "whisper-medium",
        kind: ModelKind::Asr,
        name: "Whisper medium",
        note: "느림, 정확도 높음",
        files: whisper_files!(
            "medium",
            374_196_283,
            "1c54582b4d829de0089f6cb63bbbdb3bf7555398bacaf855fbecf1a84dfd193e",
            571_059_257,
            "595d00a338a365a7bfa0ca7f296cabc639583bef770ab6130df90f49a6412747"
        ),
    },
    Spec {
        id: "whisper-turbo",
        kind: ModelKind::Asr,
        name: "Whisper turbo",
        note: "정확도 높음, CPU에서는 실시간보다 느릴 수 있음, 메모리 3GB 이상. 오류율 % 한 6.8",
        files: whisper_files!(
            "turbo",
            674_716_297,
            "b02dcdf54f348741e93fe732b67d933c8dcb6735655f710640143081db38878b",
            361_080_764,
            "20accd02388482eb3a46bd615631adfdc85e1eb2c7db9ea3f02a40ffe6b81547"
        ),
    },
    // Gemma 4, Apache-2.0, official Google GGUF repos (not gated). Facts:
    // research-gemma.md of the coordinator task (2026-11-08).
    Spec {
        id: "gemma-4-e2b-it-q4",
        kind: ModelKind::Translation,
        name: "Gemma 4 E2B",
        note: "빠름, 일반 노트북용, 메모리 약 2GB (기본)",
        files: &[FileSpec {
            urls: hf!("google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf"),
            name: LLM_FILE,
            size: 3_349_516_256,
            sha256: "fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634",
        }],
    },
    Spec {
        id: "gemma-4-12b-it-q4",
        kind: ModelKind::Translation,
        name: "Gemma 4 12B",
        note: "번역 품질 높음, 느림, 메모리 약 8GB (RAM 16GB 이상 권장)",
        files: &[FileSpec {
            urls: hf!(
                "google/gemma-4-12B-it-qat-q4_0-gguf/resolve/main/gemma-4-12b-it-qat-q4_0.gguf"
            ),
            name: LLM_FILE,
            size: 6_975_879_296,
            sha256: "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b",
        }],
    },
];

fn info(spec: &Spec) -> ModelInfo {
    ModelInfo {
        id: spec.id.into(),
        kind: spec.kind,
        name: spec.name.into(),
        size_bytes: spec.files.iter().map(|f| f.size).sum(),
        note: spec.note.into(),
    }
}

/// Display name of a catalog model ("Gemma 4 E2B"); the id if unknown.
pub fn display_name(id: &str) -> String {
    SPECS
        .iter()
        .find(|s| s.id == id)
        .map_or_else(|| id.to_string(), |s| s.name.to_string())
}

/// All models that the app can download.
pub fn catalog() -> Vec<ModelInfo> {
    SPECS.iter().map(info).collect()
}

/// Folder or file of an installed model.
pub fn path(models_dir: &Path, id: &str) -> PathBuf {
    models_dir.join(id)
}

pub fn is_installed(models_dir: &Path, id: &str) -> bool {
    match SPECS.iter().find(|s| s.id == id) {
        Some(spec) => spec_installed(models_dir, spec),
        None => path(models_dir, id).exists(),
    }
}

fn spec_installed(models_dir: &Path, spec: &Spec) -> bool {
    let dir = path(models_dir, spec.id);
    spec.files.iter().all(|f| dir.join(f.name).is_file())
}

/// Downloads a model. Writes to `<id>.part`, then renames, so a cancelled or
/// failed download leaves no partial model.
pub fn download(
    models_dir: &Path,
    id: &str,
    progress: &dyn Fn(DownloadProgress),
    cancel: &AtomicBool,
) -> anyhow::Result<()> {
    let Some(spec) = SPECS.iter().find(|s| s.id == id) else {
        bail!("알 수 없는 모델입니다: {id}");
    };
    download_spec(models_dir, spec, progress, cancel)
}

fn download_spec(
    models_dir: &Path,
    spec: &Spec,
    progress: &dyn Fn(DownloadProgress),
    cancel: &AtomicBool,
) -> anyhow::Result<()> {
    if spec_installed(models_dir, spec) {
        return Ok(());
    }
    let final_dir = path(models_dir, spec.id);
    let part_dir = models_dir.join(format!("{}.part", spec.id));
    if part_dir.exists() {
        std::fs::remove_dir_all(&part_dir).context("이전 다운로드 임시 폴더를 지울 수 없습니다")?;
    }
    std::fs::create_dir_all(&part_dir).with_context(|| {
        format!(
            "모델 폴더를 만들 수 없습니다: {}. 저장 공간과 권한을 확인하세요.",
            models_dir.display()
        )
    })?;

    let result = fetch_files(spec, &part_dir, progress, cancel);
    if let Err(e) = result {
        let _ = std::fs::remove_dir_all(&part_dir);
        return Err(e);
    }
    if final_dir.exists() {
        // An incomplete folder from an older run: replace it.
        std::fs::remove_dir_all(&final_dir).context("이전 모델 폴더를 지울 수 없습니다")?;
    }
    std::fs::rename(&part_dir, &final_dir).context("다운로드한 모델의 이름을 바꿀 수 없습니다")?;
    Ok(())
}

fn fetch_files(
    spec: &Spec,
    dir: &Path,
    progress: &dyn Fn(DownloadProgress),
    cancel: &AtomicBool,
) -> anyhow::Result<()> {
    let client = reqwest::blocking::Client::builder()
        // The GitHub API refuses requests without a User-Agent.
        .user_agent(concat!("morago/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(10))
        // The blocking client has a 30 s total timeout by default: off for big files.
        .timeout(None::<Duration>)
        .build()
        .context("다운로드 클라이언트를 만들 수 없습니다")?;
    let total: u64 = spec.files.iter().map(|f| f.size).sum();
    let mut done: u64 = 0;
    let mut reported_percent: u64 = 0;
    let report = |done: u64, reported_percent: &mut u64| {
        let percent = (done * 100).checked_div(total).unwrap_or(0);
        if percent > *reported_percent || (done == total && *reported_percent < 100) {
            *reported_percent = percent.max(*reported_percent);
            progress(DownloadProgress {
                model_id: spec.id.into(),
                done_bytes: done,
                total_bytes: Some(total),
            });
        }
    };
    progress(DownloadProgress {
        model_id: spec.id.into(),
        done_bytes: 0,
        total_bytes: Some(total),
    });
    for file in spec.files {
        let target = dir.join(file.name);
        let mut last_err = None;
        for url in file.urls {
            let before = done;
            match fetch_one(&client, url, file, &target, cancel, &mut |n| {
                done += n;
                report(done.min(total), &mut reported_percent);
            }) {
                Ok(()) => {
                    last_err = None;
                    break;
                }
                Err(Fetch::Cancelled) => bail!("다운로드를 취소했습니다."),
                Err(Fetch::Failed(e)) => {
                    log::warn!("model download from {url} failed: {e:#}");
                    done = before;
                    last_err = Some(e);
                }
            }
        }
        if let Some(e) = last_err {
            return Err(e);
        }
    }
    report(total, &mut reported_percent);
    Ok(())
}

enum Fetch {
    Cancelled,
    Failed(anyhow::Error),
}

impl From<anyhow::Error> for Fetch {
    fn from(e: anyhow::Error) -> Self {
        Fetch::Failed(e)
    }
}

/// Downloads one file from one URL into `target` and checks size and SHA-256.
fn fetch_one(
    client: &reqwest::blocking::Client,
    url: &str,
    file: &FileSpec,
    target: &Path,
    cancel: &AtomicBool,
    on_bytes: &mut dyn FnMut(u64),
) -> Result<(), Fetch> {
    use sha2::{Digest, Sha256};
    if cancel.load(Ordering::Relaxed) {
        return Err(Fetch::Cancelled);
    }
    let mut req = client.get(url);
    if url.starts_with("https://api.github.com/") {
        // GitHub release asset through the API: asks for the file, not its JSON.
        req = req.header(reqwest::header::ACCEPT, "application/octet-stream");
    }
    let mut resp = req.send().map_err(|e| {
        anyhow::anyhow!(
            "모델 서버에 연결할 수 없습니다: {}. 인터넷 연결을 확인하세요.",
            e.without_url()
        )
    })?;
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!(
            "모델을 내려받지 못했습니다 (HTTP {}): {}",
            resp.status().as_u16(),
            file.name
        )
        .into());
    }
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(target)
            .with_context(|| format!("파일을 만들 수 없습니다: {}", target.display()))?,
    );
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 256 * 1024];
    let mut written: u64 = 0;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(Fetch::Cancelled);
        }
        let n = resp
            .read(&mut buf)
            .context("다운로드 중 연결이 끊겼습니다. 다시 시도하세요.")?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])
            .context("모델 파일을 쓸 수 없습니다. 저장 공간을 확인하세요.")?;
        hasher.update(&buf[..n]);
        written += n as u64;
        on_bytes(n as u64);
    }
    out.flush().context("모델 파일을 쓸 수 없습니다.")?;
    let digest: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if written != file.size || digest != file.sha256 {
        return Err(anyhow::anyhow!(
            "다운로드한 파일이 맞지 않습니다: {}. 다시 시도하세요.",
            file.name
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::Mutex;

    #[test]
    fn catalog_has_required_models() {
        let cat = catalog();
        for id in [
            SENSE_VOICE_ID,
            QWEN3_ASR_ID,
            "whisper-base",
            "whisper-small",
            "whisper-medium",
            "whisper-turbo",
            "gemma-4-e2b-it-q4",
            "gemma-4-12b-it-q4",
        ] {
            assert!(cat.iter().any(|m| m.id == id), "missing {id}");
        }
        let mut ids: Vec<_> = cat.iter().map(|m| m.id.clone()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), cat.len(), "ids must be unique");
        assert!(cat.iter().all(|m| m.size_bytes > 0));
        let small = cat.iter().find(|m| m.id == "whisper-small").unwrap();
        assert_eq!(small.size_bytes, 112_442_483 + 262_226_114 + 816_730);
        let sv = cat.iter().find(|m| m.id == SENSE_VOICE_ID).unwrap();
        assert_eq!(sv.size_bytes, 239_233_841 + 315_894);
        // The 2024-07-17 int8 file: the 2025-09-09 one breaks ko, ja and en.
        let sv_spec = SPECS.iter().find(|s| s.id == SENSE_VOICE_ID).unwrap();
        assert!(sv_spec.files[0].urls[0].contains("-2024-07-17/"));
        assert_eq!(
            sv_spec.files[0].sha256,
            "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"
        );
        let qwen = cat.iter().find(|m| m.id == QWEN3_ASR_ID).unwrap();
        assert_eq!(
            qwen.size_bytes,
            44_148_281 + 182_491_662 + 755_914_231 + 2_776_833 + 1_671_853 + 12_487
        );
    }

    #[test]
    fn default_settings_models_are_in_catalog() {
        let s = crate::settings::Settings::default();
        let cat = catalog();
        if let crate::settings::AsrConfig::Local { model_id } = s.asr {
            assert!(cat.iter().any(|m| m.id == model_id));
        }
        if let crate::settings::TranslateConfig::Local { model_id } = s.translate {
            assert!(cat.iter().any(|m| m.id == model_id));
        }
    }

    /// Serves `body` for every GET on a local port; returns the base URL.
    fn serve(body: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                while reader.read_line(&mut line).is_ok() {
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    line.clear();
                }
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body);
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    fn sha(bytes: &[u8]) -> String {
        sha256_hex(bytes)
    }

    /// The model files in the repository are the ones the app checks.
    #[test]
    fn bundled_models_match_their_hashes() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/models");
        verify_bundled(&dir).unwrap();
        assert!(
            !catalog()
                .iter()
                .any(|m| BUNDLED.iter().any(|b| b.0 == m.id))
        );
    }

    #[test]
    fn a_damaged_bundled_model_is_reported() {
        let dir = temp_dir("bundled");
        for (id, file, _) in BUNDLED {
            std::fs::create_dir_all(dir.join(id)).unwrap();
            std::fs::write(dir.join(id).join(file), b"not a model").unwrap();
        }
        let err = verify_bundled(&dir).unwrap_err();
        assert!(err.to_string().contains("손상"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_with_another_hash_is_rejected() {
        let base = serve(vec![3u8; 10_000]);
        let url = format!("{base}/a");
        let other = sha(&[4u8; 10_000]);
        let files = [FileSpec {
            urls: &[&url],
            name: "a.bin",
            size: 10_000,
            sha256: &other,
        }];
        let spec = Spec {
            id: "test-hash",
            kind: ModelKind::Asr,
            name: "t",
            note: "t",
            files: &files,
        };
        let dir = temp_dir("hash");
        let err = download_spec(&dir, &spec, &|_| {}, &AtomicBool::new(false)).unwrap_err();
        assert!(err.to_string().contains("맞지 않습니다"), "{err}");
        assert!(!spec_installed(&dir, &spec));
        assert!(!dir.join("test-hash.part").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_catalog_file_has_a_pinned_hash() {
        for spec in SPECS {
            for f in spec.files {
                assert!(!f.urls.is_empty(), "{}", spec.id);
                assert_eq!(f.sha256.len(), 64, "{} {}", spec.id, f.name);
                assert!(f.sha256.bytes().all(|c| c.is_ascii_hexdigit()));
            }
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("morago-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn download_writes_files_reports_progress_and_renames() {
        let base = serve(vec![7u8; 100_000]);
        // The first URL of file a refuses the connection: the next one is used.
        let url_a = [String::from("http://127.0.0.1:9/a"), format!("{base}/a")];
        let url_b = [format!("{base}/b")];
        let a: Vec<&str> = url_a.iter().map(String::as_str).collect();
        let b: Vec<&str> = url_b.iter().map(String::as_str).collect();
        let hash = sha(&[7u8; 100_000]);
        let files = [
            FileSpec {
                urls: &a,
                name: "a.bin",
                size: 100_000,
                sha256: &hash,
            },
            FileSpec {
                urls: &b,
                name: "b.bin",
                size: 100_000,
                sha256: &hash,
            },
        ];
        let spec = Spec {
            id: "test-model",
            kind: ModelKind::Asr,
            name: "t",
            note: "t",
            files: &files,
        };
        let dir = temp_dir("dl");
        let events = Mutex::new(Vec::new());
        download_spec(
            &dir,
            &spec,
            &|p| events.lock().unwrap().push(p),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(spec_installed(&dir, &spec));
        assert!(!dir.join("test-model.part").exists());
        assert_eq!(
            std::fs::read(dir.join("test-model/b.bin")).unwrap().len(),
            100_000
        );
        let events = events.into_inner().unwrap();
        let last = events.last().unwrap();
        assert_eq!(last.done_bytes, 200_000);
        assert_eq!(last.total_bytes, Some(200_000));
        // About one event per percent, never more.
        assert!(events.len() <= 102, "{} events", events.len());
        assert!(
            events
                .windows(2)
                .all(|w| w[0].done_bytes <= w[1].done_bytes)
        );
        // A second call is a no-op.
        download_spec(
            &dir,
            &spec,
            &|_| panic!("no progress"),
            &AtomicBool::new(false),
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cancelled_download_leaves_nothing() {
        let base = serve(vec![1u8; 50_000]);
        let url = format!("{base}/a");
        let hash = sha(&[1u8; 50_000]);
        let files = [FileSpec {
            urls: &[&url],
            name: "a.bin",
            size: 50_000,
            sha256: &hash,
        }];
        let spec = Spec {
            id: "test-cancel",
            kind: ModelKind::Asr,
            name: "t",
            note: "t",
            files: &files,
        };
        let dir = temp_dir("cancel");
        let cancel = AtomicBool::new(false);
        // Cancel at the first progress event after the start event.
        let count = Mutex::new(0);
        let err = download_spec(
            &dir,
            &spec,
            &|_| {
                let mut c = count.lock().unwrap();
                *c += 1;
                if *c == 2 {
                    cancel.store(true, Ordering::Relaxed);
                }
            },
            &cancel,
        )
        .unwrap_err();
        assert!(err.to_string().contains("취소"), "{err}");
        assert!(!dir.join("test-cancel").exists());
        assert!(!dir.join("test-cancel.part").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn http_error_is_reported_in_korean_with_status() {
        // Port with no listener: connection refused.
        let files = [FileSpec {
            urls: &["http://127.0.0.1:9/a"],
            name: "a.bin",
            size: 1,
            sha256: "00",
        }];
        let spec = Spec {
            id: "test-fail",
            kind: ModelKind::Asr,
            name: "t",
            note: "t",
            files: &files,
        };
        let dir = temp_dir("fail");
        let err = download_spec(&dir, &spec, &|_| {}, &AtomicBool::new(false)).unwrap_err();
        assert!(err.to_string().contains("연결"), "{err}");
        assert!(!dir.join("test-fail.part").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
