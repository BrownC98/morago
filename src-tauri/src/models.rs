//! Local model catalog and downloads (ASR, VAD, speaker embedding, translation LLM).
//! Owner: speech worker. Contract (do not change signatures without the coordinator).
//!
//! Every model is a folder `<models_dir>/<id>` with fixed file names (see `SPECS`).
//! The UI lists only `Asr` and `Translation` models; the engine downloads the VAD
//! and speaker-embedding models (`VAD_MODEL_ID`, `SPEAKER_MODEL_ID`) with the ASR model.

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
    Vad,
    SpeakerEmbedding,
    Translation,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    /// Stable id, for example "whisper-small", "qwen2.5-3b-instruct-q4".
    pub id: String,
    pub kind: ModelKind,
    /// Display name, for example "Whisper small".
    pub name: String,
    pub size_bytes: u64,
    /// Short user-facing note (Korean), for example "빠름, 정확도 보통".
    pub note: String,
}

/// Id of the Silero VAD model (needed by every local session).
pub const VAD_MODEL_ID: &str = "silero-vad";
/// Id of the speaker-embedding model (needed when diarization is on).
pub const SPEAKER_MODEL_ID: &str = "speaker-campplus";

/// File names inside a model folder. `speech` and `translate` read these.
pub const VAD_FILE: &str = "model.onnx";
pub const SPEAKER_FILE: &str = "model.onnx";
pub const WHISPER_ENCODER_FILE: &str = "encoder.int8.onnx";
pub const WHISPER_DECODER_FILE: &str = "decoder.int8.onnx";
pub const WHISPER_TOKENS_FILE: &str = "tokens.txt";
pub const LLM_FILE: &str = "model.gguf";

/// One file of a model: where it comes from and its name in the folder.
/// `size` was read from the `Content-Length` header of the URL (2026-11).
pub struct FileSpec<'a> {
    pub url: &'a str,
    pub name: &'a str,
    pub size: u64,
}

pub struct Spec<'a> {
    pub id: &'a str,
    pub kind: ModelKind,
    pub name: &'a str,
    pub note: &'a str,
    pub files: &'a [FileSpec<'a>],
}

macro_rules! whisper_files {
    ($m:literal, $enc:expr, $dec:expr) => {
        &[
            FileSpec {
                url: concat!(
                    "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-",
                    $m,
                    "/resolve/main/",
                    $m,
                    "-encoder.int8.onnx"
                ),
                name: WHISPER_ENCODER_FILE,
                size: $enc,
            },
            FileSpec {
                url: concat!(
                    "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-",
                    $m,
                    "/resolve/main/",
                    $m,
                    "-decoder.int8.onnx"
                ),
                name: WHISPER_DECODER_FILE,
                size: $dec,
            },
            FileSpec {
                url: concat!(
                    "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-",
                    $m,
                    "/resolve/main/",
                    $m,
                    "-tokens.txt"
                ),
                name: WHISPER_TOKENS_FILE,
                size: 816_730,
            },
        ]
    };
}

const SPECS: &[Spec] = &[
    Spec {
        id: "whisper-base",
        kind: ModelKind::Asr,
        name: "Whisper base",
        note: "가장 빠름, 정확도 낮음 (저사양 PC)",
        files: whisper_files!("base", 29_120_534, 130_672_026),
    },
    Spec {
        id: "whisper-small",
        kind: ModelKind::Asr,
        name: "Whisper small",
        note: "빠름, 정확도 보통 (기본)",
        files: whisper_files!("small", 112_442_483, 262_226_114),
    },
    Spec {
        id: "whisper-medium",
        kind: ModelKind::Asr,
        name: "Whisper medium",
        note: "느림, 정확도 높음",
        files: whisper_files!("medium", 374_196_283, 571_059_257),
    },
    Spec {
        id: "whisper-turbo",
        kind: ModelKind::Asr,
        name: "Whisper turbo",
        note: "느림, 정확도 가장 높음 (고사양 PC)",
        files: whisper_files!("turbo", 674_716_297, 361_080_764),
    },
    Spec {
        id: VAD_MODEL_ID,
        kind: ModelKind::Vad,
        name: "Silero VAD",
        note: "음성 구간 검출",
        files: &[FileSpec {
            // k2-fsa/sherpa-onnx release "asr-models", asset silero_vad.onnx, through
            // the API: github.com/…/releases/download answered 504 on 2026-11-08
            // while the API worked.
            url: "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/assets/271935959",
            name: VAD_FILE,
            size: 643_854,
        }],
    },
    Spec {
        id: SPEAKER_MODEL_ID,
        kind: ModelKind::SpeakerEmbedding,
        name: "CAM++ 화자 임베딩 (중국어·영어)",
        note: "화자 구분",
        files: &[FileSpec {
            url: "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
            name: SPEAKER_FILE,
            size: 28_281_164,
        }],
    },
    Spec {
        id: "qwen2.5-3b-instruct-q4",
        kind: ModelKind::Translation,
        name: "Qwen2.5 3B Instruct",
        note: "번역 품질 좋음, 메모리 약 3GB (기본)",
        files: &[FileSpec {
            url: "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
            name: LLM_FILE,
            size: 2_104_932_768,
        }],
    },
    Spec {
        id: "qwen2.5-1.5b-instruct-q4",
        kind: ModelKind::Translation,
        name: "Qwen2.5 1.5B Instruct",
        note: "번역 품질 보통, 빠름, 메모리 약 1.5GB (저사양 PC)",
        files: &[FileSpec {
            url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            name: LLM_FILE,
            size: 1_117_320_736,
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
        if cancel.load(Ordering::Relaxed) {
            bail!("다운로드를 취소했습니다.");
        }
        let mut req = client.get(file.url);
        if file.url.starts_with("https://api.github.com/") {
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
            bail!(
                "모델을 내려받지 못했습니다 (HTTP {}): {}",
                resp.status().as_u16(),
                file.name
            );
        }
        let expected = resp.content_length();
        let target = dir.join(file.name);
        let mut out = std::io::BufWriter::new(
            std::fs::File::create(&target)
                .with_context(|| format!("파일을 만들 수 없습니다: {}", target.display()))?,
        );
        let mut buf = vec![0u8; 256 * 1024];
        let mut written: u64 = 0;
        loop {
            if cancel.load(Ordering::Relaxed) {
                bail!("다운로드를 취소했습니다.");
            }
            let n = resp
                .read(&mut buf)
                .context("다운로드 중 연결이 끊겼습니다. 다시 시도하세요.")?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])
                .context("모델 파일을 쓸 수 없습니다. 저장 공간을 확인하세요.")?;
            written += n as u64;
            done += n as u64;
            report(done.min(total), &mut reported_percent);
        }
        out.flush().context("모델 파일을 쓸 수 없습니다.")?;
        if written == 0 || expected.is_some_and(|e| e != written) {
            bail!(
                "다운로드한 파일이 불완전합니다: {}. 다시 시도하세요.",
                file.name
            );
        }
    }
    report(total, &mut reported_percent);
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
            "whisper-base",
            "whisper-small",
            "whisper-medium",
            "qwen2.5-3b-instruct-q4",
            "qwen2.5-1.5b-instruct-q4",
            VAD_MODEL_ID,
            SPEAKER_MODEL_ID,
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

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("morago-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn download_writes_files_reports_progress_and_renames() {
        let base = serve(vec![7u8; 100_000]);
        let url_a = format!("{base}/a");
        let url_b = format!("{base}/b");
        let files = [
            FileSpec {
                url: &url_a,
                name: "a.bin",
                size: 100_000,
            },
            FileSpec {
                url: &url_b,
                name: "b.bin",
                size: 100_000,
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
        let files = [FileSpec {
            url: &url,
            name: "a.bin",
            size: 50_000,
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
        let url = "http://127.0.0.1:9/a".to_string();
        let files = [FileSpec {
            url: &url,
            name: "a.bin",
            size: 1,
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
