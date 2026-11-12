//! Local ASR: sherpa-onnx offline recognizers (int8): SenseVoice Small,
//! Qwen3-ASR and Parakeet 일본어. SenseVoice and Qwen3-ASR detect the language
//! per segment, or use the meeting language of the settings; Parakeet knows
//! only Japanese and runs only with the meeting language ja.
//! Fun-ASR-MLT-Nano runs on transcribe.cpp (`funasr.rs`).
//!
//! The safe `sherpa-onnx` crate drops the detected language from the result,
//! so this file calls the C API through `sherpa-onnx-sys` and reads `lang`
//! from the result JSON.

use super::{Asr, custom::language_code};
use crate::models;
use crate::types::{SAMPLE_RATE, SpeechSegment, Transcript};
use anyhow::{Context, bail};
use serde::Deserialize;
use sherpa_onnx_sys as sys;
use std::ffi::{CStr, CString};
use std::path::Path;

pub struct LocalAsr {
    recognizer: *const sys::OfflineRecognizer,
    /// Qwen3-ASR takes the language per stream ("Korean"); others in `Config`.
    stream_language: Option<CString>,
    /// Meeting language (ISO 639-1); `None` = from the result.
    language: Option<String>,
    label: String,
}

// SAFETY: the sherpa-onnx C API allows one recognizer to move between threads;
// `transcribe` takes `&mut self`, so calls never overlap.
unsafe impl Send for LocalAsr {}

#[derive(Deserialize)]
struct RawResult {
    #[serde(default)]
    text: String,
    #[serde(default)]
    lang: String,
}

fn c_path(path: &Path) -> anyhow::Result<CString> {
    CString::new(path.to_string_lossy().as_bytes()).context("모델 경로가 올바르지 않습니다")
}

/// Recognizer config and the strings that its pointers point into.
struct Config {
    raw: sys::OfflineRecognizerConfig,
    _strings: Vec<CString>,
    stream_language: Option<CString>,
}

/// Builds the config of the model `model_id` in `dir`. `language` is an ISO
/// 639-1 code or `None` (detect).
fn config(model_id: &str, dir: &Path, language: Option<&str>) -> anyhow::Result<Config> {
    let file = |name: &str| c_path(&dir.join(name));
    let text = |s: &str| CString::new(s).context("설정 값이 올바르지 않습니다");
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(2) as i32;
    let mut strings: Vec<CString> = Vec::new();
    // A CString's buffer does not move when the CString moves into the Vec.
    let mut keep = |c: CString| {
        let ptr = c.as_ptr();
        strings.push(c);
        ptr
    };
    let mut stream_language = None;
    // SAFETY: the C config structs hold raw pointers, integers and floats;
    // all-zero means "unset" (sherpa-onnx then takes its defaults).
    let mut raw: sys::OfflineRecognizerConfig = unsafe { std::mem::zeroed() };
    raw.feat_config = sys::FeatureConfig {
        sample_rate: SAMPLE_RATE as i32,
        feature_dim: 80,
    };
    raw.decoding_method = keep(text("greedy_search")?);
    raw.max_active_paths = 4;
    let m = &mut raw.model_config;
    // Empty language = detect.
    let code = language.unwrap_or("");
    match model_id {
        models::SENSE_VOICE_ID => {
            m.sense_voice.model = keep(file(models::SENSE_VOICE_FILE)?);
            m.sense_voice.language = keep(text(code)?);
            m.sense_voice.use_itn = 1;
            m.tokens = keep(file(models::TOKENS_FILE)?);
        }
        models::QWEN3_ASR_ID => {
            m.qwen3_asr.conv_frontend = keep(file(models::QWEN3_FRONTEND_FILE)?);
            m.qwen3_asr.encoder = keep(file(models::ENCODER_FILE)?);
            m.qwen3_asr.decoder = keep(file(models::DECODER_FILE)?);
            // Folder with vocab.json, merges.txt, tokenizer_config.json.
            m.qwen3_asr.tokenizer = keep(c_path(dir)?);
            // A VAD segment is at most 15 s.
            m.qwen3_asr.max_new_tokens = 256;
            // The config has no language field: it is a stream option.
            if let Some(code) = language {
                stream_language = Some(text(&crate::translate::language_name(code))?);
            }
        }
        models::PARAKEET_JA_ID => {
            if language != models::only_language(model_id) {
                bail!(
                    "Parakeet 일본어 모델은 일본어만 인식합니다. 설정에서 회의 언어를 日本語로 고르거나 다른 음성 인식 모델을 고르십시오."
                );
            }
            m.nemo_ctc.model = keep(file(models::PARAKEET_FILE)?);
            m.tokens = keep(file(models::TOKENS_FILE)?);
        }
        _ => bail!("알 수 없는 음성 인식 모델입니다: {model_id}"),
    }
    m.num_threads = threads;
    m.provider = keep(text("cpu")?);
    Ok(Config {
        raw,
        _strings: strings,
        stream_language,
    })
}

/// The catalog entry of an installed local ASR model.
pub(super) fn installed_model(
    models_dir: &Path,
    model_id: &str,
) -> anyhow::Result<models::ModelInfo> {
    let Some(info) = models::catalog()
        .into_iter()
        .find(|m| m.id == model_id && m.kind == models::ModelKind::Asr)
    else {
        bail!("알 수 없는 음성 인식 모델입니다: {model_id}");
    };
    if !models::is_installed(models_dir, model_id) {
        bail!(
            "{} 모델이 설치되어 있지 않습니다. 설정에서 모델을 내려받으세요.",
            info.name
        );
    }
    Ok(info)
}

impl LocalAsr {
    pub fn new(models_dir: &Path, model_id: &str, language: Option<&str>) -> anyhow::Result<Self> {
        let info = installed_model(models_dir, model_id)?;
        let config = config(model_id, &models::path(models_dir, model_id), language)?;
        // SAFETY: the pointers in `config.raw` point into `config`, which lives
        // until the create call returns; the call copies what it needs.
        let recognizer = unsafe { sys::SherpaOnnxCreateOfflineRecognizer(&config.raw) };
        if recognizer.is_null() {
            bail!(
                "{} 모델을 불러오지 못했습니다. 모델을 다시 내려받아 보세요.",
                info.name
            );
        }
        Ok(Self {
            recognizer,
            stream_language: config.stream_language,
            language: language.map(str::to_string),
            label: format!("로컬 {}", info.name),
        })
    }
}

impl Drop for LocalAsr {
    fn drop(&mut self) {
        // SAFETY: `recognizer` came from `SherpaOnnxCreateOfflineRecognizer` and is freed once.
        unsafe { sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer) }
    }
}

impl Asr for LocalAsr {
    fn transcribe(&mut self, segment: &SpeechSegment) -> anyhow::Result<Transcript> {
        // SAFETY: stream and result strings are freed once, in this function.
        let json = unsafe {
            let stream = sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            if stream.is_null() {
                bail!("음성 인식 작업을 시작하지 못했습니다.");
            }
            if let Some(lang) = &self.stream_language {
                sys::SherpaOnnxOfflineStreamSetOption(stream, c"language".as_ptr(), lang.as_ptr());
            }
            sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                SAMPLE_RATE as i32,
                segment.samples.as_ptr(),
                segment.samples.len() as i32,
            );
            sys::SherpaOnnxDecodeOfflineStream(self.recognizer, stream);
            let raw = sys::SherpaOnnxGetOfflineStreamResultAsJson(stream);
            let json = if raw.is_null() {
                None
            } else {
                let s = CStr::from_ptr(raw).to_string_lossy().into_owned();
                sys::SherpaOnnxDestroyOfflineStreamResultJson(raw);
                Some(s)
            };
            sys::SherpaOnnxDestroyOfflineStream(stream);
            json
        };
        let Some(json) = json else {
            bail!("음성 인식 결과를 읽지 못했습니다.");
        };
        let mut t = parse_result(&json)?;
        if self.language.is_some() {
            t.language = self.language.clone();
        }
        Ok(t)
    }

    fn label(&self) -> String {
        self.label.clone()
    }
}

/// Parses the result JSON; `lang` is "<|en|>" (SenseVoice). Qwen3-ASR
/// leaves `lang` empty: then the script of the text decides.
fn parse_result(json: &str) -> anyhow::Result<Transcript> {
    let raw: RawResult =
        serde_json::from_str(json).context("음성 인식 결과를 해석하지 못했습니다")?;
    Ok(transcript(&raw.text, &raw.lang))
}

/// Text and language of a recognizer result. `lang` is "en", "<|en|>" or
/// empty: then the script of the text decides.
pub(super) fn transcript(text: &str, lang: &str) -> Transcript {
    let text = text.trim().to_string();
    let language = match lang.trim().trim_start_matches("<|").trim_end_matches("|>") {
        "" => language_from_script(&text),
        // Cantonese (SenseVoice) has no ISO 639-1 code of its own.
        "yue" => Some("zh".to_string()),
        lang => language_code(lang),
    };
    Transcript { text, language }
}

/// ko for Hangul, ja for kana; `None` for other scripts, and the translator
/// detects the language. Han without kana is also Japanese ("了解", "大丈夫").
fn language_from_script(text: &str) -> Option<String> {
    let has = |range: std::ops::RangeInclusive<char>| text.chars().any(|c| range.contains(&c));
    let code = if has('\u{AC00}'..='\u{D7A3}') || has('\u{1100}'..='\u{11FF}') {
        "ko"
    } else if has('\u{3040}'..='\u{30FF}') {
        "ja"
    } else {
        return None;
    };
    Some(code.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_language_with_or_without_markers() {
        let t = parse_result(r#"{"lang":"<|ko|>","text":" 안녕 ","tokens":[],"timestamps":[]}"#)
            .unwrap();
        assert_eq!(t.text, "안녕");
        assert_eq!(t.language.as_deref(), Some("ko"));
        let t = parse_result(r#"{"lang":"en","text":"hi"}"#).unwrap();
        assert_eq!(t.language.as_deref(), Some("en"));
        let t = parse_result(r#"{"lang":"","text":"hi"}"#).unwrap();
        assert_eq!(t.language, None);
        let t = parse_result(r#"{"lang":"<|yue|>","text":"你好"}"#).unwrap();
        assert_eq!(t.language.as_deref(), Some("zh"));
        assert!(parse_result("oops").is_err());
    }

    fn c(ptr: *const std::ffi::c_char) -> String {
        // SAFETY: the test reads pointers that `config` set to its own strings.
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn each_model_config_gets_the_meeting_language() {
        let dir = Path::new("/models/m");
        for (language, code, name) in [(Some("ja"), "ja", Some("Japanese")), (None, "", None)] {
            let sv = config(models::SENSE_VOICE_ID, dir, language).unwrap();
            assert_eq!(c(sv.raw.model_config.sense_voice.language), code);
            assert!(c(sv.raw.model_config.sense_voice.model).ends_with("model.int8.onnx"));
            assert!(sv.stream_language.is_none());
            let q = config(models::QWEN3_ASR_ID, dir, language).unwrap();
            assert!(!q.raw.model_config.qwen3_asr.encoder.is_null());
            assert_eq!(
                c(q.raw.model_config.qwen3_asr.tokenizer),
                c_path(dir).unwrap().to_str().unwrap()
            );
            assert_eq!(
                q.stream_language.as_ref().map(|s| s.to_str().unwrap()),
                name
            );
        }
        assert!(config(models::FUN_ASR_MLT_ID, dir, None).is_err());
        let p = config(models::PARAKEET_JA_ID, dir, Some("ja")).unwrap();
        assert!(c(p.raw.model_config.nemo_ctc.model).ends_with("model.int8.onnx"));
        assert!(c(p.raw.model_config.tokens).ends_with("tokens.txt"));
        assert!(p.stream_language.is_none());
        // Japanese only: another or no meeting language is refused.
        for language in [None, Some("ko"), Some("en")] {
            let err = config(models::PARAKEET_JA_ID, dir, language).err().unwrap();
            assert!(err.to_string().contains("日本語"), "{err}");
        }
    }

    #[test]
    fn an_empty_lang_takes_the_language_of_the_script() {
        for (text, lang) in [
            ("안녕하세요 Kim 씨", Some("ko")),
            ("今日は会議です", Some("ja")),
            ("今天开会", None),
            ("了解", None),
            ("Hello, team.", None),
            ("", None),
        ] {
            let json = serde_json::json!({ "lang": "", "text": text }).to_string();
            assert_eq!(
                parse_result(&json).unwrap().language.as_deref(),
                lang,
                "{text}"
            );
        }
    }

    #[test]
    #[ignore = "downloads a model and needs wav files (MORAGO_TEST_WAVS)"]
    fn parakeet_ja_on_a_real_clip() {
        use super::super::tests::{ensure_model, read_wav};
        let wavs = std::path::PathBuf::from(
            std::env::var("MORAGO_TEST_WAVS").expect("set MORAGO_TEST_WAVS to the wav folder"),
        );
        let models_dir = std::env::var("MORAGO_TEST_MODELS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("morago-test-models"));
        ensure_model(&models_dir, models::PARAKEET_JA_ID);
        let mut asr = LocalAsr::new(&models_dir, models::PARAKEET_JA_ID, Some("ja")).unwrap();
        let samples = read_wav(&wavs, "sv_ja");
        let secs = samples.len() as f64 / SAMPLE_RATE as f64;
        let seg = SpeechSegment {
            source: crate::types::SourceKind::System,
            start_ms: 0,
            end_ms: (secs * 1000.0) as u64,
            samples,
        };
        let started = std::time::Instant::now();
        let t = asr.transcribe(&seg).unwrap();
        let rtf = started.elapsed().as_secs_f64() / secs;
        println!(
            "{} sv_ja ({secs:.1} s, RTF {rtf:.2}): {}",
            asr.label(),
            t.text
        );
        assert_eq!(t.language.as_deref(), Some("ja"));
        assert!(
            t.text
                .chars()
                .any(|c| ('\u{3040}'..='\u{30FF}').contains(&c)),
            "{}",
            t.text
        );
    }
}
