//! Local ASR: sherpa-onnx offline Whisper (multilingual, int8).
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

pub struct LocalWhisper {
    recognizer: *const sys::OfflineRecognizer,
    label: String,
}

// SAFETY: the sherpa-onnx C API allows one recognizer to move between threads;
// `transcribe` takes `&mut self`, so calls never overlap.
unsafe impl Send for LocalWhisper {}

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

impl LocalWhisper {
    pub fn new(models_dir: &Path, model_id: &str) -> anyhow::Result<Self> {
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
        let dir = models::path(models_dir, model_id);
        let encoder = c_path(&dir.join(models::WHISPER_ENCODER_FILE))?;
        let decoder = c_path(&dir.join(models::WHISPER_DECODER_FILE))?;
        let tokens = c_path(&dir.join(models::WHISPER_TOKENS_FILE))?;
        let language = CString::new("").unwrap(); // empty = detect per segment
        let task = CString::new("transcribe").unwrap();
        let decoding = CString::new("greedy_search").unwrap();
        let provider = CString::new("cpu").unwrap();
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().min(4))
            .unwrap_or(2) as i32;

        // SAFETY: the C config structs hold raw pointers, integers and floats;
        // all-zero means "unset". The CStrings above outlive the create call,
        // which copies what it needs.
        let recognizer = unsafe {
            let mut config: sys::OfflineRecognizerConfig = std::mem::zeroed();
            config.feat_config = sys::FeatureConfig {
                sample_rate: SAMPLE_RATE as i32,
                feature_dim: 80,
            };
            config.decoding_method = decoding.as_ptr();
            config.max_active_paths = 4;
            config.model_config.whisper.encoder = encoder.as_ptr();
            config.model_config.whisper.decoder = decoder.as_ptr();
            config.model_config.whisper.language = language.as_ptr();
            config.model_config.whisper.task = task.as_ptr();
            config.model_config.whisper.tail_paddings = -1;
            config.model_config.tokens = tokens.as_ptr();
            config.model_config.num_threads = threads;
            config.model_config.provider = provider.as_ptr();
            sys::SherpaOnnxCreateOfflineRecognizer(&config)
        };
        if recognizer.is_null() {
            bail!(
                "{} 모델을 불러오지 못했습니다. 모델을 다시 내려받아 보세요.",
                info.name
            );
        }
        Ok(Self {
            recognizer,
            label: format!("로컬 {}", info.name),
        })
    }
}

impl Drop for LocalWhisper {
    fn drop(&mut self) {
        // SAFETY: `recognizer` came from `SherpaOnnxCreateOfflineRecognizer` and is freed once.
        unsafe { sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer) }
    }
}

impl Asr for LocalWhisper {
    fn transcribe(&mut self, segment: &SpeechSegment) -> anyhow::Result<Transcript> {
        // SAFETY: stream and result strings are freed once, in this function.
        let json = unsafe {
            let stream = sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            if stream.is_null() {
                bail!("음성 인식 작업을 시작하지 못했습니다.");
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
        match json {
            Some(json) => parse_result(&json),
            None => bail!("음성 인식 결과를 읽지 못했습니다."),
        }
    }

    fn label(&self) -> String {
        self.label.clone()
    }
}

/// Parses the result JSON; `lang` is "en" or "<|en|>" depending on the version.
fn parse_result(json: &str) -> anyhow::Result<Transcript> {
    let raw: RawResult =
        serde_json::from_str(json).context("음성 인식 결과를 해석하지 못했습니다")?;
    let lang = raw
        .lang
        .trim()
        .trim_start_matches("<|")
        .trim_end_matches("|>");
    Ok(Transcript {
        text: raw.text.trim().to_string(),
        language: language_code(lang),
    })
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
        assert!(parse_result("oops").is_err());
    }
}
