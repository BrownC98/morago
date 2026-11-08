//! Shared data types between modules. This file is the contract:
//! change it only with the coordinator's approval.

use serde::{Deserialize, Serialize};

/// Sample rate of all audio after capture (mono f32, -1.0..=1.0).
pub const SAMPLE_RATE: u32 = 16_000;

/// Where a piece of audio came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    /// Microphone input.
    Mic,
    /// System audio output (loopback): other meeting members.
    System,
}

/// A block of captured audio, already 16 kHz mono f32.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub source: SourceKind,
    /// Milliseconds since the session start of the first sample.
    pub start_ms: u64,
    pub samples: Vec<f32>,
}

/// One voiced region found by the VAD.
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub source: SourceKind,
    pub start_ms: u64,
    pub end_ms: u64,
    /// 16 kHz mono f32.
    pub samples: Vec<f32>,
}

/// Result of speech recognition for one segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    pub text: String,
    /// ISO 639-1 code ("en", "ko", "ja") when the provider detects it.
    pub language: Option<String>,
}

/// One utterance as stored and shown. The UI upserts by `id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Utterance {
    /// Sequence number in the session, starts at 1.
    pub id: u64,
    pub source: SourceKind,
    /// Speaker id given by diarization, for example "S1". Display names
    /// are in `SessionMeta::speaker_names`.
    pub speaker: String,
    pub start_ms: u64,
    pub end_ms: u64,
    /// ISO 639-1 code of the original text, if known.
    pub language: Option<String>,
    pub original: String,
    pub translation: TranslationState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum TranslationState {
    /// Translation is running.
    Pending,
    /// Translated into `target`.
    Done { target: String, text: String },
    /// Source language equals the target language: no translation.
    SameLanguage,
    /// Translation failed; `message` is for the user.
    Failed { message: String },
}

/// Metadata of one recorded session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionMeta {
    /// Folder-safe id, for example "20261108-140212-ab12".
    pub id: String,
    pub title: String,
    /// RFC 3339 local time.
    pub started_at: String,
    pub ended_at: Option<String>,
    /// ISO 639-1 target language at session start.
    pub target_language: String,
    /// Human-readable provider labels, for example "Local Whisper small".
    /// Never contains a URL with credentials or an API key.
    pub asr_label: String,
    pub translator_label: String,
    /// Speaker id -> display name set by the user ("S2" -> "나").
    #[serde(default)]
    pub speaker_names: std::collections::BTreeMap<String, String>,
}

/// A session summary for the record list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionSummary {
    pub meta: SessionMeta,
    pub duration_ms: u64,
    pub utterance_count: u64,
    pub speaker_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Markdown,
    Srt,
    Json,
}

/// Endpoint of an OpenAI-compatible service. The API key is not here:
/// it is in the OS credential store under a `secrets::KeySlot`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Endpoint {
    /// For example "https://api.openai.com/v1" or "http://localhost:11434/v1".
    pub base_url: String,
    pub model: String,
}

/// Progress of a model download, sent to the UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub model_id: String,
    pub done_bytes: u64,
    pub total_bytes: Option<u64>,
}

/// Settings › GPU 가속: where local models run.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GpuSetting {
    /// The first usable GPU, else the CPU.
    #[default]
    Auto,
    /// Only the GPU with this name, as the driver gives it.
    Device { name: String },
    /// Always the CPU.
    Off,
}

/// Where a local model runs, for the settings rows and the status bar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDevice {
    /// True: a GPU. False: the CPU.
    pub gpu: bool,
    /// For the user: the GPU and its backend, or why the CPU is used.
    pub detail: String,
}

/// Live text of speech that has not ended yet (streaming ASR, follow-up
/// "live mode"). The UI shows one partial line per source and removes it
/// when an `Utterance` of the same source arrives. The VAD + Whisper
/// pipeline does not send it.
#[allow(dead_code)] // Reserved for the follow-up live mode (human decision 2026-11-08).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartialText {
    pub source: SourceKind,
    pub start_ms: u64,
    pub text: String,
}
