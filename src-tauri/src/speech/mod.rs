//! Speech pipeline parts: VAD, speech recognition (ASR) and speaker diarization.
//! Owner: speech worker. Contract (do not change signatures without the coordinator).

use crate::settings::AsrConfig;
use crate::types::{AudioChunk, SpeechSegment, Transcript};
use std::path::Path;

/// Speech recognition for one segment. Runs on a worker thread (blocking).
pub trait Asr: Send {
    fn transcribe(&mut self, segment: &SpeechSegment) -> anyhow::Result<Transcript>;
    /// Label for the UI and session meta, for example "로컬 Whisper small".
    fn label(&self) -> String;
}

/// Builds the ASR from settings. Local models are in `models_dir`.
/// A custom endpoint reads its key with `secrets::get_key(Role::Asr, base_url)`.
pub fn build_asr(config: &AsrConfig, models_dir: &Path) -> anyhow::Result<Box<dyn Asr>> {
    let _ = (config, models_dir);
    anyhow::bail!("ASR is not implemented yet")
}

/// Voice activity detection for one source. Use one value per source.
pub struct Vad {
    _private: (),
}

impl Vad {
    pub fn new(models_dir: &Path) -> anyhow::Result<Self> {
        let _ = models_dir;
        anyhow::bail!("VAD is not implemented yet")
    }
    /// Adds audio; returns the segments that ended.
    pub fn push(&mut self, chunk: &AudioChunk) -> Vec<SpeechSegment> {
        let _ = chunk;
        Vec::new()
    }
    /// Ends the stream; returns the open segment, if any.
    pub fn flush(&mut self) -> Vec<SpeechSegment> {
        Vec::new()
    }
}

/// Online speaker diarization over one session (all sources share one value,
/// so the same voice on mic and system gets one id).
pub struct Diarizer {
    _private: (),
}

impl Diarizer {
    pub fn new(models_dir: &Path) -> anyhow::Result<Self> {
        let _ = models_dir;
        anyhow::bail!("diarization is not implemented yet")
    }
    /// Returns the speaker id ("S1", "S2", ...) for the segment.
    pub fn assign(&mut self, segment: &SpeechSegment) -> anyhow::Result<String> {
        let _ = segment;
        Ok("S1".into())
    }
}
