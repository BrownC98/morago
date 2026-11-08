//! Voice activity detection with Silero VAD (sherpa-onnx).

use crate::models;
use crate::types::{AudioChunk, SAMPLE_RATE, SpeechSegment};
use anyhow::{Context, bail};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};
use std::path::Path;

const MIN_SPEECH_S: f32 = 0.25;
const MIN_SILENCE_S: f32 = 0.5;
/// Longer speech is split so subtitles stay live.
const MAX_SEGMENT_S: f32 = 15.0;

/// Voice activity detection for one source. Use one value per source.
pub struct Vad {
    detector: VoiceActivityDetector,
    source: Option<crate::types::SourceKind>,
    /// `start_ms` of the first chunk: the origin of the detector's sample index.
    origin_ms: u64,
}

impl Vad {
    pub fn new(models_dir: &Path) -> anyhow::Result<Self> {
        let model = models::path(models_dir, models::VAD_MODEL_ID).join(models::VAD_FILE);
        if !model.is_file() {
            bail!("음성 검출 모델이 설치되어 있지 않습니다. 설정에서 모델을 내려받으세요.");
        }
        let config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(model.to_string_lossy().into_owned()),
                threshold: 0.5,
                min_silence_duration: MIN_SILENCE_S,
                min_speech_duration: MIN_SPEECH_S,
                window_size: 512,
                max_speech_duration: MAX_SEGMENT_S,
            },
            sample_rate: SAMPLE_RATE as i32,
            num_threads: 1,
            provider: Some("cpu".into()),
            ..Default::default()
        };
        let detector = VoiceActivityDetector::create(&config, 60.0)
            .context("음성 검출 모델을 불러오지 못했습니다. 모델을 다시 내려받아 보세요.")?;
        Ok(Self {
            detector,
            source: None,
            origin_ms: 0,
        })
    }

    /// Adds audio; returns the segments that ended.
    pub fn push(&mut self, chunk: &AudioChunk) -> Vec<SpeechSegment> {
        if self.source.is_none() {
            self.source = Some(chunk.source);
            self.origin_ms = chunk.start_ms;
        }
        self.detector.accept_waveform(&chunk.samples);
        self.drain(chunk.source)
    }

    /// Ends the stream; returns the open segment, if any.
    pub fn flush(&mut self) -> Vec<SpeechSegment> {
        let Some(source) = self.source else {
            return Vec::new();
        };
        self.detector.flush();
        self.drain(source)
    }

    fn drain(&mut self, source: crate::types::SourceKind) -> Vec<SpeechSegment> {
        let mut out = Vec::new();
        while let Some(seg) = self.detector.front() {
            let start_ms = self.origin_ms + seg.start().max(0) as u64 * 1000 / SAMPLE_RATE as u64;
            let samples = seg.samples().to_vec();
            let end_ms = start_ms + samples.len() as u64 * 1000 / SAMPLE_RATE as u64;
            out.push(SpeechSegment {
                source,
                start_ms,
                end_ms,
                samples,
            });
            drop(seg);
            self.detector.pop();
        }
        out
    }
}
