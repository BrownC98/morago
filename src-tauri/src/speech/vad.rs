//! Voice activity detection with Silero VAD (sherpa-onnx).

use crate::models;
use crate::types::{AudioChunk, SAMPLE_RATE, SpeechSegment};
use anyhow::{Context, bail};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};
use std::collections::VecDeque;
use std::path::Path;

const MIN_SPEECH_S: f32 = 0.25;
const MIN_SILENCE_S: f32 = 0.5;
/// Longer speech is split so subtitles stay live.
const MAX_SEGMENT_S: f32 = 15.0;
/// Audio before the detection that `open_speech` includes. The detector
/// starts a segment about 0.3 s before it detects speech.
const PRE_ROLL: usize = SAMPLE_RATE as usize / 2;
/// Most audio that `open_speech` holds: a segment and a margin.
const MAX_OPEN: usize = (MAX_SEGMENT_S as usize + 1) * SAMPLE_RATE as usize;

/// Voice activity detection for one source. Use one value per source.
pub struct Vad {
    detector: VoiceActivityDetector,
    source: Option<crate::types::SourceKind>,
    /// `start_ms` of the first chunk: the origin of the detector's sample index.
    origin_ms: u64,
    open: OpenAudio,
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
            open: OpenAudio::default(),
        })
    }

    /// Adds audio; returns the segments that ended.
    pub fn push(&mut self, chunk: &AudioChunk) -> Vec<SpeechSegment> {
        if self.source.is_none() {
            self.source = Some(chunk.source);
            self.origin_ms = chunk.start_ms;
        }
        self.detector.accept_waveform(&chunk.samples);
        let (segments, ended) = self.drain(chunk.source);
        self.open
            .push(&chunk.samples, ended, self.detector.detected());
        segments
    }

    /// Ends the stream; returns the open segment, if any.
    pub fn flush(&mut self) -> Vec<SpeechSegment> {
        let Some(source) = self.source else {
            return Vec::new();
        };
        self.detector.flush();
        self.drain(source).0
    }

    /// The audio of the speech in progress, from shortly before its detection.
    /// It never starts before the end of the last segment.
    pub fn open_speech(&self) -> Option<SpeechSegment> {
        let source = self.source?;
        if !self.detector.detected() || self.open.samples.is_empty() {
            return None;
        }
        let start_ms = self.ms(self.open.start);
        Some(SpeechSegment {
            source,
            start_ms,
            end_ms: start_ms + self.open.samples.len() as u64 * 1000 / SAMPLE_RATE as u64,
            samples: self.open.samples.iter().copied().collect(),
        })
    }

    fn ms(&self, index: u64) -> u64 {
        self.origin_ms + index * 1000 / SAMPLE_RATE as u64
    }

    /// Returns the segments that ended and the sample index after the last one.
    fn drain(&mut self, source: crate::types::SourceKind) -> (Vec<SpeechSegment>, Option<u64>) {
        let mut out = Vec::new();
        let mut ended = None;
        while let Some(seg) = self.detector.front() {
            let start = seg.start().max(0) as u64;
            let start_ms = self.ms(start);
            let samples = seg.samples().to_vec();
            let end_ms = start_ms + samples.len() as u64 * 1000 / SAMPLE_RATE as u64;
            ended = Some(start + samples.len() as u64);
            out.push(SpeechSegment {
                source,
                start_ms,
                end_ms,
                samples,
            });
            drop(seg);
            self.detector.pop();
        }
        (out, ended)
    }
}

/// The last samples of a source: a pre-roll while there is no speech, then
/// all samples since the detection.
#[derive(Default)]
struct OpenAudio {
    samples: VecDeque<f32>,
    /// Detector sample index of `samples[0]`.
    start: u64,
    /// Sample index after the last segment: the open audio starts there or later,
    /// so it never overlaps the previous utterance.
    floor: u64,
}

impl OpenAudio {
    /// Adds one chunk. `ended`: the sample index after a segment that ended in
    /// this chunk. `detected`: speech goes on after the chunk.
    fn push(&mut self, samples: &[f32], ended: Option<u64>, detected: bool) {
        if let Some(end) = ended {
            self.floor = end;
            if detected {
                // Long speech was split: start again from this chunk.
                self.start += self.samples.len() as u64;
                self.samples.clear();
            }
        }
        self.samples.extend(samples);
        let keep = if detected { MAX_OPEN } else { PRE_ROLL };
        let below_floor = self.floor.saturating_sub(self.start) as usize;
        let cut = self
            .samples
            .len()
            .saturating_sub(keep)
            .max(below_floor)
            .min(self.samples.len());
        self.samples.drain(..cut);
        self.start += cut as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: usize = 1600;

    fn push(open: &mut OpenAudio, n: usize, ended: Option<u64>, detected: bool) {
        for _ in 0..n / CHUNK {
            open.push(&[0.0; CHUNK], ended, detected);
        }
    }

    #[test]
    fn open_audio_keeps_the_pre_roll_then_the_speech() {
        let mut open = OpenAudio::default();
        push(&mut open, 10 * CHUNK, None, false);
        assert_eq!(open.samples.len(), PRE_ROLL);
        assert_eq!(open.start, (10 * CHUNK - PRE_ROLL) as u64);
        push(&mut open, 20 * CHUNK, None, true);
        assert_eq!(open.samples.len(), PRE_ROLL + 20 * CHUNK);
        // Memory stays bounded in very long speech.
        push(&mut open, 300 * CHUNK, None, true);
        assert_eq!(open.samples.len(), MAX_OPEN);
        assert_eq!(open.start + open.samples.len() as u64, 330 * CHUNK as u64);
    }

    #[test]
    fn open_audio_never_starts_before_the_last_segment_end() {
        let mut open = OpenAudio::default();
        push(&mut open, 30 * CHUNK, None, true);
        // The segment ends 0.5 s before the chunk end; new speech starts at once.
        let end = (31 * CHUNK - PRE_ROLL) as u64;
        open.push(&[0.0; CHUNK], Some(end), false);
        assert!(open.start >= end);
        push(&mut open, CHUNK, None, true);
        assert!(open.start >= end);
        assert_eq!(open.start + open.samples.len() as u64, 32 * CHUNK as u64);
    }

    #[test]
    fn a_split_of_long_speech_starts_again_at_the_current_chunk() {
        let mut open = OpenAudio::default();
        push(&mut open, 150 * CHUNK, None, true);
        let end = (150 * CHUNK) as u64;
        open.push(&[0.0; CHUNK], Some(end), true);
        assert_eq!(open.start, end);
        assert_eq!(open.samples.len(), CHUNK);
    }
}
