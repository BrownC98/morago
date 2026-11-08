//! Online speaker diarization: speaker embeddings + running-mean clustering.

use crate::models;
use crate::types::{SAMPLE_RATE, SourceKind, SpeechSegment};
use anyhow::{Context, bail};
use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};
use std::collections::HashMap;
use std::path::Path;

/// Cosine similarity at or above this joins an existing speaker.
/// Tuned with `speech::tests::real_models_pipeline`: 2 s windows of 7 voices gave
/// cosine >= 0.61 for the same voice and <= 0.28 for different voices.
pub const SIMILARITY_THRESHOLD: f32 = 0.45;
/// Segments shorter than this keep the last speaker of the same source:
/// short audio gives an unreliable embedding.
pub const MIN_EMBED_MS: u64 = 800;

/// Online clustering of embeddings by cosine similarity to running-mean centroids.
pub struct Clusterer {
    threshold: f32,
    /// Sum of unit-length member embeddings (the direction is the mean's direction).
    centroids: Vec<Vec<f32>>,
}

impl Clusterer {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            centroids: Vec::new(),
        }
    }

    /// Returns the speaker index (0-based) for the embedding and updates the centroid.
    pub fn assign(&mut self, embedding: &[f32]) -> usize {
        let unit = normalize(embedding);
        let best = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cosine(&unit, c)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        match best {
            Some((i, sim)) if sim >= self.threshold => {
                for (c, u) in self.centroids[i].iter_mut().zip(&unit) {
                    *c += u;
                }
                i
            }
            _ => {
                self.centroids.push(unit);
                self.centroids.len() - 1
            }
        }
    }
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

pub(super) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn speaker_id(index: usize) -> String {
    format!("S{}", index + 1)
}

/// Online speaker diarization over one session (all sources share one value,
/// so the same voice on mic and system gets one id).
pub struct Diarizer {
    extractor: SpeakerEmbeddingExtractor,
    clusterer: Clusterer,
    last_by_source: HashMap<SourceKind, String>,
    last_any: Option<String>,
}

impl Diarizer {
    pub fn new(models_dir: &Path) -> anyhow::Result<Self> {
        let model = models::path(models_dir, models::SPEAKER_MODEL_ID).join(models::SPEAKER_FILE);
        if !model.is_file() {
            bail!("화자 구분 모델이 설치되어 있지 않습니다. 설정에서 모델을 내려받으세요.");
        }
        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(model.to_string_lossy().into_owned()),
            num_threads: 2,
            ..Default::default()
        };
        let extractor = SpeakerEmbeddingExtractor::create(&config)
            .context("화자 구분 모델을 불러오지 못했습니다. 모델을 다시 내려받아 보세요.")?;
        Ok(Self {
            extractor,
            clusterer: Clusterer::new(SIMILARITY_THRESHOLD),
            last_by_source: HashMap::new(),
            last_any: None,
        })
    }

    /// Returns the speaker id ("S1", "S2", ...) for the segment.
    pub fn assign(&mut self, segment: &SpeechSegment) -> anyhow::Result<String> {
        let short = segment.end_ms.saturating_sub(segment.start_ms) < MIN_EMBED_MS
            || segment.samples.len() < (SAMPLE_RATE as u64 * MIN_EMBED_MS / 1000) as usize;
        if short && let Some(last) = self.last_by_source.get(&segment.source) {
            return Ok(last.clone());
        }
        let id = match self.embed(&segment.samples) {
            Some(e) => speaker_id(self.clusterer.assign(&e)),
            // No usable embedding (very short audio at the start): reuse the last speaker.
            None => self.last_any.clone().unwrap_or_else(|| speaker_id(0)),
        };
        self.last_by_source.insert(segment.source, id.clone());
        self.last_any = Some(id.clone());
        Ok(id)
    }

    pub(super) fn embed(&self, samples: &[f32]) -> Option<Vec<f32>> {
        let stream = self.extractor.create_stream()?;
        stream.accept_waveform(SAMPLE_RATE as i32, samples);
        stream.input_finished();
        if !self.extractor.is_ready(&stream) {
            return None;
        }
        self.extractor.compute(&stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vector near `axis` in 8 dimensions with a little deterministic noise.
    fn near(axis: usize, noise_seed: usize) -> Vec<f32> {
        (0..8)
            .map(|i| {
                let base = if i == axis { 1.0 } else { 0.0 };
                base + 0.05 * (((i * 7 + noise_seed * 13) % 5) as f32 - 2.0)
            })
            .collect()
    }

    #[test]
    fn same_voice_joins_and_new_voice_starts_a_speaker() {
        let mut c = Clusterer::new(SIMILARITY_THRESHOLD);
        assert_eq!(c.assign(&near(0, 0)), 0);
        assert_eq!(c.assign(&near(3, 1)), 1);
        assert_eq!(c.assign(&near(0, 2)), 0);
        assert_eq!(c.assign(&near(3, 3)), 1);
        assert_eq!(c.assign(&near(5, 4)), 2);
        assert_eq!(c.assign(&near(3, 5)), 1);
    }

    #[test]
    fn centroid_is_a_running_mean() {
        let mut c = Clusterer::new(0.8);
        assert_eq!(c.assign(&[1.0, 0.0]), 0);
        // 45 degrees from speaker 0: cosine 0.707 < 0.8, new speaker.
        assert_eq!(c.assign(&[1.0, 1.0]), 1);
        // Pulls the centroid of speaker 1 toward the y axis.
        assert_eq!(c.assign(&[0.2, 1.0]), 1);
        let y = c.centroids[1].clone();
        assert!(y[1] > y[0]);
        // Scale does not matter.
        assert_eq!(c.assign(&[100.0, 0.0]), 0);
    }

    #[test]
    fn zero_vector_does_not_panic() {
        let mut c = Clusterer::new(0.5);
        assert_eq!(c.assign(&[0.0, 0.0]), 0);
        assert_eq!(c.assign(&[1.0, 0.0]), 1);
    }

    #[test]
    fn speaker_ids_start_at_s1() {
        assert_eq!(speaker_id(0), "S1");
        assert_eq!(speaker_id(11), "S12");
    }
}
