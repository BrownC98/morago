//! Speech pipeline parts: VAD, speech recognition (ASR) and speaker diarization.
//! Owner: speech worker. Contract (do not change signatures without the coordinator).

mod custom;
mod diarize;
mod local;
mod vad;
mod wav;

use crate::secrets::{self, Role};
use crate::settings::AsrConfig;
use crate::types::{SpeechSegment, Transcript};
use std::path::Path;

// Used by the engine once it is wired.
#[allow(unused_imports)]
pub use diarize::Diarizer;
#[allow(unused_imports)]
pub use vad::Vad;

/// Speech recognition for one segment. Runs on a worker thread (blocking).
pub trait Asr: Send {
    fn transcribe(&mut self, segment: &SpeechSegment) -> anyhow::Result<Transcript>;
    /// Label for the UI and session meta, for example "로컬 SenseVoice Small".
    fn label(&self) -> String;
}

/// Builds the ASR from settings. Local models are in `models_dir`.
/// `language` (ISO 639-1, `Settings::meeting_language`): `None` = detect per
/// segment; else the ASR uses it and returns it as `Transcript::language`.
/// A custom endpoint reads its key with `secrets::get_key(Role::Asr, base_url)`.
pub fn build_asr(
    config: &AsrConfig,
    language: Option<&str>,
    models_dir: &Path,
) -> anyhow::Result<Box<dyn Asr>> {
    match config {
        AsrConfig::Local { model_id } => Ok(Box::new(local::LocalAsr::new(
            models_dir, model_id, language,
        )?)),
        AsrConfig::Custom { endpoint } => {
            let key = secrets::get_key(Role::Asr, &endpoint.base_url)?;
            Ok(Box::new(custom::CustomAsr::new(endpoint, key, language)?))
        }
    }
}

#[cfg(test)]
mod tests {
    //! `real_models_pipeline` is `#[ignore]`: it downloads models and needs wav files.
    //! Run: `MORAGO_TEST_WAVS=<dir> cargo test real_models_pipeline -- --ignored --nocapture`
    //! Optional: `MORAGO_TEST_MODELS=<dir>` (model cache), `MORAGO_TEST_ASR=<id>` (default sense-voice-small).
    //! Wav files in `<dir>` (16 kHz mono): en0, en1, sv_ko, sv_ja, sv_zh, sv_en, sv_yue.
    use super::*;
    use crate::models;
    use crate::types::{AudioChunk, SAMPLE_RATE, SourceKind};
    use std::collections::BTreeSet;
    use std::sync::atomic::AtomicBool;

    fn read_wav(dir: &Path, name: &str) -> Vec<f32> {
        let path = dir.join(format!("{name}.wav"));
        sherpa_onnx::Wave::read(path.to_str().unwrap())
            .unwrap_or_else(|| panic!("cannot read {}", path.display()))
            .samples()
            .to_vec()
    }

    fn ensure_model(models_dir: &Path, id: &str) {
        if models::is_installed(models_dir, id) {
            return;
        }
        let last = std::sync::Mutex::new(0u64);
        models::download(
            models_dir,
            id,
            &|p| {
                let pct = p.done_bytes * 100 / p.total_bytes.unwrap_or(1).max(1);
                let mut l = last.lock().unwrap();
                if pct / 10 != *l / 10 {
                    println!("download {id}: {pct}%");
                }
                *l = pct;
            },
            &AtomicBool::new(false),
        )
        .unwrap();
    }

    /// Runs VAD over `samples` (100 ms chunks) and returns the segments.
    fn segments(vad: &mut Vad, samples: &[f32]) -> Vec<SpeechSegment> {
        let mut out = Vec::new();
        let mut start = 0usize;
        for chunk in samples.chunks(1600) {
            out.extend(vad.push(&AudioChunk {
                source: SourceKind::System,
                start_ms: start as u64 * 1000 / SAMPLE_RATE as u64,
                samples: chunk.to_vec(),
            }));
            start += chunk.len();
        }
        out.extend(vad.flush());
        out
    }

    #[test]
    #[ignore = "downloads models and needs wav files (MORAGO_TEST_WAVS)"]
    fn real_models_pipeline() {
        let wavs = std::path::PathBuf::from(
            std::env::var("MORAGO_TEST_WAVS").expect("set MORAGO_TEST_WAVS to the wav folder"),
        );
        let models_dir = std::env::var("MORAGO_TEST_MODELS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("morago-test-models"));
        let asr_id =
            std::env::var("MORAGO_TEST_ASR").unwrap_or_else(|_| models::SENSE_VOICE_ID.into());
        ensure_model(&models_dir, &asr_id);
        let bundled = Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/models");
        let mut asr = build_asr(
            &AsrConfig::Local {
                model_id: asr_id.clone(),
            },
            None,
            &models_dir,
        )
        .unwrap();
        println!("ASR: {}", asr.label());
        let mut vad = Vad::new(&bundled).unwrap();
        let mut diarizer = Diarizer::new(&bundled).unwrap();

        // Several languages and voices in one stream, 1.2 s of silence between them.
        let gap = vec![0.0f32; SAMPLE_RATE as usize * 6 / 5];
        let order = ["en0", "sv_ko", "sv_ja", "sv_zh", "en1"];
        let mut stream = gap.clone();
        for name in order {
            stream.extend(read_wav(&wavs, name));
            stream.extend(&gap);
        }
        let segs = segments(&mut vad, &stream);
        println!("--- mixed stream: {} segments", segs.len());
        let mut langs = Vec::new();
        let mut speakers = BTreeSet::new();
        let started = std::time::Instant::now();
        for seg in &segs {
            let t = asr.transcribe(seg).unwrap();
            let who = diarizer.assign(seg).unwrap();
            println!(
                "[{:>6}-{:>6} ms] {} lang={:?} {}",
                seg.start_ms, seg.end_ms, who, t.language, t.text
            );
            assert!(!t.text.trim().is_empty(), "no text at {} ms", seg.start_ms);
            langs.push(t.language.clone().unwrap_or_default());
            speakers.insert(who);
        }
        println!("ASR+diarize time: {:?}", started.elapsed());
        assert!(segs.len() >= order.len(), "VAD found too few segments");
        // Qwen3-ASR and Fun-ASR give no language; Latin and Han-only text then stay unknown.
        let want_langs: &[&str] =
            if [models::QWEN3_ASR_ID, models::FUN_ASR_MLT_ID].contains(&asr_id.as_str()) {
                &["ko", "ja"]
            } else {
                &["en", "ko", "ja", "zh"]
            };
        for &want in want_langs {
            assert!(
                langs.iter().any(|l| l == want),
                "language {want} not detected: {langs:?}"
            );
        }
        assert!(
            speakers.len() >= 4,
            "expected several speakers, got {speakers:?}"
        );

        // Same voice vs different voices: two 2 s windows from each clip.
        let diarizer = Diarizer::new(&bundled).unwrap();
        let clips = ["en0", "en1", "sv_ko", "sv_ja", "sv_zh", "sv_en", "sv_yue"];
        let mut embeddings = Vec::new();
        for (clip, name) in clips.iter().enumerate() {
            let samples = read_wav(&wavs, name);
            for start_s in [0.3f32, 2.5] {
                let from = (start_s * SAMPLE_RATE as f32) as usize;
                let window = &samples[from..from + 2 * SAMPLE_RATE as usize];
                embeddings.push((clip, diarizer.embed(window).expect("embedding")));
            }
        }
        let (mut same_min, mut diff_max) = (f32::MAX, f32::MIN);
        for (i, (ci, ei)) in embeddings.iter().enumerate() {
            for (cj, ej) in &embeddings[i + 1..] {
                let sim = diarize::cosine(ei, ej);
                if ci == cj {
                    same_min = same_min.min(sim);
                } else {
                    diff_max = diff_max.max(sim);
                }
            }
        }
        println!("cosine: same voice min {same_min:.3}, different voices max {diff_max:.3}");
        println!("threshold {}", diarize::SIMILARITY_THRESHOLD);
        assert!(diff_max < diarize::SIMILARITY_THRESHOLD);
        assert!(same_min >= diarize::SIMILARITY_THRESHOLD);
    }
}
