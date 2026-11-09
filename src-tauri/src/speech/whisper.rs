//! Local Whisper on whisper.cpp (whisper-rs), on the device of `gpu::place`.
//! whisper.cpp uses the ggml of llama.cpp (`vendor/whisper-rs-sys`), so its
//! GPU backend (CUDA, Metal) too. Detects the language per segment, or uses
//! the meeting language of the settings.

use super::{Asr, local};
use crate::gpu::{self, GpuFailure, Placement};
use crate::models;
use crate::types::{SAMPLE_RATE, SpeechSegment, Transcript};
use anyhow::bail;
use std::path::Path;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

const MODEL: &str = "음성 인식 모델";

pub struct WhisperAsr {
    /// Keeps the model (context) alive.
    state: WhisperState,
    params: FullParams<'static, 'static>,
    /// Meeting language (ISO 639-1); `None` = detect per segment.
    language: Option<String>,
    label: String,
    /// GPU name when the model is on a GPU.
    gpu: Option<String>,
}

/// whisper.cpp's own string for an ISO 639-1 code ("ko"); "auto" for `None`.
fn whisper_language(language: Option<&str>) -> anyhow::Result<&'static str> {
    let Some(code) = language else {
        return Ok("auto");
    };
    match whisper_rs::get_lang_id(code).and_then(whisper_rs::get_lang_str) {
        Some(lang) => Ok(lang),
        None => bail!("Whisper가 모르는 회의 언어입니다: {code}"),
    }
}

fn params(language: Option<&str>) -> anyhow::Result<FullParams<'static, 'static>> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(2) as i32;
    let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    p.set_n_threads(threads);
    p.set_language(Some(whisper_language(language)?));
    // Each VAD segment (at most 15 s) is one window: no text of the segment
    // before, no timestamps.
    p.set_no_context(true);
    p.set_no_timestamps(true);
    p.set_single_segment(true);
    // As openai-whisper: no "(음악)", "[Music]" and other non-speech tokens.
    p.set_suppress_nst(true);
    p.set_print_progress(false);
    p.set_print_realtime(false);
    p.set_print_timestamps(false);
    p.set_print_special(false);
    Ok(p)
}

impl WhisperAsr {
    /// Loads the model on the device of `placement`. On a GPU a failure stops
    /// the meeting start (D2 B).
    pub fn new(
        models_dir: &Path,
        model_id: &str,
        language: Option<&str>,
        placement: &Placement,
    ) -> anyhow::Result<Self> {
        let info = local::installed_model(models_dir, model_id)?;
        let params = params(language)?;
        let file = models::path(models_dir, model_id).join(models::WHISPER_FILE);
        gpu::llama_backend()?;
        let mut context = WhisperContextParameters::default();
        match &placement.gpu {
            Some((index, _)) => context.use_gpu(true).gpu_device(gpu::gpu_ordinal(*index)),
            None => context.use_gpu(false),
        };
        gpu::clear_llama_errors();
        let loaded =
            WhisperContext::new_with_params(&file, context).and_then(|ctx| ctx.create_state());
        let state = match (loaded, &placement.gpu) {
            (Ok(state), _) => state,
            (Err(e), Some((_, name))) => {
                return Err(gpu::model_load_error(MODEL, name, &gpu::failure_cause(&e)));
            }
            (Err(_), None) => bail!(
                "{} 모델을 불러오지 못했습니다. 모델을 지우고 다시 내려받아 보세요.",
                info.name
            ),
        };
        let mut asr = Self {
            state,
            params,
            language: language.map(str::to_string),
            label: format!("로컬 {}", info.name),
            gpu: placement.gpu.as_ref().map(|(_, name)| name.clone()),
        };
        if let Some(name) = &asr.gpu {
            // The first run on a GPU loads its kernels and is slow: do it before the meeting.
            let name = name.clone();
            asr.run(&vec![0.0; SAMPLE_RATE as usize])
                .map_err(|e| gpu::model_load_error(MODEL, &name, &e.to_string()))?;
        }
        Ok(asr)
    }

    fn run(&mut self, samples: &[f32]) -> anyhow::Result<()> {
        gpu::clear_llama_errors();
        match (self.state.full(self.params.clone(), samples), &self.gpu) {
            (Ok(_), _) => Ok(()),
            (Err(e), Some(_)) => Err(GpuFailure(format!(
                "음성 인식 모델이 GPU에서 오류를 냈습니다: {}.",
                gpu::failure_cause(&e)
            ))
            .into()),
            (Err(e), None) => bail!("음성 인식에 실패했습니다: {e}"),
        }
    }
}

impl Asr for WhisperAsr {
    fn transcribe(&mut self, segment: &SpeechSegment) -> anyhow::Result<Transcript> {
        if segment.samples.is_empty() {
            return Ok(Transcript {
                text: String::new(),
                language: self.language.clone(),
            });
        }
        self.run(&segment.samples)?;
        let mut text = String::new();
        for seg in self.state.as_iter() {
            text.push_str(&seg.to_str_lossy()?);
        }
        let lang = match &self.language {
            Some(code) => code.as_str(),
            None => whisper_rs::get_lang_str(self.state.full_lang_id_from_state()).unwrap_or(""),
        };
        Ok(local::transcript(&text, lang))
    }

    fn label(&self) -> String {
        self.label.clone()
    }
}

#[cfg(test)]
mod tests {
    //! `whisper_on_real_clips` is `#[ignore]`: it downloads a model and needs wav files.
    //! Run: `MORAGO_TEST_WAVS=<dir> cargo test whisper_on_real_clips -- --ignored --nocapture`
    //! Optional: `MORAGO_TEST_MODELS=<dir>` (model cache), `MORAGO_TEST_WHISPER=<id>` (default whisper-small),
    //! `MORAGO_TEST_GPU`, `MORAGO_TEST_BENCH` (see `placement`).
    //! Wav files in `<dir>` (16 kHz mono): en0, sv_ko, sv_ja.
    use super::*;
    use crate::speech::tests::{ensure_model, read_wav};
    use crate::types::{GpuSetting, SourceKind};
    use std::time::Instant;

    #[test]
    fn the_meeting_language_becomes_the_whisper_language() {
        assert_eq!(whisper_language(None).unwrap(), "auto");
        for code in ["ko", "ja", "zh", "en"] {
            assert_eq!(whisper_language(Some(code)).unwrap(), code);
        }
        assert!(whisper_language(Some("xx")).is_err());
        assert!(params(Some("xx")).is_err());
    }

    #[test]
    fn a_missing_model_is_reported_in_korean() {
        let dir = tempfile::tempdir().unwrap();
        let cpu = Placement::cpu("");
        let err = WhisperAsr::new(dir.path(), "whisper-small", None, &cpu)
            .err()
            .unwrap();
        assert!(err.to_string().contains("설치되어 있지 않습니다"), "{err}");
        // A folder with files of the earlier sherpa-onnx format is not a model.
        let old = models::path(dir.path(), "whisper-small");
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join(models::ENCODER_FILE), b"old").unwrap();
        assert!(WhisperAsr::new(dir.path(), "whisper-small", None, &cpu).is_err());
    }

    fn segment(samples: Vec<f32>) -> SpeechSegment {
        SpeechSegment {
            source: SourceKind::System,
            start_ms: 0,
            end_ms: samples.len() as u64 * 1000 / SAMPLE_RATE as u64,
            samples,
        }
    }

    /// The device follows the 자동 setting (the GPU pack from `gpu::app_dir`);
    /// `MORAGO_TEST_GPU=off` runs on the CPU. `MORAGO_TEST_BENCH=<dir>`: also
    /// transcribes every wav file in `<dir>` and prints the RTF of all of them.
    fn placement(models_dir: &Path, id: &str) -> Placement {
        gpu::init(&gpu::app_dir().unwrap());
        let setting = match std::env::var("MORAGO_TEST_GPU").as_deref() {
            Ok("off") => GpuSetting::Off,
            _ => GpuSetting::Auto,
        };
        let file = models::path(models_dir, id).join(models::WHISPER_FILE);
        let need = gpu::whisper_memory(std::fs::metadata(file).unwrap().len(), false);
        let p = gpu::place_beside(&setting, need, 0).unwrap();
        println!("device: {:?}", p.status);
        p
    }

    fn free_mib(p: &Placement) -> Option<u64> {
        p.gpu
            .as_ref()
            .and_then(|(i, _)| gpu::free_memory(*i))
            .map(|b| b >> 20)
    }

    #[test]
    #[ignore = "downloads a model and needs wav files (MORAGO_TEST_WAVS)"]
    fn whisper_on_real_clips() {
        let wavs = std::path::PathBuf::from(
            std::env::var("MORAGO_TEST_WAVS").expect("set MORAGO_TEST_WAVS to the wav folder"),
        );
        let models_dir = std::env::var("MORAGO_TEST_MODELS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("morago-test-models"));
        let id = std::env::var("MORAGO_TEST_WHISPER").unwrap_or_else(|_| "whisper-small".into());
        ensure_model(&models_dir, &id);

        let place = placement(&models_dir, &id);
        let before = free_mib(&place);
        let started = Instant::now();
        let mut asr = WhisperAsr::new(&models_dir, &id, None, &place).unwrap();
        println!(
            "{}: load and warm-up {:?}; free GPU memory {before:?} -> {:?} MiB",
            asr.label(),
            started.elapsed(),
            free_mib(&place)
        );
        let mut first = None;
        for (name, lang) in [("en0", "en"), ("sv_ko", "ko"), ("sv_ja", "ja")] {
            let seg = segment(read_wav(&wavs, name));
            let secs = seg.samples.len() as f64 / SAMPLE_RATE as f64;
            let started = Instant::now();
            let t = asr.transcribe(&seg).unwrap();
            let rtf = started.elapsed().as_secs_f64() / secs;
            println!(
                "{name} ({secs:.1} s, RTF {rtf:.2}) lang={:?} {}",
                t.language, t.text
            );
            assert!(!t.text.trim().is_empty(), "{name}: no text");
            assert_eq!(t.language.as_deref(), Some(lang), "{name}");
            first.get_or_insert((seg, t.text));
        }
        // The first result after the model load is not a bad one: the same clip gives the same text later.
        let (seg, text) = first.unwrap();
        assert_eq!(asr.transcribe(&seg).unwrap().text, text);

        if let Ok(bench) = std::env::var("MORAGO_TEST_BENCH") {
            let mut files: Vec<_> = std::fs::read_dir(&bench)
                .unwrap()
                .map(|e| e.unwrap().path())
                .filter(|p| p.extension().is_some_and(|x| x == "wav"))
                .collect();
            files.sort();
            let (mut audio, mut took) = (0.0, 0.0);
            for file in &files {
                let name = file.file_stem().unwrap().to_str().unwrap();
                let seg = segment(read_wav(file.parent().unwrap(), name));
                let started = Instant::now();
                let t = asr.transcribe(&seg).unwrap();
                took += started.elapsed().as_secs_f64();
                audio += seg.samples.len() as f64 / SAMPLE_RATE as f64;
                println!("{name} {:?} {}", t.language, t.text);
            }
            println!(
                "bench {}: {} files, {audio:.1} s audio, {took:.2} s, RTF {:.3}; free GPU memory {:?} MiB",
                place.status.detail,
                files.len(),
                took / audio,
                free_mib(&place)
            );
        }
        drop(asr);

        // Meeting language: Whisper decodes in it and the result carries it.
        let place = placement(&models_dir, &id);
        let mut asr = WhisperAsr::new(&models_dir, &id, Some("ja"), &place).unwrap();
        let t = asr.transcribe(&segment(read_wav(&wavs, "sv_ja"))).unwrap();
        println!("sv_ja, meeting language ja: {}", t.text);
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
