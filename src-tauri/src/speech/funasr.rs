//! Local Fun-ASR-MLT-Nano on transcribe.cpp (GGUF), on the device of `gpu::place`.
//! transcribe.cpp uses the ggml of llama.cpp (`vendor/transcribe-cpp-sys`), so
//! its GPU backend (CUDA, Metal) too. Uses the meeting language of the
//! settings, or lets the model choose. The model gives no language: the
//! script of the text decides (`local::transcript`).

use super::{Asr, local};
use crate::gpu::{self, GpuFailure, Placement};
use crate::models;
use crate::types::{SAMPLE_RATE, SpeechSegment, Transcript};
use anyhow::{anyhow, bail};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;
use transcribe_cpp::{
    Backend, CancelToken, Itn, Model, ModelOptions, RunOptions, Session, SessionOptions,
};

const MODEL: &str = "음성 인식 모델";

pub struct FunAsr {
    session: Session,
    /// Keeps the model of `LOADED` alive.
    _model: Arc<Model>,
    turns: Arc<Turns>,
    /// The live partial ASR (`Asr::set_partial`).
    partial: bool,
    options: RunOptions,
    /// Meeting language (ISO 639-1); `None` = the model chooses.
    language: Option<String>,
    label: String,
    /// GPU name when the model is on a GPU.
    gpu: Option<String>,
}

/// The loaded model. A second `FunAsr` on the same device (the partial ASR of
/// the engine) opens a session on it, so the weights are in memory once.
/// transcribe.cpp runs one compute per model at a time: the runs of the
/// sessions take `Turns`.
struct Loaded {
    file: PathBuf,
    /// ggml device index; `None` = CPU.
    device: Option<usize>,
    model: Weak<Model>,
    turns: Arc<Turns>,
}

/// The final runs go first. A final run stops the partial run in flight
/// (transcribe.cpp checks the cancel token after each decode step, not in the
/// encoder), and no partial run starts until the final runs end.
#[derive(Default)]
struct Turns {
    /// Final runs that wait or run, and when the first of them came.
    finals: Mutex<(usize, Option<Instant>)>,
    /// The cancel token of the partial sessions.
    cancel: CancelToken,
}

impl Turns {
    fn final_starts(&self) {
        let mut f = self.finals.lock().expect("turns lock");
        if f.0 == 0 {
            f.1 = Some(Instant::now());
        }
        f.0 += 1;
        self.cancel.cancel();
    }

    fn final_ends(&self) {
        let mut f = self.finals.lock().expect("turns lock");
        f.0 -= 1;
        if f.0 == 0 {
            f.1 = None;
            self.cancel.reset();
        }
    }

    /// When the waiting final came; `None`: no final waits.
    fn final_waits(&self) -> Option<Instant> {
        self.finals.lock().expect("turns lock").1
    }
}

static LOADED: Mutex<Option<Loaded>> = Mutex::new(None);

/// transcribe.cpp takes ISO 639-1 codes ("ko") and puts them into the prompt
/// of the model. ITN on: digits, casing and punctuation, as the model card.
fn run_options(language: Option<&str>) -> RunOptions {
    RunOptions {
        language: language.map(str::to_string),
        itn: Itn::On,
        ..RunOptions::default()
    }
}

/// transcribe.cpp wants the backend of an explicit device.
fn backend(kind: &str) -> Backend {
    match kind {
        "cuda" => Backend::Cuda,
        "metal" => Backend::Metal,
        "vulkan" => Backend::Vulkan,
        "rocm" => Backend::Rocm,
        _ => Backend::Auto,
    }
}

fn load(file: &Path, placement: &Placement) -> anyhow::Result<(Arc<Model>, Arc<Turns>)> {
    let device = placement.gpu.as_ref().map(|(index, _)| *index);
    let mut loaded = LOADED.lock().expect("loaded lock");
    if let Some(l) = loaded.as_ref()
        && l.file == file
        && l.device == device
        && let Some(model) = l.model.upgrade()
    {
        return Ok((model, l.turns.clone()));
    }
    let options = match device {
        None => ModelOptions {
            backend: Backend::Cpu,
            ..ModelOptions::default()
        },
        Some(index) => {
            let dev = transcribe_cpp::devices()
                .into_iter()
                .find(|d| d.index == Some(index))
                .ok_or_else(|| anyhow!("transcribe.cpp가 이 GPU를 찾지 못했습니다"))?;
            ModelOptions {
                backend: backend(&dev.kind),
                device: Some(dev),
                ..ModelOptions::default()
            }
        }
    };
    let model = Arc::new(Model::load_with(file, &options)?);
    let turns = Arc::new(Turns::default());
    *loaded = Some(Loaded {
        file: file.to_path_buf(),
        device,
        model: Arc::downgrade(&model),
        turns: turns.clone(),
    });
    Ok((model, turns))
}

impl FunAsr {
    /// Loads the model on the device of `placement`, or opens a session on
    /// the model that is loaded there. On a GPU a failure stops the meeting
    /// start (D2 B).
    pub fn new(
        models_dir: &Path,
        model_id: &str,
        language: Option<&str>,
        placement: &Placement,
    ) -> anyhow::Result<Self> {
        let info = local::installed_model(models_dir, model_id)?;
        let file = models::path(models_dir, model_id).join(models::FUN_ASR_FILE);
        gpu::llama_backend()?;
        gpu::clear_llama_errors();
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().min(4))
            .unwrap_or(2) as i32;
        let opened = load(&file, placement).and_then(|(model, turns)| {
            let session = model.session_with(&SessionOptions {
                n_threads: threads,
                ..SessionOptions::default()
            })?;
            Ok((model, turns, session))
        });
        let (model, turns, session) = match (opened, &placement.gpu) {
            (Ok(opened), _) => opened,
            (Err(e), Some((_, name))) => {
                return Err(gpu::model_load_error(MODEL, name, &gpu::failure_cause(&e)));
            }
            (Err(_), None) => bail!(
                "{} 모델을 불러오지 못했습니다. 모델을 지우고 다시 내려받아 보세요.",
                info.name
            ),
        };
        if let Some(code) = language
            && !model.capabilities()?.languages.iter().any(|l| l == code)
        {
            bail!("Fun-ASR가 모르는 회의 언어입니다: {code}");
        }
        let mut asr = Self {
            session,
            _model: model,
            turns,
            partial: false,
            options: run_options(language),
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

    fn run(&mut self, samples: &[f32]) -> anyhow::Result<transcribe_cpp::Transcript> {
        gpu::clear_llama_errors();
        let result = if self.partial {
            if self.turns.final_waits().is_some() {
                bail!("최종 인식이 기다려서 부분 인식을 건너뜁니다");
            }
            self.session.run(samples, &self.options)
        } else {
            self.turns.final_starts();
            let result = self.session.run(samples, &self.options);
            self.turns.final_ends();
            result
        };
        let e = match result {
            Ok(t) => return Ok(t),
            Err(e) => e,
        };
        // A final run stopped this partial run: no text, no error to report.
        if self.partial && self.session.was_aborted() {
            if let Some(came) = self.turns.final_waits() {
                log::info!(
                    "partial run stopped {} ms after a final came",
                    came.elapsed().as_millis()
                );
            }
            bail!("최종 인식 때문에 부분 인식을 멈췄습니다");
        }
        // The decode stopped at its token budget or at a repetition: the text before it.
        if let Some(t) = e.partial() {
            return Ok(t.clone());
        }
        match &self.gpu {
            Some(_) => Err(GpuFailure(format!(
                "음성 인식 모델이 GPU에서 오류를 냈습니다: {}.",
                gpu::failure_cause(&e)
            ))
            .into()),
            None => bail!("음성 인식에 실패했습니다: {e}"),
        }
    }
}

impl Asr for FunAsr {
    fn transcribe(&mut self, segment: &SpeechSegment) -> anyhow::Result<Transcript> {
        if segment.samples.is_empty() {
            return Ok(Transcript {
                text: String::new(),
                language: self.language.clone(),
            });
        }
        let t = self.run(&segment.samples)?;
        let lang = self.language.as_deref().or(t.language.as_deref());
        Ok(local::transcript(&t.text, lang.unwrap_or("")))
    }

    fn label(&self) -> String {
        self.label.clone()
    }

    fn set_partial(&mut self) {
        self.session.set_cancel_token(&self.turns.cancel);
        self.partial = true;
    }
}

#[cfg(test)]
mod tests {
    //! `funasr_on_real_clips` is `#[ignore]`: it downloads the model and needs wav files.
    //! Run: `MORAGO_TEST_WAVS=<dir> cargo test funasr_on_real_clips -- --ignored --nocapture`
    //! Optional: `MORAGO_TEST_MODELS=<dir>` (model cache), `MORAGO_TEST_GPU`,
    //! `MORAGO_TEST_BENCH` (see `placement` and `bench`).
    //! Wav files in `<dir>` (16 kHz mono): en0, sv_ko, sv_ja.
    use super::*;
    use crate::settings::AsrConfig;
    use crate::speech::build_asr;
    use crate::speech::tests::{ensure_model, read_wav};
    use crate::types::{GpuSetting, SourceKind};
    use std::time::Instant;

    const ID: &str = models::FUN_ASR_MLT_ID;

    #[test]
    fn the_meeting_language_goes_to_the_model_as_its_code() {
        for code in ["ko", "ja", "zh", "en"] {
            let o = run_options(Some(code));
            assert_eq!(o.language.as_deref(), Some(code));
            assert_eq!(o.itn, Itn::On);
        }
        assert_eq!(run_options(None).language, None);
        assert_eq!(backend("cuda"), Backend::Cuda);
        assert_eq!(backend("metal"), Backend::Metal);
    }

    #[test]
    fn a_final_run_stops_partial_runs_until_all_finals_end() {
        let t = Turns::default();
        assert!(t.final_waits().is_none() && !t.cancel.is_cancelled());
        t.final_starts();
        let came = t.final_waits().unwrap();
        assert!(t.cancel.is_cancelled());
        t.final_starts();
        t.final_ends();
        assert_eq!(t.final_waits(), Some(came));
        assert!(t.cancel.is_cancelled());
        t.final_ends();
        assert!(t.final_waits().is_none() && !t.cancel.is_cancelled());
    }

    /// Settings that name Fun-ASR with only the earlier sherpa-onnx files on
    /// disk: "download it", never a load error.
    #[test]
    fn old_onnx_files_are_not_installed_and_ask_for_a_download() {
        let dir = tempfile::tempdir().unwrap();
        let config = AsrConfig::Local {
            model_id: ID.into(),
        };
        let cpu = Placement::cpu("");
        for old in [false, true] {
            if old {
                let folder = models::path(dir.path(), ID);
                std::fs::create_dir_all(&folder).unwrap();
                for f in [
                    "encoder_adaptor.int8.onnx",
                    "llm.int8.onnx",
                    "tokenizer.json",
                ] {
                    std::fs::write(folder.join(f), b"old").unwrap();
                }
            }
            let err = build_asr(&config, None, dir.path(), &cpu).err().unwrap();
            let text = err.to_string();
            assert!(
                text.contains("설치되어 있지 않습니다. 설정에서 모델을 내려받으세요."),
                "{text}"
            );
            assert!(!text.contains("불러오지 못했습니다"), "{text}");
        }
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
    /// `MORAGO_TEST_GPU=off` runs on the CPU. The memory check counts the
    /// final and the partial session, as the engine does.
    fn placement(models_dir: &Path) -> Placement {
        gpu::init(&gpu::app_dir().unwrap());
        let setting = match std::env::var("MORAGO_TEST_GPU").as_deref() {
            Ok("off") => GpuSetting::Off,
            _ => GpuSetting::Auto,
        };
        let file = models::path(models_dir, ID).join(models::FUN_ASR_FILE);
        let need = gpu::funasr_memory(std::fs::metadata(file).unwrap().len(), false);
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

    /// `MORAGO_TEST_BENCH=<dir>` with `<lang>/<id>.wav`: transcribes each
    /// file in the language of its folder (`MORAGO_TEST_BENCH_AUTO=1`: the
    /// model chooses) and prints one JSON line per file and the RTF. The
    /// `partial` session runs each file too (not timed), as in a meeting.
    fn bench(models_dir: &Path, place: &Placement, partial: &mut FunAsr) {
        let Ok(dir) = std::env::var("MORAGO_TEST_BENCH") else {
            return;
        };
        let auto = std::env::var("MORAGO_TEST_BENCH_AUTO").is_ok_and(|v| v == "1");
        let mut langs: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.is_dir())
            .collect();
        langs.sort();
        let (mut audio, mut took) = (0.0, 0.0);
        for folder in langs {
            let lang = folder.file_name().unwrap().to_str().unwrap().to_string();
            let mut asr =
                FunAsr::new(models_dir, ID, (!auto).then_some(lang.as_str()), place).unwrap();
            let mut files: Vec<_> = std::fs::read_dir(&folder)
                .unwrap()
                .map(|e| e.unwrap().path())
                .filter(|p| p.extension().is_some_and(|x| x == "wav"))
                .collect();
            files.sort();
            for file in &files {
                let id = file.file_stem().unwrap().to_str().unwrap();
                let seg = segment(read_wav(&folder, id));
                let sec = seg.samples.len() as f64 / SAMPLE_RATE as f64;
                let started = Instant::now();
                let t = asr.transcribe(&seg).unwrap();
                let dec_s = started.elapsed().as_secs_f64();
                took += dec_s;
                audio += sec;
                partial.transcribe(&seg).unwrap();
                println!(
                    "BENCH {}",
                    serde_json::json!({
                        "lang": lang, "id": id, "sec": sec, "dec_s": dec_s,
                        "detected": t.language, "hyp": t.text,
                    })
                );
            }
        }
        println!(
            "bench {}: {audio:.1} s audio, {took:.2} s, RTF {:.3}",
            place.status.detail,
            took / audio,
        );
    }

    #[test]
    #[ignore = "downloads a model and needs wav files (MORAGO_TEST_WAVS)"]
    fn funasr_on_real_clips() {
        let wavs = std::path::PathBuf::from(
            std::env::var("MORAGO_TEST_WAVS").expect("set MORAGO_TEST_WAVS to the wav folder"),
        );
        let models_dir = std::env::var("MORAGO_TEST_MODELS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("morago-test-models"));
        ensure_model(&models_dir, ID);

        let place = placement(&models_dir);
        let before = free_mib(&place);
        let started = Instant::now();
        let mut asr = FunAsr::new(&models_dir, ID, None, &place).unwrap();
        let one = free_mib(&place);
        println!(
            "{}: load and warm-up {:?}; free GPU memory {before:?} -> {one:?} MiB",
            asr.label(),
            started.elapsed(),
        );
        // The partial ASR of the engine: a second session on the same model.
        let started = Instant::now();
        let mut partial = FunAsr::new(&models_dir, ID, None, &place).unwrap();
        assert!(Arc::ptr_eq(&asr._model, &partial._model));
        println!(
            "second session {:?}; free GPU memory -> {:?} MiB",
            started.elapsed(),
            free_mib(&place)
        );
        let mut first = None;
        for (name, lang) in [("en0", None), ("sv_ko", Some("ko")), ("sv_ja", Some("ja"))] {
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
            // Latin text has no script language.
            assert_eq!(t.language.as_deref(), lang, "{name}");
            first.get_or_insert((seg, t.text));
        }
        // The first result after the model load is not a bad one, and both
        // sessions give the same text.
        let (seg, text) = first.unwrap();
        assert_eq!(asr.transcribe(&seg).unwrap().text, text);
        assert_eq!(partial.transcribe(&seg).unwrap().text, text);
        println!("free GPU memory after the runs: {:?} MiB", free_mib(&place));
        drop(asr);
        let gpu = place.gpu.as_ref().map(|(i, _)| *i);
        let ((), lowest) = gpu::lowest_free_while(gpu, || bench(&models_dir, &place, &mut partial));
        println!("lowest free GPU memory in the bench: {lowest:?} MiB");
        drop(partial);

        // Meeting language: the result carries it.
        let mut asr = FunAsr::new(&models_dir, ID, Some("en"), &place).unwrap();
        let t = asr.transcribe(&segment(read_wav(&wavs, "en0"))).unwrap();
        println!("en0, meeting language en: {}", t.text);
        assert_eq!(t.language.as_deref(), Some("en"));
        assert!(FunAsr::new(&models_dir, ID, Some("xx"), &place).is_err());
    }
}
