//! Local translation with an embedded llama.cpp model.

use super::{History, Translator, clean_output, system_prompt};
use crate::gpu::{self, GpuFailure, Placement};
use anyhow::anyhow;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::Path;

/// GPU memory that one translation context needs at most (KV cache and
/// compute buffers of a short prompt), with a margin.
const CONTEXT_MEMORY: u64 = 512 * 1024 * 1024;

pub(crate) struct LocalTranslator {
    model: LlamaModel,
    label: String,
    threads: i32,
    /// ggml device index and name when the model is on a GPU.
    gpu: Option<(usize, String)>,
    history: History,
}

impl LocalTranslator {
    /// Loads `<models_dir>/<model_id>/model.gguf` once, on the device of `placement`.
    pub(crate) fn load(
        models_dir: &Path,
        model_id: &str,
        placement: &Placement,
    ) -> anyhow::Result<Self> {
        let path = models_dir.join(model_id).join("model.gguf");
        if !path.is_file() {
            return Err(anyhow!(
                "번역 모델이 설치되지 않았습니다: {model_id}. 설정에서 모델을 내려받으세요."
            ));
        }
        let backend = gpu::llama_backend()?;
        // No device: llama.cpp uses no GPU at all, also not for big batches.
        let devices: Vec<usize> = placement.gpu.iter().map(|(i, _)| *i).collect();
        let params = LlamaModelParams::default()
            .with_n_gpu_layers(if devices.is_empty() { 0 } else { 999 })
            .with_devices(&devices)?;
        gpu::clear_llama_errors();
        let model = LlamaModel::load_from_file(backend, &path, &params).map_err(|e| {
            match &placement.gpu {
                Some((_, name)) => gpu::load_error(name, &gpu::failure_cause(&e)),
                None => anyhow!("번역 모델을 읽지 못했습니다 ({model_id}): {e}"),
            }
        })?;
        // About the physical cores: more threads than that slow ggml down.
        let threads =
            (std::thread::available_parallelism().map_or(4, |n| n.get()) / 2).clamp(2, 8) as i32;
        let translator = Self {
            model,
            label: format!("로컬 {}", crate::models::display_name(model_id)),
            threads,
            gpu: placement.gpu.clone(),
            history: History::default(),
        };
        if let Some((_, name)) = &placement.gpu {
            // The first run on a GPU loads its kernels and is slow: do it before the meeting.
            translator
                .generate("Translate into Korean.", "Hello.", &|| false)
                .map_err(|e| gpu::load_error(name, &e.to_string()))?;
        }
        Ok(translator)
    }

    /// On a GPU a llama.cpp error is a `GpuFailure`: the engine stops the session.
    fn llama_error<E>(&self, e: E) -> anyhow::Error
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        match &self.gpu {
            Some(_) => GpuFailure(format!(
                "번역 모델이 GPU에서 오류를 냈습니다: {}.",
                gpu::failure_cause(&e)
            ))
            .into(),
            None => e.into(),
        }
    }

    /// Stops with an error before the next token when `stop` returns true.
    fn generate(
        &self,
        system: &str,
        text: &str,
        stop: &dyn Fn() -> bool,
    ) -> anyhow::Result<String> {
        let backend = gpu::llama_backend()?;
        let prompt = gemma4_prompt(system, text);
        let vocab = self.model.vocab();
        let prompt_tokens = vocab.tokenize(prompt.as_bytes(), true, true);
        let text_tokens = vocab.tokenize(text.as_bytes(), false, false).len();
        // Output is at most about 3x the input tokens.
        let max_new = text_tokens * 3 + 32;
        let n_ctx = (prompt_tokens.len() + max_new + 8).min(self.model.n_ctx_train() as usize);
        if prompt_tokens.len() + 1 >= n_ctx {
            return Err(anyhow!("문장이 너무 길어 번역하지 못했습니다"));
        }
        if let Some((index, name)) = &self.gpu {
            // A failed CUDA allocation ends the process: check before the context.
            let free = gpu::free_memory(*index).unwrap_or(0);
            if free < CONTEXT_MEMORY {
                return Err(GpuFailure(format!("{name}의 GPU 메모리가 모자랍니다.")).into());
            }
        }
        let n_ctx = u32::try_from(n_ctx)?;
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx)
            .with_n_threads(self.threads)
            .with_n_threads_batch(self.threads);
        gpu::clear_llama_errors();
        let mut ctx = self
            .model
            .new_context(backend, params)
            .map_err(|e| self.llama_error(e))?;

        let mut batch = LlamaBatch::new(prompt_tokens.len().max(1), 1);
        batch.add_sequence(&prompt_tokens, 0, false)?;
        ctx.decode(&mut batch).map_err(|e| self.llama_error(e))?;

        let mut sampler = LlamaSampler::greedy();
        let mut bytes = Vec::new();
        for pos in (i32::try_from(prompt_tokens.len())?..).take(max_new) {
            if stop() {
                return Err(anyhow!("stopped"));
            }
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            if vocab.is_eog(token) || pos >= i32::try_from(n_ctx)? {
                break;
            }
            bytes.extend(vocab.token_to_piece(token, false, None));
            batch.clear();
            batch.add(token, pos, &[0], true)?;
            ctx.decode(&mut batch).map_err(|e| self.llama_error(e))?;
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn run(
        &self,
        text: &str,
        source_language: Option<&str>,
        target_language: &str,
        stop: &dyn Fn() -> bool,
    ) -> anyhow::Result<String> {
        let system = system_prompt(source_language, target_language, &self.history);
        let translation = clean_output(&self.generate(&system, text, stop)?, text);
        if translation.is_empty() {
            return Err(anyhow!("번역 모델이 빈 번역문을 만들었습니다"));
        }
        Ok(translation)
    }
}

impl Translator for LocalTranslator {
    fn translate(
        &mut self,
        text: &str,
        source_language: Option<&str>,
        target_language: &str,
    ) -> anyhow::Result<String> {
        self.history.set_target(target_language);
        let translation = self.run(text, source_language, target_language, &|| false)?;
        self.history.push(text);
        Ok(translation)
    }

    /// Uses the history as context but does not change it.
    fn translate_partial(
        &mut self,
        text: &str,
        source_language: Option<&str>,
        target_language: &str,
        stop: &dyn Fn() -> bool,
    ) -> anyhow::Result<String> {
        self.run(text, source_language, target_language, stop)
    }

    fn label(&self) -> String {
        self.label.clone()
    }
}

/// Gemma 4 chat format, thinking off (from the model's chat_template.jinja,
/// 2026-07-09). llama.cpp's legacy template function does not know Gemma 4.
/// No `<bos>`: tokenizing with `add_bos` adds it.
fn gemma4_prompt(system: &str, user: &str) -> String {
    format!(
        "<|turn>system\n{}<turn|>\n<|turn>user\n{}<turn|>\n<|turn>model\n",
        system.trim(),
        user.trim()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GpuSetting;
    use std::time::Instant;

    #[test]
    fn gemma4_prompt_format() {
        assert_eq!(
            gemma4_prompt(" sys ", "hi\n"),
            "<|turn>system\nsys<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn missing_model_gives_korean_error() {
        let dir = std::env::temp_dir().join("morago-no-such-models");
        let err = LocalTranslator::load(&dir, "gemma-4-e2b-it-q4", &Placement::cpu(""))
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("설치되지 않았습니다"), "{err}");
    }

    const MODEL_ID: &str = "gemma-4-e2b-it-q4";

    /// Downloads the model into `MORAGO_TEST_MODELS_DIR` (default: a temp dir).
    fn ensure_model() -> std::path::PathBuf {
        let dir = std::env::var_os("MORAGO_TEST_MODELS_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("morago-translate-test"));
        if !crate::models::is_installed(&dir, MODEL_ID) {
            let cancel = std::sync::atomic::AtomicBool::new(false);
            crate::models::download(&dir, MODEL_ID, &|_| {}, &cancel).unwrap();
        }
        dir
    }

    /// Run: `LLAMA_LIB_PROFILE=Release cargo test local_model -- --ignored --nocapture`
    /// (a debug build of llama.cpp is about 10x slower). `MORAGO_TEST_GPU=off`
    /// runs on the CPU; else the device follows the 자동 setting.
    /// The GPU pack comes from the app data folder (`gpu::app_dir`), as in the app.
    #[test]
    #[ignore = "downloads a 3.3 GB model and runs it"]
    fn local_model_translates() {
        gpu::init(&gpu::app_dir().unwrap());
        let dir = ensure_model();
        let setting = match std::env::var("MORAGO_TEST_GPU").as_deref() {
            Ok("off") => GpuSetting::Off,
            _ => GpuSetting::Auto,
        };
        let size = std::fs::metadata(dir.join(MODEL_ID).join("model.gguf"))
            .unwrap()
            .len();
        let placement = gpu::place(&setting, size).unwrap();
        println!("device: {:?}", placement.status);
        let model_started = Instant::now();
        let mut t = LocalTranslator::load(&dir, MODEL_ID, &placement).unwrap();
        println!(
            "label: {}; load: {:.1?}",
            t.label(),
            model_started.elapsed()
        );
        let cases = [
            (
                "Let's move the launch to the second week of December.",
                "en",
                "ko",
            ),
            (
                "Let's move the launch to the second week of December.",
                "en",
                "ja",
            ),
            ("来週の月曜日までに資料を送ります。", "ja", "en"),
            ("Can you send me the slides?", "en", "ko"),
        ];
        for (text, from, to) in cases {
            let started = Instant::now();
            let out = t.translate(text, Some(from), to).unwrap();
            let took = started.elapsed();
            let tokens = t.model.vocab().tokenize(out.as_bytes(), false, false).len();
            println!(
                "[{from}->{to}] {text}\n  => {out}\n  ({took:.1?}, {tokens} tokens, {:.1} tokens/s with the prompt)",
                tokens as f64 / took.as_secs_f64()
            );
            assert!(!out.is_empty());
        }
    }

    /// D2 B without the memory check of `gpu::place`: a model too big for the
    /// GPU gives an error (the meeting does not start), not an abort.
    /// Run: `MORAGO_TEST_BIG_GGUF=<12B model.gguf> LLAMA_LIB_PROFILE=Release
    /// cargo test big_model -- --ignored --nocapture`
    #[test]
    #[ignore = "needs a GPU and a model bigger than its memory"]
    fn big_model_on_the_gpu_fails_cleanly() {
        gpu::init(&gpu::app_dir().unwrap());
        let gguf = std::path::PathBuf::from(std::env::var_os("MORAGO_TEST_BIG_GGUF").unwrap());
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("big")).unwrap();
        std::fs::hard_link(&gguf, dir.path().join("big").join("model.gguf")).unwrap();
        let size = std::fs::metadata(&gguf).unwrap().len();
        let checked = gpu::place(&GpuSetting::Auto, size).unwrap();
        println!("place: {:?}", checked.status);
        assert!(
            !checked.status.gpu,
            "the memory check must keep it on the CPU"
        );

        let forced = gpu::place(&GpuSetting::Auto, 0).unwrap();
        assert!(forced.status.gpu, "no usable GPU: {:?}", forced.status);
        let started = Instant::now();
        let err = LocalTranslator::load(dir.path(), "big", &forced)
            .err()
            .unwrap()
            .to_string();
        println!("error after {:.1?}: {err}", started.elapsed());
        assert!(err.contains("불러오지 못했습니다"), "{err}");
        assert!(err.contains("「끔」"), "{err}");
    }

    /// With the earlier lines as context, no translation repeats one of them.
    #[test]
    #[ignore = "downloads a 3.3 GB model and runs it on the CPU"]
    fn local_model_context_does_not_leak() {
        let dir = ensure_model();
        let mut t = LocalTranslator::load(&dir, MODEL_ID, &Placement::cpu("")).unwrap();
        let lines = [
            "Let's review the contract with Hanbit before we sign.",
            "Section four is the part I'm worried about.",
            "It says they can end the deal with thirty days' notice.",
            "That's pretty short. Can we ask for ninety?",
            "We can, but they might push back on the price.",
            "Then let's offer sixty and see how they respond.",
        ];
        let mut earlier: Vec<String> = Vec::new();
        for line in lines {
            let out = t.translate(line, Some("en"), "ja").unwrap();
            println!("{line}\n  => {out}");
            assert!(!out.contains("Earlier lines"), "{out}");
            for e in earlier.iter().filter(|e| e.chars().count() >= 8) {
                assert!(!out.contains(e.as_str()), "{out} repeats {e}");
            }
            earlier.push(line.to_string());
            earlier.push(out);
        }
    }
}
