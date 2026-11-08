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
        };
        if let Some((_, name)) = &placement.gpu {
            // The first run on a GPU loads its kernels and is slow: do it before the meeting.
            translator
                .generate("Translate into Korean.", "Hello.")
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

    fn generate(&self, system: &str, text: &str) -> anyhow::Result<String> {
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
}

impl Translator for LocalTranslator {
    fn translate(
        &mut self,
        text: &str,
        source_language: Option<&str>,
        target_language: &str,
    ) -> anyhow::Result<String> {
        let system = system_prompt(source_language, target_language, &History::default());
        let raw = self.generate(&system, text)?;
        let translation = clean_output(&raw, text);
        if translation.is_empty() {
            return Err(anyhow!("번역 모델이 빈 번역문을 만들었습니다"));
        }
        Ok(translation)
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
    #[test]
    #[ignore = "downloads a 3.3 GB model and runs it"]
    fn local_model_translates() {
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
            println!(
                "[{from}->{to}] {text}\n  => {out}\n  ({:.1?})",
                started.elapsed()
            );
            assert!(!out.is_empty());
        }
    }
}
