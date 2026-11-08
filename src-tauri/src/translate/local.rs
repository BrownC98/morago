//! Local translation with an embedded llama.cpp model.

use super::{History, Translator, clean_output, system_prompt};
use anyhow::anyhow;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;

/// llama.cpp allows one backend per process.
fn backend() -> anyhow::Result<&'static LlamaBackend> {
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            LlamaBackend::init()
                .map(|mut backend| {
                    backend.void_logs();
                    backend
                })
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| anyhow!("번역 엔진을 시작하지 못했습니다: {e}"))
}

pub(crate) struct LocalTranslator {
    model: LlamaModel,
    label: String,
    threads: i32,
}

impl LocalTranslator {
    /// Loads `<models_dir>/<model_id>/model.gguf` once.
    pub(crate) fn load(models_dir: &Path, model_id: &str) -> anyhow::Result<Self> {
        let path = models_dir.join(model_id).join("model.gguf");
        if !path.is_file() {
            return Err(anyhow!(
                "번역 모델이 설치되지 않았습니다: {model_id}. 설정에서 모델을 내려받으세요."
            ));
        }
        let backend = backend()?;
        // All layers to the GPU on macOS (Metal). Other builds have no GPU backend.
        let gpu_layers = if cfg!(target_os = "macos") { 999 } else { 0 };
        let params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
        let model = LlamaModel::load_from_file(backend, &path, &params)
            .map_err(|e| anyhow!("번역 모델을 읽지 못했습니다 ({model_id}): {e}"))?;
        // About the physical cores: more threads than that slow ggml down.
        let threads =
            (std::thread::available_parallelism().map_or(4, |n| n.get()) / 2).clamp(2, 8) as i32;
        Ok(Self {
            model,
            label: format!("로컬 {}", crate::models::display_name(model_id)),
            threads,
        })
    }

    /// Stops with an error before the next token when `stop` returns true.
    fn generate(
        &self,
        system: &str,
        text: &str,
        stop: &dyn Fn() -> bool,
    ) -> anyhow::Result<String> {
        let backend = backend()?;
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
        let n_ctx = u32::try_from(n_ctx)?;
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx)
            .with_n_threads(self.threads)
            .with_n_threads_batch(self.threads);
        let mut ctx = self.model.new_context(backend, params)?;

        let mut batch = LlamaBatch::new(prompt_tokens.len().max(1), 1);
        batch.add_sequence(&prompt_tokens, 0, false)?;
        ctx.decode(&mut batch)?;

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
            ctx.decode(&mut batch)?;
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
        self.translate_partial(text, source_language, target_language, &|| false)
    }

    /// The same as `translate`: this provider keeps no history.
    fn translate_partial(
        &mut self,
        text: &str,
        source_language: Option<&str>,
        target_language: &str,
        stop: &dyn Fn() -> bool,
    ) -> anyhow::Result<String> {
        let system = system_prompt(source_language, target_language, &History::default());
        let raw = self.generate(&system, text, stop)?;
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
        let err = LocalTranslator::load(&dir, "gemma-4-e2b-it-q4")
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
    /// (a debug build of llama.cpp is about 10x slower).
    #[test]
    #[ignore = "downloads a 3.3 GB model and runs it on the CPU"]
    fn local_model_translates() {
        let dir = ensure_model();
        let model_started = Instant::now();
        let mut t = LocalTranslator::load(&dir, MODEL_ID).unwrap();
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
