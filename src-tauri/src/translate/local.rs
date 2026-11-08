//! Local translation with an embedded llama.cpp model.

use super::{History, Translator, clean_output, system_prompt};
use anyhow::{Context, anyhow};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
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
    history: History,
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
        let threads = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .min(16) as i32;
        Ok(Self {
            model,
            label: format!("로컬 {}", display_name(model_id)),
            threads,
            history: History::default(),
        })
    }

    fn generate(&self, system: &str, text: &str) -> anyhow::Result<String> {
        let backend = backend()?;
        let template = self
            .model
            .chat_template(None)
            .context("모델에 채팅 템플릿이 없습니다")?;
        let messages = vec![
            LlamaChatMessage::new("system".into(), system.into())?,
            LlamaChatMessage::new("user".into(), text.into())?,
        ];
        let prompt = self.model.apply_chat_template(&template, &messages, true)?;
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
        let system = system_prompt(source_language, target_language, &self.history);
        let raw = self.generate(&system, text)?;
        let translation = clean_output(&raw, text);
        if translation.is_empty() {
            return Err(anyhow!("번역 모델이 빈 번역문을 만들었습니다"));
        }
        self.history.push(text);
        Ok(translation)
    }

    fn label(&self) -> String {
        self.label.clone()
    }
}

/// "qwen2.5-3b-instruct-q4" -> "Qwen2.5 3B".
fn display_name(model_id: &str) -> String {
    let parts: Vec<String> = model_id
        .split('-')
        .take_while(|p| !p.eq_ignore_ascii_case("instruct") && !p.eq_ignore_ascii_case("chat"))
        .map(|p| {
            let is_size = p.len() > 1
                && p.ends_with(['b', 'B'])
                && p[..p.len() - 1]
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == '.');
            if is_size {
                p.to_ascii_uppercase()
            } else {
                let mut chars = p.chars();
                chars
                    .next()
                    .map(|c| c.to_uppercase().collect::<String>() + chars.as_str())
                    .unwrap_or_default()
            }
        })
        .collect();
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn labels_from_model_ids() {
        assert_eq!(display_name("qwen2.5-3b-instruct-q4"), "Qwen2.5 3B");
        assert_eq!(display_name("qwen2.5-1.5b-instruct-q4"), "Qwen2.5 1.5B");
    }

    #[test]
    fn missing_model_gives_korean_error() {
        let dir = std::env::temp_dir().join("morago-no-such-models");
        let err = LocalTranslator::load(&dir, "qwen2.5-3b-instruct-q4")
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("설치되지 않았습니다"), "{err}");
    }

    const MODEL_ID: &str = "qwen2.5-1.5b-instruct-q4";
    const MODEL_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf";

    /// Downloads the 1.5B model into `MORAGO_TEST_MODELS_DIR` (default: a temp dir).
    fn ensure_model() -> std::path::PathBuf {
        let dir = std::env::var_os("MORAGO_TEST_MODELS_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("morago-translate-test"));
        let file = dir.join(MODEL_ID).join("model.gguf");
        if !file.is_file() {
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            let client = reqwest::blocking::Client::builder()
                .timeout(None)
                .build()
                .unwrap();
            let mut response = client
                .get(MODEL_URL)
                .send()
                .unwrap()
                .error_for_status()
                .unwrap();
            let part = file.with_extension("part");
            let mut out = std::fs::File::create(&part).unwrap();
            response.copy_to(&mut out).unwrap();
            std::fs::rename(&part, &file).unwrap();
        }
        dir
    }

    /// Run: `cargo test local_model -- --ignored --nocapture`
    #[test]
    #[ignore = "downloads a 1 GB model and runs it on the CPU"]
    fn local_model_translates() {
        let dir = ensure_model();
        let started = Instant::now();
        let mut t = LocalTranslator::load(&dir, MODEL_ID).unwrap();
        println!("label: {}; load: {:.1?}", t.label(), started.elapsed());
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
