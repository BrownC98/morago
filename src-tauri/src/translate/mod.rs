//! Translation providers. Owner: translate worker.
//! Contract (do not change signatures without the coordinator).

use crate::settings::TranslateConfig;
use std::path::Path;

/// Translates one utterance. Runs on a worker thread (blocking).
pub trait Translator: Send {
    /// `source_language` and `target_language` are ISO 639-1 codes.
    fn translate(
        &mut self,
        text: &str,
        source_language: Option<&str>,
        target_language: &str,
    ) -> anyhow::Result<String>;
    /// Label for the UI and session meta, for example "로컬 Qwen2.5 3B".
    fn label(&self) -> String;
}

/// Builds the translator from settings. Local models are in `models_dir`.
/// A custom endpoint reads its key with `secrets::get_key(Role::Translate, base_url)`.
pub fn build_translator(
    config: &TranslateConfig,
    models_dir: &Path,
) -> anyhow::Result<Box<dyn Translator>> {
    let _ = (config, models_dir);
    anyhow::bail!("translation is not implemented yet")
}

/// False when the source language is known and equals the target.
pub fn needs_translation(source_language: Option<&str>, target_language: &str) -> bool {
    source_language.is_none_or(|s| !s.eq_ignore_ascii_case(target_language))
}
