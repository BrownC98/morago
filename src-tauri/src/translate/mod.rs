//! Translation providers. Owner: translate worker.
//! Contract (do not change signatures without the coordinator).
//!
//! Both providers keep the last two source texts of the session and send them
//! as context (the model translates only the last one), so the trait needs no
//! extra method for context.

mod custom;
mod local;

pub(crate) use local::on_gpu as local_on_gpu;

use crate::settings::TranslateConfig;
use std::collections::VecDeque;
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
    /// Translates live partial text. Never changes the context history.
    /// Returns an error soon after `stop` returns true (a final utterance
    /// waits, or the speech ended).
    fn translate_partial(
        &mut self,
        text: &str,
        source_language: Option<&str>,
        target_language: &str,
        stop: &dyn Fn() -> bool,
    ) -> anyhow::Result<String>;
    /// Label for the UI and session meta, for example "로컬 Gemma 4 E2B".
    fn label(&self) -> String;
    /// A one-time message for the user that is not an error of the last call.
    /// The engine shows it through the error path. Default: none.
    fn take_notice(&mut self) -> Option<String> {
        None
    }
}

/// Builds the translator from settings. Local models are in `models_dir`.
/// A custom endpoint reads its key with `secrets::get_key(Role::Translate, base_url)`.
pub fn build_translator(
    config: &TranslateConfig,
    models_dir: &Path,
) -> anyhow::Result<Box<dyn Translator>> {
    match config {
        TranslateConfig::Local { model_id } => Ok(Box::new(local::LocalTranslator::load(
            models_dir, model_id,
        )?)),
        TranslateConfig::Custom {
            endpoint,
            reasoning_effort,
            ..
        } => {
            let key = crate::secrets::get_key(crate::secrets::Role::Translate, &endpoint.base_url)?;
            Ok(Box::new(custom::CustomTranslator::new(
                endpoint.clone(),
                key,
                *reasoning_effort,
            )?))
        }
    }
}

/// False when the source language is known and equals the target.
pub fn needs_translation(source_language: Option<&str>, target_language: &str) -> bool {
    source_language.is_none_or(|s| !s.eq_ignore_ascii_case(target_language))
}

/// English name of an ISO 639-1 code for prompts. An unknown code is returned as is.
pub(crate) fn language_name(code: &str) -> String {
    let base = code
        .split(['-', '_'])
        .next()
        .unwrap_or(code)
        .to_ascii_lowercase();
    let name = match base.as_str() {
        "ko" => "Korean",
        "en" => "English",
        "ja" => "Japanese",
        "zh" => "Chinese",
        "es" => "Spanish",
        "fr" => "French",
        "de" => "German",
        "vi" => "Vietnamese",
        "th" => "Thai",
        "id" => "Indonesian",
        "ru" => "Russian",
        "pt" => "Portuguese",
        "it" => "Italian",
        "ar" => "Arabic",
        "hi" => "Hindi",
        "tr" => "Turkish",
        "nl" => "Dutch",
        "pl" => "Polish",
        "uk" => "Ukrainian",
        "ms" => "Malay",
        _ => return code.to_string(),
    };
    name.to_string()
}

/// Number of earlier utterances sent as context.
const CONTEXT_LINES: usize = 2;

/// The last source texts of the session.
#[derive(Default)]
pub(crate) struct History(VecDeque<String>);

impl History {
    pub(crate) fn push(&mut self, text: &str) {
        if self.0.len() == CONTEXT_LINES {
            self.0.pop_front();
        }
        self.0.push_back(text.to_string());
    }
}

/// System prompt for one request.
pub(crate) fn system_prompt(
    source_language: Option<&str>,
    target_language: &str,
    history: &History,
) -> String {
    let target = language_name(target_language);
    let mut prompt = match source_language {
        Some(source) => format!(
            "You are a professional translator for live meeting subtitles. Translate the user's text from {} into {target}.",
            language_name(source)
        ),
        None => format!(
            "You are a professional translator for live meeting subtitles. Translate the user's text into {target}."
        ),
    };
    prompt.push_str(
        " Output only the translation: no notes, no explanation, no quotation marks. Keep names, numbers and technical terms.",
    );
    if !history.0.is_empty() {
        prompt.push_str(
            "\nEarlier lines of the same meeting, only as context. Do not translate or repeat them:",
        );
        for line in &history.0 {
            prompt.push_str("\n- ");
            prompt.push_str(&line.replace('\n', " "));
        }
    }
    prompt
}

/// Removes what chat models add around a translation: reasoning blocks,
/// "Translation:" prefixes and wrapping quotes.
pub(crate) fn clean_output(raw: &str, source_text: &str) -> String {
    let mut s = raw.trim();
    if let Some(end) = s.rfind("</think>") {
        s = s[end + "</think>".len()..].trim();
    }
    // Gemma 4 12B writes an empty thought block before the answer; with
    // control tokens hidden it shows as a leading "thought" line.
    if let Some(rest) = s.strip_prefix("thought\n") {
        s = rest.trim();
    }
    const PREFIXES: [&str; 6] = [
        "translation:",
        "translated text:",
        "translated:",
        "번역:",
        "翻訳:",
        "翻译:",
    ];
    let lower = s.to_lowercase();
    for prefix in PREFIXES {
        if lower.starts_with(prefix) {
            // The lowercase form can differ in byte length for rare letters:
            // cut at the char boundary after the same number of chars.
            let cut = s
                .char_indices()
                .nth(prefix.chars().count())
                .map_or(s.len(), |(i, _)| i);
            s = s[cut..].trim();
            break;
        }
    }
    const PAIRS: [(char, char); 6] = [
        ('"', '"'),
        ('\'', '\''),
        ('“', '”'),
        ('‘', '’'),
        ('「', '」'),
        ('«', '»'),
    ];
    let source_quoted = PAIRS.iter().any(|(open, close)| {
        source_text.trim().starts_with(*open) && source_text.trim().ends_with(*close)
    });
    if !source_quoted {
        for (open, close) in PAIRS {
            if let Some(inner) = s.strip_prefix(open).and_then(|t| t.strip_suffix(close)) {
                // Keep quotes that open and close inside the text.
                if !inner.contains(open) && !inner.contains(close) {
                    s = inner.trim();
                    break;
                }
            }
        }
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_language_is_skipped() {
        assert!(!needs_translation(Some("ko"), "KO"));
        assert!(needs_translation(Some("en"), "ko"));
        assert!(needs_translation(None, "ko"));
    }

    #[test]
    fn language_names() {
        for (code, name) in [
            ("ko", "Korean"),
            ("en", "English"),
            ("ja", "Japanese"),
            ("zh", "Chinese"),
            ("es", "Spanish"),
            ("fr", "French"),
            ("de", "German"),
            ("vi", "Vietnamese"),
            ("th", "Thai"),
            ("id", "Indonesian"),
            ("ru", "Russian"),
            ("pt", "Portuguese"),
            ("it", "Italian"),
        ] {
            assert_eq!(language_name(code), name);
        }
        assert_eq!(language_name("zh-CN"), "Chinese");
        assert_eq!(language_name("EN"), "English");
        assert_eq!(language_name("xx"), "xx");
    }

    #[test]
    fn prompt_with_and_without_context() {
        let mut history = History::default();
        let p = system_prompt(Some("en"), "ko", &history);
        assert!(p.contains("from English into Korean"));
        assert!(!p.contains("Earlier lines"));
        history.push("one");
        history.push("two");
        history.push("three");
        let p = system_prompt(None, "ja", &history);
        assert!(p.contains("into Japanese"));
        assert!(p.contains("- two\n- three"));
        assert!(!p.contains("- one"));
    }

    #[test]
    fn cleans_prefixes_and_quotes() {
        assert_eq!(
            clean_output("  Translation: 안녕하세요  ", "hello"),
            "안녕하세요"
        );
        assert_eq!(clean_output("TRANSLATED TEXT:\n\"안녕\"", "hello"), "안녕");
        assert_eq!(clean_output("번역: “안녕하세요”", "hello"), "안녕하세요");
        assert_eq!(
            clean_output("<think>x</think>\nこんにちは", "hello"),
            "こんにちは"
        );
        assert_eq!(clean_output("plain", "plain"), "plain");
        // Inner quotes stay.
        assert_eq!(clean_output("\"a\" and \"b\"", "x"), "\"a\" and \"b\"");
        // A quoted source keeps its quotes.
        assert_eq!(clean_output("\"안녕\"", "\"hello\""), "\"안녕\"");
    }
}
