//! Translation providers. Owner: translate worker.
//! Contract (do not change signatures without the coordinator).
//!
//! Both providers keep the last source texts of the session in a `History` and
//! send them as context (the model translates only the last one), so the trait
//! needs no extra method for context. A translator lives for one session.

mod custom;
mod local;

use crate::gpu::Placement;
use crate::settings::TranslateConfig;
pub use custom::{check_extra_body, check_extra_body_map};
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

/// Builds the translator from settings. Local models are in `models_dir` and
/// load on the device of `placement` (`gpu::place`; a custom endpoint ignores it).
/// A custom endpoint reads its key with `secrets::get_key(Role::Translate, base_url)`.
/// `context_lines`: most earlier utterances sent as context.
pub fn build_translator(
    config: &TranslateConfig,
    context_lines: usize,
    models_dir: &Path,
    placement: &Placement,
) -> anyhow::Result<Box<dyn Translator>> {
    let history = History::new(context_lines);
    match config {
        TranslateConfig::Local { model_id } => Ok(Box::new(local::LocalTranslator::load(
            models_dir, model_id, history, placement,
        )?)),
        TranslateConfig::Custom {
            endpoint,
            extra_body,
            ..
        } => {
            custom::required_model(extra_body)?;
            let key = crate::secrets::get_key(crate::secrets::Role::Translate, &endpoint.base_url)?;
            Ok(Box::new(custom::CustomTranslator::new(
                &endpoint.base_url,
                key,
                extra_body.clone(),
                history,
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

/// The last source texts of the session, for one target language.
pub(crate) struct History {
    target: String,
    lines: VecDeque<String>,
    max: usize,
}

impl History {
    pub(crate) fn new(max: usize) -> Self {
        Self {
            target: String::new(),
            lines: VecDeque::new(),
            max,
        }
    }

    /// Oldest first.
    pub(crate) fn lines(&self) -> &VecDeque<String> {
        &self.lines
    }

    /// Clears the lines when the target language changes, as a new session does.
    pub(crate) fn set_target(&mut self, target_language: &str) {
        if !self.target.eq_ignore_ascii_case(target_language) {
            self.target = target_language.to_string();
            self.lines.clear();
        }
    }

    pub(crate) fn push(&mut self, text: &str) {
        self.lines.push_back(text.replace('\n', " "));
        while self.lines.len() > self.max {
            self.lines.pop_front();
        }
    }
}

/// System prompt for one request. `context`: earlier lines, oldest first.
/// `retry`: the last answer was not in the target language (`in_target_script`).
pub(crate) fn system_prompt<'a>(
    source_language: Option<&str>,
    target_language: &str,
    context: impl IntoIterator<Item = &'a String>,
    retry: bool,
) -> String {
    let mut context = context.into_iter().peekable();
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
    if context.peek().is_some() {
        prompt.push_str(
            "\nEarlier lines of the same meeting, only as context. Do not translate or repeat them:",
        );
        for line in context {
            prompt.push_str("\n- ");
            prompt.push_str(line);
        }
    }
    if retry {
        prompt.push_str(&format!(
            "\nYour last answer was not in {target}. Write the translation in {target} only. Do not copy or correct the source text."
        ));
    }
    prompt
}

/// True when `text` has letters that the target language never uses: kana
/// in Korean or Chinese, Hangul in Japanese or Chinese. Other targets: false.
pub(crate) fn wrong_script(text: &str, target_language: &str) -> bool {
    let kana = |c: char| matches!(c, '\u{3040}'..='\u{30FA}' | '\u{30FC}'..='\u{30FF}' | '\u{31F0}'..='\u{31FF}' | '\u{FF66}'..='\u{FF9F}');
    let hangul = |c: char| matches!(c, '\u{AC00}'..='\u{D7A3}' | '\u{1100}'..='\u{11FF}' | '\u{3130}'..='\u{318F}');
    let base = target_language.split(['-', '_']).next().unwrap_or("");
    let never: &dyn Fn(char) -> bool = match base.to_ascii_lowercase().as_str() {
        "ko" => &kana,
        "ja" => &hangul,
        "zh" => &|c| kana(c) || hangul(c),
        _ => return false,
    };
    text.chars().any(never)
}

/// Runs `attempt(false)`. When its output is in a wrong script
/// (`wrong_script`), runs `attempt(true)` once: its prompt insists on the target.
pub(crate) fn in_target_script(
    target_language: &str,
    mut attempt: impl FnMut(bool) -> anyhow::Result<String>,
) -> anyhow::Result<String> {
    let first = attempt(false)?;
    if !wrong_script(&first, target_language) {
        return Ok(first);
    }
    log::warn!("translation not in {target_language}, retrying once");
    let second = attempt(true)?;
    if wrong_script(&second, target_language) {
        return Err(anyhow::anyhow!(
            "번역문이 설정한 언어가 아닌 다른 언어로 나왔습니다"
        ));
    }
    Ok(second)
}

/// Removes what chat models add around a translation: reasoning blocks,
/// "Translation:" prefixes and wrapping quotes.
pub(crate) fn clean_output(raw: &str, source_text: &str) -> String {
    let mut s = raw.trim();
    if let Some(end) = s.rfind("</think>") {
        s = s[end + "</think>".len()..].trim();
    }
    // Gemma 4 12B writes an empty thought block before the answer. Its gguf
    // does not mark the channel tokens as control tokens, so they come as text;
    // with control tokens hidden it shows as a leading "thought" line.
    if let Some(end) = s.rfind("<channel|>") {
        s = s[end + "<channel|>".len()..].trim();
    }
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

    /// `start_session` and `test_provider` build the translator here.
    #[test]
    fn a_custom_server_without_model_is_a_korean_error() {
        for extra in [
            serde_json::json!({}),
            serde_json::json!({"model": " "}),
            serde_json::json!({"model": 3}),
        ] {
            let config = TranslateConfig::Custom {
                endpoint: crate::types::Endpoint {
                    base_url: "http://localhost:8000/v1".into(),
                    model: String::new(),
                },
                extra_body: extra.as_object().unwrap().clone(),
                partial_interval_ms: None,
            };
            let err = build_translator(&config, 4, Path::new(""), &Placement::cpu(""))
                .err()
                .unwrap();
            assert_eq!(
                err.to_string(),
                "번역 모델 이름이 없습니다. 설정 › 번역 › 추가 요청 값에 \"model\"을 적으세요."
            );
        }
    }

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
        let mut history = History::new(4);
        let p = system_prompt(Some("en"), "ko", history.lines(), false);
        assert!(p.contains("from English into Korean"));
        assert!(!p.contains("Earlier lines"));
        for line in ["one", "two", "three", "four", "five\nsix"] {
            history.push(line);
        }
        let p = system_prompt(None, "ja", history.lines(), false);
        assert!(p.contains("into Japanese"));
        assert!(p.ends_with("- two\n- three\n- four\n- five six"), "{p}");
        assert!(!p.contains("- one"));
    }

    #[test]
    fn a_retry_prompt_insists_on_the_target() {
        let p = system_prompt(Some("ja"), "ko", std::iter::empty(), true);
        assert!(
            p.ends_with(
                "Write the translation in Korean only. Do not copy or correct the source text."
            ),
            "{p}"
        );
        assert!(
            !system_prompt(Some("ja"), "ko", std::iter::empty(), false).contains("last answer")
        );
    }

    #[test]
    fn wrong_scripts_for_the_target() {
        // The two answers of session 20261109-171830 (ja -> ko).
        assert!(wrong_script(
            "そして、敷居が甚だしく乱れていて、それで、仮に移して。",
            "ko"
        ));
        assert!(wrong_script("程度の二つのところがあるわけであって", "ko"));
        assert!(!wrong_script("그 연쇄 장치로 가열해서 재료를.", "ko"));
        assert!(!wrong_script("田中 씨가 OK라고 했습니다.", "ko"));
        // The katakana middle dot is punctuation that a translation keeps.
        assert!(!wrong_script("도쿄・오사카", "ko"));
        assert!(wrong_script("会議は 내일です", "ja"));
        assert!(!wrong_script("会議は明日です", "ja"));
        assert!(wrong_script("ミーティング", "zh-CN"));
        assert!(!wrong_script("会议在明天", "zh"));
        assert!(!wrong_script("こんにちは", "en"));
    }

    #[test]
    fn a_wrong_script_retries_once() {
        let mut tries = Vec::new();
        let out = in_target_script("ko", |retry| {
            tries.push(retry);
            Ok(if retry {
                "안녕하세요"
            } else {
                "こんにちは"
            }
            .to_string())
        });
        assert_eq!(out.unwrap(), "안녕하세요");
        assert_eq!(tries, [false, true]);

        let mut n = 0;
        let out = in_target_script("ko", |_| {
            n += 1;
            Ok("こんにちは".to_string())
        });
        assert!(
            out.unwrap_err()
                .to_string()
                .contains("다른 언어로 나왔습니다")
        );
        assert_eq!(n, 2);

        let mut n = 0;
        assert_eq!(
            in_target_script("ko", |_| {
                n += 1;
                Ok("네".to_string())
            })
            .unwrap(),
            "네"
        );
        assert_eq!(n, 1);
    }

    #[test]
    fn history_keeps_the_newest_n_lines_of_any_length() {
        for n in [1, 2, 10] {
            let mut history = History::new(n);
            for i in 0..15 {
                history.push(&format!("{i} {}", "가".repeat(1000)));
            }
            let kept: Vec<&str> = history
                .lines
                .iter()
                .map(|l| l.split(' ').next().unwrap())
                .collect();
            let want: Vec<String> = (15 - n..15).map(|i| i.to_string()).collect();
            assert_eq!(kept, want);
        }
    }

    #[test]
    fn zero_context_lines_send_no_context_block() {
        let mut history = History::new(0);
        history.push("one");
        assert!(history.lines.is_empty());
        let p = system_prompt(Some("en"), "ko", history.lines(), false);
        assert!(!p.contains("Earlier lines"), "{p}");
        assert!(!p.contains("one"), "{p}");
    }

    #[test]
    fn history_clears_when_the_target_changes() {
        let mut history = History::new(4);
        history.set_target("ko");
        history.push("one");
        history.set_target("KO");
        assert_eq!(history.lines, ["one"]);
        history.set_target("ja");
        assert!(history.lines.is_empty());
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
        assert_eq!(
            clean_output("<|channel>thought\n<channel|>네, 오늘 하겠습니다.", "x"),
            "네, 오늘 하겠습니다."
        );
        assert_eq!(clean_output("thought\n안녕", "x"), "안녕");
        assert_eq!(clean_output("plain", "plain"), "plain");
        // Inner quotes stay.
        assert_eq!(clean_output("\"a\" and \"b\"", "x"), "\"a\" and \"b\"");
        // A quoted source keeps its quotes.
        assert_eq!(clean_output("\"안녕\"", "\"hello\""), "\"안녕\"");
    }
}
