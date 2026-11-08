//! Export texts: Markdown, SRT, JSON.

use crate::types::{SessionMeta, TranslationState, Utterance};
use std::fmt::Write;

/// Display name: the user's name, else "화자 N" for "SN", else the raw id.
fn speaker_name(meta: &SessionMeta, speaker: &str) -> String {
    if let Some(name) = meta.speaker_names.get(speaker) {
        return name.clone();
    }
    match speaker.strip_prefix('S') {
        Some(n) if !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()) => format!("화자 {n}"),
        _ => speaker.to_string(),
    }
}

fn translation(u: &Utterance) -> Option<&str> {
    match &u.translation {
        TranslationState::Done { text, .. } => Some(text),
        _ => None,
    }
}

/// Collapses line breaks and runs of spaces: one cue or quote line.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn clock(ms: u64) -> String {
    let s = ms / 1000;
    format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

fn srt_time(ms: u64) -> String {
    format!("{},{:03}", clock(ms), ms % 1000)
}

fn date(rfc3339: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(rfc3339)
        .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|_| rfc3339.to_string())
}

pub(super) fn markdown(meta: &SessionMeta, utterances: &[Utterance]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# {}\n", one_line(&meta.title));
    let _ = writeln!(out, "- 날짜: {}", date(&meta.started_at));
    let mut languages: Vec<&str> = Vec::new();
    for lang in utterances.iter().filter_map(|u| u.language.as_deref()) {
        if !languages.contains(&lang) {
            languages.push(lang);
        }
    }
    if !languages.is_empty() {
        let _ = writeln!(out, "- 원문 언어: {}", languages.join(", "));
    }
    let _ = writeln!(out, "- 번역 언어: {}", meta.target_language);
    let _ = writeln!(out, "- 음성 인식: {}", meta.asr_label);
    let _ = writeln!(out, "- 번역: {}", meta.translator_label);
    for u in utterances {
        // Two trailing spaces: a Markdown line break inside the paragraph.
        let _ = write!(
            out,
            "\n**{}** · {}–{}  \n{}\n",
            speaker_name(meta, &u.speaker),
            clock(u.start_ms),
            clock(u.end_ms),
            one_line(&u.original)
        );
        if let Some(text) = translation(u) {
            let _ = writeln!(out, "> {}", one_line(text));
        }
    }
    out
}

pub(super) fn srt(meta: &SessionMeta, utterances: &[Utterance]) -> String {
    let mut out = String::new();
    for (n, u) in utterances.iter().enumerate() {
        let _ = writeln!(
            out,
            "{}\n{} --> {}\n{}: {}",
            n + 1,
            srt_time(u.start_ms),
            srt_time(u.end_ms),
            speaker_name(meta, &u.speaker),
            one_line(&u.original)
        );
        if let Some(text) = translation(u) {
            let _ = writeln!(out, "{}", one_line(text));
        }
        out.push('\n');
    }
    out
}

pub(super) fn json(meta: &SessionMeta, utterances: &[Utterance]) -> anyhow::Result<String> {
    #[derive(serde::Serialize)]
    struct Export<'a> {
        meta: &'a SessionMeta,
        utterances: &'a [Utterance],
    }
    Ok(serde_json::to_string_pretty(&Export { meta, utterances })?)
}
