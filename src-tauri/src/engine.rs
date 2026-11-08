//! Session engine: capture -> VAD -> ASR + diarization -> translation
//! -> storage -> UI events. Each stage runs on its own thread and the
//! stages talk through channels, so a slow stage does not stop capture.

use crate::settings::{AsrConfig, Settings, TranslateConfig};
use crate::speech::{self, Diarizer, Vad};
use crate::store::{SessionStore, SessionWriter};
use crate::translate::{self, needs_translation};
use crate::types::{
    AudioChunk, SessionMeta, SessionSummary, SourceKind, SpeechSegment, TranslationState, Utterance,
};
use crate::{audio, models};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

/// Sends engine output to the UI. A trait so tests can run without Tauri.
pub trait Sink: Send + Sync + 'static {
    fn utterance(&self, u: &Utterance);
    fn level(&self, source: SourceKind, rms: f32);
    fn error(&self, stage: &str, source: Option<SourceKind>, message: &str);
}

#[derive(Serialize, Clone)]
struct LevelEvent {
    source: SourceKind,
    rms: f32,
}

#[derive(Serialize, Clone)]
struct ErrorEvent<'a> {
    stage: &'a str,
    source: Option<SourceKind>,
    message: &'a str,
}

impl Sink for AppHandle {
    fn utterance(&self, u: &Utterance) {
        let _ = self.emit("utterance", u);
    }
    fn level(&self, source: SourceKind, rms: f32) {
        let _ = self.emit("audio-level", LevelEvent { source, rms });
    }
    fn error(&self, stage: &str, source: Option<SourceKind>, message: &str) {
        log::warn!("engine {stage} error: {message}");
        let _ = self.emit(
            "engine-error",
            ErrorEvent {
                stage,
                source,
                message,
            },
        );
    }
}

/// A running session.
pub struct Running {
    pub meta: SessionMeta,
    pub target_language: Arc<RwLock<String>>,
    captures: Vec<audio::Capture>,
    audio_tx: Option<Sender<AudioChunk>>,
    threads: Vec<JoinHandle<()>>,
    writer: Arc<Mutex<Option<SessionWriter>>>,
    store_root: PathBuf,
}

/// Model ids that a session with these settings needs on disk.
pub fn required_models(settings: &Settings) -> Vec<String> {
    let mut ids: Vec<String> = models::catalog()
        .into_iter()
        .filter(|m| {
            m.kind == models::ModelKind::Vad
                || (settings.diarize && m.kind == models::ModelKind::SpeakerEmbedding)
        })
        .map(|m| m.id)
        .collect();
    if let AsrConfig::Local { model_id } = &settings.asr {
        ids.push(model_id.clone());
    }
    if let TranslateConfig::Local { model_id } = &settings.translate {
        ids.push(model_id.clone());
    }
    ids
}

fn new_session_id(now: &chrono::DateTime<chrono::Local>) -> String {
    let hex = uuid::Uuid::new_v4().simple().to_string();
    format!("{}-{}", now.format("%Y%m%d-%H%M%S"), &hex[..4])
}

/// Speaker id when diarization is off: one id per source.
fn source_speaker(source: SourceKind) -> String {
    match source {
        SourceKind::Mic => "S1".into(),
        SourceKind::System => "S2".into(),
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Starts a session. Blocking: loads models. Call from a blocking thread.
pub fn start(
    settings: &Settings,
    title: Option<String>,
    models_dir: &Path,
    store: &SessionStore,
    sink: Arc<dyn Sink>,
) -> anyhow::Result<Running> {
    if !settings.capture_mic && !settings.capture_system {
        anyhow::bail!("마이크와 시스템 소리가 모두 꺼져 있습니다. 하나 이상 켜십시오.");
    }
    let missing: Vec<String> = required_models(settings)
        .into_iter()
        .filter(|id| !models::is_installed(models_dir, id))
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "모델을 먼저 내려받아야 합니다: {}. 설정에서 내려받으십시오.",
            missing.join(", ")
        );
    }

    let asr = speech::build_asr(&settings.asr, models_dir)?;
    let translator = translate::build_translator(&settings.translate, models_dir)?;
    let diarizer = if settings.diarize {
        Some(Diarizer::new(models_dir)?)
    } else {
        None
    };
    let mut vads = Vec::new();
    for source in [SourceKind::Mic, SourceKind::System] {
        vads.push((source, Vad::new(models_dir)?));
    }

    let now = chrono::Local::now();
    let meta = SessionMeta {
        id: new_session_id(&now),
        title: title
            .filter(|t| !t.trim().is_empty())
            .unwrap_or_else(|| format!("회의 {}", now.format("%Y-%m-%d %H:%M"))),
        started_at: now.to_rfc3339(),
        ended_at: None,
        target_language: settings.target_language.clone(),
        asr_label: asr.label(),
        translator_label: translator.label(),
        speaker_names: Default::default(),
    };
    let writer = Arc::new(Mutex::new(Some(store.create(&meta)?)));
    let target_language = Arc::new(RwLock::new(settings.target_language.clone()));

    let (audio_tx, audio_rx) = bounded::<AudioChunk>(512);
    let (seg_tx, seg_rx) = unbounded::<SpeechSegment>();
    let (tr_tx, tr_rx) = unbounded::<Utterance>();

    let mut threads = Vec::new();
    {
        let sink = sink.clone();
        threads.push(std::thread::spawn(move || {
            vad_loop(audio_rx, vads, seg_tx, sink)
        }));
    }
    {
        let sink = sink.clone();
        let writer = writer.clone();
        let target = target_language.clone();
        threads.push(std::thread::spawn(move || {
            asr_loop(seg_rx, asr, diarizer, tr_tx, writer, target, sink)
        }));
    }
    {
        let sink = sink.clone();
        let writer = writer.clone();
        let target = target_language.clone();
        threads.push(std::thread::spawn(move || {
            translate_loop(tr_rx, translator, writer, target, sink)
        }));
    }

    let t0 = Instant::now();
    let mut captures = Vec::new();
    let mut wanted = Vec::new();
    if settings.capture_mic {
        wanted.push(SourceKind::Mic);
    }
    if settings.capture_system {
        wanted.push(SourceKind::System);
    }
    let mut errors = Vec::new();
    for source in &wanted {
        let device = match source {
            SourceKind::Mic => settings.mic_device.as_deref(),
            SourceKind::System => None,
        };
        match audio::start(*source, device, t0, audio_tx.clone()) {
            Ok(c) => captures.push(c),
            Err(e) => {
                sink.error("capture", Some(*source), &e.to_string());
                errors.push(e.to_string());
            }
        }
    }

    let running = Running {
        meta,
        target_language,
        captures,
        audio_tx: Some(audio_tx),
        threads,
        writer,
        store_root: store.root().to_path_buf(),
    };
    if running.captures.is_empty() {
        let msg = errors.join(" / ");
        let _ = stop(running);
        anyhow::bail!("소리를 받을 수 없습니다: {msg}");
    }
    Ok(running)
}

/// Stops capture, lets the queued speech finish, and closes the session file.
pub fn stop(mut running: Running) -> anyhow::Result<SessionSummary> {
    running.captures.clear();
    running.audio_tx.take();
    for t in running.threads.drain(..) {
        let _ = t.join();
    }
    if let Some(w) = running.writer.lock().expect("writer lock").take() {
        w.finish(&chrono::Local::now().to_rfc3339())?;
    }
    let store = SessionStore::new(&running.store_root);
    store
        .list()?
        .into_iter()
        .find(|s| s.meta.id == running.meta.id)
        .ok_or_else(|| anyhow::anyhow!("저장한 회의를 찾을 수 없습니다: {}", running.meta.id))
}

fn vad_loop(
    rx: Receiver<AudioChunk>,
    mut vads: Vec<(SourceKind, Vad)>,
    tx: Sender<SpeechSegment>,
    sink: Arc<dyn Sink>,
) {
    let mut last_level = [Instant::now(), Instant::now()];
    for chunk in rx.iter() {
        let i = chunk.source as usize;
        if last_level[i].elapsed() >= Duration::from_millis(100) {
            last_level[i] = Instant::now();
            sink.level(chunk.source, rms(&chunk.samples));
        }
        if let Some((_, vad)) = vads.iter_mut().find(|(s, _)| *s == chunk.source) {
            for seg in vad.push(&chunk) {
                let _ = tx.send(seg);
            }
        }
    }
    for (_, vad) in vads.iter_mut() {
        for seg in vad.flush() {
            let _ = tx.send(seg);
        }
    }
}

fn append(writer: &Mutex<Option<SessionWriter>>, u: &Utterance, sink: &dyn Sink) {
    if let Some(w) = writer.lock().expect("writer lock").as_mut()
        && let Err(e) = w.append(u)
    {
        sink.error(
            "store",
            Some(u.source),
            &format!("발화를 저장하지 못했습니다: {e}"),
        );
    }
}

fn asr_loop(
    rx: Receiver<SpeechSegment>,
    mut asr: Box<dyn speech::Asr>,
    mut diarizer: Option<Diarizer>,
    tx: Sender<Utterance>,
    writer: Arc<Mutex<Option<SessionWriter>>>,
    target: Arc<RwLock<String>>,
    sink: Arc<dyn Sink>,
) {
    let mut next_id = 1u64;
    for seg in rx.iter() {
        let transcript = match asr.transcribe(&seg) {
            Ok(t) => t,
            Err(e) => {
                sink.error("asr", Some(seg.source), &e.to_string());
                continue;
            }
        };
        let text = transcript.text.trim();
        if text.is_empty() {
            continue;
        }
        let speaker = match diarizer.as_mut() {
            Some(d) => d.assign(&seg).unwrap_or_else(|e| {
                sink.error("asr", Some(seg.source), &e.to_string());
                source_speaker(seg.source)
            }),
            None => source_speaker(seg.source),
        };
        let target_now = target.read().expect("target lock").clone();
        let translate = needs_translation(transcript.language.as_deref(), &target_now);
        let u = Utterance {
            id: next_id,
            source: seg.source,
            speaker,
            start_ms: seg.start_ms,
            end_ms: seg.end_ms,
            language: transcript.language.clone(),
            original: text.to_string(),
            translation: if translate {
                TranslationState::Pending
            } else {
                TranslationState::SameLanguage
            },
        };
        next_id += 1;
        sink.utterance(&u);
        append(&writer, &u, sink.as_ref());
        if translate {
            let _ = tx.send(u);
        }
    }
}

fn translate_loop(
    rx: Receiver<Utterance>,
    mut translator: Box<dyn translate::Translator>,
    writer: Arc<Mutex<Option<SessionWriter>>>,
    target: Arc<RwLock<String>>,
    sink: Arc<dyn Sink>,
) {
    for mut u in rx.iter() {
        let target_now = target.read().expect("target lock").clone();
        u.translation = if !needs_translation(u.language.as_deref(), &target_now) {
            TranslationState::SameLanguage
        } else {
            match translator.translate(&u.original, u.language.as_deref(), &target_now) {
                Ok(text) => TranslationState::Done {
                    target: target_now,
                    text,
                },
                Err(e) => {
                    sink.error("translate", Some(u.source), &e.to_string());
                    TranslationState::Failed {
                        message: e.to_string(),
                    }
                }
            }
        };
        sink.utterance(&u);
        append(&writer, &u, sink.as_ref());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_id_has_store_format() {
        let now = chrono::Local::now();
        let id = new_session_id(&now);
        let parts: Vec<&str> = id.split('-').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].len(), 8);
        assert_eq!(parts[1].len(), 6);
        assert_eq!(parts[2].len(), 4);
        assert!(parts[2].chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn rms_of_constant_signal() {
        assert_eq!(rms(&[]), 0.0);
        assert!((rms(&[0.5, -0.5, 0.5, -0.5]) - 0.5).abs() < 1e-6);
    }
}
