//! Session engine: capture -> VAD -> ASR + diarization -> translation
//! -> storage -> UI events. Each stage runs on its own thread and the
//! stages talk through channels, so a slow stage does not stop capture.

use crate::settings::{AsrConfig, Settings, TranslateConfig};
use crate::speech::{self, Diarizer, Vad};
use crate::store::{SessionStore, SessionWriter};
use crate::translate::{self, needs_translation};
use crate::types::{
    AudioChunk, PartialText, SessionMeta, SessionSummary, SourceKind, SpeechSegment,
    TranslationState, Utterance,
};
use crate::{audio, models};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

/// Sends engine output to the UI. A trait so tests can run without Tauri.
pub trait Sink: Send + Sync + 'static {
    fn utterance(&self, u: &Utterance);
    fn level(&self, source: SourceKind, rms: f32);
    fn error(&self, stage: &str, source: Option<SourceKind>, message: &str);
    /// Streaming ASR only (follow-up live mode). Kept as the ready interface
    /// (human decision 2026-11-08); nothing calls it yet.
    #[allow(dead_code)]
    fn partial(&self, p: &PartialText) {
        let _ = p;
    }
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
    fn partial(&self, p: &PartialText) {
        let _ = self.emit("utterance-partial", p);
    }
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

/// Downloaded model ids that a session with these settings needs on disk.
pub fn required_models(settings: &Settings) -> Vec<String> {
    let mut ids = Vec::new();
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

/// The parts of a running session before capture starts.
struct Pipeline {
    meta: SessionMeta,
    target_language: Arc<RwLock<String>>,
    audio_tx: Sender<AudioChunk>,
    threads: Vec<JoinHandle<()>>,
    writer: Arc<Mutex<Option<SessionWriter>>>,
}

/// Loads the models, creates the session file and starts the stage threads.
fn start_pipeline(
    settings: &Settings,
    title: Option<String>,
    models_dir: &Path,
    bundled_dir: &Path,
    store: &SessionStore,
    sink: &Arc<dyn Sink>,
) -> anyhow::Result<Pipeline> {
    let asr = speech::build_asr(
        &settings.asr,
        settings.meeting_language.as_deref(),
        models_dir,
    )?;
    let translator = translate::build_translator(&settings.translate, models_dir)?;
    models::verify_bundled(bundled_dir)?;
    let diarizer = if settings.diarize {
        Some(Diarizer::new(bundled_dir)?)
    } else {
        None
    };
    let mut vads = Vec::new();
    for source in [SourceKind::Mic, SourceKind::System] {
        vads.push((source, Vad::new(bundled_dir)?));
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
    let (seg_tx, seg_rx) = backlog(BACKLOG_LIMIT_MS, segment_ms);
    let (tr_tx, tr_rx) = backlog(BACKLOG_LIMIT_MS, utterance_ms);

    let mut threads = Vec::new();
    {
        let sink = Arc::clone(sink);
        threads.push(std::thread::spawn(move || {
            vad_loop(audio_rx, vads, seg_tx, sink)
        }));
    }
    {
        let sink = Arc::clone(sink);
        let writer = writer.clone();
        let target = target_language.clone();
        threads.push(std::thread::spawn(move || {
            asr_loop(seg_rx, asr, diarizer, tr_tx, writer, target, sink)
        }));
    }
    {
        let sink = Arc::clone(sink);
        let writer = writer.clone();
        let target = target_language.clone();
        threads.push(std::thread::spawn(move || {
            translate_loop(tr_rx, translator, writer, target, sink)
        }));
    }

    Ok(Pipeline {
        meta,
        target_language,
        audio_tx,
        threads,
        writer,
    })
}

/// Starts a session. Blocking: loads models. Call from a blocking thread.
pub fn start(
    settings: &Settings,
    title: Option<String>,
    models_dir: &Path,
    bundled_dir: &Path,
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

    let Pipeline {
        meta,
        target_language,
        audio_tx,
        threads,
        writer,
    } = start_pipeline(settings, title, models_dir, bundled_dir, store, &sink)?;

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
        let on_error: audio::ErrorSink = {
            let sink = sink.clone();
            let s = *source;
            Arc::new(move |m: String| sink.error("capture", Some(s), &m))
        };
        match audio::start(*source, device, t0, audio_tx.clone(), on_error) {
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
        // No audio came in: do not leave an empty record behind.
        let id = running.meta.id.clone();
        let _ = stop(running);
        let _ = store.delete(&id);
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

/// Most audio time a stage queue holds. ASR or translation slower than real
/// time would else grow memory for the whole meeting.
const BACKLOG_LIMIT_MS: u64 = 30_000;

/// A queue bounded by the audio time it holds. When it is full, the oldest
/// item leaves the queue through `on_drop`, so memory stays bounded.
struct Backlog<T> {
    tx: Sender<T>,
    /// Same channel: the producer takes the oldest item out on overflow.
    oldest: Receiver<T>,
    queued_ms: Arc<AtomicU64>,
    limit_ms: u64,
    ms: fn(&T) -> u64,
}

/// The consumer side of a `Backlog`.
struct BacklogRx<T> {
    rx: Receiver<T>,
    queued_ms: Arc<AtomicU64>,
    ms: fn(&T) -> u64,
}

fn backlog<T>(limit_ms: u64, ms: fn(&T) -> u64) -> (Backlog<T>, BacklogRx<T>) {
    let (tx, rx) = unbounded();
    let queued_ms = Arc::new(AtomicU64::new(0));
    (
        Backlog {
            tx,
            oldest: rx.clone(),
            queued_ms: queued_ms.clone(),
            limit_ms,
            ms,
        },
        BacklogRx { rx, queued_ms, ms },
    )
}

fn take_ms(queued: &AtomicU64, ms: u64) {
    let _ = queued.try_update(Ordering::Relaxed, Ordering::Relaxed, |q| {
        Some(q.saturating_sub(ms))
    });
}

impl<T> Backlog<T> {
    fn push(&self, item: T, on_drop: &mut dyn FnMut(T)) {
        let ms = (self.ms)(&item);
        while self.queued_ms.load(Ordering::Relaxed) + ms > self.limit_ms {
            match self.oldest.try_recv() {
                Ok(old) => {
                    take_ms(&self.queued_ms, (self.ms)(&old));
                    on_drop(old);
                }
                Err(_) => break,
            }
        }
        self.queued_ms.fetch_add(ms, Ordering::Relaxed);
        let _ = self.tx.send(item);
    }
}

impl<T> BacklogRx<T> {
    fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.rx
            .iter()
            .inspect(|item| take_ms(&self.queued_ms, (self.ms)(item)))
    }
}

/// Tells the user about dropped work: at once, then at most every 10 s.
struct DropWarning {
    stage: &'static str,
    text: &'static str,
    count: u64,
    last: Option<Instant>,
}

impl DropWarning {
    fn new(stage: &'static str, text: &'static str) -> Self {
        Self {
            stage,
            text,
            count: 0,
            last: None,
        }
    }

    fn dropped(&mut self, source: SourceKind, sink: &dyn Sink) {
        self.count += 1;
        if self
            .last
            .is_none_or(|t| t.elapsed() >= Duration::from_secs(10))
        {
            self.last = Some(Instant::now());
            sink.error(
                self.stage,
                Some(source),
                &format!("{} 건너뛴 발화: {}개.", self.text, self.count),
            );
        }
    }
}

fn segment_ms(s: &SpeechSegment) -> u64 {
    s.end_ms.saturating_sub(s.start_ms)
}

fn utterance_ms(u: &Utterance) -> u64 {
    u.end_ms.saturating_sub(u.start_ms)
}

fn vad_loop(
    rx: Receiver<AudioChunk>,
    mut vads: Vec<(SourceKind, Vad)>,
    tx: Backlog<SpeechSegment>,
    sink: Arc<dyn Sink>,
) {
    let mut warning = DropWarning::new(
        "asr",
        "인식이 밀립니다: 설정에서 더 작은 음성 인식 모델을 고르세요.",
    );
    let mut on_drop = |seg: SpeechSegment| {
        log::warn!(
            "ASR backlog full: dropped {} ms of speech",
            segment_ms(&seg)
        );
        warning.dropped(seg.source, sink.as_ref());
    };
    let mut last_level = [Instant::now(), Instant::now()];
    for chunk in rx.iter() {
        let i = chunk.source as usize;
        if last_level[i].elapsed() >= Duration::from_millis(100) {
            last_level[i] = Instant::now();
            sink.level(chunk.source, rms(&chunk.samples));
        }
        if let Some((_, vad)) = vads.iter_mut().find(|(s, _)| *s == chunk.source) {
            for seg in vad.push(&chunk) {
                tx.push(seg, &mut on_drop);
            }
        }
    }
    for (_, vad) in vads.iter_mut() {
        for seg in vad.flush() {
            tx.push(seg, &mut on_drop);
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
    rx: BacklogRx<SpeechSegment>,
    mut asr: Box<dyn speech::Asr>,
    mut diarizer: Option<Diarizer>,
    tx: Backlog<Utterance>,
    writer: Arc<Mutex<Option<SessionWriter>>>,
    target: Arc<RwLock<String>>,
    sink: Arc<dyn Sink>,
) {
    let mut warning = DropWarning::new(
        "translate",
        "번역이 밀립니다: 설정에서 더 작은 번역 모델을 고르세요.",
    );
    // An utterance that leaves the full translation queue keeps its original
    // text in the record and gets a Failed translation.
    let mut on_drop = |mut u: Utterance| {
        log::warn!(
            "translation backlog full: utterance {} not translated",
            u.id
        );
        u.translation = TranslationState::Failed {
            message: "번역이 밀려 건너뛰었습니다.".into(),
        };
        sink.utterance(&u);
        append(&writer, &u, sink.as_ref());
        warning.dropped(u.source, sink.as_ref());
    };
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
            tx.push(u, &mut on_drop);
        }
    }
}

fn translate_loop(
    rx: BacklogRx<Utterance>,
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
            let result = translator.translate(&u.original, u.language.as_deref(), &target_now);
            if let Some(notice) = translator.take_notice() {
                sink.error("translate", None, &notice);
            }
            match result {
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
    use std::path::PathBuf;

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

    fn seg(i: u64) -> SpeechSegment {
        SpeechSegment {
            source: SourceKind::System,
            start_ms: i * 1000,
            end_ms: i * 1000 + 1000,
            samples: Vec::new(),
        }
    }

    #[test]
    fn a_full_backlog_drops_the_oldest_and_stays_bounded() {
        let (tx, rx) = backlog(5_000, segment_ms);
        let mut dropped = Vec::new();
        for i in 0..12 {
            tx.push(seg(i), &mut |s| dropped.push(s.start_ms / 1000));
            assert!(tx.queued_ms.load(Ordering::Relaxed) <= 5_000);
        }
        drop(tx);
        let kept: Vec<u64> = rx.iter().map(|s| s.start_ms / 1000).collect();
        assert_eq!(dropped, (0..7).collect::<Vec<_>>());
        assert_eq!(kept, (7..12).collect::<Vec<_>>());
        assert_eq!(rx.queued_ms.load(Ordering::Relaxed), 0);
    }

    /// A consumer slower than real time: the queue never holds more than the
    /// limit, every item is either processed or dropped, and the user is told.
    #[test]
    fn slow_asr_drops_work_and_warns_instead_of_growing() {
        let (tx, rx) = backlog(3_000, segment_ms);
        let consumer = std::thread::spawn(move || {
            let mut done = 0u64;
            for _ in rx.iter() {
                std::thread::sleep(Duration::from_millis(20)); // "slow ASR"
                done += 1;
            }
            done
        });
        let recorder = Recorder::default();
        let mut warning = DropWarning::new("asr", "인식이 밀립니다.");
        let mut dropped = 0u64;
        for i in 0..100 {
            tx.push(seg(i), &mut |s| {
                dropped += 1;
                warning.dropped(s.source, &recorder);
            });
            assert!(tx.queued_ms.load(Ordering::Relaxed) <= 3_000);
            std::thread::sleep(Duration::from_millis(1));
        }
        drop(tx);
        let done = consumer.join().unwrap();
        assert!(dropped > 0);
        assert_eq!(done + dropped, 100);
        let errors = recorder.errors.lock().unwrap();
        assert_eq!(errors.len(), 1, "one warning within 10 s: {errors:?}");
        assert!(errors[0].contains("인식이 밀립니다"), "{errors:?}");
    }

    /// Records engine output for tests.
    #[derive(Default)]
    struct Recorder {
        utterances: Mutex<Vec<Utterance>>,
        errors: Mutex<Vec<String>>,
    }

    impl Sink for Recorder {
        fn utterance(&self, u: &Utterance) {
            self.utterances.lock().unwrap().push(u.clone());
        }
        fn level(&self, _: SourceKind, _: f32) {}
        fn error(&self, stage: &str, _: Option<SourceKind>, message: &str) {
            self.errors
                .lock()
                .unwrap()
                .push(format!("{stage}: {message}"));
        }
    }

    /// End to end without audio devices: speech clips go into the pipeline as
    /// 100 ms chunks of two sources, then the saved session is read back.
    /// Run: MORAGO_TEST_MODELS=<models folder; missing models are downloaded>
    /// MORAGO_TEST_WAVS=<folder with en0.wav, sv_ko.wav, sv_ja.wav>
    /// LLAMA_LIB_PROFILE=Release cargo test --lib engine -- --ignored --nocapture
    /// CI runs it on Windows and macOS (job e2e, manual runs). On Linux it
    /// aborts at the first onnxruntime call when llama.cpp is linked too
    /// (std::regex clash, found 2026-11-08); Linux is not a target.
    #[test]
    #[ignore]
    fn pipeline_end_to_end() {
        use crate::settings::{AsrConfig, TranslateConfig};
        let models_dir = PathBuf::from(std::env::var("MORAGO_TEST_MODELS").unwrap());
        let ids = [models::SENSE_VOICE_ID, "gemma-4-e2b-it-q4"];
        let bundled = Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/models");
        for id in ids {
            if !models::is_installed(&models_dir, id) {
                let cancel = std::sync::atomic::AtomicBool::new(false);
                models::download(&models_dir, id, &|_| {}, &cancel).unwrap();
            }
        }
        let wavs = PathBuf::from(std::env::var("MORAGO_TEST_WAVS").unwrap());
        let read = |name: &str| {
            let path = wavs.join(format!("{name}.wav"));
            sherpa_onnx::Wave::read(path.to_str().unwrap())
                .unwrap()
                .samples()
                .to_vec()
        };
        let settings = Settings {
            asr: AsrConfig::Local {
                model_id: models::SENSE_VOICE_ID.into(),
            },
            translate: TranslateConfig::Local {
                model_id: "gemma-4-e2b-it-q4".into(),
            },
            ..Settings::default()
        };
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path());
        let recorder = Arc::new(Recorder::default());
        let sink: Arc<dyn Sink> = recorder.clone();
        let p = start_pipeline(
            &settings,
            Some("e2e".into()),
            &models_dir,
            &bundled,
            &store,
            &sink,
        )
        .unwrap();

        // System: English, then Japanese. Mic: Korean in the gap between them.
        let silence = vec![0.0f32; 16_000];
        let mut system = silence.clone();
        system.extend(read("en0"));
        system.extend(vec![0.0f32; 16_000 * 6]);
        system.extend(read("sv_ja"));
        system.extend(&silence);
        let mut mic = vec![0.0f32; 16_000 * 9];
        mic.extend(read("sv_ko"));
        mic.extend(vec![0.0f32; system.len().saturating_sub(mic.len())]);
        for (i, (s, m)) in system.chunks(1600).zip(mic.chunks(1600)).enumerate() {
            let start_ms = i as u64 * 100;
            for (source, samples) in [(SourceKind::System, s), (SourceKind::Mic, m)] {
                p.audio_tx
                    .send(AudioChunk {
                        source,
                        start_ms,
                        samples: samples.to_vec(),
                    })
                    .unwrap();
            }
        }
        let running = Running {
            meta: p.meta.clone(),
            target_language: p.target_language,
            captures: Vec::new(),
            audio_tx: Some(p.audio_tx),
            threads: p.threads,
            writer: p.writer,
            store_root: root.path().to_path_buf(),
        };
        let summary = stop(running).unwrap();
        let (meta, utterances) = store.load(&p.meta.id).unwrap();
        for u in &utterances {
            println!(
                "{:>6}-{:>6} ms {:?} {} {:?}: {} => {:?}",
                u.start_ms, u.end_ms, u.source, u.speaker, u.language, u.original, u.translation
            );
        }
        println!(
            "{}",
            store
                .export(&meta.id, crate::types::ExportFormat::Markdown)
                .unwrap()
        );
        println!("errors: {:?}", recorder.errors.lock().unwrap());
        assert!(recorder.errors.lock().unwrap().is_empty());
        assert!(summary.utterance_count >= 3, "{summary:?}");
        let lang = |l: &str| {
            utterances
                .iter()
                .find(|u| u.language.as_deref() == Some(l))
                .unwrap()
        };
        assert!(matches!(
            lang("en").translation,
            TranslationState::Done { .. }
        ));
        assert!(matches!(
            lang("ja").translation,
            TranslationState::Done { .. }
        ));
        assert_eq!(lang("ko").translation, TranslationState::SameLanguage);
        assert_eq!(lang("ko").source, SourceKind::Mic);
        assert!(
            lang("en").start_ms < lang("ko").start_ms && lang("ko").start_ms < lang("ja").start_ms
        );
        assert_ne!(lang("en").speaker, lang("ko").speaker);
        // Every UI event of an utterance ends in its stored state.
        let events = recorder.utterances.lock().unwrap();
        for u in &utterances {
            let last = events.iter().rev().find(|e| e.id == u.id).unwrap();
            assert_eq!(last, u);
        }
    }
}
