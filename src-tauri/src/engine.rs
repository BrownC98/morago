//! Session engine: capture -> VAD -> ASR + diarization -> translation
//! -> storage -> UI events. Each stage runs on its own thread and the
//! stages talk through channels, so a slow stage does not stop capture.

use crate::gpu::{self, GpuFailure, Placement};
use crate::settings::{AsrConfig, Settings, TranslateConfig};
use crate::speech::{self, Diarizer, Vad};
use crate::store::audio::AudioFile;
use crate::store::{AudioRecorder, SessionStore, SessionWriter};
use crate::translate::{self, needs_translation};
use crate::types::{
    AudioChunk, ModelDevice, PartialText, SessionMeta, SessionSummary, SourceKind, SpeechSegment,
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

mod partial;

/// Sends engine output to the UI. A trait so tests can run without Tauri.
pub trait Sink: Send + Sync + 'static {
    fn utterance(&self, u: &Utterance);
    fn level(&self, source: SourceKind, rms: f32);
    fn error(&self, stage: &str, source: Option<SourceKind>, message: &str);
    /// The session cannot go on (a GPU failure): stop and save it, then tell the user.
    fn fault(&self, message: &str);
    fn partial(&self, p: &PartialText);
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
    /// Empty: the error of this stage (and source) is over.
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
        if !message.is_empty() {
            log::warn!("engine {stage} error: {message}");
        }
        let _ = self.emit(
            "engine-error",
            ErrorEvent {
                stage,
                source,
                message,
            },
        );
    }
    fn fault(&self, message: &str) {
        log::error!("engine fault: {message}");
        crate::commands::stop_after_fault(self, message);
    }
}

/// A running session.
pub struct Running {
    pub meta: SessionMeta,
    pub target_language: Arc<RwLock<String>>,
    /// Device of a local translation model.
    pub translate_device: Option<ModelDevice>,
    /// Device of a local speech recognition model.
    pub asr_device: Option<ModelDevice>,
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
    translate_device: Option<ModelDevice>,
    asr_device: Option<ModelDevice>,
}

/// Devices of the local models of a session.
pub struct Placements {
    /// `None`: a service translates.
    pub translate: Option<Placement>,
    /// `None`: a service recognizes speech.
    pub asr: Option<Placement>,
}

/// The translation model and Fun-ASR go on the device of `place`
/// (the GPU setting, no memory check); the sherpa-onnx models on the CPU.
fn place_with(
    settings: &Settings,
    place: &dyn Fn() -> anyhow::Result<Placement>,
) -> anyhow::Result<Placements> {
    let translate = match &settings.translate {
        TranslateConfig::Local { .. } => Some(place()?),
        TranslateConfig::Custom { .. } => None,
    };
    let asr = match &settings.asr {
        AsrConfig::Local { model_id } if model_id == models::FUN_ASR_MLT_ID => Some(place()?),
        AsrConfig::Local { .. } => Some(Placement::cpu("이 모델은 CPU에서만 돌립니다.")),
        AsrConfig::Custom { .. } => None,
    };
    Ok(Placements { translate, asr })
}

/// Devices for a session start. An error (the chosen GPU is missing) stops it.
pub fn place_models(settings: &Settings) -> anyhow::Result<Placements> {
    place_with(settings, &|| gpu::place(&settings.gpu))
}

/// Devices for the settings rows: an error shows as the CPU reason.
pub fn model_devices(settings: &Settings) -> Placements {
    place_with(settings, &|| {
        Ok(gpu::place(&settings.gpu).unwrap_or_else(|e| Placement::cpu(e.to_string())))
    })
    .expect("no error")
}

/// Loads the models, creates the session file and starts the stage threads.
/// `partials`: run a second ASR for live partial text if the model and its
/// device allow it (`partial::enabled`).
fn start_pipeline(
    settings: &Settings,
    title: Option<String>,
    models_dir: &Path,
    bundled_dir: &Path,
    store: &SessionStore,
    sink: &Arc<dyn Sink>,
    partials: bool,
) -> anyhow::Result<Pipeline> {
    let Placements {
        translate: translate_placement,
        asr: asr_placement,
    } = place_models(settings)?;
    let cpu = || Placement::cpu("");
    let build_asr = |placement: &Placement| {
        speech::build_asr(
            &settings.asr,
            settings.meeting_language.as_deref(),
            models_dir,
            placement,
        )
    };
    let asr_place = asr_placement.clone().unwrap_or_else(cpu);
    let asr = build_asr(&asr_place)?;
    // On the device of the final ASR: Fun-ASR on a GPU opens a second session
    // of its model.
    let partial_asr = if partials && partial::enabled(&settings.asr, asr_place.gpu.is_some()) {
        Some(build_asr(&asr_place)?)
    } else {
        None
    };
    let translator = translate::build_translator(
        &settings.translate,
        settings.translate_context_lines as usize,
        models_dir,
        translate_placement.as_ref().unwrap_or(&cpu()),
    )?;
    let translate_device = translate_placement.map(|p| p.status);
    let asr_device = asr_placement.map(|p| p.status);
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
    // A recording that cannot start does not stop the meeting.
    let recorder = if settings.record_audio {
        store
            .create_audio(&meta.id)
            .map_err(|e| sink.error("store", None, &format!("녹음을 시작하지 못했습니다: {e:#}")))
            .ok()
    } else {
        None
    };
    let target_language = Arc::new(RwLock::new(settings.target_language.clone()));

    let (audio_tx, audio_rx) = bounded::<AudioChunk>(512);
    let (seg_tx, seg_rx) = backlog(BACKLOG_LIMIT_MS, segment_ms);
    let (tr_tx, tr_rx) = backlog(BACKLOG_LIMIT_MS, utterance_ms);

    let mut threads = Vec::new();
    let partials = partial_asr.is_some();
    let (feed, partial_translations) = match partial_asr {
        Some(asr) => {
            let (feed, translations, thread) = partial::start(
                asr,
                &settings.translate,
                translate_device.as_ref().is_some_and(|d| d.gpu),
                target_language.clone(),
                Arc::clone(sink),
            );
            threads.push(thread);
            (Some(feed), translations)
        }
        None => (None, None),
    };
    {
        let sink = Arc::clone(sink);
        threads.push(std::thread::spawn(move || {
            vad_loop(audio_rx, vads, seg_tx, sink, feed, recorder)
        }));
    }
    {
        let sink = Arc::clone(sink);
        let writer = writer.clone();
        let target = target_language.clone();
        threads.push(std::thread::spawn(move || {
            asr_loop(seg_rx, asr, diarizer, tr_tx, writer, target, sink, partials)
        }));
    }
    {
        let sink = Arc::clone(sink);
        let writer = writer.clone();
        let target = target_language.clone();
        threads.push(std::thread::spawn(move || {
            translate_loop(
                tr_rx,
                translator,
                writer,
                target,
                sink,
                partial_translations,
            )
        }));
    }

    Ok(Pipeline {
        meta,
        target_language,
        audio_tx,
        threads,
        writer,
        translate_device,
        asr_device,
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
        translate_device,
        asr_device,
    } = start_pipeline(
        settings,
        title,
        models_dir,
        bundled_dir,
        store,
        &sink,
        settings.live_partials,
    )?;

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
        translate_device,
        asr_device,
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
        self.rx.iter().map(|item| self.took(item))
    }

    /// Counts an item that left `rx` out of the queue.
    fn took(&self, item: T) -> T {
        take_ms(&self.queued_ms, (self.ms)(&item));
        item
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

/// Writes a chunk to the recording. A write error (a full disk) ends the
/// recording only and tells the user once; the meeting and its text go on.
fn record<F: AudioFile>(
    recorder: &mut Option<AudioRecorder<F>>,
    chunk: &AudioChunk,
    sink: &dyn Sink,
) {
    if let Some(r) = recorder.as_mut()
        && let Err(e) = r.push(chunk)
    {
        *recorder = None;
        recording_stopped(sink, &e);
    }
}

fn recording_stopped(sink: &dyn Sink, e: &std::io::Error) {
    sink.error(
        "store",
        None,
        &format!(
            "녹음을 멈췄습니다: {e}. 디스크 공간과 권한을 확인하세요. 발화 기록은 계속 저장합니다."
        ),
    );
}

/// `recorder`: the meeting recording; it gets every chunk before the VAD and
/// is closed when `rx` ends.
fn vad_loop(
    rx: Receiver<AudioChunk>,
    mut vads: Vec<(SourceKind, Vad)>,
    tx: Backlog<SpeechSegment>,
    sink: Arc<dyn Sink>,
    mut feed: Option<partial::Feed>,
    mut recorder: Option<AudioRecorder>,
) {
    let mut warning = DropWarning::new(
        "notice",
        "인식이 밀립니다: 설정에서 더 작은 음성 인식 모델을 고르세요.",
    );
    let partials = feed.is_some();
    let mut on_drop = |seg: SpeechSegment| {
        log::warn!(
            "ASR backlog full: dropped {} ms of speech",
            segment_ms(&seg)
        );
        if partials {
            partial::clear(sink.as_ref(), &seg);
        }
        warning.dropped(seg.source, sink.as_ref());
    };
    let mut last_level = [Instant::now(), Instant::now()];
    for chunk in rx.iter() {
        record(&mut recorder, &chunk, sink.as_ref());
        let i = chunk.source as usize;
        if last_level[i].elapsed() >= Duration::from_millis(100) {
            last_level[i] = Instant::now();
            sink.level(chunk.source, rms(&chunk.samples));
        }
        if let Some((_, vad)) = vads.iter_mut().find(|(s, _)| *s == chunk.source) {
            for seg in vad.push(&chunk) {
                if let Some(f) = feed.as_mut() {
                    f.ended(seg.source);
                }
                tx.push(seg, &mut on_drop);
            }
            if let Some(f) = feed.as_mut() {
                f.offer(&chunk, || vad.open_speech());
            }
        }
    }
    if let Some(r) = recorder.take()
        && let Err(e) = r.finish()
    {
        recording_stopped(sink.as_ref(), &e);
    }
    for (_, vad) in vads.iter_mut() {
        for seg in vad.flush() {
            if let Some(f) = feed.as_mut() {
                f.ended(seg.source);
            }
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

/// `partials`: the session shows live partial text, so a final without text
/// clears its partial line.
#[allow(clippy::too_many_arguments)]
fn asr_loop(
    rx: BacklogRx<SpeechSegment>,
    mut asr: Box<dyn speech::Asr>,
    mut diarizer: Option<Diarizer>,
    tx: Backlog<Utterance>,
    writer: Arc<Mutex<Option<SessionWriter>>>,
    target: Arc<RwLock<String>>,
    sink: Arc<dyn Sink>,
    partials: bool,
) {
    let clear = |seg: &SpeechSegment| {
        if partials {
            partial::clear(sink.as_ref(), seg);
        }
    };
    let mut warning = DropWarning::new(
        "notice",
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
    // After a GPU failure the model is not used again; the session stops.
    let mut gpu_failed = false;
    for seg in rx.iter() {
        if gpu_failed {
            continue;
        }
        let transcript = match asr.transcribe(&seg) {
            Ok(t) => t,
            Err(e) if e.is::<GpuFailure>() => {
                gpu_failed = true;
                sink.fault(&e.to_string());
                continue;
            }
            Err(e) => {
                sink.error("asr", Some(seg.source), &e.to_string());
                clear(&seg);
                continue;
            }
        };
        let text = transcript.text.trim();
        if text.is_empty() {
            clear(&seg);
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
    mut partials: Option<partial::Translations>,
) {
    // After a GPU failure the model is not used again; the session stops.
    let mut gpu_failed = false;
    while let Some(mut u) = partial::next_final(
        &rx,
        translator.as_mut(),
        &target,
        partials.as_mut(),
        sink.as_ref(),
    ) {
        if partials.as_ref().is_some_and(|p| p.gpu_failed) {
            gpu_failed = true;
            partials = None;
        }
        let target_now = target.read().expect("target lock").clone();
        u.translation = if !needs_translation(u.language.as_deref(), &target_now) {
            TranslationState::SameLanguage
        } else if gpu_failed {
            TranslationState::Failed {
                message: "GPU 오류로 번역하지 않았습니다.".into(),
            }
        } else {
            let result = translator.translate(&u.original, u.language.as_deref(), &target_now);
            if let Some(notice) = translator.take_notice() {
                // Stage "notice": the UI shows it as a toast that the next
                // good result does not clear (gate review of 61c5a17).
                sink.error("notice", None, &notice);
            }
            match result {
                Ok(text) => TranslationState::Done {
                    target: target_now,
                    text,
                },
                Err(e) if e.is::<GpuFailure>() => {
                    gpu_failed = true;
                    partials = None;
                    sink.fault(&e.to_string());
                    TranslationState::Failed {
                        message: e.to_string(),
                    }
                }
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
    use crate::types::GpuSetting;
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
    pub(super) struct Recorder {
        pub(super) utterances: Mutex<Vec<Utterance>>,
        /// The time of each event in `utterances`.
        pub(super) utterance_times: Mutex<Vec<Instant>>,
        pub(super) partials: Mutex<Vec<(Instant, PartialText)>>,
        pub(super) errors: Mutex<Vec<String>>,
    }

    impl Sink for Recorder {
        fn utterance(&self, u: &Utterance) {
            self.utterances.lock().unwrap().push(u.clone());
            self.utterance_times.lock().unwrap().push(Instant::now());
        }
        fn partial(&self, p: &PartialText) {
            self.partials
                .lock()
                .unwrap()
                .push((Instant::now(), p.clone()));
        }
        fn level(&self, _: SourceKind, _: f32) {}
        fn error(&self, stage: &str, _: Option<SourceKind>, message: &str) {
            self.errors
                .lock()
                .unwrap()
                .push(format!("{stage}: {message}"));
        }
        fn fault(&self, message: &str) {
            self.errors
                .lock()
                .unwrap()
                .push(format!("fault: {message}"));
        }
    }

    /// Fails with a GPU failure at the second call.
    struct BrokenGpu {
        calls: Arc<AtomicU64>,
    }

    impl translate::Translator for BrokenGpu {
        fn translate_partial(
            &mut self,
            _: &str,
            _: Option<&str>,
            _: &str,
            _: &dyn Fn() -> bool,
        ) -> anyhow::Result<String> {
            unreachable!("no partials in this test")
        }
        fn translate(&mut self, text: &str, _: Option<&str>, _: &str) -> anyhow::Result<String> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
                Ok(format!("번역 {text}"))
            } else {
                Err(GpuFailure("GPU 메모리 부족".into()).into())
            }
        }
        fn label(&self) -> String {
            "test".into()
        }
    }

    #[test]
    fn a_gpu_failure_stops_translation_once_and_keeps_the_originals() {
        let (tx, rx) = backlog(60_000, utterance_ms);
        let calls = Arc::new(AtomicU64::new(0));
        let recorder = Arc::new(Recorder::default());
        for id in 1..=4 {
            let u = Utterance {
                id,
                source: SourceKind::System,
                speaker: "S2".into(),
                start_ms: id * 1000,
                end_ms: id * 1000 + 500,
                language: Some("en".into()),
                original: format!("hello {id}"),
                translation: TranslationState::Pending,
            };
            tx.push(u, &mut |_| {});
        }
        drop(tx);
        translate_loop(
            rx,
            Box::new(BrokenGpu {
                calls: calls.clone(),
            }),
            Arc::new(Mutex::new(None)),
            Arc::new(RwLock::new("ko".into())),
            recorder.clone(),
            None,
        );
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(*recorder.errors.lock().unwrap(), ["fault: GPU 메모리 부족"]);
        let events = recorder.utterances.lock().unwrap();
        assert_eq!(events.len(), 4);
        assert!(matches!(
            events[0].translation,
            TranslationState::Done { .. }
        ));
        for u in &events[1..] {
            assert!(
                matches!(u.translation, TranslationState::Failed { .. }),
                "{u:?}"
            );
        }
        assert_eq!(events[3].original, "hello 4");
    }

    /// 100 ms chunks of both sources, `secs` long, as captures send them.
    fn two_source_chunks(secs: u64) -> Vec<AudioChunk> {
        let mut chunks = Vec::new();
        for ms in (0..secs * 1000).step_by(100) {
            for source in [SourceKind::Mic, SourceKind::System] {
                chunks.push(AudioChunk {
                    source,
                    start_ms: ms,
                    samples: vec![0.1; 1600],
                });
            }
        }
        chunks
    }

    fn test_meta(id: &str) -> SessionMeta {
        SessionMeta {
            id: id.into(),
            title: "녹음".into(),
            started_at: "2026-11-12T10:00:00+09:00".into(),
            ended_at: None,
            target_language: "ko".into(),
            asr_label: String::new(),
            translator_label: String::new(),
            speaker_names: Default::default(),
        }
    }

    #[test]
    fn the_vad_stage_records_the_meeting() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path());
        let id = "20261112-100000-ab12";
        let _writer = store.create(&test_meta(id)).unwrap();
        let recorder = Arc::new(Recorder::default());
        let (audio_tx, audio_rx) = unbounded();
        for c in two_source_chunks(3) {
            audio_tx.send(c).unwrap();
        }
        drop(audio_tx);
        let (seg_tx, _seg_rx) = backlog(BACKLOG_LIMIT_MS, segment_ms);
        vad_loop(
            audio_rx,
            Vec::new(),
            seg_tx,
            recorder.clone(),
            None,
            Some(store.create_audio(id).unwrap()),
        );
        assert!(recorder.errors.lock().unwrap().is_empty());
        let path = store.audio(id).unwrap().unwrap();
        let mut wav = hound::WavReader::open(path).unwrap();
        assert_eq!(wav.len(), 48_000);
        let two = (0.2f32 * 32767.0).round() as i16;
        assert!(wav.samples::<i16>().all(|s| s.unwrap() == two));
    }

    #[test]
    fn a_full_disk_stops_the_recording_once() {
        let recorder = Recorder::default();
        let mut audio = Some(AudioRecorder::full_disk(44 + 40_000));
        for c in two_source_chunks(5) {
            record(&mut audio, &c, &recorder);
        }
        assert!(audio.is_none());
        let errors = recorder.errors.lock().unwrap();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            errors[0].starts_with("store: 녹음을 멈췄습니다: disk full."),
            "{errors:?}"
        );
    }

    /// Fails with a GPU failure at the second call.
    struct BrokenGpuAsr {
        calls: Arc<AtomicU64>,
    }

    impl speech::Asr for BrokenGpuAsr {
        fn transcribe(&mut self, _: &SpeechSegment) -> anyhow::Result<crate::types::Transcript> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
                Ok(crate::types::Transcript {
                    text: "안녕하세요".into(),
                    language: Some("ko".into()),
                })
            } else {
                Err(GpuFailure("음성 인식 모델이 GPU에서 오류를 냈습니다.".into()).into())
            }
        }
        fn label(&self) -> String {
            "test".into()
        }
    }

    #[test]
    fn a_gpu_failure_stops_speech_recognition_once() {
        let (tx, rx) = backlog(60_000, segment_ms);
        for i in 0..4 {
            tx.push(seg(i), &mut |_| {});
        }
        drop(tx);
        let (tr_tx, tr_rx) = backlog(60_000, utterance_ms);
        let calls = Arc::new(AtomicU64::new(0));
        let recorder = Arc::new(Recorder::default());
        asr_loop(
            rx,
            Box::new(BrokenGpuAsr {
                calls: calls.clone(),
            }),
            None,
            tr_tx,
            Arc::new(Mutex::new(None)),
            Arc::new(RwLock::new("ko".into())),
            recorder.clone(),
            false,
        );
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(
            *recorder.errors.lock().unwrap(),
            ["fault: 음성 인식 모델이 GPU에서 오류를 냈습니다."]
        );
        assert_eq!(recorder.utterances.lock().unwrap().len(), 1);
        assert_eq!(tr_rx.rx.try_iter().count(), 0);
    }

    /// English text for the first three segments of `seg`, then no text, then an error.
    struct EnglishAsr;

    impl speech::Asr for EnglishAsr {
        fn transcribe(&mut self, seg: &SpeechSegment) -> anyhow::Result<crate::types::Transcript> {
            let text = match seg.start_ms {
                0..3000 => format!("hello {}", seg.start_ms),
                3000 => String::new(),
                _ => anyhow::bail!("ASR failed"),
            };
            Ok(crate::types::Transcript {
                text,
                language: Some("en".into()),
            })
        }
        fn label(&self) -> String {
            "test".into()
        }
    }

    struct FinalsOnly;

    impl translate::Translator for FinalsOnly {
        fn translate_partial(
            &mut self,
            _: &str,
            _: Option<&str>,
            _: &str,
            _: &dyn Fn() -> bool,
        ) -> anyhow::Result<String> {
            unreachable!("live partials are off")
        }
        fn translate(&mut self, text: &str, _: Option<&str>, _: &str) -> anyhow::Result<String> {
            Ok(format!("번역 {text}"))
        }
        fn label(&self) -> String {
            "test".into()
        }
    }

    #[test]
    fn with_live_partials_off_finals_come_and_no_partial() {
        // `start` gives `settings.live_partials` as `partials`; with it false,
        // `start_pipeline` starts these stages without a feed or partial translations.
        // No clear event either, for a final without text or an ASR error.
        let (tx, rx) = backlog(60_000, segment_ms);
        for i in 0..5 {
            tx.push(seg(i), &mut |_| {});
        }
        drop(tx);
        let (tr_tx, tr_rx) = backlog(60_000, utterance_ms);
        let recorder = Arc::new(Recorder::default());
        let target = Arc::new(RwLock::new("ko".to_string()));
        asr_loop(
            rx,
            Box::new(EnglishAsr),
            None,
            tr_tx,
            Arc::new(Mutex::new(None)),
            target.clone(),
            recorder.clone(),
            false,
        );
        translate_loop(
            tr_rx,
            Box::new(FinalsOnly),
            Arc::new(Mutex::new(None)),
            target,
            recorder.clone(),
            None,
        );
        assert!(recorder.partials.lock().unwrap().is_empty());
        let done: Vec<String> = recorder
            .utterances
            .lock()
            .unwrap()
            .iter()
            .filter_map(|u| match &u.translation {
                TranslationState::Done { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(done, ["번역 hello 0", "번역 hello 1000", "번역 hello 2000"]);
        assert_eq!(*recorder.errors.lock().unwrap(), ["asr: ASR failed"]);
    }

    fn local_settings(asr: &str) -> Settings {
        Settings {
            asr: AsrConfig::Local {
                model_id: asr.into(),
            },
            translate: TranslateConfig::Local {
                model_id: "gemma-4-e2b-it-q4".into(),
            },
            ..Settings::default()
        }
    }

    #[test]
    fn translation_and_fun_asr_follow_the_gpu_setting() {
        let gpu = || {
            Ok(Placement {
                gpu: Some((0, "GPU".into())),
                status: ModelDevice {
                    gpu: true,
                    detail: "GPU · CUDA".into(),
                },
            })
        };
        let p = place_with(&local_settings(models::FUN_ASR_MLT_ID), &gpu).unwrap();
        assert!(p.translate.unwrap().status.gpu);
        assert!(p.asr.unwrap().status.gpu);
        let p = place_with(&local_settings(models::QWEN3_ASR_ID), &gpu).unwrap();
        assert!(!p.asr.unwrap().status.gpu);
    }

    #[test]
    fn device_rows_of_the_local_models() {
        let off = Settings {
            gpu: GpuSetting::Off,
            ..local_settings(models::FUN_ASR_MLT_ID)
        };
        let p = model_devices(&off);
        for device in [p.translate.unwrap().status, p.asr.unwrap().status] {
            assert!(!device.gpu);
            assert_eq!(device.detail, "GPU 가속을 껐습니다.");
        }
        let sherpa = model_devices(&local_settings(models::SENSE_VOICE_ID));
        assert_eq!(
            sherpa.asr.unwrap().status.detail,
            "이 모델은 CPU에서만 돌립니다."
        );
        let missing = Settings {
            gpu: GpuSetting::Device {
                name: "No Such GPU 9000".into(),
            },
            ..local_settings(models::FUN_ASR_MLT_ID)
        };
        let p = model_devices(&missing);
        assert!(p.asr.unwrap().status.detail.contains("No Such GPU 9000"));
        assert!(place_models(&missing).is_err());
        let service = Settings {
            asr: AsrConfig::Custom {
                endpoint: crate::types::Endpoint {
                    base_url: "https://api.openai.com/v1".into(),
                    model: "whisper-1".into(),
                },
            },
            ..off
        };
        assert!(model_devices(&service).asr.is_none());
    }

    // Tests with real models. Run: MORAGO_TEST_MODELS=<models folder; missing
    // models are downloaded> MORAGO_TEST_WAVS=<folder with en0.wav, sv_ko.wav,
    // sv_ja.wav> LLAMA_LIB_PROFILE=Release cargo test --lib <test> -- --ignored --nocapture
    // CI runs `pipeline_end_to_end` on Windows and macOS (job e2e, manual runs).
    // On Linux they abort at the first onnxruntime call when llama.cpp is
    // linked too (std::regex clash, found 2026-11-08); Linux is not a target.

    fn models_dir() -> PathBuf {
        PathBuf::from(std::env::var("MORAGO_TEST_MODELS").unwrap())
    }

    fn ensure_model(models_dir: &Path, id: &str) {
        if !models::is_installed(models_dir, id) {
            let cancel = std::sync::atomic::AtomicBool::new(false);
            models::download(models_dir, id, &|_| {}, &cancel).unwrap();
        }
    }

    /// `MORAGO_TEST_ASR` (default SenseVoice Small) and Gemma 4 E2B,
    /// installed. The devices follow the 자동 setting (the GPU pack from
    /// `gpu::app_dir`); `MORAGO_TEST_GPU=off` runs both on the CPU.
    fn e2e_settings(models_dir: &Path) -> Settings {
        gpu::init(&gpu::app_dir().unwrap());
        let asr =
            std::env::var("MORAGO_TEST_ASR").unwrap_or_else(|_| models::SENSE_VOICE_ID.into());
        let mut settings = local_settings(&asr);
        if std::env::var("MORAGO_TEST_GPU").is_ok_and(|v| v == "off") {
            settings.gpu = GpuSetting::Off;
        }
        for id in required_models(&settings) {
            ensure_model(models_dir, &id);
        }
        settings
    }

    fn bundled_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/models")
    }

    fn read_wav(name: &str) -> Vec<f32> {
        let path =
            PathBuf::from(std::env::var("MORAGO_TEST_WAVS").unwrap()).join(format!("{name}.wav"));
        sherpa_onnx::Wave::read(path.to_str().unwrap())
            .unwrap()
            .samples()
            .to_vec()
    }

    /// 100 ms chunks of two sources. System: English, then Japanese. Mic:
    /// Korean in the gap between them.
    fn e2e_chunks() -> Vec<AudioChunk> {
        let silence = vec![0.0f32; 16_000];
        let mut system = silence.clone();
        system.extend(read_wav("en0"));
        system.extend(vec![0.0f32; 16_000 * 6]);
        system.extend(read_wav("sv_ja"));
        system.extend(&silence);
        let mut mic = vec![0.0f32; 16_000 * 9];
        mic.extend(read_wav("sv_ko"));
        mic.extend(vec![0.0f32; system.len().saturating_sub(mic.len())]);
        let mut chunks = Vec::new();
        for (i, (s, m)) in system.chunks(1600).zip(mic.chunks(1600)).enumerate() {
            for (source, samples) in [(SourceKind::System, s), (SourceKind::Mic, m)] {
                chunks.push(AudioChunk {
                    source,
                    start_ms: i as u64 * 100,
                    samples: samples.to_vec(),
                });
            }
        }
        chunks
    }

    /// Sends each chunk when its audio would be complete, as a capture does.
    /// Returns the time of audio 0.
    fn feed_in_real_time(p: &Pipeline, chunks: &[AudioChunk]) -> Instant {
        let t0 = Instant::now();
        for chunk in chunks {
            let due = t0 + Duration::from_millis(chunk.start_ms + 100);
            std::thread::sleep(due.saturating_duration_since(Instant::now()));
            p.audio_tx.send(chunk.clone()).unwrap();
        }
        t0
    }

    /// Stops the pipeline and loads the saved session.
    fn finish(
        p: Pipeline,
        root: &Path,
        store: &SessionStore,
    ) -> (SessionSummary, SessionMeta, Vec<Utterance>) {
        let running = Running {
            meta: p.meta.clone(),
            target_language: p.target_language,
            translate_device: p.translate_device,
            asr_device: p.asr_device,
            captures: Vec::new(),
            audio_tx: Some(p.audio_tx),
            threads: p.threads,
            writer: p.writer,
            store_root: root.to_path_buf(),
        };
        let summary = stop(running).unwrap();
        let (meta, utterances) = store.load(&p.meta.id).unwrap();
        (summary, meta, utterances)
    }

    /// Prints the partial translation lag lines of `engine::partial` and the
    /// stopped partial runs of `speech::funasr`.
    fn log_partials() {
        let _ = env_logger::builder()
            .is_test(true)
            .format_timestamp_millis()
            .filter_module("morago_lib::engine::partial", log::LevelFilter::Info)
            .filter_module("morago_lib::speech::funasr", log::LevelFilter::Debug)
            .try_init();
    }

    fn first_gpu() -> Option<usize> {
        llama_cpp_2::list_llama_ggml_backend_devices()
            .into_iter()
            .find(|d| matches!(d.device_type, llama_cpp_2::LlamaBackendDeviceType::Gpu))
            .map(|d| d.index)
    }

    /// Free memory of each GPU, MiB.
    fn free_gpu_mib() -> Vec<usize> {
        llama_cpp_2::list_llama_ggml_backend_devices()
            .into_iter()
            .filter(|d| matches!(d.device_type, llama_cpp_2::LlamaBackendDeviceType::Gpu))
            .map(|d| d.memory_free >> 20)
            .collect()
    }

    /// End to end without audio devices: speech clips go into the pipeline as
    /// 100 ms chunks of two sources in real time, with partials, then the
    /// saved session is read back.
    #[test]
    #[ignore]
    fn pipeline_end_to_end() {
        log_partials();
        let models_dir = models_dir();
        let mut settings = e2e_settings(&models_dir);
        settings.record_audio = true;
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path());
        let recorder = Arc::new(Recorder::default());
        let sink: Arc<dyn Sink> = recorder.clone();
        let before = free_gpu_mib();
        let started = Instant::now();
        let p = start_pipeline(
            &settings,
            Some("e2e".into()),
            &models_dir,
            &bundled_dir(),
            &store,
            &sink,
            true,
        )
        .unwrap();
        println!(
            "load {:.1?}; ASR {:?}; translation {:?}; free GPU memory {before:?} -> {:?} MiB",
            started.elapsed(),
            p.asr_device,
            p.translate_device,
            free_gpu_mib()
        );
        let chunks = e2e_chunks();
        feed_in_real_time(&p, &chunks);
        let (summary, meta, utterances) = finish(p, root.path(), &store);
        // The recording has the whole timeline of both sources.
        let audio_len = chunks
            .iter()
            .map(|c| c.start_ms * 16 + c.samples.len() as u64)
            .max()
            .unwrap();
        let wav = hound::WavReader::open(store.audio(&meta.id).unwrap().unwrap()).unwrap();
        assert_eq!(u64::from(wav.len()), audio_len);
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
        let shown = recorder.partials.lock().unwrap();
        println!(
            "partials: {}, with translation: {}",
            shown.len(),
            shown
                .iter()
                .filter(|(_, p)| p.translation.is_some())
                .count()
        );
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

    /// CPU time of this process (user and kernel).
    #[cfg(windows)]
    fn cpu_time() -> Option<Duration> {
        #[repr(C)]
        #[derive(Default)]
        struct FileTime {
            low: u32,
            high: u32,
        }
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn GetProcessTimes(
                process: *mut std::ffi::c_void,
                creation: *mut FileTime,
                exit: *mut FileTime,
                kernel: *mut FileTime,
                user: *mut FileTime,
            ) -> i32;
        }
        let mut t: [FileTime; 4] = Default::default();
        let [c, e, k, u] = &mut t;
        // SAFETY: four valid out pointers; the pseudo handle needs no close.
        let ok = unsafe { GetProcessTimes(GetCurrentProcess(), c, e, k, u) };
        let ns = |f: &FileTime| ((u64::from(f.high) << 32) | u64::from(f.low)) * 100;
        (ok != 0).then(|| Duration::from_nanos(ns(&t[2]) + ns(&t[3])))
    }

    #[cfg(not(windows))]
    fn cpu_time() -> Option<Duration> {
        None
    }

    /// The clips of `pipeline_end_to_end`, with partials on and off, twice in
    /// turn (a hot CPU runs slower). Prints per utterance:
    /// first-partial delay from the speech start, partial intervals, final
    /// latency (event time minus segment end, both in audio time) and the
    /// process CPU time per wall time. The `engine::partial` log lines give the
    /// partial translation lag.
    #[test]
    #[ignore]
    fn pipeline_realtime() {
        log_partials();
        let models_dir = models_dir();
        let settings = e2e_settings(&models_dir);
        let chunks = e2e_chunks();
        for partials in [true, false, true, false] {
            let root = tempfile::tempdir().unwrap();
            let store = SessionStore::new(root.path());
            let recorder = Arc::new(Recorder::default());
            let sink: Arc<dyn Sink> = recorder.clone();
            let p = start_pipeline(
                &settings,
                None,
                &models_dir,
                &bundled_dir(),
                &store,
                &sink,
                partials,
            )
            .unwrap();
            println!(
                "ASR {:?}; translation {:?}; free GPU memory {:?} MiB",
                p.asr_device,
                p.translate_device,
                free_gpu_mib()
            );
            // Fun-ASR gives partials on a GPU only.
            let on_gpu = p.asr_device.as_ref().is_some_and(|d| d.gpu);
            let partials = partials && partial::enabled(&settings.asr, on_gpu);
            let cpu = cpu_time();
            let gpu = first_gpu();
            let (t0, lowest) = gpu::lowest_free_while(gpu, || feed_in_real_time(&p, &chunks));
            let (_, _, utterances) = finish(p, root.path(), &store);
            let wall = t0.elapsed();
            println!("lowest free GPU memory during the audio: {lowest:?} MiB");
            let cpu = cpu.zip(cpu_time()).map(|(a, b)| b - a);
            println!(
                "=== partials {}: wall {wall:.1?}, CPU {cpu:.1?}, CPU/wall {}",
                if partials { "on" } else { "off" },
                cpu.map_or("n/a".into(), |c| format!(
                    "{:.2}",
                    c.as_secs_f64() / wall.as_secs_f64()
                ))
            );
            // Audio time of an event: audio up to `ms` is in the pipeline at t0 + ms.
            let ms = |t: &Instant| t.saturating_duration_since(t0).as_millis() as i64;
            let events = recorder.utterances.lock().unwrap();
            let times = recorder.utterance_times.lock().unwrap();
            let shown = recorder.partials.lock().unwrap();
            for u in &utterances {
                let of_u = |e: &&(usize, &Utterance)| e.1.id == u.id;
                let all: Vec<(usize, &Utterance)> = events.iter().enumerate().collect();
                let first = all.iter().find(of_u).map(|(i, _)| ms(&times[*i])).unwrap();
                let done = all
                    .iter()
                    .filter(of_u)
                    .find(|(_, e)| matches!(e.translation, TranslationState::Done { .. }))
                    .map(|(i, _)| ms(&times[*i]) - u.end_ms as i64);
                let mine: Vec<(i64, &PartialText)> = shown
                    .iter()
                    .filter(|(_, p)| {
                        p.source == u.source
                            && !p.text.is_empty()
                            && p.start_ms < u.end_ms
                            && p.start_ms + 2_000 > u.start_ms
                    })
                    .map(|(t, p)| (ms(t), p))
                    .collect();
                let mut texts: Vec<i64> = Vec::new();
                let mut translations: Vec<i64> = Vec::new();
                let mut last: Option<&PartialText> = None;
                for (t, p) in &mine {
                    if last.is_none_or(|l| l.text != p.text) {
                        texts.push(*t);
                    }
                    if p.translation.is_some()
                        && last.is_none_or(|l| l.translation != p.translation)
                    {
                        translations.push(*t);
                    }
                    last = Some(p);
                }
                let gaps: Vec<i64> = texts.windows(2).map(|w| w[1] - w[0]).collect();
                println!(
                    "{:?} {:>5}-{:>5} ms {:?}: final +{} ms, translation {:?} ms; partials {} (first +{:?} ms from start), gaps {gaps:?} ms; partial translations {}",
                    u.source,
                    u.start_ms,
                    u.end_ms,
                    u.language,
                    first - u.end_ms as i64,
                    done,
                    texts.len(),
                    texts.first().map(|t| t - u.start_ms as i64),
                    translations.len(),
                );
                if let Some((_, p)) = mine.last() {
                    println!("  last partial: {} => {:?}", p.text, p.translation);
                }
                println!("  final: {} => {:?}", u.original, u.translation);
                // No partial of a speech after its final.
                assert!(mine.iter().all(|(t, _)| *t <= first), "{mine:?}");
                if partials {
                    assert!(!texts.is_empty(), "no partial for {}", u.original);
                } else {
                    assert!(mine.is_empty(), "{mine:?}");
                }
            }
            assert!(recorder.errors.lock().unwrap().is_empty());
        }
    }

    /// Partial-run speed of the local ASR models on this CPU: the first 1, 2,
    /// 4, 6, 10 and 15 s of the clips; RTF = run time / audio time (median of 3).
    /// Run like `pipeline_end_to_end`; `MORAGO_TEST_ASR=<ids, comma>` (default
    /// SenseVoice Small and Qwen3-ASR).
    #[test]
    #[ignore]
    fn partial_run_rtf() {
        let models_dir = models_dir();
        let ids = std::env::var("MORAGO_TEST_ASR")
            .unwrap_or_else(|_| format!("{},{}", models::SENSE_VOICE_ID, models::QWEN3_ASR_ID));
        let clips = [read_wav("en0"), read_wav("sv_ja"), read_wav("sv_ko")].concat();
        for id in ids.split(',') {
            ensure_model(&models_dir, id);
            let config = crate::settings::AsrConfig::Local {
                model_id: id.into(),
            };
            let mut asr = speech::build_asr(
                &config,
                models::only_language(id),
                &models_dir,
                &Placement::cpu(""),
            )
            .unwrap();
            for secs in [1u64, 2, 4, 6, 10, 15] {
                let seg = SpeechSegment {
                    source: SourceKind::System,
                    start_ms: 0,
                    end_ms: secs * 1000,
                    samples: clips[..secs as usize * 16_000].to_vec(),
                };
                asr.transcribe(&seg).unwrap();
                let mut runs: Vec<Duration> = (0..3)
                    .map(|_| {
                        let started = Instant::now();
                        asr.transcribe(&seg).unwrap();
                        started.elapsed()
                    })
                    .collect();
                runs.sort();
                println!(
                    "{id} {secs:>2} s: {:.0?}, RTF {:.3}",
                    runs[1],
                    runs[1].as_secs_f64() / secs as f64
                );
            }
        }
    }
}
