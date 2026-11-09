//! Live partial text. While speech goes on, a second ASR instance
//! transcribes the audio so far, and the translator translates the newest
//! partial text when no final utterance waits.
//!
//! No partial of a speech comes after its final: `vad_loop` ends the speech
//! (`Feed::ended`) before it queues the segment, and partial events go out
//! under the same mutex, only for the speech in progress.

use super::{BacklogRx, Sink};
use crate::models;
use crate::settings::{AsrConfig, TranslateConfig};
use crate::speech;
use crate::translate::{self, Translator, needs_translation};
use crate::types::{AudioChunk, PartialText, SAMPLE_RATE, SourceKind, SpeechSegment, Utterance};
use crossbeam_channel::{Receiver, Sender, TryRecvError, after, bounded, never, select};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Audio time between two partial ASR runs of one speech.
const OFFER_STEP_MS: u64 = 700;

/// True when the ASR is fast enough on the CPU to transcribe the growing
/// audio of each speech again every 0.7 s. RTF on 1 to 15 s of audio
/// (Windows, 4 threads, 2026-11-09):
/// - SenseVoice Small: yes, RTF 0.024 to 0.054.
/// - Qwen3-ASR 0.6B: no, RTF 0.29 to 0.62 (2 s per run at 6 s of audio).
/// - Whisper: no, RTF 0.35 on the CPU (whisper.cpp): the CPU would never rest.
/// - Custom: no, API cost.
pub(super) fn enabled(asr: &AsrConfig) -> bool {
    matches!(asr, AsrConfig::Local { model_id } if model_id == models::SENSE_VOICE_ID)
}

/// Least time between two partial translation requests of one speech;
/// `None`: no partial translation.
/// A local model translates partials only on a GPU. On the CPU they made the
/// final translations 0.9 to 2.5 s slower (Windows, 16 threads, Gemma 4 E2B,
/// 2026-11-09), also of a final with no partial work beside it.
fn translate_interval(config: &TranslateConfig) -> Option<Duration> {
    match config {
        TranslateConfig::Local { .. } => translate::local_on_gpu().then_some(Duration::ZERO),
        TranslateConfig::Custom {
            partial_interval_ms,
            ..
        } => partial_interval_ms.map(|ms| Duration::from_millis(ms.into())),
    }
}

/// Removes the partial line of a speech whose final has no text: empty text,
/// an ASR error or a dropped segment.
pub(super) fn clear(sink: &dyn Sink, seg: &SpeechSegment) {
    sink.partial(&PartialText {
        source: seg.source,
        start_ms: seg.end_ms,
        text: String::new(),
        translation: None,
    });
}

/// The speech in progress of each source.
#[derive(Default)]
struct Live(Mutex<[Speech; 2]>);

#[derive(Default)]
struct Speech {
    /// Goes up by one when a speech ends.
    generation: u64,
    /// The last partial event of the speech in progress.
    shown: Option<PartialText>,
}

impl Live {
    fn speech(&self) -> MutexGuard<'_, [Speech; 2]> {
        self.0.lock().expect("live lock")
    }

    fn generation(&self, source: SourceKind) -> u64 {
        self.speech()[source as usize].generation
    }

    fn end(&self, source: SourceKind) {
        let mut all = self.speech();
        let s = &mut all[source as usize];
        s.generation += 1;
        s.shown = None;
    }

    /// Shows a partial original; false when its speech has ended.
    fn original(
        &self,
        generation: u64,
        audio: &SpeechSegment,
        text: &str,
        sink: &dyn Sink,
    ) -> bool {
        let mut all = self.speech();
        let s = &mut all[audio.source as usize];
        if s.generation != generation {
            return false;
        }
        let p = PartialText {
            source: audio.source,
            start_ms: audio.start_ms,
            text: text.to_string(),
            translation: s.shown.take().and_then(|p| p.translation),
        };
        sink.partial(&p);
        s.shown = Some(p);
        true
    }

    /// Shows a partial translation with the newest original, if its speech goes on.
    fn translation(&self, request: &Request, text: String, sink: &dyn Sink) {
        let mut all = self.speech();
        let s = &mut all[request.source as usize];
        if s.generation != request.generation {
            return;
        }
        if let Some(p) = s.shown.as_mut() {
            p.translation = Some(text);
            sink.partial(p);
        }
    }
}

/// The open audio of one speech for the worker.
struct Job {
    generation: u64,
    audio: SpeechSegment,
}

/// Audio end of the last offer that the worker took, per source.
#[derive(Default)]
struct Schedule([Option<u64>; 2]);

impl Schedule {
    fn due(&self, source: SourceKind, end_ms: u64) -> bool {
        self.0[source as usize].is_none_or(|last| end_ms >= last + OFFER_STEP_MS)
    }

    fn accepted(&mut self, source: SourceKind, end_ms: u64) {
        self.0[source as usize] = Some(end_ms);
    }

    fn ended(&mut self, source: SourceKind) {
        self.0[source as usize] = None;
    }
}

/// The `vad_loop` side.
pub(super) struct Feed {
    live: Arc<Live>,
    /// Zero capacity: a send succeeds only when the worker waits.
    jobs: Sender<Job>,
    schedule: Schedule,
}

impl Feed {
    /// Ends the speech of `source`. Call before its segment goes to the ASR queue.
    pub(super) fn ended(&mut self, source: SourceKind) {
        self.live.end(source);
        self.schedule.ended(source);
    }

    /// Call after each chunk. Gives the open audio to the worker when it has
    /// grown by `OFFER_STEP_MS` since the last run (the first at detection).
    /// Never blocks: when the worker is busy, the next chunk offers again.
    pub(super) fn offer(
        &mut self,
        chunk: &AudioChunk,
        open: impl FnOnce() -> Option<SpeechSegment>,
    ) {
        let end_ms = chunk.start_ms + chunk.samples.len() as u64 * 1000 / SAMPLE_RATE as u64;
        if !self.schedule.due(chunk.source, end_ms) {
            return;
        }
        let Some(audio) = open() else {
            return;
        };
        let generation = self.live.generation(chunk.source);
        if self.jobs.try_send(Job { generation, audio }).is_ok() {
            self.schedule.accepted(chunk.source, end_ms);
        }
    }
}

/// A partial translation request.
struct Request {
    source: SourceKind,
    generation: u64,
    text: String,
    language: Option<String>,
    /// When the original was shown.
    shown_at: Instant,
}

/// The newest partial translation request. A new one replaces one that has not started.
struct Slot {
    request: Mutex<Option<Request>>,
    wake: Sender<()>,
}

impl Slot {
    fn put(&self, request: Request) {
        *self.request.lock().expect("slot lock") = Some(request);
        let _ = self.wake.try_send(());
    }
}

/// Starts the partial ASR thread. Returns the side for `vad_loop`, the side
/// for `translate_loop` (`None` when partial translation is off) and the thread.
pub(super) fn start(
    asr: Box<dyn speech::Asr>,
    translate: &TranslateConfig,
    target: Arc<RwLock<String>>,
    sink: Arc<dyn Sink>,
) -> (Feed, Option<Translations>, JoinHandle<()>) {
    let live = Arc::new(Live::default());
    let (jobs, jobs_rx) = bounded(0);
    let translations =
        translate_interval(translate).map(|interval| Translations::new(&live, interval));
    let slot = translations.as_ref().map(|t| t.slot.clone());
    let thread = {
        let live = live.clone();
        std::thread::spawn(move || worker(jobs_rx, asr, &live, slot, &target, sink.as_ref()))
    };
    let feed = Feed {
        live,
        jobs,
        schedule: Schedule::default(),
    };
    (feed, translations, thread)
}

fn worker(
    jobs: Receiver<Job>,
    mut asr: Box<dyn speech::Asr>,
    live: &Live,
    slot: Option<Arc<Slot>>,
    target: &RwLock<String>,
    sink: &dyn Sink,
) {
    for job in jobs.iter() {
        // The final ASR reports errors; a partial run only skips.
        let transcript = match asr.transcribe(&job.audio) {
            Ok(t) => t,
            Err(e) => {
                log::debug!("partial ASR: {e}");
                continue;
            }
        };
        let text = transcript.text.trim();
        if text.is_empty() || !live.original(job.generation, &job.audio, text, sink) {
            continue;
        }
        let target_now = target.read().expect("target lock").clone();
        if let Some(slot) = &slot
            && needs_translation(transcript.language.as_deref(), &target_now)
        {
            slot.put(Request {
                source: job.audio.source,
                generation: job.generation,
                text: text.to_string(),
                language: transcript.language,
                shown_at: Instant::now(),
            });
        }
    }
}

/// The `translate_loop` side.
pub(super) struct Translations {
    slot: Arc<Slot>,
    wake: Receiver<()>,
    live: Arc<Live>,
    interval: Duration,
    /// Speech generation and start time of the last request, per source.
    last: [Option<(u64, Instant)>; 2],
}

enum Next {
    Run(Request),
    /// A request waits for the interval.
    Wait(Duration),
    Idle,
}

impl Translations {
    fn new(live: &Arc<Live>, interval: Duration) -> Self {
        let (wake_tx, wake) = bounded(1);
        Self {
            slot: Arc::new(Slot {
                request: Mutex::new(None),
                wake: wake_tx,
            }),
            wake,
            live: live.clone(),
            interval,
            last: [None; 2],
        }
    }

    fn next(&mut self) -> Next {
        let mut slot = self.slot.request.lock().expect("slot lock");
        let Some((source, generation)) = slot.as_ref().map(|r| (r.source, r.generation)) else {
            return Next::Idle;
        };
        if self.live.generation(source) != generation {
            *slot = None;
            return Next::Idle;
        }
        if let Some((last, at)) = self.last[source as usize]
            && last == generation
            && at.elapsed() < self.interval
        {
            return Next::Wait(self.interval - at.elapsed().min(self.interval));
        }
        self.last[source as usize] = Some((generation, Instant::now()));
        slot.take().map_or(Next::Idle, Next::Run)
    }

    /// Stops when a final waits or when the speech of the request ends.
    fn run(
        &self,
        request: Request,
        translator: &mut dyn Translator,
        target: &RwLock<String>,
        final_waits: &dyn Fn() -> bool,
        sink: &dyn Sink,
    ) {
        let target_now = target.read().expect("target lock").clone();
        if !needs_translation(request.language.as_deref(), &target_now) {
            return;
        }
        let started = Instant::now();
        let stop = || final_waits() || self.live.generation(request.source) != request.generation;
        match translator.translate_partial(
            &request.text,
            request.language.as_deref(),
            &target_now,
            &stop,
        ) {
            Ok(text) => {
                log::info!(
                    "partial translation: {} ms after its original, {} ms to translate",
                    request.shown_at.elapsed().as_millis(),
                    started.elapsed().as_millis()
                );
                self.live.translation(&request, text, sink);
            }
            Err(e) => log::debug!("partial translation: {e}"),
        }
    }
}

/// The next final utterance, or `None` at the end. While none waits, runs
/// the newest partial translation. Finals always go first: a partial
/// translation stops when a final comes into the queue.
pub(super) fn next_final(
    rx: &BacklogRx<Utterance>,
    translator: &mut dyn Translator,
    target: &RwLock<String>,
    partials: Option<&mut Translations>,
    sink: &dyn Sink,
) -> Option<Utterance> {
    let Some(p) = partials else {
        return rx.iter().next();
    };
    loop {
        match rx.rx.try_recv() {
            Ok(u) => return Some(rx.took(u)),
            Err(TryRecvError::Disconnected) => return None,
            Err(TryRecvError::Empty) => {}
        }
        let timer = match p.next() {
            Next::Run(request) => {
                p.run(request, translator, target, &|| !rx.rx.is_empty(), sink);
                continue;
            }
            Next::Wait(d) => after(d),
            Next::Idle => never(),
        };
        select! {
            recv(rx.rx) -> u => return u.ok().map(|u| rx.took(u)),
            recv(p.wake) -> _ => {}
            recv(timer) -> _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::Recorder;
    use super::super::{asr_loop, backlog, segment_ms, translate_loop, utterance_ms};
    use super::*;
    use crate::types::{Transcript, TranslationState};
    use std::sync::atomic::{AtomicU64, Ordering};

    const MIC: SourceKind = SourceKind::Mic;

    fn audio(start_ms: u64, end_ms: u64) -> SpeechSegment {
        SpeechSegment {
            source: MIC,
            start_ms,
            end_ms,
            samples: vec![0.0; ((end_ms - start_ms) * 16) as usize],
        }
    }

    fn chunk(start_ms: u64) -> AudioChunk {
        AudioChunk {
            source: MIC,
            start_ms,
            samples: vec![0.0; 1600],
        }
    }

    /// Gives the text of its function.
    struct FakeAsr(fn(&SpeechSegment) -> anyhow::Result<String>);

    impl speech::Asr for FakeAsr {
        fn transcribe(&mut self, seg: &SpeechSegment) -> anyhow::Result<Transcript> {
            Ok(Transcript {
                text: (self.0)(seg)?,
                language: Some("en".into()),
            })
        }
        fn label(&self) -> String {
            "fake".into()
        }
    }

    /// The audio length as text.
    fn length(seg: &SpeechSegment) -> anyhow::Result<String> {
        Ok(format!("{} ms", seg.end_ms - seg.start_ms))
    }

    fn custom(partial_interval_ms: Option<u32>) -> TranslateConfig {
        TranslateConfig::Custom {
            endpoint: crate::types::Endpoint {
                base_url: "http://x/v1".into(),
                model: String::new(),
            },
            extra_body: serde_json::json!({"model": "m"})
                .as_object()
                .unwrap()
                .clone(),
            partial_interval_ms,
        }
    }

    fn local_translate() -> TranslateConfig {
        TranslateConfig::Local {
            model_id: "gemma-4-e2b-it-q4".into(),
        }
    }

    #[test]
    fn only_sense_voice_gives_partials() {
        let local = |id: &str| AsrConfig::Local {
            model_id: id.into(),
        };
        assert!(enabled(&local(models::SENSE_VOICE_ID)));
        for id in [
            models::QWEN3_ASR_ID,
            "whisper-base",
            "whisper-small",
            "whisper-turbo",
        ] {
            assert!(!enabled(&local(id)), "{id}");
        }
        let custom = AsrConfig::Custom {
            endpoint: crate::types::Endpoint {
                base_url: "http://x/v1".into(),
                model: "whisper-1".into(),
            },
        };
        assert!(!enabled(&custom));
    }

    #[test]
    fn partial_translation_interval_per_translator() {
        assert_eq!(
            translate_interval(&local_translate()),
            translate::local_on_gpu().then_some(Duration::ZERO)
        );
        assert_eq!(
            translate_interval(&custom(Some(1500))),
            Some(Duration::from_millis(1500))
        );
        // "끔": no partial translation requests.
        assert_eq!(translate_interval(&custom(None)), None);
        // 0: a request for each partial text, still one at a time.
        assert_eq!(translate_interval(&custom(Some(0))), Some(Duration::ZERO));
        assert_eq!(
            translate_interval(&custom(Some(u32::MAX))),
            Some(Duration::from_millis(u32::MAX.into()))
        );
    }

    #[test]
    fn schedule_offers_at_detection_then_every_700_ms_of_audio() {
        let mut s = Schedule::default();
        assert!(s.due(MIC, 300));
        s.accepted(MIC, 300);
        assert!(!s.due(MIC, 900));
        assert!(s.due(MIC, 1000));
        assert!(s.due(SourceKind::System, 900));
        s.ended(MIC);
        assert!(s.due(MIC, 1100));
    }

    /// Offers chunk `i` of open audio. When the offer is due, offers again
    /// until the worker takes it: an idle worker can be not yet at its `recv`.
    fn offer_taken(feed: &mut Feed, i: u64) {
        let end_ms = i * 100 + 100;
        let deadline = Instant::now() + Duration::from_secs(10);
        while feed.schedule.due(MIC, end_ms) {
            assert!(Instant::now() < deadline, "the worker took no job");
            feed.offer(&chunk(i * 100), || Some(audio(0, end_ms)));
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Runs when the test sends on `go` or drops it.
    struct GatedAsr {
        go: Receiver<()>,
        runs: Arc<AtomicU64>,
    }

    impl speech::Asr for GatedAsr {
        fn transcribe(&mut self, seg: &SpeechSegment) -> anyhow::Result<Transcript> {
            self.runs.fetch_add(1, Ordering::Relaxed);
            let _ = self.go.recv();
            Ok(Transcript {
                text: length(seg)?,
                language: Some("en".into()),
            })
        }
        fn label(&self) -> String {
            "gated".into()
        }
    }

    fn gated_asr() -> (Box<dyn speech::Asr>, Sender<()>, Arc<AtomicU64>) {
        let (go_tx, go) = bounded(0);
        let runs = Arc::new(AtomicU64::new(0));
        let asr = GatedAsr {
            go,
            runs: runs.clone(),
        };
        (Box::new(asr), go_tx, runs)
    }

    fn wait_runs(runs: &AtomicU64, n: u64) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while runs.load(Ordering::Relaxed) < n {
            assert!(Instant::now() < deadline, "no run");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn a_partial_is_shown_during_speech() {
        let recorder = Arc::new(Recorder::default());
        let asr = Box::new(FakeAsr(length));
        let target = Arc::new(RwLock::new("ko".to_string()));
        let (mut feed, _, thread) = start(asr, &custom(None), target, recorder.clone());
        for i in 0..15 {
            offer_taken(&mut feed, i);
        }
        drop(feed);
        thread.join().unwrap();
        let texts: Vec<String> = recorder
            .partials
            .lock()
            .unwrap()
            .iter()
            .map(|(_, p)| p.text.clone())
            .collect();
        // Audio time 0.1 s, 0.8 s, 1.5 s: once at detection, then every 0.7 s.
        assert_eq!(texts, ["100 ms", "800 ms", "1500 ms"]);
    }

    #[test]
    fn a_busy_worker_skips_runs_and_offer_never_blocks() {
        let recorder = Arc::new(Recorder::default());
        let (asr, go, runs) = gated_asr();
        let target = Arc::new(RwLock::new("ko".to_string()));
        let (mut feed, _, thread) = start(asr, &custom(None), target, recorder.clone());
        offer_taken(&mut feed, 0);
        wait_runs(&runs, 1);
        // The worker is busy: the offers due at 0.8 s ... 4.2 s of audio are skipped.
        let mut longest = Duration::ZERO;
        for i in 1..42 {
            let call = Instant::now();
            feed.offer(&chunk(i * 100), || Some(audio(0, i * 100 + 100)));
            longest = longest.max(call.elapsed());
        }
        println!("longest offer {longest:?}");
        assert!(longest < Duration::from_millis(50), "{longest:?}");
        go.send(()).unwrap();
        // Free again: the next chunk runs, with all audio so far.
        offer_taken(&mut feed, 42);
        drop(go);
        drop(feed);
        thread.join().unwrap();
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        let texts: Vec<String> = recorder
            .partials
            .lock()
            .unwrap()
            .iter()
            .map(|(_, p)| p.text.clone())
            .collect();
        assert_eq!(texts, ["100 ms", "4300 ms"]);
    }

    #[test]
    fn no_partial_after_the_end_of_its_speech() {
        let recorder = Arc::new(Recorder::default());
        let (asr, go, runs) = gated_asr();
        let target = Arc::new(RwLock::new("ko".to_string()));
        let (mut feed, translations, thread) =
            start(asr, &custom(Some(1500)), target, recorder.clone());
        offer_taken(&mut feed, 0);
        wait_runs(&runs, 1);
        // The segment ends while the worker runs.
        feed.ended(MIC);
        drop(go);
        drop(feed);
        thread.join().unwrap();
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert!(recorder.partials.lock().unwrap().is_empty());
        let request = translations.unwrap().slot.request.lock().unwrap().take();
        assert!(
            request.is_none(),
            "no translation request for an ended speech"
        );
    }

    #[test]
    fn a_final_without_text_clears_the_partial_line() {
        let replies: [fn(&SpeechSegment) -> anyhow::Result<String>; 2] =
            [|_| Ok("   ".into()), |_| Err(anyhow::anyhow!("ASR failed"))];
        for reply in replies {
            let recorder = Arc::new(Recorder::default());
            let (seg_tx, seg_rx) = backlog(60_000, segment_ms);
            let (tr_tx, _tr_rx) = backlog(60_000, utterance_ms);
            seg_tx.push(audio(1000, 2500), &mut |_| {});
            drop(seg_tx);
            let asr = FakeAsr(reply);
            asr_loop(
                seg_rx,
                Box::new(asr),
                None,
                tr_tx,
                Arc::new(Mutex::new(None)),
                Arc::new(RwLock::new("ko".into())),
                recorder.clone(),
            );
            let partials = recorder.partials.lock().unwrap();
            assert_eq!(partials.len(), 1);
            let p = &partials[0].1;
            assert_eq!((p.text.as_str(), p.start_ms, p.source), ("", 2500, MIC));
            assert!(recorder.utterances.lock().unwrap().is_empty());
        }
    }

    /// Partial translation runs until `stop` or `partial_ms`; records the calls.
    #[derive(Default)]
    struct FakeTranslator {
        partial_ms: u64,
        /// Partial calls started.
        starts: Arc<AtomicU64>,
        /// (text, start, stopped) of each partial call that returned.
        partials: Arc<Mutex<Vec<(String, Instant, bool)>>>,
    }

    impl Translator for FakeTranslator {
        fn translate(&mut self, text: &str, _: Option<&str>, _: &str) -> anyhow::Result<String> {
            Ok(format!("번역 {text}"))
        }
        fn translate_partial(
            &mut self,
            text: &str,
            _: Option<&str>,
            _: &str,
            stop: &dyn Fn() -> bool,
        ) -> anyhow::Result<String> {
            self.starts.fetch_add(1, Ordering::Relaxed);
            let started = Instant::now();
            while started.elapsed() < Duration::from_millis(self.partial_ms) {
                if stop() {
                    self.partials
                        .lock()
                        .unwrap()
                        .push((text.into(), started, true));
                    anyhow::bail!("stopped");
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            self.partials
                .lock()
                .unwrap()
                .push((text.into(), started, false));
            Ok(format!("중간 {text}"))
        }
        fn label(&self) -> String {
            "fake".into()
        }
    }

    fn final_utterance(id: u64) -> Utterance {
        Utterance {
            id,
            source: MIC,
            speaker: "S1".into(),
            start_ms: 0,
            end_ms: 2000,
            language: Some("en".into()),
            original: format!("final {id}"),
            translation: TranslationState::Pending,
        }
    }

    /// Shows the original `text` of the speech in progress and asks to translate it.
    fn ask(live: &Live, slot: &Slot, text: &str, sink: &dyn Sink) {
        let generation = live.generation(MIC);
        assert!(live.original(generation, &audio(0, 1000), text, sink));
        slot.put(Request {
            source: MIC,
            generation,
            text: text.into(),
            language: Some("en".into()),
            shown_at: Instant::now(),
        });
    }

    struct Loop {
        tx: Option<super::super::Backlog<Utterance>>,
        thread: JoinHandle<()>,
        recorder: Arc<Recorder>,
        starts: Arc<AtomicU64>,
        calls: Arc<Mutex<Vec<(String, Instant, bool)>>>,
    }

    fn wait_calls(calls: &Mutex<Vec<(String, Instant, bool)>>, n: usize) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while calls.lock().unwrap().len() < n {
            assert!(Instant::now() < deadline, "no call");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Runs `translate_loop` with a fake translator on its own thread.
    fn run_loop(translations: Translations, partial_ms: u64, recorder: Arc<Recorder>) -> Loop {
        let (tx, rx) = backlog(60_000, utterance_ms);
        let translator = FakeTranslator {
            partial_ms,
            ..Default::default()
        };
        let calls = translator.partials.clone();
        let starts = translator.starts.clone();
        let sink = recorder.clone();
        let thread = std::thread::spawn(move || {
            translate_loop(
                rx,
                Box::new(translator),
                Arc::new(Mutex::new(None)),
                Arc::new(RwLock::new("ko".into())),
                sink,
                Some(translations),
            )
        });
        Loop {
            tx: Some(tx),
            thread,
            recorder,
            starts,
            calls,
        }
    }

    impl Loop {
        fn finish(mut self) -> Vec<(String, Instant, bool)> {
            self.tx.take();
            self.thread.join().unwrap();
            self.calls.lock().unwrap().clone()
        }
    }

    #[test]
    fn a_waiting_final_stops_a_partial_translation_and_goes_first() {
        let recorder = Arc::new(Recorder::default());
        let t = Translations::new(&Arc::new(Live::default()), Duration::ZERO);
        ask(&t.live, &t.slot, "partial", recorder.as_ref());
        let l = run_loop(t, 5_000, recorder);
        wait_runs(&l.starts, 1);
        let queued = Instant::now();
        l.tx.as_ref().unwrap().push(final_utterance(1), &mut |_| {});
        while l.recorder.utterances.lock().unwrap().is_empty() {
            assert!(queued.elapsed() < Duration::from_secs(4), "final waits");
            std::thread::sleep(Duration::from_millis(1));
        }
        let waited = queued.elapsed();
        println!("final translated {waited:?} after it was queued during a partial translation");
        assert!(waited < Duration::from_secs(1), "{waited:?}");
        let recorder = l.recorder.clone();
        let calls = l.finish();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].2, "the partial translation stopped");
        let u = &recorder.utterances.lock().unwrap()[0];
        assert!(
            matches!(&u.translation, TranslationState::Done { text, .. } if text == "번역 final 1")
        );
        // The stopped partial translation is not shown.
        assert!(
            recorder
                .partials
                .lock()
                .unwrap()
                .iter()
                .all(|(_, p)| p.translation.is_none())
        );
    }

    #[test]
    fn only_the_newest_partial_request_runs_and_shows_with_the_newest_original() {
        let recorder = Arc::new(Recorder::default());
        let t = Translations::new(&Arc::new(Live::default()), Duration::ZERO);
        ask(&t.live, &t.slot, "one", recorder.as_ref());
        ask(&t.live, &t.slot, "one two", recorder.as_ref());
        let live = t.live.clone();
        let l = run_loop(t, 500, recorder.clone());
        wait_runs(&l.starts, 1);
        // A newer original arrives while "one two" is in translation.
        let generation = live.generation(MIC);
        assert!(live.original(
            generation,
            &audio(0, 1700),
            "one two three",
            recorder.as_ref()
        ));
        let calls = l.finish();
        let texts: Vec<&str> = calls.iter().map(|c| c.0.as_str()).collect();
        assert_eq!(texts, ["one two"]);
        let partials = recorder.partials.lock().unwrap();
        let last = &partials.last().unwrap().1;
        assert_eq!(last.text, "one two three");
        assert_eq!(last.translation.as_deref(), Some("중간 one two"));
    }

    #[test]
    fn the_end_of_its_speech_stops_a_partial_translation() {
        let recorder = Arc::new(Recorder::default());
        let t = Translations::new(&Arc::new(Live::default()), Duration::ZERO);
        ask(&t.live, &t.slot, "one", recorder.as_ref());
        let live = t.live.clone();
        let l = run_loop(t, 5_000, recorder.clone());
        wait_runs(&l.starts, 1);
        let ended = Instant::now();
        live.end(MIC);
        wait_calls(&l.calls, 1);
        assert!(
            ended.elapsed() < Duration::from_secs(1),
            "{:?}",
            ended.elapsed()
        );
        let calls = l.finish();
        assert!(calls[0].2, "stopped");
    }

    #[test]
    fn a_request_of_an_ended_speech_does_not_run() {
        let recorder = Arc::new(Recorder::default());
        let t = Translations::new(&Arc::new(Live::default()), Duration::ZERO);
        ask(&t.live, &t.slot, "one", recorder.as_ref());
        t.live.end(MIC);
        let l = run_loop(t, 10, recorder.clone());
        std::thread::sleep(Duration::from_millis(100));
        assert!(l.finish().is_empty());
    }

    #[test]
    fn a_custom_translator_gets_one_partial_request_per_interval_per_speech() {
        let recorder = Arc::new(Recorder::default());
        let interval = translate_interval(&custom(Some(1500))).unwrap();
        let t = Translations::new(&Arc::new(Live::default()), interval);
        let (live, slot) = (t.live.clone(), t.slot.clone());
        let l = run_loop(t, 0, recorder.clone());
        let asks = |text: &str| ask(&live, &slot, text, recorder.as_ref());
        asks("a");
        wait_calls(&l.calls, 1);
        std::thread::sleep(Duration::from_millis(100));
        asks("a b");
        std::thread::sleep(Duration::from_millis(300));
        asks("a b c");
        wait_calls(&l.calls, 2);
        // A new speech is not held back by the interval of the last one.
        let new_speech = Instant::now();
        live.end(MIC);
        asks("next");
        wait_calls(&l.calls, 3);
        let calls = l.finish();
        let texts: Vec<&str> = calls.iter().map(|c| c.0.as_str()).collect();
        assert_eq!(texts, ["a", "a b c", "next"]);
        let gap = calls[1].1 - calls[0].1;
        println!("custom partial requests {gap:?} apart");
        assert!(
            (interval..interval + Duration::from_millis(500)).contains(&gap),
            "{gap:?}"
        );
        assert!(calls[2].1 - new_speech < Duration::from_millis(500));
    }
}
