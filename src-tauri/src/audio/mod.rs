//! Audio capture: microphone and system audio (loopback).
//! Owner: audio worker. Contract (do not change signatures without the coordinator):
//! - Output is 16 kHz mono f32 (`types::SAMPLE_RATE`), resampled here.
//! - `start_ms` of each chunk counts from `t0`.
//! - Windows: WASAPI (mic) + WASAPI loopback on the default output device.
//! - macOS: Core Audio (mic) + Core Audio process tap (system audio, macOS 14.6+).
//!
//! Layout: the device callback only downmixes into a pooled buffer and queues it with the
//! capture time of its first sample (`worker::Feeder`). A worker thread resamples, cuts chunks
//! and sets `start_ms` (`pipeline`). A supervisor thread owns the stream: after a fatal stream
//! error (device unplugged, default device changed) it reopens the source and tells the engine.

mod pipeline;
mod worker;

#[cfg(target_os = "macos")]
mod macos_tap;

use crate::types::{AudioChunk, SourceKind};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    Device, Error as CpalError, ErrorKind, FromSample, InputCallbackInfo, SampleFormat,
    SizedSample, Stream, StreamInstant,
};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use pipeline::downmix_into;
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use worker::Feeder;

/// Tells the engine about a problem of one capture (Korean user message). An empty
/// message clears the error of that capture (it runs again, or sound came after a silent start).
pub type ErrorSink = Arc<dyn Fn(String) + Send + Sync>;

/// macOS: first version where system audio can be recorded (Core Audio process tap).
#[cfg(any(target_os = "macos", test))]
const MACOS_LOOPBACK_MIN: (u32, u32) = (14, 6);
#[cfg(any(target_os = "macos", test))]
const MACOS_LOOPBACK_REASON: &str = "시스템 소리는 macOS 14.6 이상에서 받을 수 있습니다.";

/// A capture time that lags the callback by more than this is not trusted.
const MAX_CAPTURE_LAG: Duration = Duration::from_secs(5);
/// Wait before the first reopen after a stream stopped, then twice as long up to `RETRY_MAX`.
const RETRY_FIRST: Duration = Duration::from_millis(500);
const RETRY_MAX: Duration = Duration::from_secs(2);
/// How often the supervisor asks a source if its device changed without an error.
const STALE_POLL: Duration = Duration::from_secs(1);

/// A running capture. Capture stops when this value is dropped. The audio that the callback
/// already queued is processed and sent before `drop` returns.
pub struct Capture {
    closing: Arc<AtomicBool>,
    supervisor: Option<JoinHandle<()>>,
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.closing.store(true, Ordering::Release);
        if let Some(supervisor) = self.supervisor.take() {
            let _ = supervisor.join();
        }
    }
}

/// Starts capture of one source. `device` is a mic name from `list_mics`
/// (`None` = default device); it is ignored for `SourceKind::System`.
/// `on_error` gets a Korean message when the capture stops (device lost) and when it recovers.
pub fn start(
    source: SourceKind,
    device: Option<&str>,
    t0: Instant,
    tx: Sender<AudioChunk>,
    on_error: ErrorSink,
) -> anyhow::Result<Capture> {
    let (feeder, inbox) = worker::queue();
    let (events_tx, events_rx) = crossbeam_channel::bounded(16);
    let ctx = Ctx {
        source,
        device: device.map(str::to_string),
        feeder,
        events: events_tx,
    };
    let first = open(&ctx, 0)?;

    let closing = Arc::new(AtomicBool::new(false));
    let out = worker::Output {
        source,
        t0,
        tx,
        report: on_error.clone(),
        closing: closing.clone(),
        // macOS: the tap gives zeros while nothing plays; that is not a permission problem.
        watch_zeros: !(cfg!(target_os = "macos") && source == SourceKind::System),
    };
    let worker = std::thread::Builder::new()
        .name(format!("audio-worker-{source:?}"))
        .spawn(move || worker::run(inbox, &out))?;
    let sup_closing = closing.clone();
    let supervisor = std::thread::Builder::new()
        .name(format!("audio-supervisor-{source:?}"))
        .spawn(move || supervise(ctx, first, events_rx, on_error, sup_closing, worker))?;
    Ok(Capture {
        closing,
        supervisor: Some(supervisor),
    })
}

/// Names of the input devices.
pub fn list_mics() -> anyhow::Result<Vec<String>> {
    let host = cpal::default_host();
    let devices = host.input_devices().map_err(|e| {
        log::warn!("list input devices: {e}");
        anyhow::anyhow!("마이크 목록을 읽을 수 없습니다.")
    })?;
    let mut names: Vec<String> = devices
        .filter_map(|d| d.description().ok().map(|d| d.name().to_string()))
        .collect();
    names.dedup();
    Ok(names)
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemAudioSupport {
    pub supported: bool,
    /// User-facing reason when not supported (Korean), for example the OS version limit.
    pub reason: Option<String>,
}

/// Tells if this OS can capture system audio.
pub fn system_audio_support() -> SystemAudioSupport {
    platform_support()
}

#[cfg(target_os = "windows")]
fn platform_support() -> SystemAudioSupport {
    // WASAPI loopback exists on Windows Vista and later; the app needs Windows 10.
    SystemAudioSupport {
        supported: true,
        reason: None,
    }
}

#[cfg(target_os = "macos")]
fn platform_support() -> SystemAudioSupport {
    let version = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| parse_version(&s));
    macos_support(version)
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn platform_support() -> SystemAudioSupport {
    SystemAudioSupport {
        supported: false,
        reason: Some("시스템 소리는 Windows와 macOS에서만 받을 수 있습니다.".into()),
    }
}

/// "14.6.1\n" → (14, 6).
#[cfg(any(target_os = "macos", test))]
fn parse_version(text: &str) -> Option<(u32, u32)> {
    let mut parts = text.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().map_or(Some(0), |m| m.parse().ok())?;
    Some((major, minor))
}

/// `None` (version unknown) is not supported: capture would start and give silence.
#[cfg(any(target_os = "macos", test))]
fn macos_support(version: Option<(u32, u32)>) -> SystemAudioSupport {
    match version {
        Some(v) if v >= MACOS_LOOPBACK_MIN => SystemAudioSupport {
            supported: true,
            reason: None,
        },
        _ => SystemAudioSupport {
            supported: false,
            reason: Some(MACOS_LOOPBACK_REASON.into()),
        },
    }
}

/// What opens a stream, again and again after a device change.
struct Ctx {
    source: SourceKind,
    /// Mic name; `None` = the default device (it follows a change of the default).
    device: Option<String>,
    feeder: Feeder,
    events: Sender<Event>,
}

/// From a stream's error callback to the supervisor.
struct Event {
    generation: u64,
    fatal: bool,
    text: String,
}

/// An open source. Dropping it stops the stream.
trait Running: Send {
    /// True when the device changed in a way that raises no stream error.
    fn stale(&self) -> bool {
        false
    }
}

struct CpalRunning(#[allow(dead_code)] Stream);

impl Running for CpalRunning {}

fn open(ctx: &Ctx, generation: u64) -> anyhow::Result<Box<dyn Running>> {
    #[cfg(target_os = "macos")]
    if ctx.source == SourceKind::System {
        let support = system_audio_support();
        if !support.supported {
            anyhow::bail!("{}", support.reason.unwrap_or_default());
        }
        return macos_tap::open(ctx.feeder.clone(), ctx.events.clone(), generation)
            .map(|t| Box::new(t) as Box<dyn Running>);
    }
    open_cpal(ctx, generation).map(|s| Box::new(CpalRunning(s)) as Box<dyn Running>)
}

fn open_cpal(ctx: &Ctx, generation: u64) -> anyhow::Result<Stream> {
    let source = ctx.source;
    let host = cpal::default_host();
    let (dev, config) = match source {
        SourceKind::Mic => {
            let dev = pick_mic(&host, ctx.device.as_deref())?;
            let config = dev
                .default_input_config()
                .map_err(|e| user_error(source, &e))?;
            (dev, config)
        }
        SourceKind::System => {
            let support = system_audio_support();
            if !support.supported {
                anyhow::bail!("{}", support.reason.unwrap_or_default());
            }
            // An input stream on an output device is a loopback stream (WASAPI).
            let dev = host.default_output_device().ok_or_else(|| {
                anyhow::anyhow!(
                    "재생 장치(스피커·헤드폰)를 찾을 수 없습니다. 출력 장치를 연결하세요."
                )
            })?;
            let config = dev
                .default_output_config()
                .map_err(|e| user_error(source, &e))?;
            (dev, config)
        }
    };
    let rate = config.sample_rate();
    let channels = usize::from(config.channels());
    let format = config.sample_format();
    let stream_config: cpal::StreamConfig = config.into();
    log::info!("audio {source:?}: {rate} Hz, {channels} ch, {format:?}");

    let feeder = ctx.feeder.clone();
    let events = ctx.events.clone();
    let stream = match format {
        SampleFormat::F32 => build::<f32>(
            &dev,
            stream_config,
            rate,
            channels,
            feeder,
            events,
            generation,
        ),
        SampleFormat::I16 => build::<i16>(
            &dev,
            stream_config,
            rate,
            channels,
            feeder,
            events,
            generation,
        ),
        SampleFormat::I32 => build::<i32>(
            &dev,
            stream_config,
            rate,
            channels,
            feeder,
            events,
            generation,
        ),
        SampleFormat::U16 => build::<u16>(
            &dev,
            stream_config,
            rate,
            channels,
            feeder,
            events,
            generation,
        ),
        other => anyhow::bail!("지원하지 않는 오디오 형식입니다: {other:?}"),
    }
    .map_err(|e| user_error(source, &e))?;
    stream.play().map_err(|e| user_error(source, &e))?;
    Ok(stream)
}

fn build<T>(
    device: &Device,
    config: cpal::StreamConfig,
    rate: u32,
    channels: usize,
    feeder: Feeder,
    events: Sender<Event>,
    generation: u64,
) -> Result<Stream, CpalError>
where
    T: SizedSample + Copy + Send + 'static,
    f32: FromSample<T>,
{
    device.build_input_stream(
        config,
        move |data: &[T], info: &InputCallbackInfo| {
            let ts = info.timestamp();
            let frames = data.len() / channels.max(1);
            let block = Duration::from_secs_f64(frames as f64 / f64::from(rate));
            let start = block_start(Instant::now(), ts.callback, ts.capture, block);
            feeder.feed(start, rate, |mono| {
                downmix_into(mono, data, channels, |s| f32::from_sample_(s));
            });
        },
        move |err| {
            // Not the audio thread's data path, but keep it short: the supervisor logs.
            let _ = events.try_send(Event {
                generation,
                fatal: is_fatal(err.kind()),
                text: err.to_string(),
            });
        },
        None,
    )
}

/// When the first sample of a callback's data was captured, on the `Instant` clock.
/// `callback - capture` is the age of that sample (hardware latency, and the wait in the OS
/// buffer after a thread stall). Without a usable pair, assume the block just ended.
fn block_start(
    now: Instant,
    callback: StreamInstant,
    capture: StreamInstant,
    block: Duration,
) -> Instant {
    let lag = callback
        .checked_duration_since(capture)
        .filter(|lag| *lag <= MAX_CAPTURE_LAG)
        .unwrap_or(block);
    now.checked_sub(lag).unwrap_or(now)
}

fn is_fatal(kind: ErrorKind) -> bool {
    !matches!(
        kind,
        ErrorKind::Xrun | ErrorKind::DeviceChanged | ErrorKind::RealtimeDenied
    )
}

/// Owns the stream. Reopens it after a fatal error, with a short retry, until `closing`.
fn supervise(
    ctx: Ctx,
    first: Box<dyn Running>,
    events: Receiver<Event>,
    report: ErrorSink,
    closing: Arc<AtomicBool>,
    worker: JoinHandle<()>,
) {
    let source = ctx.source;
    let mut generation = 0u64;
    let mut current: Option<Box<dyn Running>> = Some(first);
    let mut next_try = Instant::now();
    let mut retry_wait = RETRY_FIRST;
    let mut last_poll = Instant::now();
    while !closing.load(Ordering::Acquire) {
        let mut stopped = false;
        match events.recv_timeout(Duration::from_millis(100)) {
            Ok(ev) if ev.generation == generation => {
                if ev.fatal {
                    log::error!("audio {source:?} stream error: {}", ev.text);
                    stopped = current.is_some();
                } else {
                    log::warn!("audio {source:?}: {}", ev.text);
                }
            }
            Ok(_) | Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {}
        }
        if current.is_some() && last_poll.elapsed() >= STALE_POLL {
            last_poll = Instant::now();
            if current.as_ref().is_some_and(|c| c.stale()) {
                log::warn!("audio {source:?}: the device changed");
                stopped = true;
            }
        }
        if stopped {
            current = None; // stops the old stream
            report(stop_message(source));
            retry_wait = RETRY_FIRST;
            next_try = Instant::now() + retry_wait;
        }
        if current.is_none() && Instant::now() >= next_try {
            generation += 1;
            match open(&ctx, generation) {
                Ok(running) => {
                    log::info!("audio {source:?}: reopened");
                    current = Some(running);
                    // The capture runs again: clear the stop error (an empty message).
                    report(String::new());
                }
                Err(e) => {
                    log::warn!("audio {source:?}: reopen failed: {e}");
                    retry_wait = (retry_wait * 2).min(RETRY_MAX);
                    next_try = Instant::now() + retry_wait;
                }
            }
        }
    }
    // Stop the stream and every sender: the worker then drains the queue and flushes.
    drop(current);
    drop(ctx);
    let _ = worker.join();
}

fn pick_mic(host: &cpal::Host, name: Option<&str>) -> anyhow::Result<Device> {
    let Some(name) = name else {
        return host.default_input_device().ok_or_else(|| {
            anyhow::anyhow!("마이크를 찾을 수 없습니다. 마이크를 연결하고 다시 시도하세요.")
        });
    };
    let found = host
        .input_devices()
        .map_err(|e| user_error(SourceKind::Mic, &e))?
        .find(|d| d.description().is_ok_and(|d| d.name() == name));
    found.ok_or_else(|| {
        anyhow::anyhow!("마이크 \"{name}\"을(를) 찾을 수 없습니다. 연결을 확인하세요.")
    })
}

#[derive(Debug, PartialEq, Eq)]
enum Failure {
    Permission,
    NotAvailable,
    Busy,
    Unsupported,
    Other,
}

/// Windows returns `E_ACCESSDENIED` when the microphone privacy setting is off. cpal does not
/// map it to `PermissionDenied`, so the text is checked too.
fn classify(kind: ErrorKind, text: &str) -> Failure {
    let text = text.to_lowercase();
    if kind == ErrorKind::PermissionDenied
        || text.contains("0x80070005")
        || text.contains("access is denied")
        || text.contains("access denied")
    {
        return Failure::Permission;
    }
    match kind {
        ErrorKind::DeviceNotAvailable => Failure::NotAvailable,
        ErrorKind::DeviceBusy => Failure::Busy,
        ErrorKind::UnsupportedConfig => Failure::Unsupported,
        _ => Failure::Other,
    }
}

/// Korean message for a stream that could not open. The technical text goes to the log.
fn user_error(source: SourceKind, err: &CpalError) -> anyhow::Error {
    log::error!("audio {source:?}: {err}");
    let failure = classify(err.kind(), &err.to_string());
    anyhow::anyhow!(failure_message(source, &failure, cfg!(target_os = "macos")))
}

fn failure_message(source: SourceKind, failure: &Failure, macos: bool) -> String {
    let who = match source {
        SourceKind::Mic => "마이크",
        SourceKind::System => "시스템 소리",
    };
    match failure {
        Failure::Permission => match (source, macos) {
            (SourceKind::Mic, true) => "마이크 권한이 없습니다. 시스템 설정 > 개인정보 보호 및 보안 > 마이크에서 morago를 켜고 앱을 다시 시작하세요.".into(),
            (SourceKind::System, true) => "시스템 소리 녹음 권한이 없습니다. 시스템 설정 > 개인정보 보호 및 보안 > 화면 및 시스템 오디오 녹음에서 morago를 켜고 앱을 다시 시작하세요.".into(),
            (SourceKind::Mic, false) => "마이크 권한이 없습니다. 설정 > 개인 정보 및 보안 > 마이크에서 '마이크 액세스'와 '데스크톱 앱이 마이크에 액세스하도록 허용'을 켜세요.".into(),
            (SourceKind::System, false) => "시스템 소리를 받을 권한이 없습니다. 장치 설정을 확인하세요.".into(),
        },
        Failure::NotAvailable => {
            format!("{who} 장치를 사용할 수 없습니다. 장치 연결을 확인하세요.")
        }
        Failure::Busy => {
            format!("{who} 장치를 다른 앱이 쓰고 있습니다. 그 앱을 닫고 다시 시도하세요.")
        }
        Failure::Unsupported => {
            format!("{who} 장치가 지원하지 않는 형식입니다. 다른 장치를 고르세요.")
        }
        Failure::Other => format!("{who}를 시작하지 못했습니다. 장치 설정을 확인하고 다시 시도하세요."),
    }
}

fn stop_message(source: SourceKind) -> String {
    match source {
        SourceKind::Mic => "마이크 연결이 끊겨 받기가 멈췄습니다. 다시 연결을 시도합니다.".into(),
        SourceKind::System => "재생 장치가 바뀌었거나 연결이 끊겨 시스템 소리 받기가 멈췄습니다. 다시 연결을 시도합니다.".into(),
    }
}

/// Exact zeros for the first seconds: a denied permission (macOS) or a muted device.
fn silent_message(source: SourceKind, macos: bool) -> String {
    match (source, macos) {
        (SourceKind::Mic, true) => "마이크에서 3초 동안 소리가 전혀 들어오지 않습니다. 시스템 설정 > 개인정보 보호 및 보안 > 마이크에서 morago가 켜져 있는지 확인하세요.".into(),
        (SourceKind::Mic, false) => "마이크에서 3초 동안 소리가 전혀 들어오지 않습니다. 마이크가 꺼져 있거나 설정 > 개인 정보 및 보안 > 마이크의 접근 권한이 꺼져 있는지 확인하세요.".into(),
        (SourceKind::System, true) => "시스템 소리에서 3초 동안 소리가 전혀 들어오지 않습니다. 재생 중인 소리가 없거나 권한이 꺼져 있을 수 있습니다. 시스템 설정 > 개인정보 보호 및 보안 > 화면 및 시스템 오디오 녹음에서 morago를 확인하세요.".into(),
        (SourceKind::System, false) => "시스템 소리에서 3초 동안 소리가 전혀 들어오지 않습니다. 재생 중인 소리가 없거나 출력 장치 설정에 문제가 있을 수 있습니다.".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_macos_versions() {
        assert_eq!(parse_version("14.6.1\n"), Some((14, 6)));
        assert_eq!(parse_version("15.0"), Some((15, 0)));
        assert_eq!(parse_version("26"), Some((26, 0)));
        assert_eq!(parse_version("abc"), None);
    }

    #[test]
    fn macos_limit_is_14_6() {
        assert!(!macos_support(Some((14, 5))).supported);
        assert!(!macos_support(Some((13, 9))).supported);
        assert!(macos_support(Some((14, 6))).supported);
        assert!(macos_support(Some((15, 0))).supported);
        assert!(macos_support(Some((26, 1))).supported);
        let unknown = macos_support(None);
        assert!(!unknown.supported);
        assert!(unknown.reason.unwrap().contains("14.6"));
    }

    #[test]
    fn permission_message_tells_where_to_allow() {
        let mic = failure_message(SourceKind::Mic, &Failure::Permission, true);
        assert!(mic.contains("개인정보 보호 및 보안 > 마이크"));
        let sys = failure_message(SourceKind::System, &Failure::Permission, true);
        assert!(sys.contains("화면 및 시스템 오디오 녹음"));
        let win = failure_message(SourceKind::Mic, &Failure::Permission, false);
        assert!(win.contains("마이크 액세스"));
    }

    #[test]
    fn windows_access_denied_is_a_permission_failure() {
        assert_eq!(
            classify(
                ErrorKind::BackendError,
                "HRESULT 0x80070005: Access is denied."
            ),
            Failure::Permission
        );
        assert_eq!(
            classify(ErrorKind::PermissionDenied, ""),
            Failure::Permission
        );
        assert_eq!(classify(ErrorKind::DeviceBusy, "busy"), Failure::Busy);
        assert_eq!(classify(ErrorKind::BackendError, "other"), Failure::Other);
    }

    #[test]
    fn fatal_errors() {
        assert!(is_fatal(ErrorKind::DeviceNotAvailable));
        assert!(is_fatal(ErrorKind::StreamInvalidated));
        assert!(is_fatal(ErrorKind::PermissionDenied));
        assert!(!is_fatal(ErrorKind::Xrun));
        assert!(!is_fatal(ErrorKind::DeviceChanged));
    }

    #[test]
    fn capture_start_uses_the_capture_time_not_the_arrival() {
        // A stall: 12 packets of 10 ms are delivered in one go at `now`. The k-th packet was
        // captured (120 - 10 k) ms before the callback, so the starts are 10 ms apart.
        let now = Instant::now() + Duration::from_secs(10);
        let cb = StreamInstant::from_millis(5_000);
        let block = Duration::from_millis(10);
        let starts: Vec<Instant> = (0..12)
            .map(|k| {
                let capture = StreamInstant::from_millis(5_000 - (120 - 10 * k));
                block_start(now, cb, capture, block)
            })
            .collect();
        for w in starts.windows(2) {
            assert_eq!(w[1] - w[0], Duration::from_millis(10));
        }
        assert_eq!(now - starts[11], Duration::from_millis(10));
    }

    #[test]
    fn capture_start_falls_back_to_the_block_length() {
        let now = Instant::now() + Duration::from_secs(10);
        let block = Duration::from_millis(10);
        // The capture time is after the callback time: not usable.
        let a = block_start(
            now,
            StreamInstant::from_millis(100),
            StreamInstant::from_millis(200),
            block,
        );
        assert_eq!(now - a, block);
        // The capture time is far in the past: not trusted.
        let b = block_start(
            now,
            StreamInstant::from_millis(100_000),
            StreamInstant::from_millis(1),
            block,
        );
        assert_eq!(now - b, block);
    }

    #[test]
    fn two_sources_with_different_latency_agree_on_the_wall_clock() {
        // The same instant of sound reaches a mic (30 ms latency) and a loopback (5 ms latency)
        // at different callback times; both starts equal the sound's real time.
        let sound = Instant::now() + Duration::from_secs(1);
        let block = Duration::from_millis(10);
        let mic_now = sound + Duration::from_millis(30);
        let sys_now = sound + Duration::from_millis(5);
        let mic = block_start(
            mic_now,
            StreamInstant::from_millis(1_000_030),
            StreamInstant::from_millis(1_000_000),
            block,
        );
        let sys = block_start(
            sys_now,
            StreamInstant::from_millis(2_000_005),
            StreamInstant::from_millis(2_000_000),
            block,
        );
        assert_eq!(mic, sound);
        assert_eq!(sys, sound);
    }

    #[test]
    fn user_messages_are_korean() {
        for s in [SourceKind::Mic, SourceKind::System] {
            for text in [stop_message(s), silent_message(s, false)] {
                assert!(text.chars().any(|c| ('가'..='힣').contains(&c)));
            }
        }
    }
}
