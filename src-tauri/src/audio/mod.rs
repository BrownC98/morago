//! Audio capture: microphone and system audio (loopback).
//! Owner: audio worker. Contract (do not change signatures without the coordinator):
//! - Output is 16 kHz mono f32 (`types::SAMPLE_RATE`), resampled here.
//! - `start_ms` of each chunk counts from `t0`.
//! - Windows: WASAPI (mic) + WASAPI loopback on the default output device.
//! - macOS: Core Audio (mic) + Core Audio process tap (system audio, macOS 14.6+).
//!
//! Both are reached through `cpal`: an input stream on the default *output* device is a loopback
//! stream on Windows and macOS. The data callback only downmixes and queues the block; a worker
//! thread resamples, cuts chunks and sets timestamps (see `pipeline`).

mod pipeline;

use crate::types::{AudioChunk, SourceKind};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Error as CpalError, ErrorKind, FromSample, SampleFormat, SizedSample, Stream};
use crossbeam_channel::{RecvTimeoutError, SendTimeoutError, Sender, TrySendError};
use pipeline::{Processor, downmix};
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Blocks between the device callback and the worker. A block is about 10 ms.
const BLOCK_QUEUE: usize = 256;
/// With no block for this long, the worker sends the audio it holds (loopback sends no packets
/// while nothing plays, so the end of a sentence would wait for the next sound).
const IDLE_FLUSH: Duration = Duration::from_millis(150);
/// How long the worker waits for room in the output channel before it drops a chunk.
const SEND_TIMEOUT: Duration = Duration::from_millis(200);

/// macOS: first version where cpal can record system audio (Core Audio process tap).
#[cfg(any(target_os = "macos", test))]
const MACOS_LOOPBACK_MIN: (u32, u32) = (14, 6);
const MACOS_LOOPBACK_REASON: &str = "시스템 소리는 macOS 14.6 이상에서 받을 수 있습니다.";

/// A running capture. Capture stops when this value is dropped.
pub struct Capture {
    stream: Option<Stream>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        // The stream owns the callback and its sender: dropping it ends the worker loop.
        drop(self.stream.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Starts capture of one source. `device` is a mic name from `list_mics`
/// (`None` = default device); it is ignored for `SourceKind::System`.
pub fn start(
    source: SourceKind,
    device: Option<&str>,
    t0: Instant,
    tx: Sender<AudioChunk>,
) -> anyhow::Result<Capture> {
    let host = cpal::default_host();
    let (dev, config) = match source {
        SourceKind::Mic => {
            let dev = pick_mic(&host, device)?;
            let config = dev
                .default_input_config()
                .map_err(|e| user_error(source, &e))?;
            (dev, config)
        }
        SourceKind::System => {
            let support = system_audio_support();
            if !support.supported {
                anyhow::bail!(
                    "{}",
                    support
                        .reason
                        .unwrap_or_else(|| "지원하지 않습니다.".into())
                );
            }
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
    let in_rate = config.sample_rate();
    let channels = usize::from(config.channels());
    let format = config.sample_format();
    let stream_config: cpal::StreamConfig = config.into();
    log::info!(
        "audio {source:?}: {} Hz, {channels} ch, {format:?}",
        in_rate
    );

    let processor = Processor::new(source, in_rate)?;
    let stop = Arc::new(AtomicBool::new(false));
    let (block_tx, block_rx) = crossbeam_channel::bounded::<Block>(BLOCK_QUEUE);
    let sink = Sink {
        source,
        channels,
        tx: block_tx,
        stop: stop.clone(),
        dropped: Arc::new(AtomicU32::new(0)),
    };

    let stream = match format {
        SampleFormat::F32 => build::<f32>(&dev, stream_config, sink),
        SampleFormat::I16 => build::<i16>(&dev, stream_config, sink),
        SampleFormat::I32 => build::<i32>(&dev, stream_config, sink),
        SampleFormat::U16 => build::<u16>(&dev, stream_config, sink),
        other => anyhow::bail!("지원하지 않는 오디오 형식입니다: {other:?}"),
    }
    .map_err(|e| user_error(source, &e))?;

    let worker_stop = stop.clone();
    let worker = std::thread::Builder::new()
        .name(format!("audio-{source:?}"))
        .spawn(move || run_worker(processor, block_rx, tx, t0, worker_stop))?;
    // The stream is created paused on some hosts: play() starts it.
    let capture = Capture {
        stream: Some(stream),
        stop,
        worker: Some(worker),
    };
    if let Some(stream) = &capture.stream {
        stream.play().map_err(|e| user_error(source, &e))?;
    }
    Ok(capture)
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

struct Block {
    arrival: Instant,
    mono: Vec<f32>,
}

/// What the device callback needs. The callback never blocks.
struct Sink {
    source: SourceKind,
    channels: usize,
    tx: Sender<Block>,
    stop: Arc<AtomicBool>,
    dropped: Arc<AtomicU32>,
}

fn build<T>(device: &Device, config: cpal::StreamConfig, sink: Sink) -> Result<Stream, CpalError>
where
    T: SizedSample + Copy + Send + 'static,
    f32: FromSample<T>,
{
    let source = sink.source;
    let channels = sink.channels;
    let stop_err = sink.stop.clone();
    device.build_input_stream(
        config,
        move |data: &[T], _| {
            if sink.stop.load(Ordering::Acquire) {
                return;
            }
            let arrival = Instant::now();
            let mono = downmix(data, channels, |s| f32::from_sample_(s));
            match sink.tx.try_send(Block { arrival, mono }) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    // Log the first drop of a burst only: the callback must stay fast.
                    if sink.dropped.fetch_add(1, Ordering::Relaxed) == 0 {
                        log::warn!("audio {:?}: worker is behind, dropping blocks", sink.source);
                    }
                }
                Err(TrySendError::Disconnected(_)) => sink.stop.store(true, Ordering::Release),
            }
        },
        move |err| {
            log::error!("audio {source:?} stream error: {err}");
            if is_fatal(err.kind()) {
                // Stops this stream: the callback ignores data and the worker ends.
                stop_err.store(true, Ordering::Release);
            }
        },
        None,
    )
}

fn is_fatal(kind: ErrorKind) -> bool {
    !matches!(
        kind,
        ErrorKind::Xrun | ErrorKind::DeviceChanged | ErrorKind::RealtimeDenied
    )
}

fn run_worker(
    mut processor: Processor,
    rx: crossbeam_channel::Receiver<Block>,
    tx: Sender<AudioChunk>,
    t0: Instant,
    stop: Arc<AtomicBool>,
) {
    let receiver_gone = std::cell::Cell::new(false);
    let mut emit = |chunk: AudioChunk| match tx.send_timeout(chunk, SEND_TIMEOUT) {
        Ok(()) => {}
        Err(SendTimeoutError::Timeout(_)) => {
            log::warn!("audio: output channel is full, chunk dropped")
        }
        Err(SendTimeoutError::Disconnected(_)) => receiver_gone.set(true),
    };
    while !stop.load(Ordering::Acquire) {
        let result = match rx.recv_timeout(IDLE_FLUSH) {
            Ok(block) => {
                let end_ms = block.arrival.saturating_duration_since(t0).as_secs_f64() * 1000.0;
                processor.push(end_ms, &block.mono, &mut emit)
            }
            Err(RecvTimeoutError::Timeout) => processor.flush(&mut emit),
            Err(RecvTimeoutError::Disconnected) => break,
        };
        if let Err(e) = result {
            log::error!("audio worker: {e}");
            break;
        }
        if receiver_gone.get() {
            return;
        }
    }
    if let Err(e) = processor.flush(&mut emit) {
        log::error!("audio worker flush: {e}");
    }
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

/// Korean message for a stream that could not open. The technical text goes to the log.
fn user_error(source: SourceKind, err: &CpalError) -> anyhow::Error {
    log::error!("audio {source:?}: {err}");
    anyhow::anyhow!(user_message(source, err.kind(), cfg!(target_os = "macos")))
}

fn user_message(source: SourceKind, kind: ErrorKind, macos: bool) -> String {
    let who = match source {
        SourceKind::Mic => "마이크",
        SourceKind::System => "시스템 소리",
    };
    match kind {
        ErrorKind::PermissionDenied => match (source, macos) {
            (SourceKind::Mic, true) => "마이크 권한이 없습니다. 시스템 설정 > 개인정보 보호 및 보안 > 마이크에서 Morago를 켜고 앱을 다시 시작하세요.".into(),
            (SourceKind::System, true) => "시스템 소리 녹음 권한이 없습니다. 시스템 설정 > 개인정보 보호 및 보안 > 화면 및 시스템 오디오 녹음에서 Morago를 켜고 앱을 다시 시작하세요.".into(),
            (SourceKind::Mic, false) => "마이크 권한이 없습니다. 설정 > 개인 정보 및 보안 > 마이크에서 마이크 접근과 데스크톱 앱의 접근을 허용하세요.".into(),
            (SourceKind::System, false) => "시스템 소리를 받을 권한이 없습니다. 장치 설정을 확인하세요.".into(),
        },
        ErrorKind::DeviceNotAvailable => {
            format!("{who} 장치를 사용할 수 없습니다. 장치 연결을 확인하세요.")
        }
        ErrorKind::DeviceBusy => {
            format!("{who} 장치를 다른 앱이 쓰고 있습니다. 그 앱을 닫고 다시 시도하세요.")
        }
        ErrorKind::UnsupportedConfig => {
            format!("{who} 장치가 지원하지 않는 형식입니다. 다른 장치를 고르세요.")
        }
        _ => format!("{who}를 시작하지 못했습니다. 장치 설정을 확인하고 다시 시도하세요."),
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
        let mic = user_message(SourceKind::Mic, ErrorKind::PermissionDenied, true);
        assert!(mic.contains("개인정보 보호 및 보안 > 마이크"));
        let sys = user_message(SourceKind::System, ErrorKind::PermissionDenied, true);
        assert!(sys.contains("화면 및 시스템 오디오 녹음"));
        let win = user_message(SourceKind::Mic, ErrorKind::PermissionDenied, false);
        assert!(win.contains("설정"));
    }

    #[test]
    fn fatal_errors() {
        assert!(is_fatal(ErrorKind::DeviceNotAvailable));
        assert!(is_fatal(ErrorKind::PermissionDenied));
        assert!(!is_fatal(ErrorKind::Xrun));
    }
}
