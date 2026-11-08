//! Audio capture: microphone and system audio (loopback).
//! Owner: audio worker. Contract (do not change signatures without the coordinator):
//! - Output is 16 kHz mono f32 (`types::SAMPLE_RATE`), resampled here.
//! - `start_ms` of each chunk counts from `t0`.
//! - Windows: WASAPI (mic) + WASAPI loopback on the default output device.
//! - macOS: Core Audio (mic) + Core Audio process tap (system audio, macOS 14.6+).

use crate::types::{AudioChunk, SourceKind};
use crossbeam_channel::Sender;
use serde::Serialize;
use std::time::Instant;

/// A running capture. Capture stops when this value is dropped.
pub struct Capture {
    _private: (),
}

/// Starts capture of one source. `device` is a mic name from `list_mics`
/// (`None` = default device); it is ignored for `SourceKind::System`.
pub fn start(
    source: SourceKind,
    device: Option<&str>,
    t0: Instant,
    tx: Sender<AudioChunk>,
) -> anyhow::Result<Capture> {
    let _ = (source, device, t0, tx);
    anyhow::bail!("audio capture is not implemented yet")
}

/// Names of the input devices.
pub fn list_mics() -> anyhow::Result<Vec<String>> {
    Ok(Vec::new())
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemAudioSupport {
    pub supported: bool,
    /// User-facing reason when not supported (Korean), for example the OS version limit.
    pub reason: Option<String>,
}

/// Tells if this OS can capture system audio.
pub fn system_audio_support() -> SystemAudioSupport {
    SystemAudioSupport {
        supported: false,
        reason: Some("아직 구현되지 않았습니다".into()),
    }
}
