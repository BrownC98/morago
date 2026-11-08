//! Manual test of the audio capture on a real machine.
//!
//! `cargo run --example capture -- <mic|system> <seconds> <out.wav>`
//!
//! Writes the 16 kHz mono chunks to a WAV file. A gap in the timeline (loopback while nothing
//! plays) is filled with silence, so the length of the file equals the wall-clock time.

use crossbeam_channel::RecvTimeoutError;
use std::time::{Duration, Instant};
use types::{SAMPLE_RATE, SourceKind};

// `morago_lib` keeps its modules private, so the example compiles the capture module itself.
#[allow(dead_code)]
#[path = "../src/audio/mod.rs"]
mod audio;
#[allow(dead_code)]
#[path = "../src/types.rs"]
mod types;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [kind, secs, out] = args.as_slice() else {
        anyhow::bail!("usage: capture <mic|system> <seconds> <out.wav>");
    };
    let source = match kind.as_str() {
        "mic" => SourceKind::Mic,
        "system" => SourceKind::System,
        other => anyhow::bail!("unknown source {other:?}: use mic or system"),
    };
    let secs: u64 = secs.parse()?;

    println!("mics: {:?}", audio::list_mics()?);
    println!("system audio: {:?}", audio::system_audio_support());

    let (tx, rx) = crossbeam_channel::bounded(256);
    let t0 = Instant::now();
    let capture = audio::start(source, None, t0, tx)?;
    println!("recording {secs} s ...");

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut out_wav = Output {
        wav: hound::WavWriter::create(out, spec)?,
        written: 0,
        peak: 0.0,
        chunks: 0,
    };
    let deadline = t0 + Duration::from_secs(secs);
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        match rx.recv_timeout(left) {
            Ok(chunk) => out_wav.put(&chunk)?,
            Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
        }
    }
    drop(capture); // stops the stream; the worker sends the audio it still holds
    while let Ok(chunk) = rx.try_recv() {
        out_wav.put(&chunk)?;
    }
    let Output {
        wav,
        written,
        peak,
        chunks,
    } = out_wav;
    wav.finalize()?;
    println!(
        "{chunks} chunks, {:.2} s of audio, peak {peak:.3} -> {out}",
        written as f64 / f64::from(SAMPLE_RATE)
    );
    if peak < 1e-4 {
        println!("warning: the file is silent. Check the permission and that sound is playing.");
    }
    Ok(())
}

struct Output {
    wav: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    /// Samples written so far.
    written: u64,
    peak: f32,
    chunks: u32,
}

impl Output {
    fn put(&mut self, chunk: &types::AudioChunk) -> anyhow::Result<()> {
        let start = chunk.start_ms * u64::from(SAMPLE_RATE) / 1000;
        for _ in 0..start.saturating_sub(self.written) {
            self.wav.write_sample(0.0f32)?;
            self.written += 1;
        }
        // Skip samples that overlap what is written (rounding of start_ms).
        let skip = self.written.saturating_sub(start) as usize;
        for &s in chunk.samples.iter().skip(skip) {
            self.peak = self.peak.max(s.abs());
            self.wav.write_sample(s)?;
            self.written += 1;
        }
        self.chunks += 1;
        Ok(())
    }
}
