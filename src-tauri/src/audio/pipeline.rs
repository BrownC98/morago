//! Device-independent part of the capture: downmix, resample to 16 kHz, cut into chunks and
//! place each chunk on the session timeline. No audio device is needed, so it runs in unit tests.

use crate::types::{AudioChunk, SAMPLE_RATE, SourceKind};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

/// Target chunk length. The resampler may round it to its own block size.
const CHUNK_MS: usize = 100;
/// A block that starts more than this far from where the previous block ended opens a new
/// segment (a gap, for example WASAPI loopback while nothing plays). Smaller jumps are jitter.
pub const GAP_TOLERANCE_MS: f64 = 80.0;

/// Averages interleaved frames to one channel and converts each sample with `conv`.
pub fn downmix<T: Copy>(data: &[T], channels: usize, conv: impl Fn(T) -> f32) -> Vec<f32> {
    let channels = channels.max(1);
    if channels == 1 {
        return data.iter().map(|&s| conv(s)).collect();
    }
    data.chunks_exact(channels)
        .map(|frame| frame.iter().map(|&s| conv(s)).sum::<f32>() / channels as f32)
        .collect()
}

/// Mono sample rate converter with a fixed output chunk size.
enum Converter {
    /// Input is already 16 kHz.
    Passthrough { chunk: usize, out: Vec<f32> },
    Fft {
        resampler: Fft<f32>,
        out: Vec<f32>,
        /// Output frames still to drop: the resampler delay at the start of a segment.
        skip: usize,
    },
}

impl Converter {
    fn new(in_rate: usize) -> anyhow::Result<Self> {
        let out_rate = SAMPLE_RATE as usize;
        let chunk = out_rate * CHUNK_MS / 1000;
        if in_rate == out_rate {
            return Ok(Self::Passthrough {
                chunk,
                out: vec![0.0; chunk],
            });
        }
        let resampler = Fft::<f32>::new(in_rate, out_rate, chunk, 1, FixedSync::Output)
            .map_err(|e| anyhow::anyhow!("resampler: {e}"))?;
        let out = vec![0.0; resampler.output_frames_max()];
        let skip = resampler.output_delay();
        Ok(Self::Fft {
            resampler,
            out,
            skip,
        })
    }

    /// Input frames that the next `process` call needs.
    fn chunk_in(&self) -> usize {
        match self {
            Self::Passthrough { chunk, .. } => *chunk,
            Self::Fft { resampler, .. } => resampler.input_frames_next(),
        }
    }

    /// Converts exactly `chunk_in()` frames. The result is aligned with the input: the
    /// resampler delay is cut away from the start of a segment, so the result of the first call
    /// is shorter by that delay.
    fn process(&mut self, input: &[f32]) -> anyhow::Result<&[f32]> {
        match self {
            Self::Passthrough { out, .. } => {
                out[..input.len()].copy_from_slice(input);
                Ok(&out[..input.len()])
            }
            Self::Fft {
                resampler,
                out,
                skip,
            } => {
                let frames = input.len();
                let in_adapter = InterleavedSlice::new(input, 1, frames)
                    .map_err(|e| anyhow::anyhow!("resampler input: {e}"))?;
                let out_frames = out.len();
                let mut out_adapter = InterleavedSlice::new_mut(out.as_mut_slice(), 1, out_frames)
                    .map_err(|e| anyhow::anyhow!("resampler output: {e}"))?;
                let (_, written) = resampler
                    .process_into_buffer(&in_adapter, &mut out_adapter, None)
                    .map_err(|e| anyhow::anyhow!("resampler: {e}"))?;
                let cut = (*skip).min(written);
                *skip -= cut;
                Ok(&out[cut..written])
            }
        }
    }

    fn reset(&mut self) {
        if let Self::Fft {
            resampler, skip, ..
        } = self
        {
            resampler.reset();
            *skip = resampler.output_delay();
        }
    }
}

/// A stretch of audio without gaps. Times are exact integers of frames from its start.
struct Segment {
    start_ms: f64,
    in_frames: u64,
    out_frames: u64,
}

/// Turns blocks of mono audio at the device rate into 16 kHz chunks with timestamps.
pub struct Processor {
    source: SourceKind,
    in_rate: u64,
    converter: Converter,
    /// Mono input that does not fill a chunk yet.
    pending: Vec<f32>,
    segment: Option<Segment>,
}

impl Processor {
    pub fn new(source: SourceKind, in_rate: u32) -> anyhow::Result<Self> {
        Ok(Self {
            source,
            in_rate: u64::from(in_rate),
            converter: Converter::new(in_rate as usize)?,
            pending: Vec::new(),
            segment: None,
        })
    }

    /// Adds one block that ended `end_ms` after the session start (callback time).
    pub fn push(
        &mut self,
        end_ms: f64,
        mono: &[f32],
        emit: &mut dyn FnMut(AudioChunk),
    ) -> anyhow::Result<()> {
        if mono.is_empty() {
            return Ok(());
        }
        let block_ms = mono.len() as f64 * 1000.0 / self.in_rate as f64;
        let observed_start = (end_ms - block_ms).max(0.0);
        let new_segment = match &self.segment {
            None => true,
            Some(seg) => {
                let expected = seg.start_ms + seg.in_frames as f64 * 1000.0 / self.in_rate as f64;
                (observed_start - expected).abs() > GAP_TOLERANCE_MS
            }
        };
        if new_segment {
            self.flush(emit)?;
            self.segment = Some(Segment {
                start_ms: observed_start,
                in_frames: 0,
                out_frames: 0,
            });
        }
        if let Some(seg) = self.segment.as_mut() {
            seg.in_frames += mono.len() as u64;
        }
        self.pending.extend_from_slice(mono);
        let chunk_in = self.converter.chunk_in();
        while self.pending.len() >= chunk_in {
            let out = self.converter.process(&self.pending[..chunk_in])?.to_vec();
            self.emit_samples(out, emit);
            self.pending.drain(..chunk_in);
        }
        Ok(())
    }

    /// Ends the current segment: converts the rest of the input (zero padded) and the resampler
    /// tail, so the last audio before a gap is not held back. The next block starts a new
    /// segment.
    pub fn flush(&mut self, emit: &mut dyn FnMut(AudioChunk)) -> anyhow::Result<()> {
        let Some(seg) = self.segment.as_ref() else {
            return Ok(());
        };
        let want_total = seg.in_frames * u64::from(SAMPLE_RATE) / self.in_rate;
        let need = want_total.saturating_sub(seg.out_frames) as usize;
        let chunk_in = self.converter.chunk_in();
        let mut tail: Vec<f32> = Vec::new();
        while tail.len() < need {
            self.pending.resize(chunk_in, 0.0);
            let out = self.converter.process(&self.pending[..chunk_in])?;
            tail.extend_from_slice(out);
            self.pending.clear();
        }
        tail.truncate(need);
        self.emit_samples(tail, emit);
        self.pending.clear();
        self.converter.reset();
        self.segment = None;
        Ok(())
    }

    fn emit_samples(&mut self, samples: Vec<f32>, emit: &mut dyn FnMut(AudioChunk)) {
        let Some(seg) = self.segment.as_mut() else {
            return;
        };
        if samples.is_empty() {
            return;
        }
        let offset_ms = seg.out_frames as f64 * 1000.0 / f64::from(SAMPLE_RATE);
        seg.out_frames += samples.len() as u64;
        emit(AudioChunk {
            source: self.source,
            start_ms: (seg.start_ms + offset_ms).round() as u64,
            samples,
        });
    }

    /// True when input is waiting that has not produced a chunk yet.
    pub fn has_pending(&self) -> bool {
        self.segment.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, secs: f64, hz: f64) -> Vec<f32> {
        let n = (f64::from(rate) * secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    /// Feeds `samples` in blocks of `block` frames as if they arrived in real time from `t0_ms`.
    fn feed(
        p: &mut Processor,
        rate: u32,
        t0_ms: f64,
        samples: &[f32],
        block: usize,
        out: &mut Vec<AudioChunk>,
    ) {
        let mut t = t0_ms;
        for b in samples.chunks(block) {
            t += b.len() as f64 * 1000.0 / f64::from(rate);
            p.push(t, b, &mut |c| out.push(c)).unwrap();
        }
    }

    fn total(out: &[AudioChunk]) -> usize {
        out.iter().map(|c| c.samples.len()).sum()
    }

    #[test]
    fn downmix_averages_channels() {
        let stereo = [1.0f32, 0.0, 0.5, 0.5, -1.0, 1.0];
        assert_eq!(downmix(&stereo, 2, |s| s), vec![0.5, 0.5, 0.0]);
        let mono = [0.25f32, -0.25];
        assert_eq!(downmix(&mono, 1, |s| s), vec![0.25, -0.25]);
    }

    #[test]
    fn downmix_converts_integer_samples() {
        let data = [i16::MAX, i16::MAX, 0, 0];
        let out = downmix(&data, 2, |s| f32::from(s) / 32768.0);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.99997).abs() < 1e-4);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn downmix_drops_partial_frame() {
        assert_eq!(downmix(&[1.0f32, 1.0, 1.0], 2, |s| s).len(), 1);
    }

    #[test]
    fn resample_length_matches_rate() {
        for rate in [48_000u32, 44_100, 96_000, 8_000, 16_000] {
            let mut p = Processor::new(SourceKind::Mic, rate).unwrap();
            let mut out = Vec::new();
            feed(&mut p, rate, 0.0, &sine(rate, 2.0, 440.0), 480, &mut out);
            p.flush(&mut |c| out.push(c)).unwrap();
            // 2 s at 16 kHz, exact to the frame.
            assert_eq!(total(&out), 32_000, "rate {rate}");
            assert!(out.iter().all(|c| c.source == SourceKind::Mic));
        }
    }

    #[test]
    fn resample_keeps_tone_frequency() {
        let mut p = Processor::new(SourceKind::Mic, 48_000).unwrap();
        let mut out = Vec::new();
        feed(
            &mut p,
            48_000,
            0.0,
            &sine(48_000, 1.0, 1000.0),
            480,
            &mut out,
        );
        p.flush(&mut |c| out.push(c)).unwrap();
        let all: Vec<f32> = out.iter().flat_map(|c| c.samples.clone()).collect();
        // Count rising zero crossings in the middle: 1000 Hz → 1000 per second at 16 kHz.
        let mid = &all[4000..12_000];
        let crossings = mid.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        assert!((crossings as i64 - 500).abs() <= 1, "crossings {crossings}");
        let peak = mid.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
        assert!((peak - 1.0).abs() < 0.05, "peak {peak}");
    }

    #[test]
    fn chunks_are_about_100_ms() {
        let mut p = Processor::new(SourceKind::System, 44_100).unwrap();
        let mut out = Vec::new();
        feed(
            &mut p,
            44_100,
            0.0,
            &sine(44_100, 1.0, 300.0),
            441,
            &mut out,
        );
        assert!(out.len() >= 8);
        for c in &out[1..] {
            assert!(
                (1200..=2000).contains(&c.samples.len()),
                "len {}",
                c.samples.len()
            );
        }
    }

    #[test]
    fn continuous_audio_has_contiguous_timestamps() {
        let mut p = Processor::new(SourceKind::Mic, 48_000).unwrap();
        let mut out = Vec::new();
        feed(
            &mut p,
            48_000,
            500.0,
            &sine(48_000, 1.0, 440.0),
            480,
            &mut out,
        );
        p.flush(&mut |c| out.push(c)).unwrap();
        let mut expected = out[0].start_ms;
        assert!(expected.abs_diff(500) <= 1, "first start {expected}");
        for c in &out {
            assert!(c.start_ms.abs_diff(expected) <= 1);
            expected = c.start_ms + (c.samples.len() as u64 * 1000 / 16_000);
        }
    }

    #[test]
    fn jitter_does_not_open_a_segment() {
        let mut p = Processor::new(SourceKind::Mic, 16_000).unwrap();
        let mut out = Vec::new();
        let block = vec![0.1f32; 160];
        let mut t = 0.0;
        for i in 0..100 {
            // Arrival jitters by ±20 ms around the nominal time.
            t += 10.0;
            let jitter = if i % 2 == 0 { 20.0 } else { -20.0 };
            p.push(t + jitter, &block, &mut |c| out.push(c)).unwrap();
        }
        p.flush(&mut |c| out.push(c)).unwrap();
        assert_eq!(total(&out), 16_000);
        let mut next = out[0].start_ms;
        for c in &out {
            assert_eq!(c.start_ms, next);
            next += c.samples.len() as u64 * 1000 / 16_000;
        }
    }

    #[test]
    fn gap_keeps_timeline_and_flushes_tail() {
        let mut p = Processor::new(SourceKind::System, 48_000).unwrap();
        let mut out = Vec::new();
        // 1.05 s of audio from t = 0, then silence (no packets) until t = 5 s, then 0.5 s.
        feed(
            &mut p,
            48_000,
            0.0,
            &sine(48_000, 1.05, 440.0),
            480,
            &mut out,
        );
        let before_gap = out.len();
        feed(
            &mut p,
            48_000,
            5_000.0,
            &sine(48_000, 0.5, 440.0),
            480,
            &mut out,
        );
        p.flush(&mut |c| out.push(c)).unwrap();
        // The partial chunk before the gap came out when the gap was seen, with its true start.
        let first_total: usize = out[..=before_gap].iter().map(|c| c.samples.len()).sum();
        assert_eq!(first_total, 16_800);
        let after: Vec<&AudioChunk> = out.iter().filter(|c| c.start_ms >= 4_900).collect();
        assert!(!after.is_empty());
        assert!(
            after[0].start_ms.abs_diff(5_000) <= 1,
            "{}",
            after[0].start_ms
        );
        let after_total: usize = after.iter().map(|c| c.samples.len()).sum();
        assert_eq!(after_total, 8_000);
        // Nothing lands inside the gap.
        assert!(
            out.iter()
                .all(|c| c.start_ms <= 1_100 || c.start_ms >= 4_900)
        );
    }

    #[test]
    fn idle_flush_releases_pending_audio() {
        let mut p = Processor::new(SourceKind::Mic, 16_000).unwrap();
        let mut out = Vec::new();
        p.push(30.0, &vec![0.2f32; 480], &mut |c| out.push(c))
            .unwrap();
        assert!(out.is_empty());
        assert!(p.has_pending());
        p.flush(&mut |c| out.push(c)).unwrap();
        assert_eq!(total(&out), 480);
        assert_eq!(out[0].start_ms, 0);
        assert!(!p.has_pending());
    }

    #[test]
    fn start_never_negative() {
        let mut p = Processor::new(SourceKind::Mic, 16_000).unwrap();
        let mut out = Vec::new();
        // The block is longer than the time since t0 (callback delivered a big first buffer).
        p.push(10.0, &vec![0.0f32; 1600], &mut |c| out.push(c))
            .unwrap();
        assert_eq!(out[0].start_ms, 0);
    }
}
