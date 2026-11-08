//! Device-independent part of the capture: downmix, resample to 16 kHz, cut into chunks and
//! place each chunk on the session timeline. No audio device is needed, so it runs in unit tests.

use crate::types::{AudioChunk, SAMPLE_RATE, SourceKind};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

/// Target chunk length. The resampler may round it to its own block size.
const CHUNK_MS: usize = 100;
/// How far a block may start from where the previous one ended before the timeline is fixed.
/// Later: a gap (WASAPI loopback while nothing plays) opens a new segment. Earlier: the device
/// clock runs faster than the wall clock, so the leading audio of the block is dropped.
/// Smaller differences are jitter.
pub const TOLERANCE_MS: f64 = 80.0;
/// A stream that gives only exact zeros for this long is reported once (denied permission).
const ZERO_WATCH_SECS: u64 = 3;

/// Appends the interleaved frames to `out` as mono (channel average), converting each sample
/// with `conv`. A partial frame at the end is dropped. Does not allocate if `out` has room.
pub fn downmix_into<T: Copy>(
    out: &mut Vec<f32>,
    data: &[T],
    channels: usize,
    conv: impl Fn(T) -> f32,
) {
    let channels = channels.max(1);
    if channels == 1 {
        out.extend(data.iter().map(|&s| conv(s)));
        return;
    }
    out.extend(
        data.chunks_exact(channels)
            .map(|frame| frame.iter().map(|&s| conv(s)).sum::<f32>() / channels as f32),
    );
}

/// Mono sample rate converter with a fixed output chunk size.
struct Converter {
    /// `None` when the input is already 16 kHz.
    resampler: Option<Fft<f32>>,
    chunk: usize,
    out: Vec<f32>,
    /// Output frames still to drop: the resampler delay at the start of a segment.
    skip: usize,
}

impl Converter {
    fn new(in_rate: usize) -> anyhow::Result<Self> {
        let out_rate = SAMPLE_RATE as usize;
        let chunk = out_rate * CHUNK_MS / 1000;
        if in_rate == out_rate {
            return Ok(Self {
                resampler: None,
                chunk,
                out: vec![0.0; chunk],
                skip: 0,
            });
        }
        let resampler = Fft::<f32>::new(in_rate, out_rate, chunk, 1, FixedSync::Output)
            .map_err(|e| anyhow::anyhow!("resampler: {e}"))?;
        Ok(Self {
            out: vec![0.0; resampler.output_frames_max()],
            skip: resampler.output_delay(),
            chunk,
            resampler: Some(resampler),
        })
    }

    /// Input frames that the next `process` call needs.
    fn chunk_in(&self) -> usize {
        self.resampler
            .as_ref()
            .map_or(self.chunk, |r| r.input_frames_next())
    }

    /// Converts exactly `chunk_in()` frames. The result is aligned with the input: the
    /// resampler delay is cut away from the start of a segment, so the result of the first call
    /// is shorter by that delay.
    fn process(&mut self, input: &[f32]) -> anyhow::Result<&[f32]> {
        let Some(resampler) = self.resampler.as_mut() else {
            self.out[..input.len()].copy_from_slice(input);
            return Ok(&self.out[..input.len()]);
        };
        let frames = input.len();
        let in_adapter = InterleavedSlice::new(input, 1, frames)
            .map_err(|e| anyhow::anyhow!("resampler input: {e}"))?;
        let out_frames = self.out.len();
        let mut out_adapter = InterleavedSlice::new_mut(self.out.as_mut_slice(), 1, out_frames)
            .map_err(|e| anyhow::anyhow!("resampler output: {e}"))?;
        let (_, written) = resampler
            .process_into_buffer(&in_adapter, &mut out_adapter, None)
            .map_err(|e| anyhow::anyhow!("resampler: {e}"))?;
        let cut = self.skip.min(written);
        self.skip -= cut;
        Ok(&self.out[cut..written])
    }

    fn reset(&mut self) {
        if let Some(resampler) = self.resampler.as_mut() {
            resampler.reset();
            self.skip = resampler.output_delay();
        }
    }
}

/// A stretch of audio without gaps. Times are exact integers of frames from its start.
struct Segment {
    start_ms: f64,
    in_frames: u64,
    out_frames: u64,
}

/// Reports once when a stream gives exact digital zeros for its first seconds. A real
/// microphone or tap never does; a denied permission (macOS) does.
pub struct ZeroWatch {
    frames: u64,
    done: bool,
}

impl ZeroWatch {
    pub fn new() -> Self {
        Self {
            frames: 0,
            done: false,
        }
    }

    /// True once, when the first `ZERO_WATCH_SECS` seconds were all exact zeros.
    pub fn feed(&mut self, mono: &[f32], rate: u32) -> bool {
        if self.done {
            return false;
        }
        if mono.iter().any(|&s| s != 0.0) {
            self.done = true;
            return false;
        }
        self.frames += mono.len() as u64;
        if self.frames >= u64::from(rate) * ZERO_WATCH_SECS {
            self.done = true;
            return true;
        }
        false
    }
}

/// Turns blocks of mono audio at the device rate into 16 kHz chunks with timestamps.
///
/// Rules for `start_ms` of the chunks of one source: never goes backwards, never overlaps the
/// previous chunk, and follows the capture time of the blocks except for jitter below
/// `TOLERANCE_MS`.
pub struct Processor {
    source: SourceKind,
    in_rate: u64,
    converter: Converter,
    /// Mono input that does not fill a chunk yet.
    pending: Vec<f32>,
    segment: Option<Segment>,
    /// Where the next input sample is expected on the timeline. Kept after a flush.
    end_ms: Option<f64>,
    /// Smallest `start_ms` the next chunk may have (end of the previous chunk).
    min_start_ms: u64,
}

impl Processor {
    pub fn new(source: SourceKind, in_rate: u32) -> anyhow::Result<Self> {
        Ok(Self {
            source,
            in_rate: u64::from(in_rate),
            converter: Converter::new(in_rate as usize)?,
            pending: Vec::new(),
            segment: None,
            end_ms: None,
            min_start_ms: 0,
        })
    }

    pub fn rate(&self) -> u32 {
        self.in_rate as u32
    }

    /// Adds one block whose first sample was captured `start_ms` after the session start.
    pub fn push(
        &mut self,
        start_ms: f64,
        mono: &[f32],
        emit: &mut dyn FnMut(AudioChunk),
    ) -> anyhow::Result<()> {
        let mut mono = mono;
        let observed = start_ms.max(0.0);
        let mut seg_start = observed;
        if let Some(expected) = self.end_ms {
            let diff = observed - expected;
            if diff > TOLERANCE_MS {
                // Gap: nothing was captured for a while.
                self.flush(emit)?;
            } else {
                seg_start = expected;
                if diff < -TOLERANCE_MS {
                    // The device clock is fast: drop the leading audio that is already late.
                    let skip = ((-diff) * self.in_rate as f64 / 1000.0).ceil() as usize;
                    if skip >= mono.len() {
                        return Ok(());
                    }
                    mono = &mono[skip..];
                }
            }
        }
        if mono.is_empty() {
            return Ok(());
        }
        let seg = self.segment.get_or_insert(Segment {
            start_ms: seg_start,
            in_frames: 0,
            out_frames: 0,
        });
        seg.in_frames += mono.len() as u64;
        self.end_ms = Some(seg.start_ms + seg.in_frames as f64 * 1000.0 / self.in_rate as f64);
        self.pending.extend_from_slice(mono);
        // The input size of the next conversion changes after each call: ask every time.
        while self.pending.len() >= self.converter.chunk_in() {
            let chunk_in = self.converter.chunk_in();
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
        let mut tail: Vec<f32> = Vec::new();
        while tail.len() < need {
            let chunk_in = self.converter.chunk_in();
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
        let start_ms = ((seg.start_ms + offset_ms).round() as u64).max(self.min_start_ms);
        self.min_start_ms = start_ms + samples.len() as u64 * 1000 / u64::from(SAMPLE_RATE);
        emit(AudioChunk {
            source: self.source,
            start_ms,
            samples,
        });
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

    /// Feeds `samples` in blocks of `block` frames. The first block starts at `t0_ms`; each next
    /// block starts `block` frames later by the device clock, scaled by `clock` (1.0 = the device
    /// clock equals the wall clock, above 1.0 = the wall clock runs ahead of the samples).
    fn feed_clock(
        p: &mut Processor,
        rate: u32,
        t0_ms: f64,
        clock: f64,
        samples: &[f32],
        block: usize,
        out: &mut Vec<AudioChunk>,
    ) -> f64 {
        let mut t = t0_ms;
        for b in samples.chunks(block) {
            p.push(t, b, &mut |c| out.push(c)).unwrap();
            t += b.len() as f64 * 1000.0 / f64::from(rate) * clock;
        }
        t
    }

    fn feed(
        p: &mut Processor,
        rate: u32,
        t0_ms: f64,
        samples: &[f32],
        block: usize,
        out: &mut Vec<AudioChunk>,
    ) {
        feed_clock(p, rate, t0_ms, 1.0, samples, block, out);
    }

    fn downmix<T: Copy>(data: &[T], channels: usize, conv: impl Fn(T) -> f32) -> Vec<f32> {
        let mut out = Vec::new();
        downmix_into(&mut out, data, channels, conv);
        out
    }

    fn total(out: &[AudioChunk]) -> usize {
        out.iter().map(|c| c.samples.len()).sum()
    }

    /// start_ms never goes backwards and a chunk never starts before the previous one ends.
    fn assert_ordered(out: &[AudioChunk]) {
        let mut end = 0u64;
        for (i, c) in out.iter().enumerate() {
            assert!(
                c.start_ms >= end,
                "chunk {i} starts {} < end {end}",
                c.start_ms
            );
            end = c.start_ms + c.samples.len() as u64 * 1000 / 16_000;
        }
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
    fn large_blocks_cross_the_input_size_change() {
        // One block holds several conversions; the input size of the Fft resampler changes
        // between conversions (48 kHz: 4800 frames, then 5586 and others).
        for (rate, block) in [(48_000u32, 4_800usize), (44_100, 4_410), (48_000, 20_000)] {
            let mut p = Processor::new(SourceKind::Mic, rate).unwrap();
            let mut out = Vec::new();
            feed(&mut p, rate, 0.0, &sine(rate, 12.0, 440.0), block, &mut out);
            p.flush(&mut |c| out.push(c)).unwrap();
            assert_eq!(total(&out), 192_000, "rate {rate} block {block}");
            assert_ordered(&out);
        }
    }

    #[test]
    fn output_does_not_depend_on_block_size() {
        let input = sine(48_000, 3.0, 700.0);
        let run = |block: usize| {
            let mut p = Processor::new(SourceKind::Mic, 48_000).unwrap();
            let mut out = Vec::new();
            feed(&mut p, 48_000, 0.0, &input, block, &mut out);
            p.flush(&mut |c| out.push(c)).unwrap();
            out.iter()
                .flat_map(|c| c.samples.clone())
                .collect::<Vec<f32>>()
        };
        let a = run(480);
        let b = run(7_777);
        assert_eq!(a.len(), b.len());
        let worst = a
            .iter()
            .zip(&b)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(worst < 1e-4, "worst difference {worst}");
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
        assert!(
            out[0].start_ms.abs_diff(500) <= 1,
            "first {}",
            out[0].start_ms
        );
        assert_ordered(&out);
        let mut expected = out[0].start_ms;
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
        for i in 0..100 {
            // Capture times jitter by ±20 ms around the nominal time.
            let jitter = if i % 2 == 0 { 20.0 } else { -20.0 };
            p.push(f64::from(i) * 10.0 + 40.0 + jitter, &block, &mut |c| {
                out.push(c)
            })
            .unwrap();
        }
        p.flush(&mut |c| out.push(c)).unwrap();
        assert_eq!(total(&out), 16_000);
        assert_ordered(&out);
        let mut next = out[0].start_ms;
        for c in &out {
            assert_eq!(c.start_ms, next);
            next += c.samples.len() as u64 * 1000 / 16_000;
        }
    }

    #[test]
    fn burst_after_a_stall_keeps_the_timeline() {
        // The capture thread stalls for 300 ms. When it wakes, the 30 queued 10 ms packets
        // arrive together. Each carries its own capture time, so they are contiguous.
        let mut p = Processor::new(SourceKind::System, 16_000).unwrap();
        let mut out = Vec::new();
        let block = vec![0.1f32; 160];
        for i in 0..10 {
            p.push(f64::from(i) * 10.0, &block, &mut |c| out.push(c))
                .unwrap();
        }
        // Packets 10..40 delivered late, in one go: starts are their capture times.
        for i in 10..40 {
            p.push(f64::from(i) * 10.0, &block, &mut |c| out.push(c))
                .unwrap();
        }
        p.flush(&mut |c| out.push(c)).unwrap();
        assert_eq!(total(&out), 40 * 160);
        assert_ordered(&out);
        assert_eq!(out[0].start_ms, 0);
        let mut next = 0;
        for c in &out {
            assert_eq!(c.start_ms, next, "false gap");
            next += c.samples.len() as u64 / 16;
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
        assert_ordered(&out);
    }

    #[test]
    fn idle_flush_then_more_audio_does_not_overlap() {
        let mut p = Processor::new(SourceKind::Mic, 16_000).unwrap();
        let mut out = Vec::new();
        p.push(0.0, &vec![0.2f32; 480], &mut |c| out.push(c))
            .unwrap();
        assert!(out.is_empty());
        p.flush(&mut |c| out.push(c)).unwrap();
        assert_eq!(total(&out), 480);
        assert_eq!(out[0].start_ms, 0);
        // The next block is 20 ms *before* where the flushed audio ended (clock fast):
        // the chunk must not start before 30 ms.
        p.push(10.0, &vec![0.2f32; 1600], &mut |c| out.push(c))
            .unwrap();
        p.flush(&mut |c| out.push(c)).unwrap();
        assert_ordered(&out);
        assert!(out[1].start_ms >= 30);
    }

    #[test]
    fn slow_device_clock_opens_segments_but_stays_ordered() {
        // The samples run 1 % slower than the wall clock: the timestamps move ahead by 0.1 ms
        // per 10 ms block. Every 80 ms of drift a new segment opens.
        let mut p = Processor::new(SourceKind::Mic, 16_000).unwrap();
        let mut out = Vec::new();
        let input = vec![0.1f32; 16_000 * 30];
        feed_clock(&mut p, 16_000, 0.0, 1.01, &input, 160, &mut out);
        p.flush(&mut |c| out.push(c)).unwrap();
        assert_eq!(total(&out), input.len(), "no audio lost");
        assert_ordered(&out);
        // The last chunk is near the wall-clock time of the last block (30 s × 1.01).
        let last = out.last().unwrap();
        let end = last.start_ms + last.samples.len() as u64 / 16;
        assert!(end.abs_diff(30_300) <= 100, "end {end}");
    }

    #[test]
    fn fast_device_clock_drops_audio_to_follow_the_wall_clock() {
        // The samples run 1 % faster than the wall clock. Without a fix, start_ms would run
        // 300 ms ahead after 30 s. The leading audio of late blocks is dropped instead.
        let mut p = Processor::new(SourceKind::Mic, 16_000).unwrap();
        let mut out = Vec::new();
        let input = vec![0.1f32; 16_000 * 30];
        feed_clock(&mut p, 16_000, 0.0, 0.99, &input, 160, &mut out);
        p.flush(&mut |c| out.push(c)).unwrap();
        assert_ordered(&out);
        let dropped = input.len() - total(&out);
        // About 1 % of the audio (300 ms) is dropped, in steps of at most a block or two.
        assert!((3_000..=6_000).contains(&dropped), "dropped {dropped}");
        let last = out.last().unwrap();
        let end = last.start_ms + last.samples.len() as u64 / 16;
        assert!(end.abs_diff(29_700) <= 100, "end {end}");
    }

    #[test]
    fn start_never_negative() {
        let mut p = Processor::new(SourceKind::Mic, 16_000).unwrap();
        let mut out = Vec::new();
        // The block starts before t0 (capture time earlier than the session start).
        p.push(-20.0, &vec![0.0f32; 1600], &mut |c| out.push(c))
            .unwrap();
        assert_eq!(out[0].start_ms, 0);
    }

    #[test]
    fn zero_watch_reports_once_after_three_seconds_of_zeros() {
        let mut w = ZeroWatch::new();
        let zeros = vec![0.0f32; 4_800];
        let mut hits = 0;
        for _ in 0..40 {
            if w.feed(&zeros, 48_000) {
                hits += 1;
            }
        }
        assert_eq!(hits, 1);
    }

    #[test]
    fn zero_watch_is_quiet_when_sound_arrives() {
        let mut w = ZeroWatch::new();
        let zeros = vec![0.0f32; 4_800];
        for _ in 0..10 {
            assert!(!w.feed(&zeros, 48_000));
        }
        let mut quiet = zeros.clone();
        quiet[10] = 1e-5;
        assert!(!w.feed(&quiet, 48_000));
        for _ in 0..100 {
            assert!(!w.feed(&zeros, 48_000));
        }
    }
}
