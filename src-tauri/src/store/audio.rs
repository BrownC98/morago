//! Meeting recording: `<session>/audio.wav`, one mono WAV, 16 kHz 16-bit PCM.
//! The chunks of both sources go on the session timeline (`start_ms`):
//! sample `n` of the file is at `n / 16` ms, so the file lines up with the
//! utterance times. Gaps are silence, overlaps are summed and clamped.
//!
//! Crash safety: the data goes to the file in order, and the header sizes are
//! written and synced every `SYNC_INTERVAL`. `repair` sets the sizes from the
//! file length after a crash, so the file plays up to the kill.

use super::SYNC_INTERVAL;
use crate::types::{AudioChunk, SAMPLE_RATE};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, Instant};

pub(crate) const AUDIO: &str = "audio.wav";
const HEADER_LEN: u64 = 44;
/// Largest data chunk that the 32-bit RIFF size can hold, in whole samples
/// (bytes); about 37 hours.
const MAX_DATA: u64 = (u32::MAX as u64 - 36) & !1;
const RATE: u64 = SAMPLE_RATE as u64;
/// Audio time held back before it goes to the file, so a source that comes
/// a little later than the other still mixes in.
const HOLD: u64 = RATE;
/// A chunk that starts this near the end of the previous chunk of its source
/// continues it: `start_ms` is rounded to whole ms.
const JITTER: u64 = RATE / 50;

/// A file that the recorder can write and sync. A trait so tests can fail
/// a write.
pub trait AudioFile: Write + Seek {
    fn sync(&mut self) -> io::Result<()>;
}

impl AudioFile for File {
    fn sync(&mut self) -> io::Result<()> {
        self.sync_data()
    }
}

/// Mixes and writes the audio of one running session. A write error ends the
/// recording (the caller drops the recorder); the file then plays up to the
/// last good write after `repair`.
pub struct AudioRecorder<F: AudioFile = File> {
    file: F,
    /// Samples in the file.
    written: u64,
    /// Mixed samples from `written` on, not yet in the file.
    pending: VecDeque<f32>,
    /// For each source: where its next chunk continues (sample index).
    next: [Option<u64>; 2],
    /// End of the newest chunk (sample index).
    end: u64,
    late_logged: bool,
    last_sync: Instant,
    sync_every: Duration,
}

impl AudioRecorder<File> {
    /// Makes a new `path`; an existing file is an error.
    pub fn create(path: &Path) -> io::Result<Self> {
        Self::new(File::create_new(path)?, SYNC_INTERVAL)
    }
}

impl<F: AudioFile> AudioRecorder<F> {
    fn new(mut file: F, sync_every: Duration) -> io::Result<Self> {
        file.write_all(&header(0))?;
        file.sync()?;
        Ok(Self {
            file,
            written: 0,
            pending: VecDeque::new(),
            next: [None; 2],
            end: 0,
            late_logged: false,
            last_sync: Instant::now(),
            sync_every,
        })
    }

    /// Mixes one chunk in. Audio older than `HOLD` before the newest chunk
    /// goes to the file.
    pub fn push(&mut self, chunk: &AudioChunk) -> io::Result<()> {
        let len = chunk.samples.len() as u64;
        if len == 0 {
            return Ok(());
        }
        let i = chunk.source as usize;
        let at = chunk.start_ms * RATE / 1000;
        let pos = match self.next[i] {
            Some(n) if at.abs_diff(n) <= JITTER => n,
            _ => at,
        };
        self.next[i] = Some(pos + len);
        if pos > self.written + HOLD {
            // Far ahead: write the audio before the hold window first (silence
            // for a gap), so `pending` stays bounded.
            self.write_until(pos - HOLD)?;
        }
        let skip = self.written.saturating_sub(pos);
        if skip > 0 && !self.late_logged {
            self.late_logged = true;
            log::warn!(
                "recording: {:?} audio came more than {} ms late; dropped {skip} samples (logged once)",
                chunk.source,
                HOLD * 1000 / RATE
            );
        }
        if skip < len {
            let off = (pos + skip - self.written) as usize;
            let samples = &chunk.samples[skip as usize..];
            if self.pending.len() < off + samples.len() {
                self.pending.resize(off + samples.len(), 0.0);
            }
            for (slot, s) in self.pending.range_mut(off..).zip(samples) {
                *slot += s;
            }
        }
        self.end = self.end.max(pos + len);
        if self.end > self.written + HOLD {
            self.write_until(self.end - HOLD)?;
        }
        if self.last_sync.elapsed() >= self.sync_every {
            self.sync()?;
        }
        Ok(())
    }

    /// Writes the rest and the final header.
    pub fn finish(mut self) -> io::Result<()> {
        self.write_until(self.end)?;
        self.sync()
    }

    /// Writes mixed samples (silence where none) up to sample `target`.
    fn write_until(&mut self, target: u64) -> io::Result<()> {
        if target * 2 > MAX_DATA {
            return Err(io::Error::other(
                "녹음 파일이 WAV 최대 크기(약 37시간)에 닿았습니다",
            ));
        }
        let mut bytes = Vec::new();
        while self.written < target {
            let n = (target - self.written).min(RATE);
            bytes.clear();
            for _ in 0..n {
                let s = self.pending.pop_front().unwrap_or(0.0);
                bytes.extend_from_slice(&pcm(s).to_le_bytes());
            }
            self.file.write_all(&bytes)?;
            self.written += n;
        }
        Ok(())
    }

    /// Writes the header sizes and syncs data and header to the disk.
    fn sync(&mut self) -> io::Result<()> {
        let data = self.written * 2;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&header(data))?;
        self.file.seek(SeekFrom::Start(HEADER_LEN + data))?;
        self.file.sync()?;
        self.last_sync = Instant::now();
        Ok(())
    }
}

fn pcm(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16
}

/// The 44-byte header of a mono 16 kHz 16-bit PCM WAV with `data` bytes.
fn header(data: u64) -> [u8; HEADER_LEN as usize] {
    let data = data.min(MAX_DATA) as u32;
    let mut h = [0u8; HEADER_LEN as usize];
    let mut put = |at: usize, bytes: &[u8]| h[at..at + bytes.len()].copy_from_slice(bytes);
    put(0, b"RIFF");
    put(4, &(36 + data).to_le_bytes());
    put(8, b"WAVEfmt ");
    put(16, &16u32.to_le_bytes());
    put(20, &1u16.to_le_bytes()); // PCM
    put(22, &1u16.to_le_bytes()); // mono
    put(24, &SAMPLE_RATE.to_le_bytes());
    put(28, &(SAMPLE_RATE * 2).to_le_bytes()); // bytes per second
    put(32, &2u16.to_le_bytes()); // bytes per frame
    put(34, &16u16.to_le_bytes()); // bits per sample
    put(36, b"data");
    put(40, &data.to_le_bytes());
    h
}

/// Makes a recording that a crash left behind playable: cuts a part sample at
/// the end and sets the header sizes from the file length. A file without
/// samples (only a header, or less) is removed. Returns whether a file
/// remains. Do not call it while the recorder of the file runs.
pub(crate) fn repair(path: &Path) -> io::Result<bool> {
    let mut file = match OpenOptions::new().read(true).write(true).open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e),
    };
    let len = file.metadata()?.len();
    let mut old = [0u8; HEADER_LEN as usize];
    if len >= HEADER_LEN {
        file.read_exact(&mut old)?;
        if &old[..4] != b"RIFF" || &old[8..16] != b"WAVEfmt " || &old[36..40] != b"data" {
            // Not a file of this recorder: leave it as it is.
            return Ok(true);
        }
    }
    let data = (len.saturating_sub(HEADER_LEN) & !1).min(MAX_DATA);
    if data == 0 {
        drop(file);
        std::fs::remove_file(path)?;
        return Ok(false);
    }
    if len != HEADER_LEN + data {
        file.set_len(HEADER_LEN + data)?;
    }
    let new = header(data);
    if old != new || len != HEADER_LEN + data {
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&new)?;
        file.sync_data()?;
    }
    Ok(true)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::types::SourceKind;
    use std::io::Cursor;

    /// 100 ms chunks of one source from `from_ms` to `to_ms`; sample `k` of
    /// the stretch is `value(k)`.
    fn chunks(
        source: SourceKind,
        from_ms: u64,
        to_ms: u64,
        value: impl Fn(u64) -> f32,
    ) -> Vec<AudioChunk> {
        (from_ms..to_ms)
            .step_by(100)
            .map(|ms| AudioChunk {
                source,
                start_ms: ms,
                samples: (0..1600).map(|k| value((ms - from_ms) * 16 + k)).collect(),
            })
            .collect()
    }

    fn record(path: &Path, sync_every: Duration, chunks: &[AudioChunk]) -> AudioRecorder {
        let mut r = AudioRecorder::new(File::create_new(path).unwrap(), sync_every).unwrap();
        for c in chunks {
            r.push(c).unwrap();
        }
        r
    }

    fn read(path: &Path) -> Vec<i16> {
        let mut reader = hound::WavReader::open(path).unwrap();
        let spec = reader.spec();
        assert_eq!(
            (spec.channels, spec.sample_rate, spec.bits_per_sample),
            (1, 16_000, 16)
        );
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);
        reader.samples::<i16>().map(Result::unwrap).collect()
    }

    /// The value of sample `k`: a ramp that does not repeat within 100 ms.
    fn ramp(k: u64) -> f32 {
        (k % 2000) as f32 / 4000.0
    }

    #[test]
    fn round_trip_reads_back_with_hound() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(AUDIO);
        let r = record(
            &path,
            SYNC_INTERVAL,
            &chunks(SourceKind::Mic, 0, 3000, ramp),
        );
        r.finish().unwrap();
        let samples = read(&path);
        assert_eq!(samples.len(), 48_000);
        for (k, s) in samples.iter().enumerate() {
            assert_eq!(*s, pcm(ramp(k as u64)), "sample {k}");
        }
        assert!(repair(&path).unwrap());
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 44 + 96_000);
    }

    #[test]
    fn a_gap_becomes_silence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(AUDIO);
        let mut c = chunks(SourceKind::System, 200, 300, |_| 0.5);
        c.extend(chunks(SourceKind::System, 2500, 2600, |_| 0.5));
        record(&path, SYNC_INTERVAL, &c).finish().unwrap();
        let s = read(&path);
        assert_eq!(s.len(), 2600 * 16);
        let half = pcm(0.5);
        assert!(s[..3200].iter().all(|&v| v == 0));
        assert!(s[3200..4800].iter().all(|&v| v == half));
        assert!(s[4800..40_000].iter().all(|&v| v == 0));
        assert!(s[40_000..].iter().all(|&v| v == half));
    }

    #[test]
    fn two_sources_are_summed_and_clamped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(AUDIO);
        let mic = chunks(SourceKind::Mic, 0, 2000, |_| 0.25);
        let system = chunks(SourceKind::System, 0, 1000, |_| 0.5);
        let loud = chunks(SourceKind::System, 1000, 2000, |_| 0.9);
        // The system source comes 500 ms after the mic: it still mixes in.
        let mut r = AudioRecorder::new(File::create_new(&path).unwrap(), SYNC_INTERVAL).unwrap();
        for (k, m) in mic.iter().enumerate() {
            r.push(m).unwrap();
            if k >= 5 {
                let s = system.iter().chain(&loud).nth(k - 5).unwrap();
                r.push(s).unwrap();
            }
        }
        for s in system.iter().chain(&loud).skip(15) {
            r.push(s).unwrap();
        }
        r.finish().unwrap();
        let s = read(&path);
        assert_eq!(s.len(), 32_000);
        assert!(s[..16_000].iter().all(|&v| v == pcm(0.75)));
        assert!(s[16_000..].iter().all(|&v| v == i16::MAX));
    }

    #[test]
    fn a_late_chunk_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(AUDIO);
        let mut r = record(
            &path,
            SYNC_INTERVAL,
            &chunks(SourceKind::Mic, 0, 3000, |_| 0.25),
        );
        // 2 s are in the file now. Wholly late, then late in part.
        r.push(&chunks(SourceKind::System, 500, 600, |_| 0.5)[0])
            .unwrap();
        r.push(&chunks(SourceKind::System, 1900, 2000, |_| 0.5)[0])
            .unwrap();
        r.push(&chunks(SourceKind::System, 1950, 2050, |_| 0.5)[0])
            .unwrap();
        r.finish().unwrap();
        let s = read(&path);
        assert_eq!(s.len(), 48_000);
        assert!(s[..32_000].iter().all(|&v| v == pcm(0.25)));
        assert!(s[32_000..32_800].iter().all(|&v| v == pcm(0.75)));
        assert!(s[32_800..].iter().all(|&v| v == pcm(0.25)));
    }

    #[test]
    fn rounded_starts_continue_the_source() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(AUDIO);
        // 1601 samples are 100.06 ms; the next chunk says 100 ms.
        let a = AudioChunk {
            source: SourceKind::Mic,
            start_ms: 0,
            samples: vec![0.25; 1601],
        };
        let b = AudioChunk {
            start_ms: 100,
            ..a.clone()
        };
        let mut r = AudioRecorder::new(File::create_new(&path).unwrap(), SYNC_INTERVAL).unwrap();
        r.push(&a).unwrap();
        r.push(&b).unwrap();
        r.finish().unwrap();
        let s = read(&path);
        assert_eq!(s.len(), 3202);
        assert!(s.iter().all(|&v| v == pcm(0.25)));
    }

    #[test]
    fn the_buffer_stays_bounded() {
        let mut r = AudioRecorder::full_disk(usize::MAX);
        for c in chunks(SourceKind::Mic, 0, 60_000, |_| 0.1) {
            r.push(&c).unwrap();
            assert!(r.pending.len() as u64 <= HOLD + 1600);
        }
        // A chunk ten minutes ahead: the gap goes to the file as silence.
        r.push(&chunks(SourceKind::System, 600_000, 600_100, |_| 0.1)[0])
            .unwrap();
        assert!(r.pending.len() as u64 <= HOLD + 1600);
        assert_eq!(r.written, 600_100 * 16 - HOLD);
        r.finish().unwrap();
    }

    /// The app is killed: no `finish`, the last samples came after the last
    /// header sync, and a write was cut in the middle of a sample.
    #[test]
    fn a_killed_recording_plays_after_repair() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(AUDIO);
        let mut r = record(
            &path,
            Duration::ZERO,
            &chunks(SourceKind::Mic, 0, 3000, ramp),
        );
        assert_eq!(read(&path).len(), 32_000);
        r.sync_every = Duration::from_secs(3600);
        for c in chunks(SourceKind::Mic, 3000, 5000, |k| ramp(k + 48_000)) {
            r.push(&c).unwrap();
        }
        std::mem::forget(r);
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0x12])
            .unwrap();
        // The header still has the size of the last sync.
        let mut stale = [0u8; 44];
        File::open(&path).unwrap().read_exact(&mut stale).unwrap();
        assert_eq!(stale[40..44], 64_000u32.to_le_bytes());

        assert!(repair(&path).unwrap());
        let samples = read(&path);
        assert_eq!(samples.len(), 64_000);
        for (k, s) in samples.iter().enumerate() {
            assert_eq!(*s, pcm(ramp(k as u64)), "sample {k}");
        }
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 44 + 128_000);
        // A second repair changes nothing.
        let before = std::fs::read(&path).unwrap();
        assert!(repair(&path).unwrap());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn a_recording_without_samples_is_removed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(AUDIO);
        AudioRecorder::create(&path).unwrap().finish().unwrap();
        assert!(!repair(&path).unwrap());
        assert!(!path.exists());
        // A header cut by a kill.
        std::fs::write(&path, &header(0)[..20]).unwrap();
        assert!(!repair(&path).unwrap());
        assert!(!path.exists());
        assert!(!repair(&path).unwrap());
    }

    /// Takes `room` bytes, then fails like a full disk.
    pub(crate) struct FullDisk {
        pub file: Cursor<Vec<u8>>,
        pub room: usize,
    }

    impl Write for FullDisk {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.file.get_ref().len() + buf.len() > self.room {
                return Err(io::Error::new(io::ErrorKind::StorageFull, "disk full"));
            }
            self.file.write(buf)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Seek for FullDisk {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            self.file.seek(pos)
        }
    }

    impl AudioFile for FullDisk {
        fn sync(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl AudioRecorder<FullDisk> {
        pub(crate) fn full_disk(room: usize) -> Self {
            let disk = FullDisk {
                file: Cursor::new(Vec::new()),
                room,
            };
            AudioRecorder::new(disk, SYNC_INTERVAL).unwrap()
        }
    }

    #[test]
    fn a_full_disk_fails_the_push() {
        let mut r = AudioRecorder::full_disk(44 + 40_000);
        let mut result = Ok(());
        for c in chunks(SourceKind::Mic, 0, 3000, |_| 0.1) {
            result = r.push(&c);
            if result.is_err() {
                break;
            }
        }
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::StorageFull);
    }
}
