//! The queue between a device callback and the worker thread, and the worker loop.
//!
//! The callback side (`Feeder::feed`) never blocks, logs or allocates: sample buffers come from
//! a fixed pool, a full queue only increments a counter, and the worker reports the counter.
//!
//! One exception, accepted: `try_send` on a crossbeam channel can take the channel's internal
//! waker lock to wake the worker, which is parked in `recv_timeout` most of the time. That lock
//! is held for a few instructions and only by the worker's park and the callback's wake-up, never
//! across I/O, so the callback does not wait on anything slow.

use super::pipeline::{Processor, ZeroEvent, ZeroWatch};
use super::{ErrorSink, silent_message};
use crate::types::{AudioChunk, SourceKind};
use crossbeam_channel::{Receiver, RecvTimeoutError, SendTimeoutError, Sender, TrySendError};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Blocks in flight between the callback and the worker. A block is about 10 ms.
const QUEUE_BLOCKS: usize = 128;
/// Samples reserved in each pooled buffer (100 ms at 48 kHz). A larger block grows its buffer once.
const BLOCK_CAPACITY: usize = 4_800;
/// With no block for this long, the worker sends the audio it holds (loopback sends no packets
/// while nothing plays, so the end of a sentence would wait for the next sound).
const IDLE_FLUSH: Duration = Duration::from_millis(150);
/// How long the worker waits for room in the output channel before it drops a chunk.
const SEND_TIMEOUT: Duration = Duration::from_millis(200);

/// Mono samples at the device rate, with the capture time of the first sample.
pub struct Block {
    pub start: Instant,
    pub rate: u32,
    pub mono: Vec<f32>,
}

/// Callback side of the queue.
#[derive(Clone)]
pub struct Feeder {
    tx: Sender<Block>,
    pool_rx: Receiver<Vec<f32>>,
    pool_tx: Sender<Vec<f32>>,
    dropped: Arc<AtomicU32>,
}

impl Feeder {
    /// Queues one block. `fill` writes the mono samples into an empty pooled buffer. Never
    /// blocks, logs or allocates (a buffer may grow once for a larger block).
    pub fn feed(&self, start: Instant, rate: u32, fill: impl FnOnce(&mut Vec<f32>)) {
        let Ok(mut mono) = self.pool_rx.try_recv() else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        mono.clear();
        fill(&mut mono);
        if let Err(full) = self.tx.try_send(Block { start, rate, mono }) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            let block = match full {
                TrySendError::Full(b) | TrySendError::Disconnected(b) => b,
            };
            let _ = self.pool_tx.try_send(block.mono);
        }
    }
}

/// Worker side of the queue.
pub struct Inbox {
    rx: Receiver<Block>,
    pool_tx: Sender<Vec<f32>>,
    dropped: Arc<AtomicU32>,
}

pub fn queue() -> (Feeder, Inbox) {
    let (tx, rx) = crossbeam_channel::bounded(QUEUE_BLOCKS);
    let (pool_tx, pool_rx) = crossbeam_channel::bounded(QUEUE_BLOCKS);
    for _ in 0..QUEUE_BLOCKS {
        let _ = pool_tx.try_send(Vec::with_capacity(BLOCK_CAPACITY));
    }
    let dropped = Arc::new(AtomicU32::new(0));
    (
        Feeder {
            tx,
            pool_rx,
            pool_tx: pool_tx.clone(),
            dropped: dropped.clone(),
        },
        Inbox {
            rx,
            pool_tx,
            dropped,
        },
    )
}

pub struct Output {
    pub source: SourceKind,
    pub t0: Instant,
    pub tx: Sender<AudioChunk>,
    pub report: ErrorSink,
    /// Set by the worker when nobody reads the chunks any more (the engine stopped).
    pub closing: Arc<AtomicBool>,
    /// Report a stream that starts with exact zeros (denied permission). Off for macOS system
    /// audio: the tap delivers zeros while nothing plays, which is normal.
    pub watch_zeros: bool,
}

/// Runs until every `Feeder` is gone and the queue is empty. The audio that was already queued
/// is processed and flushed first.
pub fn run(inbox: Inbox, out: &Output) {
    let mut processor: Option<Processor> = None;
    let mut watch = ZeroWatch::new();
    let receiver_gone = std::cell::Cell::new(false);
    let mut emit = |chunk: AudioChunk| match out.tx.send_timeout(chunk, SEND_TIMEOUT) {
        Ok(()) => {}
        Err(SendTimeoutError::Timeout(_)) => {
            log::warn!(
                "audio {:?}: output channel is full, chunk dropped",
                out.source
            );
        }
        Err(SendTimeoutError::Disconnected(_)) => receiver_gone.set(true),
    };
    loop {
        let dropped = inbox.dropped.swap(0, Ordering::Relaxed);
        if dropped > 0 {
            log::warn!(
                "audio {:?}: worker was behind, {dropped} blocks dropped",
                out.source
            );
        }
        match inbox.rx.recv_timeout(IDLE_FLUSH) {
            Ok(block) => {
                handle(&block, out, &mut processor, &mut watch, &mut emit);
                let _ = inbox.pool_tx.try_send(block.mono);
            }
            Err(RecvTimeoutError::Timeout) => {
                if let Some(p) = processor.as_mut()
                    && let Err(e) = p.flush(&mut emit)
                {
                    log::error!("audio {:?}: {e}", out.source);
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
        if receiver_gone.get() {
            out.closing.store(true, Ordering::Release);
            return;
        }
    }
    if let Some(p) = processor.as_mut()
        && let Err(e) = p.flush(&mut emit)
    {
        log::error!("audio {:?}: {e}", out.source);
    }
}

fn handle(
    block: &Block,
    out: &Output,
    processor: &mut Option<Processor>,
    watch: &mut ZeroWatch,
    emit: &mut dyn FnMut(AudioChunk),
) {
    if out.watch_zeros {
        match watch.feed(&block.mono, block.rate) {
            Some(ZeroEvent::Silent) => {
                log::warn!("audio {:?}: only digital silence at the start", out.source);
                (out.report)(silent_message(out.source, cfg!(target_os = "macos")));
            }
            Some(ZeroEvent::Sound) => {
                log::info!("audio {:?}: sound after the silent start", out.source);
                (out.report)(String::new());
            }
            None => {}
        }
    }
    if processor.as_ref().is_some_and(|p| p.rate() != block.rate) {
        // The device changed to another sample rate: finish the old audio first.
        if let Some(mut old) = processor.take()
            && let Err(e) = old.flush(emit)
        {
            log::error!("audio {:?}: {e}", out.source);
        }
    }
    if processor.is_none() {
        match Processor::new(out.source, block.rate) {
            Ok(p) => *processor = Some(p),
            Err(e) => {
                log::error!("audio {:?}: {e}", out.source);
                return;
            }
        }
    }
    let start_ms = block.start.saturating_duration_since(out.t0).as_secs_f64() * 1000.0;
    if let Some(p) = processor.as_mut()
        && let Err(e) = p.push(start_ms, &block.mono, emit)
    {
        // Start again from the next block rather than end the capture.
        log::error!("audio {:?}: {e}", out.source);
        *processor = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn output(
        source: SourceKind,
        t0: Instant,
    ) -> (Output, Receiver<AudioChunk>, Arc<Mutex<Vec<String>>>) {
        let (tx, rx) = crossbeam_channel::bounded(1024);
        let reports = Arc::new(Mutex::new(Vec::new()));
        let r = reports.clone();
        let out = Output {
            source,
            t0,
            tx,
            report: Arc::new(move |m| r.lock().unwrap().push(m)),
            closing: Arc::new(AtomicBool::new(false)),
            watch_zeros: true,
        };
        (out, rx, reports)
    }

    #[test]
    fn queued_blocks_are_processed_and_flushed_after_the_feeders_go() {
        let t0 = Instant::now();
        let (feeder, inbox) = queue();
        // 25 blocks of 10 ms wait in the queue before the worker runs.
        for i in 0..25u32 {
            feeder.feed(t0 + Duration::from_millis(u64::from(i) * 10), 16_000, |v| {
                v.extend(std::iter::repeat_n(0.1, 160));
            });
        }
        drop(feeder);
        let (out, rx, _) = output(SourceKind::Mic, t0);
        run(inbox, &out);
        let total: usize = rx.try_iter().map(|c| c.samples.len()).sum();
        assert_eq!(total, 25 * 160, "no queued audio may be lost");
    }

    #[test]
    fn idle_flush_sends_audio_without_waiting_for_more() {
        let t0 = Instant::now();
        let (feeder, inbox) = queue();
        feeder.feed(t0, 16_000, |v| v.extend(std::iter::repeat_n(0.1, 480)));
        let (out, rx, _) = output(SourceKind::System, t0);
        let handle = std::thread::scope(|s| {
            let worker = s.spawn(|| run(inbox, &out));
            // The feeder stays alive: only the idle timer can release the 30 ms.
            let chunk = rx.recv_timeout(Duration::from_secs(2)).expect("idle flush");
            drop(feeder);
            worker.join().unwrap();
            chunk
        });
        assert_eq!(handle.samples.len(), 480);
        assert_eq!(handle.source, SourceKind::System);
    }

    #[test]
    fn full_pool_counts_drops_and_the_worker_reports_them_later() {
        let t0 = Instant::now();
        let (feeder, inbox) = queue();
        for _ in 0..QUEUE_BLOCKS + 10 {
            feeder.feed(t0, 16_000, |v| v.extend(std::iter::repeat_n(0.1, 160)));
        }
        assert_eq!(inbox.dropped.load(Ordering::Relaxed), 10);
        drop(feeder);
        let (out, rx, _) = output(SourceKind::Mic, t0);
        run(inbox, &out);
        assert!(rx.try_iter().count() > 0);
    }

    #[test]
    fn silent_start_is_reported_once() {
        let t0 = Instant::now();
        let (feeder, inbox) = queue();
        for i in 0..50u32 {
            feeder.feed(
                t0 + Duration::from_millis(u64::from(i) * 100),
                16_000,
                |v| {
                    v.extend(std::iter::repeat_n(0.0, 1_600));
                },
            );
        }
        drop(feeder);
        let (out, _rx, reports) = output(SourceKind::Mic, t0);
        run(inbox, &out);
        assert_eq!(reports.lock().unwrap().len(), 1);
    }

    #[test]
    fn sound_after_a_silent_start_clears_the_report() {
        let t0 = Instant::now();
        let (feeder, inbox) = queue();
        for i in 0..50u32 {
            let level = if i < 40 { 0.0 } else { 0.1 };
            feeder.feed(
                t0 + Duration::from_millis(u64::from(i) * 100),
                16_000,
                |v| {
                    v.extend(std::iter::repeat_n(level, 1_600));
                },
            );
        }
        drop(feeder);
        let (out, _rx, reports) = output(SourceKind::Mic, t0);
        run(inbox, &out);
        let reports = reports.lock().unwrap();
        assert_eq!(reports.len(), 2);
        assert!(!reports[0].is_empty());
        assert!(reports[1].is_empty(), "an empty message clears the error");
    }

    #[test]
    fn zero_watch_can_be_off() {
        let t0 = Instant::now();
        let (feeder, inbox) = queue();
        for i in 0..50u32 {
            feeder.feed(
                t0 + Duration::from_millis(u64::from(i) * 100),
                16_000,
                |v| {
                    v.extend(std::iter::repeat_n(0.0, 1_600));
                },
            );
        }
        drop(feeder);
        let (mut out, _rx, reports) = output(SourceKind::System, t0);
        out.watch_zeros = false;
        run(inbox, &out);
        assert!(reports.lock().unwrap().is_empty());
    }

    #[test]
    fn rate_change_flushes_the_old_audio() {
        let t0 = Instant::now();
        let (feeder, inbox) = queue();
        feeder.feed(t0, 48_000, |v| v.extend(std::iter::repeat_n(0.1, 4_800)));
        feeder.feed(t0 + Duration::from_millis(100), 16_000, |v| {
            v.extend(std::iter::repeat_n(0.1, 1_600));
        });
        drop(feeder);
        let (out, rx, _) = output(SourceKind::Mic, t0);
        run(inbox, &out);
        let total: usize = rx.try_iter().map(|c| c.samples.len()).sum();
        assert_eq!(total, 3_200);
    }

    #[test]
    fn worker_ends_when_nobody_reads_the_chunks() {
        let t0 = Instant::now();
        let (feeder, inbox) = queue();
        let (out, rx, _) = output(SourceKind::Mic, t0);
        drop(rx);
        for i in 0..20u32 {
            feeder.feed(t0 + Duration::from_millis(u64::from(i) * 10), 16_000, |v| {
                v.extend(std::iter::repeat_n(0.1, 160));
            });
        }
        run(inbox, &out); // the feeder is alive: the worker must stop by itself
        assert!(out.closing.load(Ordering::Acquire));
    }
}
