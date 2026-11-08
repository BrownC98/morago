//! Session storage and export. Owner: secrets/storage worker (high risk).
//! Layout: `<sessions_dir>/<id>/meta.json` + `utterances.jsonl`.
//! `utterances.jsonl` is append-only; a later line with the same `id`
//! replaces the earlier one (translation updates), so a crash loses at most
//! the last line. Contract (do not change signatures without the coordinator).

mod export;

use crate::types::{ExportFormat, SessionMeta, SessionSummary, Utterance};
use anyhow::Context;
use std::collections::{BTreeMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const META: &str = "meta.json";
const UTTERANCES: &str = "utterances.jsonl";
/// `append` calls `sync_data` at most this often; `finish` always does.
const SYNC_INTERVAL: Duration = Duration::from_secs(2);

/// Serializes read-modify-write of meta.json (`update_meta` and `finish`
/// can run at the same time on different threads).
static META_LOCK: Mutex<()> = Mutex::new(());

pub struct SessionStore {
    root: PathBuf,
}

/// Appends utterances of one running session.
pub struct SessionWriter {
    dir: PathBuf,
    meta: SessionMeta,
    file: File,
    last_sync: Instant,
    /// A failed write can leave a part line; the next line then starts on
    /// a new line, so only the part line is lost.
    torn: bool,
}

impl SessionWriter {
    /// Appends (or replaces by id) one utterance and flushes it to disk.
    pub fn append(&mut self, utterance: &Utterance) -> anyhow::Result<()> {
        let mut line = Vec::new();
        if self.torn {
            line.push(b'\n');
        }
        serde_json::to_writer(&mut line, utterance)?;
        line.push(b'\n');
        // One write call per line: a crash can tear only the last line.
        self.torn = true;
        self.file
            .write_all(&line)
            .context("발화를 기록 파일에 쓸 수 없습니다. 디스크 공간과 권한을 확인하세요")?;
        self.torn = false;
        if self.last_sync.elapsed() >= SYNC_INTERVAL {
            self.file
                .sync_data()
                .context("발화 기록을 디스크에 쓸 수 없습니다")?;
            self.last_sync = Instant::now();
        }
        Ok(())
    }
    /// Writes `ended_at` into meta.json.
    pub fn finish(self, ended_at: &str) -> anyhow::Result<()> {
        self.file
            .sync_data()
            .context("발화 기록을 디스크에 쓸 수 없습니다")?;
        let _guard = META_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // The title or speaker names can change while recording.
        let mut meta = read_meta(&self.dir).unwrap_or_else(|_| self.meta.clone());
        meta.ended_at = Some(ended_at.to_string());
        write_meta(&self.dir, &meta)
    }
}

impl Drop for SessionWriter {
    fn drop(&mut self) {
        // A writer dropped without `finish` (error path) still syncs its lines.
        let _ = self.file.sync_data();
    }
}

impl SessionStore {
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
        }
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn create(&self, meta: &SessionMeta) -> anyhow::Result<SessionWriter> {
        let dir = self.dir(&meta.id)?;
        std::fs::create_dir_all(&self.root).with_context(|| {
            format!(
                "기록 폴더를 만들 수 없습니다({}). 설정에서 다른 폴더를 고르세요",
                self.root.display()
            )
        })?;
        std::fs::create_dir(&dir)
            .with_context(|| format!("기록 폴더를 만들 수 없습니다({})", dir.display()))?;
        write_meta(&dir, meta)?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join(UTTERANCES))
            .context("발화 기록 파일을 만들 수 없습니다")?;
        Ok(SessionWriter {
            dir,
            meta: meta.clone(),
            file,
            last_sync: Instant::now(),
            torn: false,
        })
    }
    /// Newest first. A broken session folder is skipped, not an error.
    pub fn list(&self) -> anyhow::Result<Vec<SessionSummary>> {
        let entries = match std::fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "기록 폴더를 읽을 수 없습니다({}): {e}",
                    self.root.display()
                ));
            }
        };
        let mut list = Vec::new();
        for entry in entries.flatten() {
            let Some(id) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if !valid_id(&id) {
                continue;
            }
            match self.load(&id) {
                Ok((meta, utterances)) => list.push(summary(meta, &utterances)),
                Err(e) => log::warn!("skipped broken session folder {id}: {e:#}"),
            }
        }
        list.sort_by_cached_key(|s| {
            let started = chrono::DateTime::parse_from_rfc3339(&s.meta.started_at).ok();
            std::cmp::Reverse((started, s.meta.id.clone()))
        });
        Ok(list)
    }
    pub fn load(&self, id: &str) -> anyhow::Result<(SessionMeta, Vec<Utterance>)> {
        let dir = self.dir(id)?;
        let meta = read_meta(&dir)?;
        // Bytes, not a string: a crash can cut a UTF-8 character in two.
        let bytes = match std::fs::read(dir.join(UTTERANCES)) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e).context("발화 기록 파일을 읽을 수 없습니다"),
        };
        Ok((meta, parse_utterances(id, &bytes)))
    }
    /// Changes title and speaker names in meta.json.
    pub fn update_meta(&self, meta: &SessionMeta) -> anyhow::Result<()> {
        let dir = self.dir(&meta.id)?;
        let _guard = META_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut stored = read_meta(&dir)?;
        stored.title = meta.title.clone();
        stored.speaker_names = meta.speaker_names.clone();
        write_meta(&dir, &stored)
    }
    pub fn delete(&self, id: &str) -> anyhow::Result<()> {
        let dir = self.dir(id)?;
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => {
                Err(e).context("기록을 지울 수 없습니다. 다른 프로그램이 파일을 쓰는지 확인하세요")
            }
        }
    }
    /// Text of the export file. Uses speaker display names.
    pub fn export(&self, id: &str, format: ExportFormat) -> anyhow::Result<String> {
        let (meta, utterances) = self.load(id)?;
        Ok(match format {
            ExportFormat::Markdown => export::markdown(&meta, &utterances),
            ExportFormat::Srt => export::srt(&meta, &utterances),
            ExportFormat::Json => export::json(&meta, &utterances)?,
        })
    }

    /// The session folder. The id comes from the UI: only the exact id form
    /// passes, so the path cannot leave `root`.
    fn dir(&self, id: &str) -> anyhow::Result<PathBuf> {
        if !valid_id(id) {
            anyhow::bail!("잘못된 기록 id입니다");
        }
        Ok(self.root.join(id))
    }
}

/// `^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$`
fn valid_id(id: &str) -> bool {
    let b = id.as_bytes();
    b.len() == 20
        && b.iter().enumerate().all(|(i, &c)| match i {
            8 | 15 => c == b'-',
            0..=14 => c.is_ascii_digit(),
            _ => c.is_ascii_digit() || (b'a'..=b'f').contains(&c),
        })
}

fn parse_utterances(id: &str, bytes: &[u8]) -> Vec<Utterance> {
    let mut by_id = BTreeMap::new();
    for (n, line) in bytes.split(|&b| b == b'\n').enumerate() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        match serde_json::from_slice::<Utterance>(line) {
            Ok(u) => {
                by_id.insert(u.id, u);
            }
            // A crash during a write leaves a torn last line. The log has
            // no text from the line (meeting content).
            Err(e) => log::warn!(
                "session {id}: skipped broken line {} ({})",
                n + 1,
                json_error_place(&e)
            ),
        }
    }
    by_id.into_values().collect()
}

/// Kind and place of a JSON error, without the values that serde_json
/// puts in its message.
pub(crate) fn json_error_place(e: &serde_json::Error) -> String {
    format!(
        "{:?} error at line {} column {}",
        e.classify(),
        e.line(),
        e.column()
    )
}

fn summary(meta: SessionMeta, utterances: &[Utterance]) -> SessionSummary {
    let speakers: HashSet<&str> = utterances
        .iter()
        .map(|u| u.speaker.as_str())
        .filter(|s| !s.is_empty())
        .collect();
    SessionSummary {
        meta,
        duration_ms: utterances.iter().map(|u| u.end_ms).max().unwrap_or(0),
        utterance_count: utterances.len() as u64,
        speaker_count: speakers.len() as u64,
    }
}

fn read_meta(dir: &Path) -> anyhow::Result<SessionMeta> {
    let bytes = std::fs::read(dir.join(META)).context("기록 정보 파일을 읽을 수 없습니다")?;
    serde_json::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("기록 정보 파일이 손상되었습니다({})", json_error_place(&e)))
}

fn write_meta(dir: &Path, meta: &SessionMeta) -> anyhow::Result<()> {
    write_atomic(&dir.join(META), &serde_json::to_vec_pretty(meta)?)
        .context("기록 정보 파일을 쓸 수 없습니다")
}

/// Replaces `path` with `bytes` in one step: temp file in the same folder,
/// fsync, rename (replaces an existing file on Windows too). The parent
/// folder is made if missing.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    std::fs::create_dir_all(dir)?;
    let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
    tmp.write_all(bytes)?;
    tmp.as_file().sync_all()?;
    tmp.persist(path).map_err(|e| e.error)?;
    #[cfg(unix)]
    File::open(dir)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests;
