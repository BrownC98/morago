//! Session storage and export. Owner: secrets/storage worker (high risk).
//! Layout: `<sessions_dir>/<id>/meta.json` + `utterances.jsonl`.
//! `utterances.jsonl` is append-only; a later line with the same `id`
//! replaces the earlier one (translation updates), so a crash loses at most
//! the last line. Contract (do not change signatures without the coordinator).

use crate::types::{ExportFormat, SessionMeta, SessionSummary, Utterance};
use std::path::{Path, PathBuf};

pub struct SessionStore {
    root: PathBuf,
}

/// Appends utterances of one running session.
pub struct SessionWriter {
    _private: (),
}

impl SessionWriter {
    /// Appends (or replaces by id) one utterance and flushes it to disk.
    pub fn append(&mut self, utterance: &Utterance) -> anyhow::Result<()> {
        let _ = utterance;
        anyhow::bail!("not implemented yet")
    }
    /// Writes `ended_at` into meta.json.
    pub fn finish(self, ended_at: &str) -> anyhow::Result<()> {
        let _ = ended_at;
        anyhow::bail!("not implemented yet")
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
        let _ = meta;
        anyhow::bail!("not implemented yet")
    }
    /// Newest first. A broken session folder is skipped, not an error.
    pub fn list(&self) -> anyhow::Result<Vec<SessionSummary>> {
        Ok(Vec::new())
    }
    pub fn load(&self, id: &str) -> anyhow::Result<(SessionMeta, Vec<Utterance>)> {
        let _ = id;
        anyhow::bail!("not implemented yet")
    }
    /// Changes title and speaker names in meta.json.
    pub fn update_meta(&self, meta: &SessionMeta) -> anyhow::Result<()> {
        let _ = meta;
        anyhow::bail!("not implemented yet")
    }
    pub fn delete(&self, id: &str) -> anyhow::Result<()> {
        let _ = id;
        anyhow::bail!("not implemented yet")
    }
    /// Text of the export file. Uses speaker display names.
    pub fn export(&self, id: &str, format: ExportFormat) -> anyhow::Result<String> {
        let _ = (id, format);
        anyhow::bail!("not implemented yet")
    }
}
