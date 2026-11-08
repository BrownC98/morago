//! Local model catalog and downloads (ASR, VAD, speaker embedding, translation LLM).
//! Owner: speech worker. Contract (do not change signatures without the coordinator).

use crate::types::DownloadProgress;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Asr,
    Vad,
    SpeakerEmbedding,
    Translation,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    /// Stable id, for example "whisper-small", "qwen2.5-3b-instruct-q4".
    pub id: String,
    pub kind: ModelKind,
    /// Display name, for example "Whisper small".
    pub name: String,
    pub size_bytes: u64,
    /// Short user-facing note (Korean), for example "빠름, 정확도 보통".
    pub note: String,
}

/// All models that the app can download.
pub fn catalog() -> Vec<ModelInfo> {
    Vec::new()
}

/// Folder or file of an installed model.
pub fn path(models_dir: &Path, id: &str) -> PathBuf {
    models_dir.join(id)
}

pub fn is_installed(models_dir: &Path, id: &str) -> bool {
    path(models_dir, id).exists()
}

/// Downloads and unpacks a model. Writes to a temp name, then renames,
/// so a cancelled or failed download leaves no partial model.
pub fn download(
    models_dir: &Path,
    id: &str,
    progress: &dyn Fn(DownloadProgress),
    cancel: &AtomicBool,
) -> anyhow::Result<()> {
    let _ = (models_dir, id, progress, cancel);
    anyhow::bail!("model download is not implemented yet")
}
