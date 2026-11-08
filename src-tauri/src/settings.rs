//! User settings, saved as JSON in the app config folder.
//! Owner: secrets/storage worker (high risk). Holds no API key: keys are in `secrets`.
//! Contract (do not change the types without the coordinator).

use crate::types::Endpoint;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AsrConfig {
    /// Local model from `models::catalog()` (kind Asr).
    Local { model_id: String },
    /// OpenAI-compatible `POST {base_url}/audio/transcriptions`.
    Custom { endpoint: Endpoint },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TranslateConfig {
    /// Local LLM from `models::catalog()` (kind Translation).
    Local { model_id: String },
    /// OpenAI-compatible `POST {base_url}/chat/completions`.
    Custom { endpoint: Endpoint },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Theme {
    Light,
    Dark,
    /// Follows the OS color scheme live.
    #[default]
    System,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub theme: Theme,
    /// ISO 639-1 code, default "ko".
    pub target_language: String,
    pub asr: AsrConfig,
    pub translate: TranslateConfig,
    pub diarize: bool,
    pub capture_mic: bool,
    pub capture_system: bool,
    /// `None` = default input device.
    pub mic_device: Option<String>,
    /// `None` = `<documents>/morago/sessions`.
    pub sessions_dir: Option<PathBuf>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            theme: Theme::System,
            target_language: "ko".into(),
            asr: AsrConfig::Local {
                model_id: "whisper-small".into(),
            },
            translate: TranslateConfig::Local {
                model_id: "qwen2.5-3b-instruct-q4".into(),
            },
            diarize: true,
            capture_mic: true,
            capture_system: true,
            mic_device: None,
            sessions_dir: None,
        }
    }
}

/// Reads settings; a missing file gives the defaults.
pub fn load(path: &Path) -> anyhow::Result<Settings> {
    let _ = path;
    Ok(Settings::default())
}

/// Writes settings atomically (temp file + rename).
pub fn save(path: &Path, settings: &Settings) -> anyhow::Result<()> {
    let _ = (path, settings);
    anyhow::bail!("settings save is not implemented yet")
}
