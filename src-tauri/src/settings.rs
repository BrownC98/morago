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
    Custom {
        endpoint: Endpoint,
        /// `None` = default: the request has no `reasoning_effort` key.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_effort: Option<ReasoningEffort>,
    },
}

/// Value of the `reasoning_effort` request field of a custom translation server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Sent as `"none"`.
    #[serde(rename = "none")]
    Off,
    Minimal,
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
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
    /// Meeting language for speech recognition, ISO 639-1 ("ko", "ja", "zh",
    /// "en"). `None` = detect per segment (default). When set, every ASR uses
    /// it and gives it as the utterance language.
    pub meeting_language: Option<String>,
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
                model_id: crate::models::SENSE_VOICE_ID.into(),
            },
            meeting_language: None,
            translate: TranslateConfig::Local {
                model_id: "gemma-4-e2b-it-q4".into(),
            },
            diarize: true,
            capture_mic: true,
            capture_system: true,
            mic_device: None,
            sessions_dir: None,
        }
    }
}

/// Reads settings; a missing file gives the defaults. A corrupt file is kept
/// as `settings.json.bad-<timestamp>` and the defaults are returned.
/// Unknown fields are ignored; missing fields take the defaults.
pub fn load(path: &Path) -> anyhow::Result<Settings> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Settings::default()),
        Err(e) => {
            return Err(anyhow::anyhow!(
                "설정 파일을 읽을 수 없습니다({}): {e}",
                path.display()
            ));
        }
    };
    // Bytes, so that bad UTF-8 or UTF-16 goes the corrupt-file way too.
    // A UTF-8 BOM (some Windows editors) is not damage.
    let json = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes);
    match serde_json::from_slice(json) {
        Ok(settings) => Ok(settings),
        Err(e) => {
            // The log has the error place only: serde_json messages carry values.
            let reason = crate::store::json_error_place(&e);
            let backup = backup_path(path);
            // A rename keeps the bytes; the next save writes a new file.
            match std::fs::rename(path, &backup) {
                Ok(()) => log::warn!(
                    "settings file is corrupt ({reason}); kept it as {} and used the defaults",
                    backup.display()
                ),
                Err(re) => log::error!(
                    "settings file is corrupt ({reason}); could not keep a copy at {}: {re}",
                    backup.display()
                ),
            }
            Ok(Settings::default())
        }
    }
}

/// Writes settings atomically (temp file + fsync + rename).
pub fn save(path: &Path, settings: &Settings) -> anyhow::Result<()> {
    let json = serde_json::to_vec_pretty(settings)?;
    crate::store::write_atomic(path, &json)
        .map_err(|e| anyhow::anyhow!("설정을 저장할 수 없습니다({}): {e}", path.display()))
}

fn backup_path(path: &Path) -> PathBuf {
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S%3f");
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".bad-{stamp}"));
    path.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn custom() -> Settings {
        Settings {
            theme: Theme::Dark,
            target_language: "en".into(),
            asr: AsrConfig::Custom {
                endpoint: Endpoint {
                    base_url: "http://localhost:8000/v1".into(),
                    model: "whisper-1".into(),
                },
            },
            meeting_language: Some("ja".into()),
            translate: TranslateConfig::Local {
                model_id: "m".into(),
            },
            diarize: false,
            capture_mic: true,
            capture_system: false,
            mic_device: Some("USB 마이크".into()),
            sessions_dir: Some(PathBuf::from("/tmp/세션")),
        }
    }

    #[test]
    fn round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("settings.json");
        save(&path, &custom()).unwrap();
        assert_eq!(load(&path).unwrap(), custom());
        // Overwrite an existing file.
        save(&path, &Settings::default()).unwrap();
        assert_eq!(load(&path).unwrap(), Settings::default());
        let names: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["settings.json"], "no temp file left");
    }

    fn translate_custom(effort: Option<ReasoningEffort>) -> Settings {
        Settings {
            translate: TranslateConfig::Custom {
                endpoint: Endpoint {
                    base_url: "http://localhost:8000/v1".into(),
                    model: "gpt-test".into(),
                },
                reasoning_effort: effort,
            },
            ..Settings::default()
        }
    }

    #[test]
    fn reasoning_effort_round_trip_with_lowercase_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        for (effort, text) in [
            (ReasoningEffort::Off, "none"),
            (ReasoningEffort::Minimal, "minimal"),
            (ReasoningEffort::Low, "low"),
            (ReasoningEffort::Medium, "medium"),
            (ReasoningEffort::High, "high"),
        ] {
            let s = translate_custom(Some(effort));
            save(&path, &s).unwrap();
            let file = std::fs::read_to_string(&path).unwrap();
            assert!(
                file.contains(&format!("\"reasoning_effort\": \"{text}\"")),
                "{file}"
            );
            assert_eq!(effort.as_str(), text);
            assert_eq!(load(&path).unwrap(), s);
        }
    }

    #[test]
    fn default_reasoning_effort_is_not_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        save(&path, &translate_custom(None)).unwrap();
        let file = std::fs::read_to_string(&path).unwrap();
        assert!(!file.contains("reasoning_effort"), "{file}");
        assert_eq!(load(&path).unwrap(), translate_custom(None));
    }

    #[test]
    fn a_file_from_before_reasoning_effort_loads_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(
            &path,
            r#"{"translate":{"kind":"custom","endpoint":{"base_url":"http://localhost:8000/v1","model":"gpt-test"}}}"#,
        )
        .unwrap();
        assert_eq!(load(&path).unwrap(), translate_custom(None));
    }

    #[test]
    fn a_file_with_a_whisper_model_keeps_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        assert_eq!(
            Settings::default().asr,
            AsrConfig::Local {
                model_id: crate::models::SENSE_VOICE_ID.into()
            }
        );
        for id in [
            "whisper-base",
            "whisper-small",
            "whisper-medium",
            "whisper-turbo",
        ] {
            std::fs::write(
                &path,
                format!(r#"{{"asr":{{"kind":"local","model_id":"{id}"}}}}"#),
            )
            .unwrap();
            let s = load(&path).unwrap();
            assert_eq!(
                s.asr,
                AsrConfig::Local {
                    model_id: id.into()
                }
            );
            // A file from before the meeting language setting: detect.
            assert_eq!(s.meeting_language, None);
            assert!(crate::models::catalog().iter().any(|m| m.id == id));
        }
    }

    #[test]
    fn missing_file_gives_defaults() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            load(&dir.path().join("settings.json")).unwrap(),
            Settings::default()
        );
    }

    /// Loads `bytes`: defaults come back and a backup keeps the exact bytes.
    fn assert_kept(bytes: &[u8]) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, bytes).unwrap();
        assert_eq!(load(&path).unwrap(), Settings::default());
        let backups: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(backups.len(), 1);
        assert!(backups[0].starts_with("settings.json.bad-"), "{backups:?}");
        let kept = std::fs::read(dir.path().join(&backups[0])).unwrap();
        assert_eq!(kept, bytes);
    }

    #[test]
    fn corrupt_file_is_kept_and_defaults_used() {
        assert_kept(b"{\"theme\": \"dark\", ");
    }

    #[test]
    fn invalid_utf8_file_is_kept_and_defaults_used() {
        assert_kept(b"{\"theme\": \"dark\", \"mic_device\": \"\xff\xfe\"}");
    }

    #[test]
    fn utf16_file_is_kept_and_defaults_used() {
        let mut bytes = vec![0xFF, 0xFE];
        for unit in r#"{"theme":"dark"}"#.encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        assert_kept(&bytes);
    }

    #[test]
    fn utf8_bom_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, b"\xEF\xBB\xBF{\"theme\":\"dark\"}").unwrap();
        assert_eq!(load(&path).unwrap().theme, Theme::Dark);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn unknown_fields_ignored_and_missing_fields_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, r#"{"theme":"light","future_option":42}"#).unwrap();
        let s = load(&path).unwrap();
        assert_eq!(s.theme, Theme::Light);
        assert_eq!(s.target_language, "ko");
        assert_eq!(s.asr, Settings::default().asr);
    }

    #[test]
    fn an_api_key_in_the_file_is_not_saved_again() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(
            &path,
            r#"{"api_key":"sk-test-LEAK1234",
                "asr":{"kind":"custom","endpoint":{"base_url":"http://h/v1","model":"w","api_key":"sk-test-LEAK1234"}}}"#,
        )
        .unwrap();
        let s = load(&path).unwrap();
        assert!(matches!(s.asr, AsrConfig::Custom { .. }));
        save(&path, &s).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("LEAK1234"), "{text}");
        assert!(!text.contains("api_key"), "{text}");
        assert!(!text.to_lowercase().contains("key"), "{text}");
    }
}
