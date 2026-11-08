//! Tauri commands. The contract is docs/ipc.md.
//! Errors go to the UI as Korean text; they never contain an API key.

use crate::engine::{self, Running};
use crate::models::{self, ModelInfo};
use crate::secrets::{self, Role};
use crate::settings::{self, Settings};
use crate::speech;
use crate::store::SessionStore;
use crate::translate;
use crate::types::{
    DownloadProgress, ExportFormat, SessionMeta, SessionSummary, SourceKind, SpeechSegment,
    Utterance,
};
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};

type CmdResult<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

pub struct AppState {
    settings_path: PathBuf,
    models_dir: PathBuf,
    default_sessions_dir: PathBuf,
    settings: Mutex<Settings>,
    running: Mutex<Option<Running>>,
    downloads: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

impl AppState {
    pub fn new(app: &AppHandle) -> anyhow::Result<Self> {
        let path = app.path();
        let settings_path = path.app_config_dir()?.join("settings.json");
        let models_dir = path.app_data_dir()?.join("models");
        let default_sessions_dir = path
            .document_dir()
            .unwrap_or(path.app_data_dir()?)
            .join("morago")
            .join("sessions");
        let settings = settings::load(&settings_path)?;
        Ok(Self {
            settings_path,
            models_dir,
            default_sessions_dir,
            settings: Mutex::new(settings),
            running: Mutex::new(None),
            downloads: Mutex::new(HashMap::new()),
        })
    }

    /// Stops a running session at app exit.
    pub fn stop_on_exit(&self) {
        if let Some(r) = self.running.lock().expect("running lock").take()
            && let Err(e) = engine::stop(r)
        {
            log::error!("session stop at exit failed: {e}");
        }
    }

    fn settings(&self) -> Settings {
        self.settings.lock().expect("settings lock").clone()
    }

    fn store(&self) -> SessionStore {
        let s = self.settings();
        SessionStore::new(
            s.sessions_dir
                .as_ref()
                .unwrap_or(&self.default_sessions_dir),
        )
    }
}

async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> anyhow::Result<T> + Send + 'static,
) -> CmdResult<T> {
    tauri::async_runtime::spawn_blocking(f)
        .await
        .map_err(err)?
        .map_err(err)
}

#[tauri::command]
pub fn get_settings(state: State<'_, AppState>) -> Settings {
    state.settings()
}

#[tauri::command]
pub fn save_settings(state: State<'_, AppState>, settings: Settings) -> CmdResult<()> {
    settings::save(&state.settings_path, &settings).map_err(err)?;
    if let Some(r) = state.running.lock().expect("running lock").as_ref() {
        *r.target_language.write().expect("target lock") = settings.target_language.clone();
    }
    *state.settings.lock().expect("settings lock") = settings;
    Ok(())
}

#[tauri::command]
pub fn set_api_key(role: Role, base_url: String, key: String) -> CmdResult<()> {
    secrets::set_key(role, &base_url, &key).map_err(err)
}

#[tauri::command]
pub fn delete_api_key(role: Role, base_url: String) -> CmdResult<()> {
    secrets::delete_key(role, &base_url).map_err(err)
}

#[tauri::command]
pub fn api_key_hint(role: Role, base_url: String) -> CmdResult<Option<String>> {
    secrets::key_hint(role, &base_url).map_err(err)
}

#[derive(Serialize)]
pub struct TestResult {
    ok: bool,
    message: String,
}

#[tauri::command]
pub async fn test_provider(state: State<'_, AppState>, role: Role) -> CmdResult<TestResult> {
    let settings = state.settings();
    let models_dir = state.models_dir.clone();
    let result = blocking(move || -> anyhow::Result<String> {
        match role {
            Role::Asr => {
                let mut asr = speech::build_asr(&settings.asr, &models_dir)?;
                // One second of silence: checks the address, key and model name.
                let seg = SpeechSegment {
                    source: SourceKind::Mic,
                    start_ms: 0,
                    end_ms: 1000,
                    samples: vec![0.0; 16_000],
                };
                asr.transcribe(&seg)?;
                Ok(format!("{}: 연결되었습니다.", asr.label()))
            }
            Role::Translate => {
                let mut t = translate::build_translator(&settings.translate, &models_dir)?;
                let out = t.translate("Hello.", Some("en"), &settings.target_language)?;
                Ok(format!("{}: 연결되었습니다. 시험 번역: {out}", t.label()))
            }
        }
    })
    .await;
    Ok(match result {
        Ok(message) => TestResult { ok: true, message },
        Err(message) => TestResult { ok: false, message },
    })
}

#[tauri::command]
pub fn list_mics() -> CmdResult<Vec<String>> {
    crate::audio::list_mics().map_err(err)
}

#[tauri::command]
pub fn system_audio_support() -> crate::audio::SystemAudioSupport {
    crate::audio::system_audio_support()
}

#[derive(Serialize)]
pub struct ModelRow {
    #[serde(flatten)]
    info: ModelInfo,
    installed: bool,
}

#[tauri::command]
pub fn list_models(state: State<'_, AppState>) -> Vec<ModelRow> {
    models::catalog()
        .into_iter()
        .map(|info| ModelRow {
            installed: models::is_installed(&state.models_dir, &info.id),
            info,
        })
        .collect()
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct DownloadDone {
    model_id: String,
    error: Option<String>,
}

/// Downloads the model and the helper models (VAD, speaker embedding) that
/// a session needs, so the user downloads only what the settings list shows.
#[tauri::command]
pub fn download_model(app: AppHandle, state: State<'_, AppState>, id: String) -> CmdResult<()> {
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let mut d = state.downloads.lock().expect("downloads lock");
        if d.contains_key(&id) {
            return Ok(());
        }
        d.insert(id.clone(), cancel.clone());
    }
    let models_dir = state.models_dir.clone();
    let mut ids: Vec<String> = models::catalog()
        .into_iter()
        .filter(|m| {
            matches!(
                m.kind,
                models::ModelKind::Vad | models::ModelKind::SpeakerEmbedding
            ) && !models::is_installed(&models_dir, &m.id)
        })
        .map(|m| m.id)
        .collect();
    ids.push(id.clone());
    std::thread::spawn(move || {
        let progress_app = app.clone();
        let progress = move |p: DownloadProgress| {
            let _ = progress_app.emit("model-download", p);
        };
        let mut error = None;
        for one in &ids {
            if models::is_installed(&models_dir, one) {
                continue;
            }
            if let Err(e) = models::download(&models_dir, one, &progress, &cancel) {
                error = Some(e.to_string());
                break;
            }
        }
        if let Some(state) = app.try_state::<AppState>() {
            state.downloads.lock().expect("downloads lock").remove(&id);
        }
        let _ = app.emit(
            "model-download-done",
            DownloadDone {
                model_id: id,
                error,
            },
        );
    });
    Ok(())
}

#[tauri::command]
pub fn cancel_download(state: State<'_, AppState>, id: String) {
    if let Some(c) = state.downloads.lock().expect("downloads lock").get(&id) {
        c.store(true, Ordering::Relaxed);
    }
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct SessionState {
    running: bool,
    session_id: Option<String>,
    started_at: Option<String>,
}

#[tauri::command]
pub async fn start_session(
    app: AppHandle,
    state: State<'_, AppState>,
    title: Option<String>,
) -> CmdResult<SessionMeta> {
    if state.running.lock().expect("running lock").is_some() {
        return Err("회의가 이미 진행 중입니다.".into());
    }
    let settings = state.settings();
    let models_dir = state.models_dir.clone();
    let store = state.store();
    let sink = Arc::new(app.clone());
    let running =
        blocking(move || engine::start(&settings, title, &models_dir, &store, sink)).await?;
    let meta = running.meta.clone();
    let mut slot = state.running.lock().expect("running lock");
    if slot.is_some() {
        drop(slot);
        let _ = engine::stop(running);
        return Err("회의가 이미 진행 중입니다.".into());
    }
    *slot = Some(running);
    let _ = app.emit(
        "session-state",
        SessionState {
            running: true,
            session_id: Some(meta.id.clone()),
            started_at: Some(meta.started_at.clone()),
        },
    );
    Ok(meta)
}

#[tauri::command]
pub async fn stop_session(app: AppHandle, state: State<'_, AppState>) -> CmdResult<SessionSummary> {
    let running = state
        .running
        .lock()
        .expect("running lock")
        .take()
        .ok_or("진행 중인 회의가 없습니다.")?;
    let result = blocking(move || engine::stop(running)).await;
    let _ = app.emit(
        "session-state",
        SessionState {
            running: false,
            session_id: None,
            started_at: None,
        },
    );
    result
}

#[tauri::command]
pub fn list_sessions(state: State<'_, AppState>) -> CmdResult<Vec<SessionSummary>> {
    state.store().list().map_err(err)
}

#[derive(Serialize)]
pub struct LoadedSession {
    meta: SessionMeta,
    utterances: Vec<Utterance>,
}

#[tauri::command]
pub fn load_session(state: State<'_, AppState>, id: String) -> CmdResult<LoadedSession> {
    let (meta, utterances) = state.store().load(&id).map_err(err)?;
    Ok(LoadedSession { meta, utterances })
}

#[tauri::command]
pub fn update_session_meta(state: State<'_, AppState>, meta: SessionMeta) -> CmdResult<()> {
    state.store().update_meta(&meta).map_err(err)
}

#[tauri::command]
pub fn delete_session(state: State<'_, AppState>, id: String) -> CmdResult<()> {
    let running_id = state
        .running
        .lock()
        .expect("running lock")
        .as_ref()
        .map(|r| r.meta.id.clone());
    if running_id.as_deref() == Some(id.as_str()) {
        return Err("진행 중인 회의는 지울 수 없습니다. 먼저 중지하십시오.".into());
    }
    state.store().delete(&id).map_err(err)
}

#[tauri::command]
pub fn export_session(
    state: State<'_, AppState>,
    id: String,
    format: ExportFormat,
    path: PathBuf,
) -> CmdResult<()> {
    let text = state.store().export(&id, format).map_err(err)?;
    std::fs::write(&path, text)
        .map_err(|e| format!("파일을 쓰지 못했습니다({}): {e}", path.display()))
}
