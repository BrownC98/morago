//! Tauri commands. The contract is docs/ipc.md.
//! Errors go to the UI as Korean text; they never contain an API key.

use crate::engine::{self, Running, Sink};
use crate::gpu;
use crate::models::{self, ModelInfo};
use crate::secrets::{self, Role};
use crate::settings::{self, AsrConfig, Settings, TranslateConfig};
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
use std::sync::{Arc, Condvar, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};

type CmdResult<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Where the one session is in its life.
enum Session {
    Idle,
    Running(Box<Running>),
    /// `engine::stop` drains the queues; the id stays protected.
    Stopping(String),
}

impl Session {
    fn id(&self) -> Option<&str> {
        match self {
            Session::Idle => None,
            Session::Running(r) => Some(&r.meta.id),
            Session::Stopping(id) => Some(id),
        }
    }
}

pub struct AppState {
    settings_path: PathBuf,
    models_dir: PathBuf,
    /// Models shipped in the installer (Tauri resources `models/`).
    bundled_dir: PathBuf,
    default_sessions_dir: PathBuf,
    settings: Mutex<Settings>,
    /// The session and its stop state; `stopped` wakes waiters when a stop ends.
    session: Mutex<Session>,
    stopped: Condvar,
    downloads: Mutex<HashMap<String, Arc<AtomicBool>>>,
    /// The GPU pack (`gpu.rs`).
    gpu_dir: PathBuf,
}

impl AppState {
    pub fn new(app: &AppHandle) -> anyhow::Result<Self> {
        let path = app.path();
        let settings_path = path.app_config_dir()?.join("settings.json");
        let models_dir = path.app_data_dir()?.join("models");
        let gpu_dir = gpu::app_dir().ok_or_else(|| anyhow::anyhow!("no app data folder"))?;
        gpu::init(&gpu_dir);
        let bundled_dir = path.resource_dir()?.join("models");
        let default_sessions_dir = path
            .document_dir()
            .unwrap_or(path.app_data_dir()?)
            .join("morago")
            .join("sessions");
        let settings = settings::load(&settings_path)?;
        Ok(Self {
            settings_path,
            models_dir,
            bundled_dir,
            default_sessions_dir,
            settings: Mutex::new(settings),
            session: Mutex::new(Session::Idle),
            stopped: Condvar::new(),
            downloads: Mutex::new(HashMap::new()),
            gpu_dir,
        })
    }

    /// At app exit: stops a running session, or waits for a stop that is
    /// still draining the queues, so the record is complete.
    pub fn stop_on_exit(&self) {
        if let Some(r) = self.begin_stop()
            && let Err(e) = self.finish_stop(r)
        {
            log::error!("session stop at exit failed: {e}");
        }
        let mut s = self.session.lock().expect("session lock");
        while matches!(*s, Session::Stopping(_)) {
            s = self.stopped.wait(s).expect("session lock");
        }
    }

    /// Moves a running session to `Stopping`; `None` if none runs.
    fn begin_stop(&self) -> Option<Running> {
        let mut s = self.session.lock().expect("session lock");
        match std::mem::replace(&mut *s, Session::Idle) {
            Session::Running(r) => {
                *s = Session::Stopping(r.meta.id.clone());
                Some(*r)
            }
            other => {
                *s = other;
                None
            }
        }
    }

    /// Drains and closes the session (blocking), then wakes waiters.
    fn finish_stop(&self, r: Running) -> anyhow::Result<SessionSummary> {
        let result = engine::stop(r);
        *self.session.lock().expect("session lock") = Session::Idle;
        self.stopped.notify_all();
        result
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
    if let AsrConfig::Custom { endpoint } = &settings.asr {
        check_base_url(&endpoint.base_url)?;
    }
    if let TranslateConfig::Custom { endpoint, .. } = &settings.translate {
        check_base_url(&endpoint.base_url)?;
    }
    settings::save(&state.settings_path, &settings).map_err(err)?;
    if let Session::Running(r) = &*state.session.lock().expect("session lock") {
        *r.target_language.write().expect("target lock") = settings.target_language.clone();
    }
    *state.settings.lock().expect("settings lock") = settings;
    Ok(())
}

/// A base URL is saved in plain text, so it must not carry credentials.
/// An empty URL is allowed while the user fills in the form.
fn check_base_url(url: &str) -> CmdResult<()> {
    let url = url.trim();
    if url.is_empty() {
        return Ok(());
    }
    let parsed = reqwest::Url::parse(url).map_err(|_| {
        format!("서비스 주소가 올바르지 않습니다: {url}. 예: https://api.openai.com/v1")
    })?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err("서비스 주소는 http:// 또는 https://로 시작해야 합니다.".into());
    }
    if !parsed.username().is_empty() || parsed.password().is_some() || parsed.query().is_some() {
        return Err(
            "서비스 주소에 사용자 이름, 비밀번호, 키(?key=…)를 넣지 마십시오. API 키 칸에 넣으십시오."
                .into(),
        );
    }
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
                let placement = engine::place_models(&settings, &models_dir)?.asr;
                let mut asr = speech::build_asr(
                    &settings.asr,
                    settings.meeting_language.as_deref(),
                    &models_dir,
                    &placement.unwrap_or_else(|| gpu::Placement::cpu("")),
                )?;
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
                let placement = engine::place_models(&settings, &models_dir)?.translate;
                let mut t = translate::build_translator(
                    &settings.translate,
                    &models_dir,
                    &placement.unwrap_or_else(|| gpu::Placement::cpu("")),
                )?;
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
    /// Files on disk: installed, or files that the app cannot use (an older format).
    on_disk: bool,
}

#[tauri::command]
pub fn list_models(state: State<'_, AppState>) -> Vec<ModelRow> {
    models::catalog()
        .into_iter()
        .map(|info| ModelRow {
            installed: models::is_installed(&state.models_dir, &info.id),
            on_disk: models::on_disk(&state.models_dir, &info.id),
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

/// Downloads a model from the catalog. Progress and the end come as events.
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
    let gpu_dir = state.gpu_dir.clone();
    std::thread::spawn(move || {
        let progress_app = app.clone();
        let progress = move |p: DownloadProgress| {
            let _ = progress_app.emit("model-download", p);
        };
        let result = if id == gpu::PACK_ID {
            gpu::download_pack(&gpu_dir, &progress, &cancel)
        } else {
            models::download(&models_dir, &id, &progress, &cancel)
        };
        let error = result.err().map(|e| e.to_string());
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

/// Deletes the files of a model. Refused while a meeting runs or the model downloads.
#[tauri::command]
pub fn delete_model(state: State<'_, AppState>, id: String) -> CmdResult<()> {
    if !matches!(*state.session.lock().expect("session lock"), Session::Idle) {
        return Err("회의 중에는 모델을 지울 수 없습니다. 회의를 마친 뒤 지우십시오.".into());
    }
    if state
        .downloads
        .lock()
        .expect("downloads lock")
        .contains_key(&id)
    {
        return Err("내려받는 중인 모델은 지울 수 없습니다. 먼저 취소하십시오.".into());
    }
    models::delete(&state.models_dir, &id).map_err(err)
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
    if !matches!(*state.session.lock().expect("session lock"), Session::Idle) {
        return Err("회의가 이미 진행 중이거나 저장하는 중입니다.".into());
    }
    let settings = state.settings();
    let models_dir = state.models_dir.clone();
    let bundled_dir = state.bundled_dir.clone();
    let store = state.store();
    let sink = Arc::new(app.clone());
    let running =
        blocking(move || engine::start(&settings, title, &models_dir, &bundled_dir, &store, sink))
            .await?;
    let meta = running.meta.clone();
    let mut slot = state.session.lock().expect("session lock");
    if !matches!(*slot, Session::Idle) {
        drop(slot);
        let _ = engine::stop(running);
        return Err("회의가 이미 진행 중이거나 저장하는 중입니다.".into());
    }
    *slot = Session::Running(Box::new(running));
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
    let running = state.begin_stop().ok_or_else(|| {
        match &*state.session.lock().expect("session lock") {
            Session::Stopping(_) => "회의를 저장하는 중입니다. 잠시 기다리십시오.",
            _ => "진행 중인 회의가 없습니다.",
        }
        .to_string()
    })?;
    let app2 = app.clone();
    let result = blocking(move || app2.state::<AppState>().finish_stop(running)).await;
    emit_stopped(&app);
    result
}

fn emit_stopped(app: &AppHandle) {
    let _ = app.emit(
        "session-state",
        SessionState {
            running: false,
            session_id: None,
            started_at: None,
        },
    );
}

/// `Sink::fault`: stops and saves the session on its own thread (the caller is
/// a pipeline thread that the stop joins), then tells the UI with stage "session".
pub fn stop_after_fault(app: &AppHandle, message: &str) {
    let app = app.clone();
    let message = message.to_string();
    std::thread::spawn(move || {
        let state = app.state::<AppState>();
        let saved = match state.begin_stop().map(|r| state.finish_stop(r)) {
            Some(Ok(s)) => format!("회의를 멈추고 「{}」 기록을 저장했습니다.", s.meta.title),
            Some(Err(e)) => format!("회의를 멈췄지만 저장하지 못했습니다: {e}."),
            None => "회의를 멈췄습니다.".into(),
        };
        emit_stopped(&app);
        app.error(
            "session",
            None,
            &format!("{saved} {message} {}", gpu::CHOOSE_OTHER),
        );
    });
}

/// GPUs, the GPU pack and the devices of the local models: the running
/// session's devices, else the devices that the settings give.
#[tauri::command]
pub async fn gpu_status(state: State<'_, AppState>) -> CmdResult<gpu::GpuStatus> {
    let settings = state.settings();
    let running = match &*state.session.lock().expect("session lock") {
        Session::Running(r) => Some((r.translate_device.clone(), r.asr_device.clone())),
        _ => None,
    };
    let models_dir = state.models_dir.clone();
    blocking(move || {
        let (translate, asr) = running.unwrap_or_else(|| {
            let p = engine::model_devices(&settings, &models_dir);
            (p.translate.map(|p| p.status), p.asr.map(|p| p.status))
        });
        Ok(gpu::status(translate, asr))
    })
    .await
}

/// The GPU pack is loaded at start: a restart uses a new pack.
#[tauri::command]
pub fn restart_app(app: AppHandle, state: State<'_, AppState>) -> CmdResult<()> {
    if !matches!(*state.session.lock().expect("session lock"), Session::Idle) {
        return Err("회의 중에는 다시 시작할 수 없습니다. 회의를 마친 뒤 다시 시작하세요.".into());
    }
    app.restart()
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
    if state.session.lock().expect("session lock").id() == Some(id.as_str()) {
        return Err("진행 중이거나 저장하는 중인 회의는 지울 수 없습니다.".into());
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

#[cfg(test)]
mod tests {
    use super::check_base_url;

    #[test]
    fn a_stopping_session_keeps_its_id_protected() {
        use super::Session;
        assert_eq!(Session::Idle.id(), None);
        assert_eq!(
            Session::Stopping("20261108-101500-ab12".into()).id(),
            Some("20261108-101500-ab12")
        );
    }

    #[test]
    fn base_url_without_credentials() {
        assert!(check_base_url("").is_ok());
        assert!(check_base_url("https://api.openai.com/v1").is_ok());
        assert!(check_base_url("http://localhost:11434/v1").is_ok());
        assert!(check_base_url("https://user:pw@example.com/v1").is_err());
        assert!(check_base_url("https://example.com/v1?key=abc").is_err());
        assert!(check_base_url("ftp://example.com").is_err());
        assert!(check_base_url("example.com").is_err());
    }
}
