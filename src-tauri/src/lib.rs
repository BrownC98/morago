mod audio;
mod commands;
mod engine;
mod gpu;
mod models;
mod secrets;
mod settings;
mod speech;
mod store;
mod translate;
mod types;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            let state = commands::AppState::new(app.handle())?;
            app.manage(state);
            #[cfg(feature = "update-test")]
            update_test(app.handle().clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_settings,
            commands::save_settings,
            commands::check_extra_body,
            commands::set_api_key,
            commands::delete_api_key,
            commands::api_key_hint,
            commands::test_provider,
            commands::list_mics,
            commands::system_audio_support,
            commands::list_models,
            commands::download_model,
            commands::cancel_download,
            commands::delete_model,
            commands::start_session,
            commands::stop_session,
            commands::list_sessions,
            commands::load_session,
            commands::update_session_meta,
            commands::delete_session,
            commands::export_session,
            commands::export_session_audio,
            commands::open_sessions_dir,
            commands::gpu_status,
            commands::restart_app,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            // Close the session file on quit so the record is complete.
            if let tauri::RunEvent::Exit = event
                && let Some(state) = app.try_state::<commands::AppState>()
            {
                state.stop_on_exit();
            }
        });
}

/// CI update test (scripts/update-test.sh): if MORAGO_UPDATE_TEST names a log
/// file, check for an update, install it and start again, with no user step.
#[cfg(feature = "update-test")]
fn update_test(app: tauri::AppHandle) {
    use std::io::Write;
    use tauri_plugin_updater::UpdaterExt;
    let Some(path) = std::env::var_os("MORAGO_UPDATE_TEST") else {
        return;
    };
    let log = move |line: String| {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = writeln!(f, "{line}");
        }
    };
    tauri::async_runtime::spawn(async move {
        let update = match app.updater() {
            Ok(updater) => updater.check().await,
            Err(e) => Err(e),
        };
        match update {
            Ok(Some(update)) => {
                log(format!(
                    "found {} (current {})",
                    update.version, update.current_version
                ));
                // On Windows the installer starts here and this process ends.
                match update.download_and_install(|_, _| {}, || {}).await {
                    Ok(()) => {
                        log("installed; restart".into());
                        app.restart();
                    }
                    Err(e) => log(format!("install failed: {e}")),
                }
            }
            Ok(None) => log("no update".into()),
            Err(e) => log(format!("check failed: {e}")),
        }
    });
}
