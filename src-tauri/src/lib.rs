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
        .setup(|app| {
            let state = commands::AppState::new(app.handle())?;
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_settings,
            commands::save_settings,
            commands::set_api_key,
            commands::delete_api_key,
            commands::api_key_hint,
            commands::test_provider,
            commands::list_mics,
            commands::system_audio_support,
            commands::list_models,
            commands::download_model,
            commands::cancel_download,
            commands::start_session,
            commands::stop_session,
            commands::list_sessions,
            commands::load_session,
            commands::update_session_meta,
            commands::delete_session,
            commands::export_session,
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
