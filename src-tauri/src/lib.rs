// Module stubs are unused until the engine integration (coordinator). Remove this then.
#![allow(dead_code)]

mod audio;
mod engine;
mod models;
mod secrets;
mod settings;
mod speech;
mod store;
mod translate;
mod types;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
