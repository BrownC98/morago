# UI ↔ core contract (Tauri commands and events)

Rust types are in `src-tauri/src/types.rs`, `settings.rs`, `models.rs`, `secrets.rs`. JSON follows their serde attributes. The UI keeps TypeScript copies in `src/api.ts`. Change this file only with the coordinator.

## Commands (`invoke(name, args)`)

| Command | Args | Returns |
|---|---|---|
| `get_settings` | – | `Settings` |
| `save_settings` | `{ settings: Settings }` | – (a running session uses the new `target_language` from the next utterance; refuses a custom translation `extra_body` that `check_extra_body` refuses) |
| `check_extra_body` | `{ text }` | – (rejects with the reason if the text is not a JSON object for `extra_body`, has an API key, sets `messages` or `stream`, or has a `model` that is not a non-empty string; blank text is `{}`. `extra_body.model` is the translation model; `endpoint.model` of translate is always `""`) |
| `set_api_key` | `{ role: "asr"\|"translate", baseUrl, key }` | – |
| `delete_api_key` | `{ role, baseUrl }` | – |
| `api_key_hint` | `{ role, baseUrl }` | `string \| null` (for example `"••••3f9a"`) |
| `test_provider` | `{ role }` | `{ ok: boolean, message: string }` (uses the saved settings and key) |
| `list_mics` | – | `string[]` |
| `system_audio_support` | – | `{ supported: boolean, reason: string \| null }` |
| `list_models` | – | `(ModelInfo & { installed: boolean })[]` |
| `download_model` | `{ id }` | – (returns at once; progress by events) |
| `cancel_download` | `{ id }` | – |
| `start_session` | `{ title: string \| null }` | `SessionMeta` |
| `stop_session` | – | `SessionSummary` |
| `list_sessions` | – | `SessionSummary[]` (newest first) |
| `load_session` | `{ id }` | `{ meta: SessionMeta, utterances: Utterance[] }` |
| `update_session_meta` | `{ meta: SessionMeta }` | – (title, speaker names) |
| `delete_session` | `{ id }` | – |
| `export_session` | `{ id, format: "markdown"\|"srt"\|"json"\|"csv", path }` | – (writes the file; UI picks `path` with the dialog plugin `save()`) |

Errors: a command rejects with a Korean user-facing string that names the problem and the fix. It never contains an API key.

## Events (`listen(name)`)

| Event | Payload |
|---|---|
| `utterance` | `Utterance`. Upsert by `id`: the first event has `translation.state = "pending"`, a later one the result. |
| `utterance-partial` | `PartialText` `{ source, start_ms, text, translation: string \| null }`: live text of speech that has not ended, about every 0.7 s, and its provisional translation (can be of an older, shorter text). Sent only with the local SenseVoice Small ASR; partial translations with a local model only on a GPU (macOS Metal), or with a custom server at most once per `partial_interval_ms` of the translate settings (`null`: none). Show one grey line per source. Remove it when an `utterance` of that source has `end_ms > start_ms` (the final of that speech; not an update of an older utterance). An empty `text` (any ASR: the speech gave no utterance) removes the line of that source if the line has a smaller `start_ms`. No partial of a speech comes after its utterance. |
| `audio-level` | `{ source: "mic"\|"system", rms: number }` (0..1, about 10 per second while running) |
| `session-state` | `{ running: boolean, sessionId: string \| null, startedAt: string \| null }` |
| `engine-error` | `{ stage: "capture"\|"asr"\|"translate"\|"store"\|"model"\|"notice", source: "mic"\|"system"\|null, message: string }`. Stage `notice` (a setting the server does not support, skipped work when a queue is full) is shown as a toast; the other stages show in the status bar until the next good result. |
| `model-download` | `DownloadProgress` |
| `model-download-done` | `{ modelId: string, error: string \| null }` |
