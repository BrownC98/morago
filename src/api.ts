// Typed wrappers for the Tauri commands and events in docs/ipc.md.
// The types copy the serde JSON of src-tauri/src/{types,settings,models,secrets}.rs.
// In a plain browser (no Tauri) the calls go to the in-memory mock in mock.ts.

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export type SourceKind = "mic" | "system";

export type TranslationState =
  | { state: "pending" }
  | { state: "done"; target: string; text: string }
  | { state: "same_language" }
  | { state: "failed"; message: string };

export interface Utterance {
  id: number;
  source: SourceKind;
  /** Speaker id from diarization, for example "S1". */
  speaker: string;
  start_ms: number;
  end_ms: number;
  language: string | null;
  original: string;
  translation: TranslationState;
}

export interface SessionMeta {
  id: string;
  title: string;
  started_at: string;
  ended_at: string | null;
  target_language: string;
  asr_label: string;
  translator_label: string;
  speaker_names: Record<string, string>;
}

export interface SessionSummary {
  meta: SessionMeta;
  duration_ms: number;
  utterance_count: number;
  speaker_count: number;
}

export type ExportFormat = "markdown" | "srt" | "json";

export interface Endpoint {
  base_url: string;
  model: string;
}

/**
 * `AsrConfig` and `TranslateConfig` have the same JSON shape. Only translate has
 * `extra_body` (absent = `{}`): its `model` is the translation model, and the
 * translate `endpoint.model` is always "". Only translate has `partial_interval_ms`
 * (least time between two partial translation requests of one speech; null = off,
 * absent = 1500).
 */
export type ProviderConfig =
  | { kind: "local"; model_id: string }
  | { kind: "custom"; endpoint: Endpoint; extra_body?: Record<string, unknown>; partial_interval_ms?: number | null };

export type ThemeSetting = "light" | "dark" | "system";

/** 자동: the first usable GPU, else the CPU. */
export type GpuSetting = { kind: "auto" } | { kind: "device"; name: string } | { kind: "off" };

export interface ModelDevice {
  gpu: boolean;
  /** "NVIDIA GeForce RTX 5060 Laptop GPU · CUDA", or why the model runs on the CPU. */
  detail: string;
}

export interface GpuStatus {
  gpus: string[];
  hint: string | null;
  /** The "GPU 구성 요소" download; null when no GPU can use it. */
  pack: {
    id: string;
    size_bytes: number;
    installed: boolean;
    needs_restart: boolean;
    note: string;
  } | null;
  /** Device of the local translation model; null when a service translates. */
  translate: ModelDevice | null;
}

export interface Settings {
  theme: ThemeSetting;
  target_language: string;
  asr: ProviderConfig;
  /** ISO 639-1 code for speech recognition; null = detect per utterance. */
  meeting_language: string | null;
  translate: ProviderConfig;
  diarize: boolean;
  capture_mic: boolean;
  capture_system: boolean;
  mic_device: string | null;
  sessions_dir: string | null;
  gpu: GpuSetting;
}

export type ModelKind = "asr" | "translation";

export interface ModelInfo {
  id: string;
  kind: ModelKind;
  name: string;
  size_bytes: number;
  note: string;
  installed: boolean;
  /** Files on disk: installed, or files of an older format that the app does not use. */
  on_disk: boolean;
}

export interface DownloadProgress {
  model_id: string;
  done_bytes: number;
  total_bytes: number | null;
}

export type KeyRole = "asr" | "translate";
/** "session": the core stopped and saved the session after a fault (a GPU error). */
export type EngineStage = "capture" | "asr" | "translate" | "store" | "model" | "notice" | "session";

export interface PartialText {
  source: SourceKind;
  start_ms: number;
  /** Empty: remove the line of a speech that started before `start_ms`. */
  text: string;
  /** Provisional translation; it can be of an older, shorter `text`. */
  translation: string | null;
}

export interface EventMap {
  utterance: Utterance;
  "utterance-partial": PartialText;
  "audio-level": { source: SourceKind; rms: number };
  "session-state": { running: boolean; sessionId: string | null; startedAt: string | null };
  "engine-error": { stage: EngineStage; source: SourceKind | null; message: string };
  "model-download": DownloadProgress;
  "model-download-done": { modelId: string; error: string | null };
}

export interface MockBackend {
  invoke(cmd: string, args: Record<string, unknown>): Promise<unknown>;
  listen(event: string, cb: (payload: unknown) => void): () => void;
  save(defaultPath: string): string | null;
}

export const IS_MOCK = typeof window !== "undefined" && !("__TAURI_INTERNALS__" in window);
const mockPromise: Promise<MockBackend> | null = IS_MOCK
  ? import("./mock").then((m) => m.backend)
  : null;

/** A command error is a Korean string. It never holds an API key. */
export function errText(e: unknown): string {
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message;
  return "알 수 없는 오류가 났습니다.";
}

async function call<T>(cmd: string, args: Record<string, unknown> = {}): Promise<T> {
  if (mockPromise) return (await (await mockPromise).invoke(cmd, args)) as T;
  return invoke<T>(cmd, args);
}

/** Subscribes to a core event. Returns an unsubscribe function (async). */
export function onEvent<K extends keyof EventMap>(
  name: K,
  cb: (payload: EventMap[K]) => void,
): Promise<() => void> {
  if (mockPromise) {
    return mockPromise.then((m) => m.listen(name, (p) => cb(p as EventMap[K])));
  }
  return listen<EventMap[K]>(name, (e) => cb(e.payload));
}

/** Asks for a save path. Returns null when the user cancels. */
export async function pickSavePath(
  defaultPath: string,
  filterName: string,
  extension: string,
): Promise<string | null> {
  if (mockPromise) return (await mockPromise).save(defaultPath);
  const { save } = await import("@tauri-apps/plugin-dialog");
  return save({ defaultPath, filters: [{ name: filterName, extensions: [extension] }] });
}

export const api = {
  getSettings: () => call<Settings>("get_settings"),
  saveSettings: (settings: Settings) => call<void>("save_settings", { settings }),
  /** Rejects with the Korean reason when `save_settings` would refuse the text as `extra_body`. */
  checkExtraBody: (text: string) => call<void>("check_extra_body", { text }),
  /** The key goes to the core and is not kept in the UI. */
  setApiKey: (role: KeyRole, baseUrl: string, key: string) =>
    call<void>("set_api_key", { role, baseUrl, key }),
  deleteApiKey: (role: KeyRole, baseUrl: string) => call<void>("delete_api_key", { role, baseUrl }),
  apiKeyHint: (role: KeyRole, baseUrl: string) =>
    call<string | null>("api_key_hint", { role, baseUrl }),
  testProvider: (role: KeyRole) => call<{ ok: boolean; message: string }>("test_provider", { role }),
  listMics: () => call<string[]>("list_mics"),
  systemAudioSupport: () =>
    call<{ supported: boolean; reason: string | null }>("system_audio_support"),
  listModels: () => call<ModelInfo[]>("list_models"),
  downloadModel: (id: string) => call<void>("download_model", { id }),
  cancelDownload: (id: string) => call<void>("cancel_download", { id }),
  deleteModel: (id: string) => call<void>("delete_model", { id }),
  startSession: (title: string | null) => call<SessionMeta>("start_session", { title }),
  stopSession: () => call<SessionSummary>("stop_session"),
  listSessions: () => call<SessionSummary[]>("list_sessions"),
  loadSession: (id: string) =>
    call<{ meta: SessionMeta; utterances: Utterance[] }>("load_session", { id }),
  updateSessionMeta: (meta: SessionMeta) => call<void>("update_session_meta", { meta }),
  deleteSession: (id: string) => call<void>("delete_session", { id }),
  exportSession: (id: string, format: ExportFormat, path: string) =>
    call<void>("export_session", { id, format, path }),
  gpuStatus: () => call<GpuStatus>("gpu_status"),
  restartApp: () => call<void>("restart_app"),
};
