// In-memory stand-in for the Tauri core. Used only in a plain browser (no Tauri), to develop the UI
// and to take screenshots. It follows docs/ipc.md. Query flags: ?mock=many,nomodel,nosystem,fail
//   many     600 utterances at once on start (scroll and render check)
//   nomodel  the selected local ASR model is not installed
//   nosystem system audio is not supported
//   fail     start_session rejects

import type {
  MockBackend,
  ModelInfo,
  SessionMeta,
  SessionSummary,
  Settings,
  TranslationState,
  Utterance,
} from "./api";

const flags = new Set((new URLSearchParams(location.search).get("mock") ?? "").split(","));
const listeners = new Map<string, Set<(p: unknown) => void>>();
const emit = (event: string, payload: unknown) => listeners.get(event)?.forEach((cb) => cb(payload));

let settings: Settings = {
  theme: "system",
  target_language: "ko",
  asr: { kind: "local", model_id: "whisper-small" },
  translate: { kind: "custom", endpoint: { base_url: "https://api.openai.com/v1", model: "gpt-4o-mini" } },
  diarize: true,
  capture_mic: true,
  capture_system: true,
  mic_device: null,
  sessions_dir: null,
};

const MB = 1024 ** 2;
const models: ModelInfo[] = [
  { id: "whisper-base", kind: "asr", name: "Whisper base", size_bytes: 150 * MB, note: "빠름, 정확도 보통", installed: true },
  { id: "whisper-small", kind: "asr", name: "Whisper small", size_bytes: 480 * MB, note: "대부분의 노트북에 맞음", installed: !flags.has("nomodel") },
  { id: "whisper-medium", kind: "asr", name: "Whisper medium", size_bytes: 1.5 * 1024 * MB, note: "정확도 높음, 느린 컴퓨터는 지연됨", installed: false },
  { id: "qwen2.5-3b-instruct-q4", kind: "translation", name: "Qwen2.5 3B", size_bytes: 2 * 1024 * MB, note: "가볍고 빠름", installed: true },
  { id: "qwen2.5-7b-instruct-q4", kind: "translation", name: "Qwen2.5 7B", size_bytes: 4.7 * 1024 * MB, note: "번역 품질 높음, 메모리 8GB 이상", installed: false },
];

const keys = new Map<string, string>();
keys.set("translate|https://api.openai.com/v1", "sk-mock-0000000000003f9a");
const keyHint = (role: string, url: string) => {
  const k = keys.get(`${role}|${url}`);
  return k ? `••••••••••••${k.slice(-4)}` : null;
};

// ---- sample content ----
type Line = { speaker: string; source: "mic" | "system"; language: string; original: string; tr: string | "same" | "fail" };
const SCRIPT: Line[] = [
  { speaker: "S1", source: "system", language: "en", original: "Let's start with the Q3 roadmap. We moved the launch to the second week of December.", tr: "3분기 로드맵부터 시작하죠. 출시를 12월 둘째 주로 옮겼습니다." },
  { speaker: "S2", source: "mic", language: "ko", original: "그럼 베타 테스트 일정도 같이 밀리나요?", tr: "same" },
  { speaker: "S3", source: "system", language: "ja", original: "ベータは予定通り11月末に始めます。フィードバックは二週間で集めます。", tr: "베타는 예정대로 11월 말에 시작합니다. 피드백은 2주 동안 모읍니다." },
  { speaker: "S1", source: "system", language: "en", original: "Good. Then marketing needs the final screenshots by November 20th.", tr: "좋습니다. 그러면 마케팅 팀은 11월 20일까지 최종 스크린샷이 필요합니다." },
  { speaker: "S2", source: "mic", language: "ko", original: "스크린샷은 이번 주 금요일까지 드리겠습니다.", tr: "same" },
  { speaker: "S3", source: "system", language: "ja", original: "価格表は来週までに確認をお願いします。", tr: "fail" },
  { speaker: "S1", source: "system", language: "en", original: "One more thing: the partner demo moves to Thursday afternoon.", tr: "한 가지 더, 파트너 시연은 목요일 오후로 옮깁니다." },
];

function fmtId(d: Date, n: string) {
  const p = (x: number) => String(x).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}-${n}`;
}

function done(l: Line): TranslationState {
  if (l.tr === "same") return { state: "same_language" };
  if (l.tr === "fail") return { state: "failed", message: "번역 서비스에 연결하지 못했습니다. 설정 › 번역에서 주소를 확인하세요." };
  return { state: "done", target: "ko", text: l.tr };
}

function seed(n: number): Utterance[] {
  const out: Utterance[] = [];
  let t = 4000;
  for (let i = 0; i < n; i++) {
    const l = SCRIPT[i % SCRIPT.length];
    out.push({
      id: i + 1,
      source: l.source,
      speaker: l.speaker,
      start_ms: t,
      end_ms: t + 4000 + (i % 3) * 2000,
      language: l.language,
      original: l.original,
      translation: done(l),
    });
    t += 7000 + (i % 4) * 1500;
  }
  return out;
}

const store = new Map<string, { meta: SessionMeta; utterances: Utterance[] }>();
function addSession(id: string, title: string, started: string, minutesLong: number, count: number, names: Record<string, string> = {}) {
  const start = new Date(started);
  const end = new Date(start.getTime() + minutesLong * 60000);
  const utterances = seed(count);
  store.set(id, {
    meta: {
      id,
      title,
      started_at: started,
      ended_at: end.toISOString(),
      target_language: "ko",
      asr_label: "로컬 Whisper small",
      translator_label: "로컬 Qwen2.5 3B",
      speaker_names: names,
    },
    utterances,
  });
}
addSession("20261108-140212-ab12", "3분기 로드맵 회의", "2026-11-08T14:02:12+09:00", 48, 21, { S1: "민지", S2: "나" });
addSession("20261106-103000-c3d4", "파트너사 주간 점검", "2026-11-06T10:30:00+09:00", 31, 14);
addSession("20261104-161500-e5f6", "회의 2026-11-04 16:15", "2026-11-04T16:15:00+09:00", 12, 7);

function summary(id: string): SessionSummary {
  const s = store.get(id)!;
  const last = s.utterances[s.utterances.length - 1];
  return {
    meta: s.meta,
    duration_ms: last ? last.end_ms : 0,
    utterance_count: s.utterances.length,
    speaker_count: new Set(s.utterances.map((u) => u.speaker)).size,
  };
}

// ---- running session ----
let run: { id: string; startedAt: number; timers: number[]; utterances: Utterance[]; meta: SessionMeta } | null = null;

function stopTimers() {
  run?.timers.forEach((t) => clearInterval(t));
}

function startRun(): SessionMeta {
  const now = new Date();
  const id = fmtId(now, "m0ck");
  const meta: SessionMeta = {
    id,
    title: `회의 ${now.toLocaleDateString("sv-SE")} ${now.toTimeString().slice(0, 5)}`,
    started_at: now.toISOString(),
    ended_at: null,
    target_language: settings.target_language,
    asr_label: settings.asr.kind === "local" ? "로컬 Whisper small" : `사용자 지정 ${settings.asr.endpoint.model}`,
    translator_label: settings.translate.kind === "local" ? "로컬 Qwen2.5 3B" : `사용자 지정 ${settings.translate.endpoint.model}`,
    speaker_names: {},
  };
  run = { id, startedAt: now.getTime(), timers: [], utterances: [], meta };
  const r = run;
  emit("session-state", { running: true, sessionId: id, startedAt: meta.started_at });

  // levels, about 10 per second
  let phase = 0;
  r.timers.push(
    window.setInterval(() => {
      phase += 0.35;
      const mic = settings.capture_mic ? Math.max(0, Math.sin(phase) * 0.12 + Math.random() * 0.05) : 0;
      const sys = settings.capture_system && !flags.has("nosystem") ? Math.max(0, Math.sin(phase / 2 + 1) * 0.1 + Math.random() * 0.05) : 0;
      emit("audio-level", { source: "mic", rms: mic });
      emit("audio-level", { source: "system", rms: sys });
    }, 100),
  );

  if (flags.has("many")) {
    for (const u of seed(600)) {
      r.utterances.push(u);
      emit("utterance", u);
    }
    return meta;
  }

  let n = 0;
  const speak = () => {
    const l = SCRIPT[n % SCRIPT.length];
    const elapsed = Date.now() - r.startedAt;
    // Live partial text first (reserved event), then the finished utterance 1.2 s later.
    emit("utterance-partial", { source: l.source, start_ms: Math.max(0, elapsed - 1000), text: l.original.slice(0, Math.ceil(l.original.length * 0.55)) });
    window.setTimeout(() => {
      if (run !== r) return;
      n++;
      const end = Date.now() - r.startedAt;
      const u: Utterance = {
        id: n,
        source: l.source,
        speaker: l.speaker,
        start_ms: Math.max(0, end - 4000),
        end_ms: end,
        language: l.language,
        original: l.original,
        translation: l.tr === "same" ? { state: "same_language" } : { state: "pending" },
      };
      r.utterances.push(u);
      emit("utterance", u);
      if (l.tr !== "same") {
        window.setTimeout(() => {
          if (run !== r) return;
          const d = { ...u, translation: done(l) };
          r.utterances[r.utterances.indexOf(u)] = d;
          emit("utterance", d);
          if (l.tr === "fail") emit("engine-error", { stage: "translate", source: null, message: "번역 서비스에 연결하지 못했습니다." });
        }, 1400);
      }
    }, 1200);
  };
  window.setTimeout(() => run === r && speak(), 400);
  r.timers.push(window.setInterval(speak, 2600));
  return meta;
}

// ---- downloads ----
const downloading = new Map<string, number>();
function download(id: string) {
  const m = models.find((x) => x.id === id);
  if (!m || downloading.has(id)) return;
  let done = 0;
  const t = window.setInterval(() => {
    done = Math.min(m.size_bytes, done + m.size_bytes * 0.06);
    emit("model-download", { model_id: id, done_bytes: Math.round(done), total_bytes: m.size_bytes });
    if (done >= m.size_bytes) {
      clearInterval(t);
      downloading.delete(id);
      m.installed = true;
      emit("model-download-done", { modelId: id, error: null });
    }
  }, 350);
  downloading.set(id, t);
}

const delay = <T,>(v: T, ms = 60) => new Promise<T>((r) => setTimeout(() => r(v), ms));

async function invoke(cmd: string, a: Record<string, unknown>): Promise<unknown> {
  switch (cmd) {
    case "get_settings":
      return delay(settings);
    case "save_settings":
      settings = a.settings as Settings;
      return delay(undefined);
    case "set_api_key":
      keys.set(`${a.role}|${a.baseUrl}`, String(a.key));
      return delay(undefined);
    case "delete_api_key":
      keys.delete(`${a.role}|${a.baseUrl}`);
      return delay(undefined);
    case "api_key_hint":
      return delay(keyHint(String(a.role), String(a.baseUrl)));
    case "test_provider":
      return delay({ ok: true, message: "응답이 정상입니다(모의)." }, 700);
    case "list_mics":
      return delay(["MacBook Pro 마이크", "USB 헤드셋"]);
    case "system_audio_support":
      return delay(
        flags.has("nosystem")
          ? { supported: false, reason: "시스템 소리는 macOS 14.6 이상에서 받을 수 있습니다. 이 컴퓨터는 macOS 13입니다." }
          : { supported: true, reason: null },
      );
    case "list_models":
      return delay(models.map((m) => ({ ...m })));
    case "download_model":
      download(String(a.id));
      return delay(undefined);
    case "cancel_download": {
      const t = downloading.get(String(a.id));
      if (t) {
        clearInterval(t);
        downloading.delete(String(a.id));
        emit("model-download-done", { modelId: a.id, error: null });
      }
      return delay(undefined);
    }
    case "start_session":
      if (flags.has("fail")) return Promise.reject("마이크를 열 수 없습니다. 운영체제의 마이크 권한을 확인하세요.");
      return delay(startRun());
    case "stop_session": {
      if (!run) return Promise.reject("진행 중인 회의가 없습니다.");
      stopTimers();
      const r = run;
      run = null;
      const meta = { ...r.meta, ended_at: new Date().toISOString() };
      store.set(meta.id, { meta, utterances: r.utterances });
      emit("session-state", { running: false, sessionId: null, startedAt: null });
      return delay(summary(meta.id));
    }
    case "list_sessions":
      return delay(
        [...store.keys()].map(summary).sort((x, y) => y.meta.started_at.localeCompare(x.meta.started_at)),
      );
    case "load_session": {
      const s = store.get(String(a.id));
      return s ? delay({ meta: s.meta, utterances: s.utterances }) : Promise.reject("회의 기록을 찾을 수 없습니다.");
    }
    case "update_session_meta": {
      const m = a.meta as SessionMeta;
      const s = store.get(m.id);
      if (s) s.meta = m;
      return delay(undefined);
    }
    case "delete_session":
      store.delete(String(a.id));
      return delay(undefined);
    case "export_session":
      return delay(undefined, 200);
    default:
      return Promise.reject(`알 수 없는 명령: ${cmd}`);
  }
}

export const backend: MockBackend = {
  invoke,
  listen(event, cb) {
    if (!listeners.has(event)) listeners.set(event, new Set());
    listeners.get(event)!.add(cb);
    return () => listeners.get(event)?.delete(cb);
  },
  save: (defaultPath) => `/mock/${defaultPath}`,
};
