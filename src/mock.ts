// In-memory stand-in for the Tauri core. Used only in a plain browser (no Tauri), to develop the UI
// and to take screenshots. It follows docs/ipc.md. Query flags: ?mock=many,nomodel,nosystem,fail,nogpu,gpufail
//   many     600 utterances at once on start (scroll and render check)
//   nomodel  the selected local ASR model is not installed
//   nosystem system audio is not supported
//   fail     start_session rejects
//   nogpu    no NVIDIA GPU
//   gpufail  start_session rejects when translation would run on the GPU

import type {
  GpuStatus,
  MockBackend,
  ModelInfo,
  SessionMeta,
  SessionSummary,
  Settings,
  TranslationState,
  Utterance,
} from "./api";

const MB = 1024 ** 2;
const flags = new Set((new URLSearchParams(location.search).get("mock") ?? "").split(","));
const listeners = new Map<string, Set<(p: unknown) => void>>();
const emit = (event: string, payload: unknown) => listeners.get(event)?.forEach((cb) => cb(payload));

let settings: Settings = {
  theme: "system",
  target_language: "ko",
  asr: { kind: "local", model_id: "sense-voice-small" },
  meeting_language: null,
  translate: {
    kind: "custom",
    endpoint: { base_url: "https://api.openai.com/v1", model: "" },
    extra_body: { model: "gpt-4o-mini" },
    partial_interval_ms: 1500,
  },
  translate_context_lines: 4,
  diarize: true,
  capture_mic: true,
  capture_system: true,
  mic_device: null,
  sessions_dir: null,
  gpu: { kind: "auto" },
};

// GPU: one NVIDIA GPU; ?mock=nogpu has none, ?mock=gpufail fails the start on the GPU.
const GPU_NAME = "NVIDIA GeForce RTX 5060 Laptop GPU";
const pack = { id: "cuda-12.9.1", size_bytes: 528 * MB, installed: false, restart: false };
function gpuStatus(): GpuStatus {
  const gpus = flags.has("nogpu") ? [] : [GPU_NAME];
  const local = settings.translate.kind === "local";
  const chosen = settings.gpu.kind === "device" ? settings.gpu.name : GPU_NAME;
  const onGpu = gpus.includes(chosen) && settings.gpu.kind !== "off" && pack.installed && !pack.restart;
  const detail =
    settings.gpu.kind === "off"
      ? "GPU 가속을 껐습니다."
      : gpus.length === 0
        ? "NVIDIA GPU가 없습니다."
        : !pack.installed
          ? `GPU 구성 요소를 내려받으면 ${chosen}를 씁니다.`
          : pack.restart
            ? `앱을 다시 시작하면 ${chosen}를 씁니다.`
            : `${chosen} · CUDA`;
  const asr = settings.asr;
  return {
    gpus,
    hint: "GPU 가속은 NVIDIA GPU(CUDA)만 씁니다.",
    pack:
      gpus.length === 0
        ? null
        : {
            id: pack.id,
            size_bytes: pack.size_bytes,
            installed: pack.installed,
            needs_restart: pack.restart,
            note: "NVIDIA GPU에서 음성 인식과 번역을 돌리는 CUDA 런타임입니다. NVIDIA 서버에서 내려받고, 풀면 0.7 GB입니다. NVIDIA 드라이버 528.33 이상(RTX 50 시리즈는 570.65 이상)이 필요합니다.",
          },
    translate: local ? { gpu: onGpu, detail } : null,
    asr:
      asr.kind !== "local"
        ? null
        : asr.model_id.startsWith("whisper-")
          ? { gpu: onGpu, detail }
          : { gpu: false, detail: "이 모델은 CPU에서만 돌립니다." },
  };
}

const models: ModelInfo[] = [
  { id: "sense-voice-small", kind: "asr", name: "SenseVoice Small", size_bytes: 229 * MB, note: "가볍고 빠름, 메모리 약 1GB (기본). 오류율 % 한 6.9 · 일 7.2 · 중 4.5 · 영 8.7", installed: !flags.has("nomodel"), on_disk: !flags.has("nomodel") },
  { id: "qwen3-asr-0.6b", kind: "asr", name: "Qwen3-ASR 0.6B", size_bytes: 941 * MB, note: "정확도 높음(한국어·중국어), 느림, 메모리 약 2.5GB. 오류율 % 한 5.1 · 일 12.5 · 중 2.7 · 영 8.1", installed: false, on_disk: false },
  { id: "whisper-base", kind: "asr", name: "Whisper base", size_bytes: 78 * MB, note: "빠름, 정확도 낮음", installed: true, on_disk: true },
  { id: "whisper-small", kind: "asr", name: "Whisper small", size_bytes: 252 * MB, note: "SenseVoice보다 느림, 메모리 약 0.6GB. 오류율 % 한 7.5 · 일 12.7 · 중 10.6 · 영 11.2", installed: false, on_disk: true },
  { id: "whisper-medium", kind: "asr", name: "Whisper medium", size_bytes: 514 * MB, note: "느림, 정확도 높음", installed: false, on_disk: false },
  { id: "whisper-turbo", kind: "asr", name: "Whisper turbo", size_bytes: 547 * MB, note: "정확도 높음, CPU에서는 실시간보다 느림, 메모리 약 1GB. 오류율 % 한 7.4 · 일 4.8 · 중 5.3 · 영 9.3", installed: false, on_disk: false },
  { id: "gemma-4-e2b-it-q4", kind: "translation", name: "Gemma 4 E2B", size_bytes: 3.1 * 1024 * MB, note: "빠름, 일반 노트북용, 메모리 약 2GB (기본)", installed: true, on_disk: true },
  { id: "gemma-4-12b-it-q4", kind: "translation", name: "Gemma 4 12B", size_bytes: 6.5 * 1024 * MB, note: "번역 품질 높음, 느림, 메모리 약 8GB (RAM 16GB 이상 권장)", installed: false, on_disk: false },
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
      asr_label: "로컬 SenseVoice Small",
      translator_label: "로컬 Gemma 4 E2B",
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
    asr_label: settings.asr.kind === "local" ? "로컬 SenseVoice Small" : `사용자 지정 ${settings.asr.endpoint.model}`,
    translator_label: settings.translate.kind === "local" ? "로컬 Gemma 4 E2B" : `사용자 지정 ${settings.translate.extra_body?.model}`,
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
  const STEPS = 5;
  const cut = (s: string, k: number) => s.slice(0, Math.ceil((s.length * k) / STEPS));
  const speak = () => {
    const l = SCRIPT[n % SCRIPT.length];
    const start = Date.now() - r.startedAt;
    // Like the engine: the partial text grows every 0.7 s, its translation one step behind,
    // then the finished utterance replaces it.
    for (let k = 1; k < STEPS; k++) {
      window.setTimeout(() => {
        if (run !== r) return;
        const translated = l.tr !== "same" && l.tr !== "fail" && k > 1;
        emit("utterance-partial", {
          source: l.source,
          start_ms: start,
          text: cut(l.original, k),
          translation: translated ? cut(l.tr, k - 1) : null,
        });
      }, (k - 1) * 700);
    }
    window.setTimeout(() => {
      if (run !== r) return;
      n++;
      const end = Date.now() - r.startedAt;
      const u: Utterance = {
        id: n,
        source: l.source,
        speaker: l.speaker,
        start_ms: start,
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
    }, (STEPS - 1) * 700);
  };
  window.setTimeout(() => run === r && speak(), 400);
  r.timers.push(window.setInterval(speak, 3600));
  return meta;
}

// ---- downloads ----
const downloading = new Map<string, number>();
function download(id: string) {
  const m = id === pack.id ? pack : models.find((x) => x.id === id);
  if (!m || downloading.has(id)) return;
  let done = 0;
  const t = window.setInterval(() => {
    done = Math.min(m.size_bytes, done + m.size_bytes * 0.06);
    emit("model-download", { model_id: id, done_bytes: Math.round(done), total_bytes: m.size_bytes });
    if (done >= m.size_bytes) {
      clearInterval(t);
      downloading.delete(id);
      m.installed = true;
      if ("on_disk" in m) m.on_disk = true;
      else pack.restart = true;
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
    case "check_extra_body": {
      // Two of the core rules, enough to show both states in a browser.
      const text = String(a.text).trim();
      let value: unknown = {};
      try {
        if (text) value = JSON.parse(text);
      } catch {
        throw "JSON 형식이 아닙니다 (1번째 줄)";
      }
      if (typeof value !== "object" || value === null || Array.isArray(value)) throw "{ }로 감싼 JSON 객체를 적으세요";
      return delay(undefined);
    }
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
    case "delete_model": {
      if (run) return Promise.reject("회의 중에는 모델을 지울 수 없습니다. 회의를 마친 뒤 지우십시오.");
      const m = models.find((x) => x.id === a.id);
      if (m) m.installed = m.on_disk = false;
      return delay(undefined);
    }
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
      if (flags.has("gpufail") && gpuStatus().translate?.gpu)
        return Promise.reject(
          `번역 모델을 ${GPU_NAME}에 불러오지 못했습니다. 원인: GPU 메모리 부족. 설정 › GPU 가속에서 다른 GPU나 「끔」을 고른 뒤 다시 시작하세요.`,
        );
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
    case "gpu_status":
      return delay(gpuStatus());
    case "restart_app":
      pack.restart = false;
      return delay(undefined);
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
