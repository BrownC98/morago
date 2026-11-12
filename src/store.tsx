// App-wide state: settings, models, the live session, toasts. One React context, no state library.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  api,
  checkUpdate as fetchUpdate,
  errText,
  type AppUpdate,
  onEvent,
  type EngineStage,
  type GpuStatus,
  type ModelInfo,
  type PartialText,
  type SessionMeta,
  type Settings,
  type SourceKind,
  type Utterance,
} from "./api";

export type Tab = "live" | "records" | "settings";
export type SaveState = "idle" | "saving" | "saved" | "error";

export interface Toast {
  id: number;
  kind: "error" | "info";
  text: string;
  action?: { label: string; run: () => void };
}

/** App update: one state for the start toast and the settings card. */
export type UpdateState =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "none" }
  | { kind: "error" }
  /** `blocked`: install was pressed during a meeting. */
  | { kind: "available"; version: string; blocked: boolean }
  /** `pct` is null when the server sends no length. */
  | { kind: "installing"; version: string; pct: number | null };

/** Screen text of an update state. Empty for "idle". */
export function updateText(u: UpdateState, running: boolean): string {
  switch (u.kind) {
    case "idle":
      return "";
    case "checking":
      return "확인하는 중…";
    case "none":
      return "최신 버전입니다.";
    case "error":
      return "확인하지 못했습니다. 인터넷 연결을 확인하세요.";
    case "available":
      return u.blocked && running ? "회의를 마친 뒤 설치하세요." : `새 버전 ${u.version}이 있습니다.`;
    case "installing":
      return u.pct === null ? "내려받는 중…" : `내려받는 중… ${u.pct}%`;
  }
}

interface Download {
  done: number;
  total: number | null;
}

export interface App {
  tab: Tab;
  setTab: (t: Tab) => void;
  settings: Settings | null;
  updateSettings: (patch: Partial<Settings>) => void;
  /** Resolves when the queued saves finish: false if the last one failed. */
  settled: () => Promise<boolean>;
  saveState: SaveState;
  models: ModelInfo[];
  downloads: Record<string, Download>;
  downloadErrors: Record<string, string>;
  startDownload: (id: string) => void;
  cancelDownload: (id: string) => void;
  deleteModel: (id: string) => void;
  mics: string[];
  systemAudio: { supported: boolean; reason: string | null } | null;
  running: boolean;
  startedAt: string | null;
  meta: SessionMeta | null;
  starting: boolean;
  /** Stop pressed: the core finishes queued speech and translations. */
  stopping: boolean;
  utterances: Utterance[];
  /** Live text of speech that has not ended: one per source. */
  partials: Partial<Record<SourceKind, PartialText>>;
  engineErrors: Partial<Record<EngineStage, string>>;
  /** Why the last session did not start, or why the core stopped it. */
  sessionNotice: string | null;
  gpu: GpuStatus | null;
  refreshGpu: () => void;
  startSession: () => Promise<void>;
  stopSession: () => Promise<void>;
  /** Bumps when the session list changes (stop, delete, rename). */
  sessionsVersion: number;
  bumpSessions: () => void;
  openRecordId: string | null;
  openRecord: (id: string | null) => void;
  /** Section of the settings tab to scroll to. */
  settingsTarget: string | null;
  goSettings: (section: string) => void;
  toast: Toast | null;
  notify: (kind: Toast["kind"], text: string, action?: Toast["action"]) => void;
  dismissToast: () => void;
  update: UpdateState;
  /** The update toast shows (from the check at app start until the user closes it). */
  updateToast: boolean;
  dismissUpdateToast: () => void;
  checkUpdate: () => void;
  installUpdate: () => void;
}

const Ctx = createContext<App | null>(null);
// Levels change about ten times a second: a separate context keeps the rest of the UI still.
const LevelsCtx = createContext<Record<SourceKind, number>>({ mic: 0, system: 0 });

export function useLevels() {
  return useContext(LevelsCtx);
}

export function useApp(): App {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp outside AppProvider");
  return v;
}

/** Upsert by id. New ids arrive in order, so search from the end. */
function upsert(list: Utterance[], u: Utterance): Utterance[] {
  const last = list[list.length - 1];
  if (!last || u.id > last.id) return [...list, u];
  const next = list.slice();
  for (let i = next.length - 1; i >= 0; i--) {
    if (next[i].id === u.id) {
      next[i] = u;
      return next;
    }
    if (next[i].id < u.id) {
      next.splice(i + 1, 0, u);
      return next;
    }
  }
  next.unshift(u);
  return next;
}

/** Removes the partial line of `source` if its speech started before `endMs`. */
function dropPartial(p: App["partials"], source: SourceKind, endMs: number): App["partials"] {
  const line = p[source];
  if (!line || line.start_ms >= endMs) return p;
  const n = { ...p };
  delete n[source];
  return n;
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [tab, setTab] = useState<Tab>("live");
  const [settings, setSettings] = useState<Settings | null>(null);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [downloads, setDownloads] = useState<Record<string, Download>>({});
  const [downloadErrors, setDownloadErrors] = useState<Record<string, string>>({});
  const [mics, setMics] = useState<string[]>([]);
  const [systemAudio, setSystemAudio] = useState<App["systemAudio"]>(null);
  const [running, setRunning] = useState(false);
  const [startedAt, setStartedAt] = useState<string | null>(null);
  const [meta, setMeta] = useState<SessionMeta | null>(null);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [utterances, setUtterances] = useState<Utterance[]>([]);
  const [levels, setLevels] = useState<Record<SourceKind, number>>({ mic: 0, system: 0 });
  const [partials, setPartials] = useState<App["partials"]>({});
  const [engineErrors, setEngineErrors] = useState<App["engineErrors"]>({});
  const [sessionNotice, setSessionNotice] = useState<string | null>(null);
  const [gpu, setGpu] = useState<GpuStatus | null>(null);
  const [sessionsVersion, setSessionsVersion] = useState(0);
  const [openRecordId, setOpenRecordId] = useState<string | null>(null);
  const [settingsTarget, setSettingsTarget] = useState<string | null>(null);
  const [toast, setToast] = useState<Toast | null>(null);
  const toastId = useRef(0);
  /** Source of the capture error in the status bar: only that source may clear it. */
  const captureFrom = useRef<SourceKind | null>(null);
  const [update, setUpdate] = useState<UpdateState>({ kind: "idle" });
  const [updateToast, setUpdateToast] = useState(false);
  const updateRef = useRef<AppUpdate | null>(null);
  // On Windows the updater ends the process when the download ends: no meeting may start before that.
  const installingRef = useRef(false);
  const saveChain = useRef<Promise<boolean>>(Promise.resolve(true));
  const settingsRef = useRef<Settings | null>(null);

  const notify = useCallback<App["notify"]>((kind, text, action) => {
    setToast({ id: ++toastId.current, kind, text, action });
  }, []);
  const dismissToast = useCallback(() => setToast(null), []);

  const refreshModels = useCallback(() => {
    api.listModels().then(setModels, (e) => notify("error", errText(e)));
  }, [notify]);

  const refreshGpu = useCallback(() => {
    api.gpuStatus().then(setGpu, () => setGpu(null));
  }, []);

  // Initial load.
  useEffect(() => {
    api.getSettings().then(
      (s) => {
        settingsRef.current = s;
        setSettings(s);
      },
      (e) => notify("error", errText(e)),
    );
    refreshModels();
    refreshGpu();
    api.listMics().then(setMics, () => setMics([]));
    api.systemAudioSupport().then(setSystemAudio, () => setSystemAudio(null));
  }, [notify, refreshModels, refreshGpu]);

  const setFound = useCallback((u: AppUpdate | null) => {
    updateRef.current = u;
    setUpdate(u ? { kind: "available", version: u.version, blocked: false } : { kind: "none" });
  }, []);

  // One update check at app start. A failure shows nothing.
  useEffect(() => {
    fetchUpdate().then((u) => {
      setFound(u);
      if (u) setUpdateToast(true);
    }, () => {});
  }, [setFound]);

  // Core events.
  useEffect(() => {
    const unsubs = [
      onEvent("utterance", (u) => {
        setUtterances((l) => upsert(l, u));
        // Only the utterance of that speech replaces the line, not a later update of an older one.
        setPartials((p) => dropPartial(p, u.source, u.end_ms));
        // A fresh result from a stage clears that stage's error.
        setEngineErrors((e) => {
          const clearAsr = "asr" in e;
          const clearTr = u.translation.state === "done" && "translate" in e;
          if (!clearAsr && !clearTr) return e;
          const n = { ...e };
          if (clearAsr) delete n.asr;
          if (clearTr) delete n.translate;
          return n;
        });
      }),
      onEvent("utterance-partial", (p) =>
        setPartials((x) => (p.text ? { ...x, [p.source]: p } : dropPartial(x, p.source, p.start_ms))),
      ),
      onEvent("audio-level", (p) => setLevels((l) => ({ ...l, [p.source]: p.rms }))),
      onEvent("session-state", (p) => {
        setRunning(p.running);
        setStartedAt(p.startedAt);
        if (!p.running) {
          setLevels({ mic: 0, system: 0 });
          setPartials({});
        }
        refreshGpu();
      }),
      onEvent("engine-error", (p) => {
        // A notice (a setting the server ignores, skipped work) must stay readable:
        // a toast, not the status-bar error that the next good result clears.
        if (p.stage === "notice") notify("info", p.message);
        else if (p.stage === "session") {
          setSessionNotice(p.message);
          setSessionsVersion((v) => v + 1);
        } else if (p.message) {
          if (p.stage === "capture") captureFrom.current = p.source;
          setEngineErrors((e) => ({ ...e, [p.stage]: p.message }));
        } else if (p.stage !== "capture" || captureFrom.current === p.source) {
          setEngineErrors((e) => {
            const n = { ...e };
            delete n[p.stage];
            return n;
          });
        }
      }),
      onEvent("model-download", (p) =>
        setDownloads((d) => ({ ...d, [p.model_id]: { done: p.done_bytes, total: p.total_bytes } })),
      ),
      onEvent("model-download-done", (p) => {
        setDownloads((d) => {
          const n = { ...d };
          delete n[p.modelId];
          return n;
        });
        setDownloadErrors((d) => {
          const n = { ...d };
          if (p.error) n[p.modelId] = p.error;
          else delete n[p.modelId];
          return n;
        });
        refreshModels();
        refreshGpu();
      }),
    ];
    return () => {
      for (const u of unsubs) u.then((f) => f());
    };
  }, [refreshModels, refreshGpu, notify]);

  // Saves are sent one after another, so the last change wins.
  const updateSettings = useCallback(
    (patch: Partial<Settings>) => {
      const cur = settingsRef.current;
      if (!cur) return;
      const next = { ...cur, ...patch };
      settingsRef.current = next;
      setSettings(next);
      setSaveState("saving");
      saveChain.current = saveChain.current.then(() =>
        api.saveSettings(next).then(
          () => {
            setSaveState(settingsRef.current === next ? "saved" : "saving");
            if ("gpu" in patch || "translate" in patch || "asr" in patch) refreshGpu();
            return true;
          },
          (e) => {
            setSaveState("error");
            notify("error", errText(e));
            return false;
          },
        ),
      );
    },
    [notify, refreshGpu],
  );
  const settled = useCallback(() => saveChain.current, []);

  const startDownload = useCallback(
    (id: string) => {
      setDownloadErrors((d) => {
        const n = { ...d };
        delete n[id];
        return n;
      });
      setDownloads((d) => ({ ...d, [id]: { done: 0, total: null } }));
      api.downloadModel(id).catch((e) => {
        setDownloads((d) => {
          const n = { ...d };
          delete n[id];
          return n;
        });
        setDownloadErrors((d) => ({ ...d, [id]: errText(e) }));
      });
    },
    [],
  );

  const cancelDownload = useCallback(
    (id: string) => {
      api.cancelDownload(id).catch((e) => notify("error", errText(e)));
    },
    [notify],
  );

  const deleteModel = useCallback(
    (id: string) => {
      api.deleteModel(id).then(refreshModels, (e) => notify("error", errText(e)));
    },
    [notify, refreshModels],
  );

  const startSession = useCallback(async () => {
    if (installingRef.current) return;
    setStarting(true);
    // Clear first: the core can send utterances before the command returns.
    setUtterances([]);
    setPartials({});
    setEngineErrors({});
    setSessionNotice(null);
    try {
      const m = await api.startSession(null);
      setMeta(m);
      setRunning(true);
      setStartedAt((s) => s ?? m.started_at);
    } catch (e) {
      setSessionNotice(`회의를 시작하지 못했습니다. ${errText(e)}`);
    } finally {
      setStarting(false);
    }
  }, []);

  const bumpSessions = useCallback(() => setSessionsVersion((v) => v + 1), []);

  const openRecord = useCallback((id: string | null) => {
    setOpenRecordId(id);
    if (id) setTab("records");
  }, []);

  const stopSession = useCallback(async () => {
    setStopping(true);
    try {
      const sum = await api.stopSession();
      setRunning(false);
      setLevels({ mic: 0, system: 0 });
      bumpSessions();
      notify("info", `「${sum.meta.title}」을 저장했습니다.`, {
        label: "기록에서 보기",
        run: () => openRecord(sum.meta.id),
      });
    } catch (e) {
      notify("error", errText(e));
    } finally {
      setStopping(false);
    }
  }, [bumpSessions, notify, openRecord]);

  const checkUpdate = useCallback(() => {
    setUpdate({ kind: "checking" });
    fetchUpdate().then(setFound, () => setUpdate({ kind: "error" }));
  }, [setFound]);

  const installUpdate = useCallback(async () => {
    const u = updateRef.current;
    if (!u) return;
    const { version } = u;
    if (running || starting) {
      setUpdate({ kind: "available", version, blocked: true });
      return;
    }
    installingRef.current = true;
    setUpdate({ kind: "installing", version, pct: null });
    let total = 0;
    let done = 0;
    let shown = -1;
    try {
      await u.downloadAndInstall((e) => {
        if (e.event === "Started") total = e.data.contentLength ?? 0;
        else if (e.event === "Progress") done += e.data.chunkLength;
        if (!total) return;
        const pct = Math.min(100, Math.floor((done / total) * 100));
        if (pct === shown) return;
        shown = pct;
        setUpdate({ kind: "installing", version, pct });
      });
      // Windows: the installer ends this process before here. macOS: start the new bundle.
      await api.restartApp();
    } catch (e) {
      installingRef.current = false;
      setUpdate({ kind: "available", version, blocked: false });
      notify("error", errText(e));
    }
  }, [running, starting, notify]);

  const dismissUpdateToast = useCallback(() => setUpdateToast(false), []);

  const goSettings = useCallback((section: string) => {
    setSettingsTarget(section);
    setTab("settings");
  }, []);

  const value = useMemo<App>(
    () => ({
      tab,
      setTab,
      settings,
      updateSettings,
      settled,
      saveState,
      models,
      downloads,
      downloadErrors,
      startDownload,
      cancelDownload,
      deleteModel,
      mics,
      systemAudio,
      running,
      startedAt,
      meta,
      starting,
      stopping,
      utterances,
      partials,
      engineErrors,
      sessionNotice,
      gpu,
      refreshGpu,
      startSession,
      stopSession,
      sessionsVersion,
      bumpSessions,
      openRecordId,
      openRecord,
      settingsTarget,
      goSettings,
      toast,
      notify,
      dismissToast,
      update,
      updateToast,
      dismissUpdateToast,
      checkUpdate,
      installUpdate,
    }),
    [
      tab,
      settings,
      updateSettings,
      settled,
      saveState,
      models,
      downloads,
      downloadErrors,
      startDownload,
      cancelDownload,
      deleteModel,
      mics,
      systemAudio,
      running,
      startedAt,
      meta,
      starting,
      stopping,
      utterances,
      partials,
      engineErrors,
      sessionNotice,
      gpu,
      refreshGpu,
      startSession,
      stopSession,
      sessionsVersion,
      bumpSessions,
      openRecordId,
      openRecord,
      settingsTarget,
      goSettings,
      toast,
      notify,
      dismissToast,
      update,
      updateToast,
      dismissUpdateToast,
      checkUpdate,
      installUpdate,
    ],
  );

  return (
    <Ctx.Provider value={value}>
      <LevelsCtx.Provider value={levels}>{children}</LevelsCtx.Provider>
    </Ctx.Provider>
  );
}
