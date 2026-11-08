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
  errText,
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

interface Download {
  done: number;
  total: number | null;
}

export interface App {
  tab: Tab;
  setTab: (t: Tab) => void;
  settings: Settings | null;
  updateSettings: (patch: Partial<Settings>) => void;
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
  const saveChain = useRef<Promise<void>>(Promise.resolve());
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
        } else setEngineErrors((e) => ({ ...e, [p.stage]: p.message }));
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
            if ("gpu" in patch || "translate" in patch) refreshGpu();
          },
          (e) => {
            setSaveState("error");
            notify("error", errText(e));
          },
        ),
      );
    },
    [notify, refreshGpu],
  );

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
    }),
    [
      tab,
      settings,
      updateSettings,
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
    ],
  );

  return (
    <Ctx.Provider value={value}>
      <LevelsCtx.Provider value={levels}>{children}</LevelsCtx.Provider>
    </Ctx.Provider>
  );
}
