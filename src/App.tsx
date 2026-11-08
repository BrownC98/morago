import { useEffect, useRef, type KeyboardEvent } from "react";
import { LiveTab, Timer } from "./LiveTab";
import { RecordsTab } from "./RecordsTab";
import { SettingsTab } from "./SettingsTab";
import { AppProvider, useApp, type Tab } from "./store";
import { useTheme } from "./theme";
import { Icon } from "./ui";

const TABS: { id: Tab; label: string }[] = [
  { id: "live", label: "회의" },
  { id: "records", label: "기록" },
  { id: "settings", label: "설정" },
];

function Shell() {
  const app = useApp();
  const { tab, setTab, toast } = app;
  useTheme(app.settings?.theme);
  useEffect(() => {
    if (!toast || toast.kind !== "info") return;
    const t = setTimeout(app.dismissToast, 8000);
    return () => clearTimeout(t);
  }, [toast, app.dismissToast]);
  const refs = useRef<Record<Tab, HTMLButtonElement | null>>({ live: null, records: null, settings: null });

  const onKey = (e: KeyboardEvent) => {
    const i = TABS.findIndex((t) => t.id === tab);
    let n = -1;
    if (e.key === "ArrowRight") n = (i + 1) % TABS.length;
    else if (e.key === "ArrowLeft") n = (i + TABS.length - 1) % TABS.length;
    else if (e.key === "Home") n = 0;
    else if (e.key === "End") n = TABS.length - 1;
    if (n < 0) return;
    e.preventDefault();
    setTab(TABS[n].id);
    refs.current[TABS[n].id]?.focus();
  };

  return (
    <div className="page">
      <header className="top">
        <h1 className="brand">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 7h10M4 12h7M4 17h12M17 9l3 3-3 3" />
          </svg>
          morago
        </h1>
        <div className="tabs" role="tablist" aria-label="화면" onKeyDown={onKey}>
          {TABS.map((t) => (
            <button
              key={t.id}
              ref={(el) => {
                refs.current[t.id] = el;
              }}
              type="button"
              role="tab"
              id={`tab-${t.id}`}
              aria-controls={`panel-${t.id}`}
              aria-selected={tab === t.id}
              tabIndex={tab === t.id ? 0 : -1}
              className="tab"
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        <span className="sp" />
        <Timer />
      </header>
      <main>
        <div role="tabpanel" id="panel-live" aria-labelledby="tab-live" hidden={tab !== "live"}>
          <LiveTab active={tab === "live"} />
        </div>
        <div role="tabpanel" id="panel-records" aria-labelledby="tab-records" hidden={tab !== "records"}>
          <RecordsTab active={tab === "records"} />
        </div>
        <div role="tabpanel" id="panel-settings" aria-labelledby="tab-settings" hidden={tab !== "settings"}>
          <SettingsTab active={tab === "settings"} />
        </div>
      </main>
      {toast && (
        <div className="toast" data-kind={toast.kind} role={toast.kind === "error" ? "alert" : "status"} key={toast.id}>
          <span>{toast.text}</span>
          {toast.action && (
            <button
              type="button"
              className="ab sm"
              onClick={() => {
                toast.action?.run();
                app.dismissToast();
              }}
            >
              {toast.action.label}
            </button>
          )}
          <button type="button" className="ab ghost sm icon" aria-label="알림 닫기" onClick={app.dismissToast}>
            <Icon name="close" />
          </button>
        </div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  );
}
