import { useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  errText,
  pickSavePath,
  type ExportFormat,
  type SessionMeta,
  type SessionSummary,
  type Utterance,
} from "./api";
import {
  dateTime,
  languageName,
  minutes,
  number,
  sessionFileName,
  speakerBase,
  speakerNumber,
  timeOnly,
} from "./format";
import { useApp } from "./store";
import { ConfirmDialog, Icon } from "./ui";
import { UtteranceRow } from "./UtteranceRow";

const EXPORTS: { format: ExportFormat; label: string; ext: string; filter: string }[] = [
  { format: "markdown", label: "Markdown", ext: "md", filter: "Markdown" },
  { format: "srt", label: "SRT 자막", ext: "srt", filter: "SRT 자막" },
  { format: "json", label: "JSON", ext: "json", filter: "JSON" },
];

interface Detail {
  meta: SessionMeta;
  utterances: Utterance[];
}

/** Text of a session for the content search. */
function searchText(utterances: Utterance[]): string {
  return utterances
    .map((u) => `${u.original}\n${u.translation.state === "done" ? u.translation.text : ""}`)
    .join("\n")
    .toLowerCase();
}

export function RecordsTab({ active }: { active: boolean }) {
  const app = useApp();
  const { openRecordId, sessionsVersion, notify, settings } = app;
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [query, setQuery] = useState("");
  const [content, setContent] = useState<Record<string, string>>({});
  const [detail, setDetail] = useState<Detail | null>(null);
  const [loading, setLoading] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [namingSpeakers, setNamingSpeakers] = useState<Record<string, string> | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);

  // The list reloads each time the tab opens or the session list changes.
  useEffect(() => {
    if (!active) return;
    let stale = false;
    api.listSessions().then(
      (l) => {
        if (stale) return;
        setSessions(l);
        // Sessions that changed must be indexed again.
        setContent((c) => {
          const keep: Record<string, string> = {};
          for (const s of l) if (s.meta.id in c) keep[s.meta.id] = c[s.meta.id];
          return keep;
        });
      },
      (e) => notify("error", errText(e)),
    );
    return () => {
      stale = true;
    };
  }, [active, sessionsVersion, notify]);

  // Select the first session when none is selected, or when the selected one is gone.
  useEffect(() => {
    if (!sessions) return;
    if (openRecordId && sessions.some((s) => s.meta.id === openRecordId)) return;
    app.openRecord(sessions[0]?.meta.id ?? null);
  }, [sessions, openRecordId]);

  // Load the selected session.
  useEffect(() => {
    if (!openRecordId || !active) return;
    let stale = false;
    setLoading(true);
    api.loadSession(openRecordId).then(
      (d) => {
        if (stale) return;
        setDetail(d);
        setLoading(false);
        setContent((c) => ({ ...c, [d.meta.id]: searchText(d.utterances) }));
      },
      (e) => {
        if (stale) return;
        setLoading(false);
        notify("error", errText(e));
      },
    );
    return () => {
      stale = true;
    };
  }, [openRecordId, active, sessionsVersion, notify]);

  useEffect(() => {
    setRenaming(null);
    setNamingSpeakers(null);
  }, [openRecordId]);

  // Search by content: load the sessions that are not indexed yet, one after another.
  const q = query.trim().toLowerCase();
  const [indexing, setIndexing] = useState(false);
  useEffect(() => {
    if (!q || !sessions) return;
    const todo = sessions.filter((s) => !(s.meta.id in content));
    if (todo.length === 0) return;
    let stale = false;
    setIndexing(true);
    (async () => {
      for (const s of todo) {
        if (stale) return;
        try {
          const d = await api.loadSession(s.meta.id);
          if (!stale) setContent((c) => ({ ...c, [s.meta.id]: searchText(d.utterances) }));
        } catch {
          if (!stale) setContent((c) => ({ ...c, [s.meta.id]: "" }));
        }
      }
      if (!stale) setIndexing(false);
    })();
    return () => {
      stale = true;
      setIndexing(false);
    };
  }, [q, sessions, content]);

  const shown = useMemo(() => {
    if (!sessions) return [];
    if (!q) return sessions;
    return sessions.filter(
      (s) => s.meta.title.toLowerCase().includes(q) || (content[s.meta.id] ?? "").includes(q),
    );
  }, [sessions, q, content]);

  const saveMeta = useCallback(
    async (meta: SessionMeta) => {
      try {
        await api.updateSessionMeta(meta);
        setDetail((d) => (d ? { ...d, meta } : d));
        app.bumpSessions();
      } catch (e) {
        notify("error", errText(e));
      }
    },
    [app, notify],
  );

  const doExport = async (x: (typeof EXPORTS)[number]) => {
    if (!detail) return;
    try {
      const path = await pickSavePath(sessionFileName(detail.meta.title, x.ext), x.filter, x.ext);
      if (!path) return;
      await api.exportSession(detail.meta.id, x.format, path);
      notify("info", `${x.label} 파일로 내보냈습니다.`);
    } catch (e) {
      notify("error", errText(e));
    }
  };

  const doDelete = async () => {
    if (!detail) return;
    setConfirmDelete(false);
    try {
      await api.deleteSession(detail.meta.id);
      setDetail(null);
      setSessions((l) => l && l.filter((x) => x.meta.id !== detail.meta.id));
      app.openRecord(null);
      app.bumpSessions();
      notify("info", `「${detail.meta.title}」을 지웠습니다.`);
    } catch (e) {
      notify("error", errText(e));
    }
  };

  const speakers = useMemo(() => {
    const ids = [...new Set((detail?.utterances ?? []).map((u) => u.speaker))];
    return ids.sort((a, b) => (speakerNumber(a) ?? 1e9) - (speakerNumber(b) ?? 1e9) || a.localeCompare(b));
  }, [detail]);

  const current = detail && detail.meta.id === openRecordId ? detail : null;
  const dir = settings?.sessions_dir ?? "문서/morago/sessions";

  return (
    <div className="panel">
      <div className="rec">
        <div className="list">
          <label className="search">
            <Icon name="search" />
            <span className="sr">회의 이름이나 내용으로 찾기</span>
            <input
              type="search"
              placeholder="회의 이름이나 내용으로 찾기"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          {indexing && <p className="fhint pad">내용을 찾는 중…</p>}
          <nav aria-label="회의 목록" className="items">
            {shown.map((s) => (
              <button
                type="button"
                key={s.meta.id}
                className="it"
                aria-current={s.meta.id === openRecordId ? "true" : undefined}
                onClick={() => app.openRecord(s.meta.id)}
              >
                <span className="t">{s.meta.title}</span>
                <span className="m">
                  {dateTime(s.meta.started_at)} · {minutes(s.duration_ms)} · 화자 {number(s.speaker_count)}명 · 발화{" "}
                  {number(s.utterance_count)}개
                </span>
              </button>
            ))}
          </nav>
          {sessions && sessions.length > 0 && shown.length === 0 && (
            <p className="fhint pad">「{query.trim()}」이(가) 들어간 회의가 없습니다.</p>
          )}
        </div>
        <div className="det">
          {sessions && sessions.length === 0 ? (
            <div className="guide center">
              <h2>저장된 회의가 없습니다</h2>
              <p>회의 탭에서 시작해 중지하면 원문과 번역이 여기에 저장됩니다.</p>
              <button type="button" className="ab primary" onClick={() => app.setTab("live")}>
                회의 탭으로 가기
              </button>
            </div>
          ) : !current ? (
            <p className="empty-load">{loading ? "불러오는 중…" : "왼쪽에서 회의를 고르세요."}</p>
          ) : (
            <>
              <div className="dhead">
                <div className="dtitle">
                  {renaming === null ? (
                    <h2>{current.meta.title}</h2>
                  ) : (
                    <form
                      className="row2"
                      onSubmit={(e) => {
                        e.preventDefault();
                        const t = renaming.trim();
                        if (t && t !== current.meta.title) void saveMeta({ ...current.meta, title: t });
                        setRenaming(null);
                      }}
                    >
                      <input
                        className="inp"
                        aria-label="회의 이름"
                        value={renaming}
                        autoFocus
                        onChange={(e) => setRenaming(e.target.value)}
                        onKeyDown={(e) => e.key === "Escape" && setRenaming(null)}
                      />
                      <button type="submit" className="ab sm primary">
                        저장
                      </button>
                      <button type="button" className="ab sm ghost" onClick={() => setRenaming(null)}>
                        취소
                      </button>
                    </form>
                  )}
                  <span className="m">
                    {dateTime(current.meta.started_at)}
                    {current.meta.ended_at ? `–${timeOnly(current.meta.ended_at)}` : ""} · 번역 언어{" "}
                    {languageName(current.meta.target_language)} · {current.meta.asr_label} ·{" "}
                    {current.meta.translator_label}
                  </span>
                </div>
                {renaming === null && (
                  <button type="button" className="ab ghost sm" onClick={() => setRenaming(current.meta.title)}>
                    이름 바꾸기
                  </button>
                )}
                <button
                  type="button"
                  className="ab sm"
                  aria-expanded={namingSpeakers !== null}
                  onClick={() => setNamingSpeakers(namingSpeakers ? null : { ...current.meta.speaker_names })}
                >
                  화자 이름 바꾸기
                </button>
                <span className="exp" role="group" aria-label="내보내기">
                  {EXPORTS.map((x, i) => (
                    <button type="button" className="ab sm" key={x.format} onClick={() => void doExport(x)}>
                      {i === 0 && <Icon name="download" />}
                      {x.label}
                    </button>
                  ))}
                </span>
                <button type="button" className="ab ghost sm danger" onClick={() => setConfirmDelete(true)}>
                  삭제
                </button>
              </div>
              {namingSpeakers && (
                <form
                  className="spkpanel"
                  onSubmit={(e) => {
                    e.preventDefault();
                    const names: Record<string, string> = {};
                    for (const [k, v] of Object.entries(namingSpeakers)) if (v.trim()) names[k] = v.trim();
                    void saveMeta({ ...current.meta, speaker_names: names });
                    setNamingSpeakers(null);
                  }}
                >
                  <p className="fhint">
                    이름은 이 회의의 모든 발화와 내보낸 파일에 적용됩니다. 비워 두면 「화자 N」으로 보입니다.
                  </p>
                  <div className="spkgrid">
                    {speakers.map((id) => (
                      <label key={id} className="fld2">
                        <span>{speakerBase(id)}</span>
                        <input
                          className="inp"
                          value={namingSpeakers[id] ?? ""}
                          placeholder="이름"
                          onChange={(e) => setNamingSpeakers({ ...namingSpeakers, [id]: e.target.value })}
                        />
                      </label>
                    ))}
                  </div>
                  <div className="row2">
                    <button type="submit" className="ab sm primary">
                      저장
                    </button>
                    <button type="button" className="ab sm ghost" onClick={() => setNamingSpeakers(null)}>
                      취소
                    </button>
                  </div>
                </form>
              )}
              <div className="feed" tabIndex={0} aria-label="발화 목록">
                {current.utterances.map((u) => (
                  <UtteranceRow key={u.id} u={u} names={current.meta.speaker_names} />
                ))}
                {current.utterances.length === 0 && <p className="listening">이 회의에는 발화가 없습니다.</p>}
              </div>
            </>
          )}
        </div>
      </div>
      <div className="bar">
        <span>저장 위치: {dir}</span>
        <span className="sp" />
        {sessions && <span>회의 {number(sessions.length)}개</span>}
      </div>
      <ConfirmDialog
        open={confirmDelete}
        title="이 회의를 지울까요?"
        body={`「${current?.meta.title ?? ""}」의 원문, 번역, 화자 기록이 모두 지워지고 되돌릴 수 없습니다.`}
        confirmLabel="지우기"
        onConfirm={() => void doDelete()}
        onCancel={() => setConfirmDelete(false)}
      />
    </div>
  );
}
