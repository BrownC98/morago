import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { EngineStage, ModelDevice } from "./api";
import { clock, languageName, number, providerLabel, TARGET_LANGUAGES } from "./format";
import { useApp, useLevels } from "./store";
import { Icon, Select, Switch } from "./ui";
import { PartialRow, UtteranceRow } from "./UtteranceRow";

const BAR_HEIGHTS = [5, 9, 12, 14];

/** Four bars. rms is small in real speech, so the scale is a square root. */
function Meter({ rms, on }: { rms: number; on: boolean }) {
  const lit = on ? Math.round(Math.min(1, Math.sqrt(rms) * 1.5) * 4) : 0;
  return (
    <span className="lvl" aria-hidden="true">
      {BAR_HEIGHTS.map((h, i) => (
        <i key={h} className={i < lit ? undefined : "z"} style={{ height: h }} />
      ))}
    </span>
  );
}

export function Timer() {
  const { running, startedAt } = useApp();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    setNow(Date.now());
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [running]);
  if (!running || !startedAt) return null;
  return (
    <span className="timer" role="timer" aria-label="회의 시간">
      {clock(now - Date.parse(startedAt))}
    </span>
  );
}

const STAGE_NAME: Record<EngineStage, string> = {
  capture: "소리 입력",
  asr: "음성 인식",
  translate: "번역",
  store: "저장",
  model: "모델",
  notice: "알림",
  session: "회의",
};

export function LiveTab({ active }: { active: boolean }) {
  const app = useApp();
  const levels = useLevels();
  const { settings, running, utterances, partials, models, systemAudio, engineErrors, meta } = app;
  const feedRef = useRef<HTMLDivElement>(null);
  const stuck = useRef(true);
  const [away, setAway] = useState(false);

  const toBottom = useCallback(() => {
    const el = feedRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    stuck.current = true;
    setAway(false);
  }, []);

  const onScroll = useCallback(() => {
    const el = feedRef.current;
    if (!el) return;
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
    stuck.current = near;
    setAway(!near);
  }, []);

  useLayoutEffect(() => {
    if (stuck.current) {
      const el = feedRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    } else if (utterances.length > 0) {
      setAway(true);
    }
  }, [utterances, partials, active]);

  useEffect(() => {
    if (running) toBottom();
  }, [running, toBottom]);

  if (!settings) return <div className="panel"><p className="empty-load">설정을 불러오는 중…</p></div>;

  const sysOk = systemAudio?.supported !== false;
  const langs = TARGET_LANGUAGES.includes(settings.target_language)
    ? TARGET_LANGUAGES
    : [settings.target_language, ...TARGET_LANGUAGES];

  const missing = (
    [
      ["set-asr", "음성 인식", settings.asr],
      ["set-translate", "번역", settings.translate],
    ] as const
  ).flatMap(([section, name, cfg]) => {
    if (cfg.kind !== "local") return [];
    const m = models.find((x) => x.id === cfg.model_id);
    return m && !m.installed ? [{ section, name, model: m.name }] : [];
  });

  const asrLabel = running && meta ? meta.asr_label : providerLabel(settings.asr, models);
  const trLabel = running && meta ? meta.translator_label : providerLabel(settings.translate, models);
  const device = (d: ModelDevice | null | undefined) => (d ? (d.gpu ? " · GPU" : " · CPU") : "");
  const asrDevice = device(app.gpu?.asr);
  const trDevice = device(app.gpu?.translate);
  const otherErrors = (["capture", "store", "model"] as const).filter((s) => engineErrors[s]);
  const idle = !running && utterances.length === 0;

  return (
    <div className="panel">
      <div className="live">
        <div className="ctl">
          <span className="row2" style={{ gap: 16 }}>
            <Switch
              checked={settings.capture_mic}
              disabled={running}
              title={running ? "회의 중에는 바꿀 수 없습니다" : undefined}
              onChange={(v) => app.updateSettings({ capture_mic: v })}
            >
              마이크
              <Meter rms={levels.mic} on={running && settings.capture_mic} />
            </Switch>
            <Switch
              checked={sysOk && settings.capture_system}
              disabled={running || !sysOk}
              describedBy={sysOk ? undefined : "sys-reason"}
              title={running ? "회의 중에는 바꿀 수 없습니다" : undefined}
              onChange={(v) => app.updateSettings({ capture_system: v })}
            >
              시스템 소리
              <Meter rms={levels.system} on={running && sysOk && settings.capture_system} />
            </Switch>
          </span>
          <span className="sp" />
          <Select
            inline
            label="번역할 언어"
            value={settings.target_language}
            onChange={(v) => app.updateSettings({ target_language: v })}
            options={langs.map((c) => ({ value: c, label: languageName(c) }))}
          />
          {running ? (
            <button
              type="button"
              className="ab stop"
              disabled={app.stopping}
              onClick={() => void app.stopSession()}
            >
              <Icon name="stop" />
              {app.stopping ? "남은 발화를 저장하는 중…" : "중지하고 저장"}
            </button>
          ) : (
            <button
              type="button"
              className="ab primary"
              disabled={app.starting || missing.length > 0 || (!settings.capture_mic && !(sysOk && settings.capture_system))}
              onClick={() => void app.startSession()}
            >
              <Icon name="play" />
              회의 시작
            </button>
          )}
        </div>
        {!sysOk && (
          <p id="sys-reason" className="ctl-note">
            시스템 소리를 받을 수 없습니다. {systemAudio?.reason}
          </p>
        )}
        {!running && !settings.capture_mic && !(sysOk && settings.capture_system) && (
          <p className="ctl-note">마이크나 시스템 소리 중 하나는 켜야 시작할 수 있습니다.</p>
        )}
        <div className="feedwrap">
          <div
            ref={feedRef}
            className="feed"
            onScroll={onScroll}
            aria-live="polite"
            aria-relevant="additions text"
            aria-label="발화 목록"
            role="log"
            tabIndex={0}
          >
            {!running && app.sessionNotice && (
              <div className="notice" role="alert">
                <span>{app.sessionNotice}</span>
              </div>
            )}
            {!running &&
              missing.map((m) => (
                <div className="notice" key={m.section}>
                  <span>
                    선택한 {m.name} 모델({m.model})을 아직 내려받지 않았습니다. 내려받아야 회의를 시작할 수 있습니다.
                  </span>
                  <button type="button" className="ab sm" onClick={() => app.goSettings(m.section)}>
                    설정에서 내려받기
                  </button>
                </div>
              ))}
            {idle && (
              <div className="guide">
                <h2>회의를 시작해 보세요</h2>
                <ol>
                  <li>들을 소리를 고릅니다. 마이크는 내 목소리, 시스템 소리는 화상회의 상대의 목소리입니다.</li>
                  <li>번역할 언어를 고릅니다. 회의 중에도 바꿀 수 있습니다.</li>
                  <li>
                    <b>회의 시작</b>을 누릅니다. 말이 들어오면 원문과 번역이 이 자리에 쌓이고, 중지하면 기록에 저장됩니다.
                  </li>
                </ol>
              </div>
            )}
            {running && utterances.length === 0 && Object.keys(partials).length === 0 && (
              <p className="listening">듣고 있습니다. 말하기 시작하면 여기에 나타납니다.</p>
            )}
            {utterances.map((u, i) => (
              <UtteranceRow key={u.id} u={u} names={meta?.speaker_names ?? {}} latest={running && i === utterances.length - 1} />
            ))}
            {(Object.values(partials)).map((p) => (
              <PartialRow key={p.source} p={p} />
            ))}
          </div>
          {away && utterances.length > 0 && (
            <button type="button" className="ab primary jump" onClick={toBottom}>
              <Icon name="down" />
              최신 발화로 이동
            </button>
          )}
        </div>
      </div>
      <div className="bar">
        <span className="bi">
          <span className={engineErrors.asr ? "dot bad" : "dot"} aria-hidden="true" />
          음성 인식: {asrLabel}
          {asrDevice}
          {engineErrors.asr && <span className="berr" title={engineErrors.asr}> · {engineErrors.asr}</span>}
        </span>
        <span className="bi">
          <span className={engineErrors.translate ? "dot bad" : "dot"} aria-hidden="true" />
          번역: {trLabel}
          {trDevice}
          {engineErrors.translate && <span className="berr" title={engineErrors.translate}> · {engineErrors.translate}</span>}
        </span>
        {otherErrors.map((s) => (
          <span className="bi" key={s}>
            <span className="dot bad" aria-hidden="true" />
            <span className="berr" title={engineErrors[s]}>
              {STAGE_NAME[s]} 오류: {engineErrors[s]}
            </span>
          </span>
        ))}
        <span className="sp" />
        {(running || utterances.length > 0) && <span>자동 저장: 발화 {number(utterances.length)}개</span>}
      </div>
    </div>
  );
}
