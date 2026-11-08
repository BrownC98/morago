import { useEffect, useRef, useState } from "react";
import {
  api,
  errText,
  type Endpoint,
  type KeyRole,
  type ModelInfo,
  type ProviderConfig,
  type ReasoningEffort,
  type ThemeSetting,
} from "./api";
import { bytes, languageName, TARGET_LANGUAGES } from "./format";
import { useApp } from "./store";
import { ConfirmDialog, Radios, Select, Switch } from "./ui";

const SECTIONS = [
  { id: "set-lang", label: "언어" },
  { id: "set-theme", label: "화면" },
  { id: "set-asr", label: "음성 인식" },
  { id: "set-translate", label: "번역" },
  { id: "set-input", label: "소리 입력" },
  { id: "set-save", label: "저장" },
  { id: "set-about", label: "정보" },
];

const THEMES: { value: ThemeSetting; label: string }[] = [
  { value: "light", label: "라이트" },
  { value: "dark", label: "다크" },
  { value: "system", label: "시스템 설정 따름" },
];

function ModelRow({ model, selected, onUse }: { model: ModelInfo; selected?: boolean; onUse?: () => void }) {
  const { downloads, downloadErrors, startDownload, cancelDownload } = useApp();
  const dl = downloads[model.id];
  const err = downloadErrors[model.id];
  const pct = dl && dl.total ? Math.min(100, Math.round((dl.done / dl.total) * 100)) : null;
  return (
    <div className="mrow">
      <span className="n">{model.name}</span>
      <span className="s">{bytes(model.size_bytes)}</span>
      {dl ? (
        <>
          <span className="prog">
            <span
              className="pb"
              role="progressbar"
              aria-label={`${model.name} 내려받기`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={pct ?? undefined}
            >
              <i style={{ transform: `scaleX(${(pct ?? 0) / 100})` }} />
            </span>
            <span>{pct === null ? "…" : `${pct}%`}</span>
          </span>
          <button type="button" className="ab ghost sm" onClick={() => cancelDownload(model.id)}>
            취소
          </button>
        </>
      ) : (
        <>
          <span className="fhint">
            {model.note}
            {err && <span className="errtext"> · {err}</span>}
          </span>
          {model.installed ? (
            selected ? (
              <span className="ok">사용 중</span>
            ) : onUse ? (
              <button type="button" className="ab sm" onClick={onUse} aria-label={`${model.name} 사용하기`}>
                사용하기
              </button>
            ) : (
              <span className="ok">내려받음</span>
            )
          ) : (
            <span className="row2">
              {selected && <span className="fhint">선택됨</span>}
              <button
                type="button"
                className="ab sm"
                onClick={() => startDownload(model.id)}
                aria-label={`${model.name} 내려받기 (${bytes(model.size_bytes)})`}
              >
                {err ? "다시 내려받기" : "내려받기"}
              </button>
            </span>
          )}
        </>
      )}
    </div>
  );
}

/** Password input and the buttons for one key. The key text lives only in the input: not in React state. */
function KeyField({ role, baseUrl }: { role: KeyRole; baseUrl: string }) {
  const { notify } = useApp();
  const [hint, setHint] = useState<string | null | undefined>(undefined);
  const [replacing, setReplacing] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let stale = false;
    setHint(undefined);
    setReplacing(false);
    if (!baseUrl) return;
    api.apiKeyHint(role, baseUrl).then(
      (h) => !stale && setHint(h),
      () => !stale && setHint(null),
    );
    return () => {
      stale = true;
    };
  }, [role, baseUrl]);

  const save = async () => {
    const el = input.current;
    if (!el || !el.value) return;
    setBusy(true);
    try {
      await api.setApiKey(role, baseUrl, el.value);
      el.value = "";
      setHint(await api.apiKeyHint(role, baseUrl));
      setReplacing(false);
      notify("info", "API 키를 저장했습니다.");
    } catch (e) {
      notify("error", errText(e));
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    setConfirmDelete(false);
    try {
      await api.deleteApiKey(role, baseUrl);
      setHint(null);
      setReplacing(false);
      notify("info", "API 키를 지웠습니다.");
    } catch (e) {
      notify("error", errText(e));
    }
  };

  if (!baseUrl) return <span className="fhint">서비스 주소를 먼저 넣으세요.</span>;
  const showInput = hint === null || replacing;
  return (
    <>
      {showInput ? (
        <form
          className="row2 grow"
          onSubmit={(e) => {
            e.preventDefault();
            void save();
          }}
        >
          <input
            ref={input}
            className="inp mono grow"
            type="password"
            autoComplete="off"
            spellCheck={false}
            autoCapitalize="off"
            aria-label="API 키"
            placeholder={replacing ? "새 API 키" : "API 키 (로컬 서버는 비워 둠)"}
            autoFocus={replacing}
          />
          <button type="submit" className="ab sm" disabled={busy}>
            저장
          </button>
          {replacing && (
            <button
              type="button"
              className="ab ghost sm"
              onClick={() => {
                if (input.current) input.current.value = "";
                setReplacing(false);
              }}
            >
              취소
            </button>
          )}
        </form>
      ) : (
        <span className="row2 grow">
          <output className="inp mono grow" aria-label="저장된 API 키">
            {hint ?? "…"}
          </output>
          <button type="button" className="ab sm" onClick={() => setReplacing(true)}>
            바꾸기
          </button>
          <button type="button" className="ab ghost sm danger" onClick={() => setConfirmDelete(true)}>
            삭제
          </button>
        </span>
      )}
      <ConfirmDialog
        open={confirmDelete}
        title="API 키를 지울까요?"
        body="저장한 키가 이 컴퓨터의 자격 증명 저장소에서 지워집니다. 다시 쓰려면 키를 새로 넣어야 합니다."
        confirmLabel="지우기"
        onConfirm={() => void remove()}
        onCancel={() => setConfirmDelete(false)}
      />
    </>
  );
}

const EFFORTS: { value: ReasoningEffort | ""; label: string }[] = [
  { value: "", label: "기본값(보내지 않음)" },
  { value: "none", label: "none" },
  { value: "minimal", label: "minimal" },
  { value: "low", label: "low" },
  { value: "medium", label: "medium" },
  { value: "high", label: "high" },
];

/** `partial_interval_ms`: off, then 0.5 s to 10 s in 0.5 s steps. */
const INTERVALS: { value: string; label: string }[] = [
  { value: "off", label: "끔" },
  ...Array.from({ length: 20 }, (_, i) => {
    const ms = (i + 1) * 500;
    return { value: String(ms), label: `${ms / 1000}초${ms === 1500 ? "(기본값)" : ""}` };
  }),
];

function EndpointForm({
  role,
  endpoint,
  effort,
  interval,
  onCommit,
  onEffort,
  onInterval,
}: {
  role: KeyRole;
  endpoint: Endpoint;
  effort?: ReasoningEffort;
  interval?: number | null;
  onCommit: (e: Endpoint) => void;
  onEffort: (e: ReasoningEffort | undefined) => void;
  onInterval: (ms: number | null) => void;
}) {
  const [url, setUrl] = useState(endpoint.base_url);
  const [model, setModel] = useState(endpoint.model);
  const [test, setTest] = useState<{ ok: boolean; message: string } | "busy" | null>(null);
  const baseUrl = endpoint.base_url.trim();

  useEffect(() => {
    setUrl(endpoint.base_url);
    setModel(endpoint.model);
  }, [endpoint.base_url, endpoint.model]);

  const commit = () => {
    const next = { base_url: url.trim(), model: model.trim() };
    if (next.base_url !== endpoint.base_url || next.model !== endpoint.model) onCommit(next);
  };
  const enter = (e: React.KeyboardEvent) => e.key === "Enter" && (e.target as HTMLElement).blur();

  const runTest = async () => {
    commit();
    setTest("busy");
    try {
      setTest(await api.testProvider(role));
    } catch (e) {
      setTest({ ok: false, message: errText(e) });
    }
  };

  const id = `${role}-ep`;
  return (
    <div className="fld wide">
      <label htmlFor={`${id}-url`}>서비스 주소</label>
      <input
        id={`${id}-url`}
        className="inp mono"
        value={url}
        placeholder="https://api.openai.com/v1"
        spellCheck={false}
        autoComplete="off"
        translate="no"
        inputMode="url"
        onChange={(e) => setUrl(e.target.value)}
        onBlur={commit}
        onKeyDown={enter}
      />
      <label htmlFor={`${id}-key`}>API 키</label>
      <div className="row2" id={`${id}-key`}>
        <KeyField role={role} baseUrl={baseUrl} />
        <button type="button" className="ab sm" disabled={test === "busy" || !baseUrl} onClick={() => void runTest()}>
          연결 시험
        </button>
      </div>
      <label htmlFor={`${id}-model`}>모델</label>
      <input
        id={`${id}-model`}
        className="inp mono"
        value={model}
        placeholder={role === "asr" ? "whisper-1" : "gpt-4o-mini"}
        spellCheck={false}
        autoComplete="off"
        translate="no"
        onChange={(e) => setModel(e.target.value)}
        onBlur={commit}
        onKeyDown={enter}
      />
      {role === "translate" && (
        <>
          <span className="flab" aria-hidden="true">추론 강도</span>
          <Select
            label="추론 강도"
            width={260}
            value={effort ?? ""}
            onChange={(v) => onEffort(v === "" ? undefined : (v as ReasoningEffort))}
            options={EFFORTS}
          />
          <span className="flab" aria-hidden="true">중간 번역 간격</span>
          <Select
            label="중간 번역 간격"
            width={260}
            value={interval === null ? "off" : String(interval ?? 1500)}
            onChange={(v) => onInterval(v === "off" ? null : Number(v))}
            options={INTERVALS}
          />
          <span />
          <p className="fhint">
            말하는 동안 보이는 회색 번역을 이 간격마다 한 번 요청합니다. 간격이 짧을수록 요청이 많아져 요금이
            늘어납니다.
          </p>
        </>
      )}
      <span />
      <p className={test && test !== "busy" && !test.ok ? "fhint errtext" : "fhint"} role="status">
        {test === "busy" ? "연결을 시험하는 중…" : test ? `${test.ok ? "연결됨. " : "연결 실패. "}${test.message}` : ""}
      </p>
    </div>
  );
}

function ProviderCard({ role }: { role: KeyRole }) {
  const { settings, updateSettings, models } = useApp();
  const cfg = settings ? (role === "asr" ? settings.asr : settings.translate) : null;
  const wantKind = role === "asr" ? "asr" : "translation";
  const list = models.filter((m) => m.kind === wantKind);
  // Remember the other variant, so a switch back does not lose what the user typed.
  const lastEndpoint = useRef<Endpoint>({ base_url: "", model: "" });
  const lastEffort = useRef<ReasoningEffort | undefined>(undefined);
  const lastInterval = useRef<number | null | undefined>(undefined);
  const lastLocal = useRef<string | null>(null);
  if (cfg?.kind === "custom") {
    lastEndpoint.current = cfg.endpoint;
    lastEffort.current = cfg.reasoning_effort;
    lastInterval.current = cfg.partial_interval_ms;
  }
  if (cfg?.kind === "local") lastLocal.current = cfg.model_id;
  if (!cfg) return null;
  const set = (c: ProviderConfig) => updateSettings(role === "asr" ? { asr: c } : { translate: c });
  // An undefined value is an absent key in JSON: the default, as in the settings file.
  const custom = (
    endpoint: Endpoint,
    effort: ReasoningEffort | undefined,
    interval: number | null | undefined,
  ): ProviderConfig =>
    role === "translate"
      ? { kind: "custom", endpoint, reasoning_effort: effort, partial_interval_ms: interval }
      : { kind: "custom", endpoint };

  return (
    <div className="card">
      <Radios
        label={role === "asr" ? "음성 인식 방법" : "번역 방법"}
        value={cfg.kind}
        onChange={(k) => {
          if (k === "custom") set(custom(lastEndpoint.current, lastEffort.current, lastInterval.current));
          else set({ kind: "local", model_id: lastLocal.current ?? list.find((m) => m.installed)?.id ?? list[0]?.id ?? "" });
        }}
        options={[
          { value: "local", label: role === "asr" ? "이 컴퓨터(로컬 음성 인식)" : "이 컴퓨터(로컬 언어 모델)" },
          { value: "custom", label: "사용자 지정 서비스" },
        ]}
      />
      {cfg.kind === "local" ? (
        <div>
          {list.map((m) => (
            <ModelRow
              key={m.id}
              model={m}
              selected={cfg.model_id === m.id}
              onUse={() => set({ kind: "local", model_id: m.id })}
            />
          ))}
          {list.length === 0 && <p className="fhint">내려받을 수 있는 모델 목록을 불러오지 못했습니다.</p>}
        </div>
      ) : (
        <>
          <EndpointForm
            role={role}
            endpoint={cfg.endpoint}
            effort={cfg.reasoning_effort}
            interval={cfg.partial_interval_ms}
            onCommit={(endpoint) => set(custom(endpoint, cfg.reasoning_effort, cfg.partial_interval_ms))}
            onEffort={(effort) => set(custom(cfg.endpoint, effort, cfg.partial_interval_ms))}
            onInterval={(ms) => set(custom(cfg.endpoint, cfg.reasoning_effort, ms))}
          />
          <span className="fhint">
            OpenAI 호환 주소를 씁니다(Ollama, LM Studio, vLLM도 가능). API 키는 Windows 자격 증명 관리자, macOS 키체인에만
            저장하고 설정 파일, 기록, 내보낸 파일에는 넣지 않습니다.
          </span>
        </>
      )}
    </div>
  );
}

export function SettingsTab({ active }: { active: boolean }) {
  const app = useApp();
  const { settings, mics, systemAudio, saveState, settingsTarget } = app;
  const body = useRef<HTMLDivElement>(null);
  const [current, setCurrent] = useState(SECTIONS[0].id);

  const go = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ block: "start" });
    setCurrent(id);
  };

  useEffect(() => {
    if (active && settingsTarget) go(settingsTarget);
  }, [active, settingsTarget]);

  const onScroll = () => {
    const el = body.current;
    if (!el) return;
    const top = el.getBoundingClientRect().top + 40;
    let cur = SECTIONS[0].id;
    for (const s of SECTIONS) {
      const e = document.getElementById(s.id);
      if (e && e.getBoundingClientRect().top <= top) cur = s.id;
    }
    if (el.scrollTop + el.clientHeight >= el.scrollHeight - 2) cur = SECTIONS[SECTIONS.length - 1].id;
    setCurrent(cur);
  };

  if (!settings) return <div className="panel"><p className="empty-load">설정을 불러오는 중…</p></div>;
  const langs = TARGET_LANGUAGES.includes(settings.target_language)
    ? TARGET_LANGUAGES
    : [settings.target_language, ...TARGET_LANGUAGES];

  return (
    <div className="panel">
      <div className="set">
        <nav className="snav" aria-label="설정 항목">
          {SECTIONS.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              aria-current={current === s.id ? "true" : undefined}
              onClick={(e) => {
                e.preventDefault();
                go(s.id);
              }}
            >
              {s.label}
            </a>
          ))}
        </nav>
        <div className="sbody" ref={body} onScroll={onScroll}>
          <section className="grp" id="set-lang" aria-labelledby="h-lang">
            <h2 id="h-lang">언어</h2>
            <div className="card">
              <div className="fld">
                <span className="flab" aria-hidden="true">번역할 언어</span>
                <Select
                  label="번역할 언어"
                  width={260}
                  value={settings.target_language}
                  onChange={(v) => app.updateSettings({ target_language: v })}
                  options={langs.map((c) => ({ value: c, label: languageName(c) }))}
                />
              </div>
              <span className="fhint">회의 탭의 번역 언어와 같은 설정입니다. 어느 쪽에서 바꿔도 같습니다.</span>
            </div>
          </section>

          <section className="grp" id="set-theme" aria-labelledby="h-theme">
            <h2 id="h-theme">화면</h2>
            <div className="card">
              <div className="fld">
                <span className="flab">
                  화면 색
                </span>
                <Radios
                  label="화면 색"
                  value={settings.theme}
                  onChange={(theme) => app.updateSettings({ theme })}
                  options={THEMES}
                />
              </div>
              <span className="fhint">시스템 설정 따름을 고르면 컴퓨터의 라이트·다크 설정이 바뀔 때 바로 따라갑니다.</span>
            </div>
          </section>

          <section className="grp" id="set-asr" aria-labelledby="h-asr">
            <h2 id="h-asr">음성 인식</h2>
            <ProviderCard role="asr" />
            <div className="card">
              <Switch checked={settings.diarize} onChange={(diarize) => app.updateSettings({ diarize })}>
                목소리로 화자 나누기
              </Switch>
              <span className="fhint">마이크와 시스템 소리 모두 목소리로 화자를 나눕니다(화자 1, 2, …).</span>
            </div>
          </section>

          <section className="grp" id="set-translate" aria-labelledby="h-tr">
            <h2 id="h-tr">번역</h2>
            <ProviderCard role="translate" />
          </section>

          <section className="grp" id="set-input" aria-labelledby="h-in">
            <h2 id="h-in">소리 입력</h2>
            <div className="card">
              <div className="fld">
                <span className="flab" aria-hidden="true">마이크</span>
                <Select
                  label="마이크"
                  width={340}
                  value={settings.mic_device ?? ""}
                  onChange={(v) => app.updateSettings({ mic_device: v || null })}
                  options={[
                    { value: "", label: "기본 장치" },
                    ...(settings.mic_device && !mics.includes(settings.mic_device)
                      ? [{ value: settings.mic_device, label: `${settings.mic_device} (연결 안 됨)` }]
                      : []),
                    ...mics.map((m) => ({ value: m, label: m })),
                  ]}
                />
              </div>
              <div className="fld">
                <span className="flab" aria-hidden="true">회의 언어</span>
                <Select
                  label="회의 언어"
                  width={340}
                  value={settings.meeting_language ?? ""}
                  onChange={(v) => app.updateSettings({ meeting_language: v || null })}
                  options={[
                    { value: "", label: "자동 감지" },
                    { value: "ko", label: "한국어" },
                    { value: "ja", label: "日本語" },
                    { value: "zh", label: "中文" },
                    { value: "en", label: "English" },
                  ]}
                />
              </div>
              <span className="fhint">
                한 언어로만 말하는 회의는 언어를 고르면 음성 인식이 더 정확합니다. 다음 회의부터 적용됩니다.
              </span>
              {systemAudio && !systemAudio.supported ? (
                <div className="notice" role="note">
                  {systemAudio.reason}
                </div>
              ) : (
                <span className="fhint">
                  시스템 소리를 받을 수 있습니다. 처음 시작할 때 운영체제가 권한을 물을 수 있습니다.
                </span>
              )}
            </div>
          </section>

          <section className="grp" id="set-save" aria-labelledby="h-save">
            <h2 id="h-save">저장</h2>
            <div className="card">
              <div className="fld">
                <span className="flab">회의 기록 위치</span>
                <span className="inp mono">{settings.sessions_dir ?? "문서/morago/sessions"}</span>
              </div>
              <span className="fhint">회의를 중지하면 이 폴더에 원문과 번역이 저장됩니다.</span>
            </div>
          </section>

          <section className="grp" id="set-about" aria-labelledby="h-about">
            <h2 id="h-about">정보</h2>
            <div className="card">
              <span className="fhint">모델과 라이선스</span>
              <ul className="notice-list">
                <li translate="no">Silero VAD: MIT License, Copyright (c) 2020-present Silero Team</li>
                <li translate="no">CAM++ speaker embedding (3D-Speaker): Apache License 2.0</li>
                <li translate="no">
                  SenseVoice Small (FunAudioLLM, Alibaba Group): FunASR Model License 1.1
                  (https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)
                </li>
                <li translate="no">Qwen3-ASR 0.6B (Qwen, Alibaba Cloud): Apache License 2.0</li>
                <li translate="no">Whisper (OpenAI): MIT License</li>
                <li>
                  로컬 번역은 Google DeepMind의 Gemma 4를 씁니다. Apache License 2.0
                  (https://ai.google.dev/gemma/docs/gemma_4_license). 모델 파일은 Google의 Hugging Face 저장소에서 바꾸지 않고 내려받습니다.
                </li>
              </ul>
            </div>
          </section>
        </div>
      </div>
      <div className="bar" role="status">
        <span>
          {saveState === "saving"
            ? "저장하는 중…"
            : saveState === "error"
              ? "설정을 저장하지 못했습니다."
              : "바꾼 설정은 바로 저장됩니다"}
        </span>
      </div>
    </div>
  );
}
