import { useEffect, useRef, useState } from "react";
import {
  api,
  appVersion,
  errText,
  type Endpoint,
  type KeyRole,
  type ModelInfo,
  type ProviderConfig,
  type ThemeSetting,
} from "./api";
import { bytes, languageName, TARGET_LANGUAGES } from "./format";
import { updateText, useApp } from "./store";
import { ConfirmDialog, Radios, Select, Switch } from "./ui";

const SECTIONS = [
  { id: "set-lang", label: "언어" },
  { id: "set-theme", label: "화면" },
  { id: "set-asr", label: "음성 인식" },
  { id: "set-translate", label: "번역" },
  { id: "set-gpu", label: "GPU 가속" },
  { id: "set-input", label: "소리 입력" },
  { id: "set-save", label: "저장" },
  { id: "set-about", label: "정보" },
];

const THEMES: { value: ThemeSetting; label: string }[] = [
  { value: "light", label: "라이트" },
  { value: "dark", label: "다크" },
  { value: "system", label: "시스템 설정 따름" },
];

function UpdateCard() {
  const { update, running, checkUpdate, installUpdate } = useApp();
  const [version, setVersion] = useState("");
  useEffect(() => {
    appVersion().then(setVersion, () => setVersion(""));
  }, []);
  const busy = update.kind === "checking" || update.kind === "installing";
  const text = updateText(update, running);
  return (
    <div className="card">
      <span className="row2">
        <span className="fhint">현재 버전</span>
        <span className="ver" translate="no">{version}</span>
        <button type="button" className="ab sm" disabled={busy} onClick={checkUpdate}>
          업데이트 확인
        </button>
      </span>
      <span className="row2" role="status">
        {text && <span className={update.kind === "error" ? "fhint errtext" : update.kind === "available" ? "fhint ink" : "fhint"}>{text}</span>}
        {update.kind === "available" && (
          <button type="button" className="ab sm primary" onClick={installUpdate}>
            설치하고 다시 시작
          </button>
        )}
      </span>
    </div>
  );
}

const MEETING_LANGUAGES = [
  { value: "", label: "자동 감지" },
  { value: "ko", label: "한국어" },
  { value: "ja", label: "日本語" },
  { value: "zh", label: "中文" },
  { value: "en", label: "English" },
];

/** Why a single-language model cannot run with the meeting language, or null. */
function languageBlock(model: ModelInfo, meetingLanguage: string | null | undefined): string | null {
  if (!model.only_language || model.only_language === meetingLanguage) return null;
  const name = MEETING_LANGUAGES.find((l) => l.value === model.only_language)?.label ?? model.only_language;
  return `회의 언어를 ${name}로 고르면 쓸 수 있습니다.`;
}

function ModelRow({
  model,
  selected,
  onUse,
  blocked,
}: {
  model: ModelInfo;
  selected?: boolean;
  onUse?: () => void;
  /** Reason why the model cannot run now (meeting language). */
  blocked?: string | null;
}) {
  const { downloads, downloadErrors, startDownload, cancelDownload, deleteModel } = useApp();
  const [confirmDelete, setConfirmDelete] = useState(false);
  const dl = downloads[model.id];
  const err = downloadErrors[model.id];
  const pct = dl && dl.total ? Math.min(100, Math.round((dl.done / dl.total) * 100)) : null;
  const del = model.on_disk && (
    <button
      type="button"
      className="ab ghost sm danger"
      onClick={() => setConfirmDelete(true)}
      aria-label={`${model.name} 파일 삭제`}
    >
      삭제
    </button>
  );
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
            {model.on_disk && !model.installed && <> · 이전 형식의 파일이 있습니다. 이 파일은 쓰지 않습니다.</>}
            {blocked && <span className={selected ? "errtext" : undefined}> · {blocked}</span>}
            {err && <span className="errtext"> · {err}</span>}
          </span>
          {model.installed ? (
            <span className="row2">
              {selected ? (
                <span className="ok">사용 중</span>
              ) : onUse ? (
                <button
                  type="button"
                  className="ab sm"
                  onClick={onUse}
                  disabled={!!blocked}
                  aria-label={`${model.name} 사용하기`}
                >
                  사용하기
                </button>
              ) : (
                <span className="ok">내려받음</span>
              )}
              {del}
            </span>
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
              {del}
            </span>
          )}
        </>
      )}
      <ConfirmDialog
        open={confirmDelete}
        title={`${model.name} 모델 파일을 지울까요?`}
        body={
          model.installed
            ? "이 컴퓨터에서 모델 파일이 지워집니다. 다시 쓰려면 다시 내려받아야 합니다."
            : "앱이 쓰지 않는 이전 형식의 파일입니다. 이 컴퓨터에서 지워집니다."
        }
        confirmLabel="지우기"
        onConfirm={() => {
          setConfirmDelete(false);
          deleteModel(model.id);
        }}
        onCancel={() => setConfirmDelete(false)}
      />
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

type ExtraBody = Record<string, unknown>;

/** `extra_body` of a new custom translation server. */
const DEFAULT_EXTRA_BODY: ExtraBody = { model: "gpt-5-mini", reasoning_effort: "low" };

const isEmpty = (body?: ExtraBody) => !body || Object.keys(body).length === 0;

/** Text of a saved `extra_body`: blank for an empty object. */
const extraBodyText = (body?: ExtraBody) => (isEmpty(body) ? "" : JSON.stringify(body, null, 2));

/** `extra_body` of a custom translation server as JSON text. The core checks it, live and on save. */
function ExtraBodyField({
  initial,
  saved,
  onText,
  onCommit,
  commitRef,
}: {
  initial: string;
  saved?: ExtraBody;
  onText: (text: string) => void;
  onCommit: (body: ExtraBody) => void;
  /** Set to `commit`: true when the text is valid and its save is queued. */
  commitRef: React.MutableRefObject<(() => Promise<boolean>) | null>;
}) {
  const [text, setText] = useState(initial);
  const [error, setError] = useState<string | null>(null);
  // Only the newest check sets `error`.
  const checkId = useRef(0);
  const liveCheck = useRef<number | undefined>(undefined);

  useEffect(() => {
    const n = ++checkId.current;
    liveCheck.current = window.setTimeout(() => {
      api.checkExtraBody(text).then(
        () => n === checkId.current && setError(null),
        (e) => n === checkId.current && setError(errText(e)),
      );
    }, 300);
    return () => window.clearTimeout(liveCheck.current);
  }, [text]);

  const commit = async () => {
    // A live check that starts later must not cancel this commit.
    window.clearTimeout(liveCheck.current);
    const n = ++checkId.current;
    try {
      await api.checkExtraBody(text);
    } catch (e) {
      if (n === checkId.current) setError(errText(e));
      return false;
    }
    if (n !== checkId.current) return false;
    setError(null);
    const body = (text.trim() ? JSON.parse(text) : {}) as ExtraBody;
    if (JSON.stringify(body) !== JSON.stringify(saved ?? {})) onCommit(body);
    return true;
  };
  useEffect(() => {
    commitRef.current = commit;
  });

  return (
    <>
      <label htmlFor="translate-extra" className="top-lab">
        추가 요청 값
      </label>
      <div className="stack">
        <textarea
          id="translate-extra"
          className="inp mono"
          value={text}
          rows={5}
          spellCheck={false}
          autoComplete="off"
          translate="no"
          aria-invalid={error !== null}
          aria-describedby="translate-extra-msg"
          onChange={(e) => {
            setText(e.target.value);
            onText(e.target.value);
          }}
          onBlur={() => void commit()}
        />
        <p id="translate-extra-msg" className={error ? "fhint errtext" : "fhint"} role="status">
          {error ?? 'JSON 객체로 적습니다. "model"이 번역 모델입니다. 번역 요청마다 합쳐 보냅니다.'}
        </p>
      </div>
    </>
  );
}

/** Largest whole number that the core stores (Rust `u32`). */
const MAX_U32 = 4_294_967_295;

/**
 * Label, a whole-number input and a hint, as cells of a `.fld` grid. Saves on blur or Enter;
 * text that is not an integer from `min` shows an error and is not saved.
 */
function WholeNumberField({
  id,
  label,
  unit,
  value,
  min,
  disabled,
  before,
  hint,
  onCommit,
}: {
  id: string;
  label: string;
  unit?: string;
  value: number;
  min: number;
  disabled?: boolean;
  before?: React.ReactNode;
  hint: string;
  onCommit: (n: number) => void;
}) {
  const [text, setText] = useState(String(value));
  const [error, setError] = useState(false);

  useEffect(() => {
    setText(String(value));
    setError(false);
  }, [value, disabled]);

  const commit = () => {
    const t = text.trim();
    const n = Number(t);
    const ok = /^\d+$/.test(t) && n >= min && n <= MAX_U32;
    setError(!ok);
    if (ok && n !== value) onCommit(n);
  };

  return (
    <>
      <label htmlFor={id}>{label}</label>
      <div className="row2">
        {before}
        <input
          id={id}
          className="inp"
          style={{ width: 120 }}
          aria-invalid={error}
          aria-describedby={`${id}-msg`}
          inputMode="numeric"
          autoComplete="off"
          disabled={disabled}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLElement).blur()}
        />
        {unit && <span>{unit}</span>}
      </div>
      <span />
      <p id={`${id}-msg`} className={error ? "fhint errtext" : "fhint"} role="status">
        {error ? `${min} 이상의 정수로 적습니다. 저장하지 않았습니다.` : hint}
      </p>
    </>
  );
}

/** `partial_interval_ms` of a custom translation server: off (`null`) or whole ms from 0.
 * `live`: the live partials setting; when it is off, the field has no effect. */
function IntervalField({
  interval,
  live,
  onInterval,
}: {
  interval?: number | null;
  live: boolean;
  onInterval: (ms: number | null) => void;
}) {
  const lastOn = useRef(1500);
  if (interval != null) lastOn.current = interval;
  const on = interval !== null;
  return (
    <WholeNumberField
      id="translate-interval"
      label="중간 번역 간격"
      unit="ms"
      value={lastOn.current}
      min={0}
      disabled={!live || !on}
      before={
        <Switch checked={on} disabled={!live} onChange={(v) => onInterval(v ? lastOn.current : null)}>
          {on ? "켬" : "끔"}
        </Switch>
      }
      hint={
        live
          ? "말하는 동안 보이는 회색 번역을 이 간격마다 한 번 요청합니다(기본값 1500). 0이면 글이 바뀔 때마다 요청합니다. 값이 작을수록 요청이 많아져 요금이 늘어납니다."
          : "음성 인식의 실시간 전사가 꺼져 있어 중간 번역을 요청하지 않습니다."
      }
      onCommit={onInterval}
    />
  );
}

/** Address, key and test of a custom service. ASR has a 모델 row; `children` go after it. */
function EndpointForm({
  role,
  endpoint,
  onCommit,
  beforeTest,
  children,
}: {
  role: KeyRole;
  endpoint: Endpoint;
  onCommit: (e: Endpoint) => void;
  /** Commits the fields in `children`: false stops the test. */
  beforeTest?: () => Promise<boolean>;
  children?: React.ReactNode;
}) {
  const { settled } = useApp();
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
      // The core tests the saved settings, so the test waits for the fields on screen to be saved.
      if (!(await (beforeTest?.() ?? true)) || !(await settled())) {
        setTest(null);
        return;
      }
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
      {role === "asr" && (
        <>
          <label htmlFor={`${id}-model`}>모델</label>
          <input
            id={`${id}-model`}
            className="inp mono"
            value={model}
            placeholder="whisper-1"
            spellCheck={false}
            autoComplete="off"
            translate="no"
            onChange={(e) => setModel(e.target.value)}
            onBlur={commit}
            onKeyDown={enter}
          />
        </>
      )}
      {children}
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
  const lastExtra = useRef<ExtraBody | undefined>(undefined);
  // The typed text, also when it is not valid and so not saved.
  const extraText = useRef<string | null>(null);
  const commitExtra = useRef<(() => Promise<boolean>) | null>(null);
  const lastInterval = useRef<number | null | undefined>(undefined);
  const lastLocal = useRef<string | null>(null);
  if (cfg?.kind === "custom") {
    lastEndpoint.current = cfg.endpoint;
    lastExtra.current = cfg.extra_body;
    lastInterval.current = cfg.partial_interval_ms;
  }
  if (cfg?.kind === "local") lastLocal.current = cfg.model_id;
  if (!cfg) return null;
  const set = (c: ProviderConfig) => updateSettings(role === "asr" ? { asr: c } : { translate: c });
  // An empty `extra_body` is the absent key, as in the settings file.
  const custom = (endpoint: Endpoint, extra: ExtraBody | undefined, interval: number | null | undefined): ProviderConfig =>
    role === "translate"
      ? {
          kind: "custom",
          endpoint,
          ...(isEmpty(extra) ? {} : { extra_body: extra }),
          partial_interval_ms: interval,
        }
      : { kind: "custom", endpoint };

  return (
    <div className="card">
      <Radios
        label={role === "asr" ? "음성 인식 방법" : "번역 방법"}
        value={cfg.kind}
        onChange={(k) => {
          if (k === "custom") {
            const extra = isEmpty(lastExtra.current) ? DEFAULT_EXTRA_BODY : lastExtra.current;
            // A draft that is not the saved body would hide the default.
            if (extra === DEFAULT_EXTRA_BODY) extraText.current = null;
            set(custom(lastEndpoint.current, extra, lastInterval.current));
          }
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
              blocked={languageBlock(m, settings?.meeting_language)}
            />
          ))}
          {list.length === 0 && <p className="fhint">내려받을 수 있는 모델 목록을 불러오지 못했습니다.</p>}
        </div>
      ) : (
        <>
          <EndpointForm
            role={role}
            endpoint={cfg.endpoint}
            onCommit={(endpoint) => set(custom(endpoint, cfg.extra_body, cfg.partial_interval_ms))}
            beforeTest={role === "translate" ? () => commitExtra.current?.() ?? Promise.resolve(true) : undefined}
          >
            {role === "translate" && (
              <>
                <ExtraBodyField
                  initial={extraText.current ?? extraBodyText(cfg.extra_body)}
                  saved={cfg.extra_body}
                  onText={(t) => (extraText.current = t)}
                  onCommit={(body) => set(custom(cfg.endpoint, body, cfg.partial_interval_ms))}
                  commitRef={commitExtra}
                />
                <IntervalField
                  interval={cfg.partial_interval_ms}
                  live={settings?.live_partials ?? true}
                  onInterval={(ms) => set(custom(cfg.endpoint, cfg.extra_body, ms))}
                />
              </>
            )}
          </EndpointForm>
          <span className="fhint">
            OpenAI 호환 주소를 씁니다(Ollama, LM Studio, vLLM도 가능). API 키는 Windows 자격 증명 관리자, macOS 키체인에만
            저장하고 설정 파일, 기록, 내보낸 파일에는 넣지 않습니다.
          </span>
        </>
      )}
    </div>
  );
}

function DeviceRow({ name, device, detail }: { name: string; device: string; detail: string }) {
  return (
    <div className="mrow">
      <span className="n">{name}</span>
      <span className="s">{device}</span>
      <span className="fhint">{detail}</span>
      <span />
    </div>
  );
}

function GpuCard() {
  const { settings, updateSettings, gpu, notify } = useApp();
  if (!settings) return null;
  const chosen = settings.gpu;
  const names = gpu?.gpus ?? [];
  const pack = gpu?.pack;
  const options = [
    { value: "auto", label: "자동" },
    ...names.map((n) => ({ value: `device:${n}`, label: n })),
    ...(chosen.kind === "device" && !names.includes(chosen.name)
      ? [{ value: `device:${chosen.name}`, label: `${chosen.name} (찾을 수 없음)` }]
      : []),
    { value: "off", label: "끔" },
  ];
  const tr = gpu?.translate;
  const asr = gpu?.asr;
  return (
    <div className="card">
      <div className="fld">
        <span className="flab">GPU 사용</span>
        <Radios
          label="GPU 사용"
          value={chosen.kind === "device" ? `device:${chosen.name}` : chosen.kind}
          onChange={(v) =>
            updateSettings({
              gpu: v.startsWith("device:")
                ? { kind: "device", name: v.slice("device:".length) }
                : { kind: v === "off" ? "off" : "auto" },
            })
          }
          options={options}
        />
      </div>
      <span className="fhint">
        자동: 쓸 수 있는 GPU가 있으면 쓰고, 없으면 CPU에서 돌립니다. GPU 이름을 고르면 그 GPU만 씁니다. 끔: 늘 CPU에서
        돌립니다. 바꾼 값은 다음 회의를 시작할 때 적용합니다. {gpu?.hint}
      </span>
      <div>
        {pack &&
          (pack.installed && pack.needs_restart ? (
            <div className="mrow">
              <span className="n">GPU 구성 요소</span>
              <span className="s">{bytes(pack.size_bytes)}</span>
              <span className="fhint">내려받았습니다. 앱을 다시 시작하면 씁니다.</span>
              <button
                type="button"
                className="ab sm"
                onClick={() => api.restartApp().catch((e) => notify("error", errText(e)))}
              >
                다시 시작
              </button>
            </div>
          ) : (
            <ModelRow
              model={{
                id: pack.id,
                kind: "translation",
                name: "GPU 구성 요소",
                size_bytes: pack.size_bytes,
                note: pack.note,
                installed: pack.installed,
                on_disk: false,
              }}
            />
          ))}
        <DeviceRow
          name="음성 인식"
          {...(asr
            ? { device: asr.gpu ? "GPU" : "CPU", detail: asr.detail }
            : { device: "서비스", detail: "사용자 지정 서비스가 인식합니다." })}
        />
        <DeviceRow
          name="번역"
          {...(tr
            ? { device: tr.gpu ? "GPU" : "CPU", detail: tr.detail }
            : { device: "서비스", detail: "사용자 지정 서비스가 번역합니다." })}
        />
        <DeviceRow name="목소리 구간·화자" device="CPU" detail="작은 모델이라 CPU로 충분합니다" />
      </div>
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
              <Switch
                checked={settings.live_partials}
                onChange={(live_partials) => app.updateSettings({ live_partials })}
                describedBy="live-partials-hint"
              >
                실시간 전사
              </Switch>
              <span className="fhint" id="live-partials-hint">
                말하는 동안 인식 중인 글과 번역을 회색 줄로 먼저 보입니다. 끄면 말이 끝난 뒤에 글과 번역만 보이고 CPU·GPU를
                덜 씁니다. 다음 회의부터 적용합니다. 로컬 SenseVoice Small, Parakeet 일본어, GPU에서 도는
                Fun-ASR-MLT-Nano만 회색 줄을 보입니다.
              </span>
            </div>
          </section>

          <section className="grp" id="set-translate" aria-labelledby="h-tr">
            <h2 id="h-tr">번역</h2>
            <ProviderCard role="translate" />
            <div className="card">
              <div className="fld wide">
                <WholeNumberField
                  id="translate-context"
                  label="번역 문맥 줄 수"
                  unit="줄"
                  value={settings.translate_context_lines}
                  min={0}
                  hint="번역할 때 앞 발화를 이 줄 수만큼 문맥으로 함께 보냅니다(기본값 4). 0이면 문맥 없이 보냅니다. 줄이 많을수록 번역이 느려지고 사용자 지정 서비스의 요금이 늘어납니다. 로컬 모델에 다 들어가지 않으면 오래된 줄부터 뺍니다."
                  onCommit={(translate_context_lines) => app.updateSettings({ translate_context_lines })}
                />
              </div>
            </div>
          </section>

          <section className="grp" id="set-gpu" aria-labelledby="h-gpu">
            <h2 id="h-gpu">GPU 가속</h2>
            <GpuCard />
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
                  options={MEETING_LANGUAGES}
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
            <UpdateCard />
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
                <li translate="no">
                  Fun-ASR-MLT-Nano (FunAudioLLM, Alibaba Group): FunASR Model License 1.1
                  (https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE). GGUF conversion by handy-computer
                </li>
                <li translate="no">
                  Parakeet 일본어: NVIDIA parakeet-tdt_ctc-0.6b-ja, CC BY 4.0
                  (https://creativecommons.org/licenses/by/4.0/). Converted to ONNX int8 by k2-fsa (sherpa-onnx)
                </li>
                <li translate="no">ggml (The ggml authors): MIT License</li>
                <li translate="no">transcribe.cpp (handy-computer): MIT License</li>
                <li>
                  로컬 번역은 Google DeepMind의 Gemma 4를 씁니다. Apache License 2.0
                  (https://ai.google.dev/gemma/docs/gemma_4_license). 모델 파일은 Google의 Hugging Face 저장소에서 바꾸지 않고 내려받습니다.
                </li>
                {app.gpu?.pack && (
                  <li>
                    GPU 구성 요소는 NVIDIA CUDA 런타임과 cuBLAS입니다. NVIDIA 서버에서 바꾸지 않고 내려받고, NVIDIA CUDA Toolkit
                    EULA(https://docs.nvidia.com/cuda/eula/)를 따릅니다.
                  </li>
                )}
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
