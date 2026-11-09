import type { ModelInfo, ProviderConfig, SourceKind, Utterance } from "./api";

const pad = (n: number) => String(n).padStart(2, "0");

/** hh:mm:ss from milliseconds. */
export function clock(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`;
}

export function range(u: Pick<Utterance, "start_ms" | "end_ms">): string {
  return `${clock(u.start_ms)}–${clock(u.end_ms)}`;
}

const dateFmt = new Intl.DateTimeFormat("ko-KR", {
  year: "numeric",
  month: "long",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});
const timeFmt = new Intl.DateTimeFormat("ko-KR", { hour: "2-digit", minute: "2-digit", hour12: false });

export function dateTime(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : dateFmt.format(d);
}

export function timeOnly(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : timeFmt.format(d);
}

export function minutes(ms: number): string {
  const m = Math.round(ms / 60000);
  return m < 1 ? "1분 미만" : `${new Intl.NumberFormat("ko-KR").format(m)}분`;
}

export function bytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  return `${Math.round(n / 1024 ** 2)} MB`;
}

export const number = (n: number) => new Intl.NumberFormat("ko-KR").format(n);

/** Language codes offered as the translation target. */
export const TARGET_LANGUAGES = ["ko", "en", "ja", "zh", "es", "fr", "de", "ru", "vi", "th", "id"];

const names = new Intl.DisplayNames(["ko"], { type: "language" });
export function languageName(code: string): string {
  try {
    return names.of(code) ?? code;
  } catch {
    return code;
  }
}

export function sourceLabel(s: SourceKind): string {
  return s === "mic" ? "마이크" : "시스템 소리";
}

/** "S2" -> 2. Ids without a number give null. */
export function speakerNumber(id: string): number | null {
  const m = /(\d+)\s*$/.exec(id);
  return m ? Number(m[1]) : null;
}

export function speakerBase(id: string): string {
  const n = speakerNumber(id);
  return n === null ? id : `화자 ${n}`;
}

/** Display name: the user's name with the base name ("민지 (화자 1)"), or the base name. */
export function speakerLabel(id: string, names: Record<string, string>): string {
  const name = names[id]?.trim();
  return name ? `${name} (${speakerBase(id)})` : speakerBase(id);
}

/** Color slot 1..6 for a speaker id. */
export function speakerSlot(id: string): number {
  const n = speakerNumber(id);
  if (n === null) {
    let h = 0;
    for (const c of id) h = (h * 31 + c.charCodeAt(0)) >>> 0;
    return (h % 6) + 1;
  }
  return ((n - 1 + 6) % 6) + 1;
}

/** Label of the engine, for the status bar. Local models show their catalog name. */
export function providerLabel(cfg: ProviderConfig, models: ModelInfo[]): string {
  if (cfg.kind === "local") {
    const m = models.find((x) => x.id === cfg.model_id);
    return `로컬 ${m?.name ?? cfg.model_id}`;
  }
  const model = typeof cfg.extra_body?.model === "string" ? cfg.extra_body.model : cfg.endpoint.model;
  return `사용자 지정 ${model || "(모델 없음)"}`;
}

export function sessionFileName(title: string, ext: string): string {
  const safe = title.replace(/[\\/:*?"<>|]+/g, "-").trim() || "회의";
  return `${safe}.${ext}`;
}
