import { memo } from "react";
import type { PartialText, Utterance } from "./api";
import { clock, range, sourceLabel, speakerLabel, speakerSlot } from "./format";

/** One utterance: speaker, source, time, original on top, translation below (decision D1 = A). */
export const UtteranceRow = memo(function UtteranceRow({
  u,
  names,
  latest,
}: {
  u: Utterance;
  names: Record<string, string>;
  latest?: boolean;
}) {
  const t = u.translation;
  return (
    <article className="utt" data-latest={latest || undefined}>
      <div className="who-c">
        <span className="spk" style={{ ["--c" as string]: `var(--spk-${speakerSlot(u.speaker)})` }}>
          {speakerLabel(u.speaker, names)}
        </span>
        <span className="src">{sourceLabel(u.source)}</span>
        <span className="tm">{range(u)}</span>
      </div>
      <div className="txt">
        <p className="orig" lang={u.language ?? undefined}>
          {u.language && <span className="lg">{u.language.toUpperCase()}</span>}
          {u.original}
        </p>
        {t.state === "done" && <p className="tr">{t.text}</p>}
        {t.state === "pending" && <p className="tr wait">번역하는 중…</p>}
        {t.state === "same_language" && <p className="tr same">같은 언어라 번역하지 않음</p>}
        {t.state === "failed" && <p className="tr fail">{t.message}</p>}
      </div>
    </article>
  );
});

/** Live text and provisional translation of speech that has not ended. Grey; replaced by the utterance when it ends. */
export function PartialRow({ p }: { p: PartialText }) {
  return (
    <article className="utt partial">
      <div className="who-c">
        <span className="spk">듣는 중…</span>
        <span className="src">{sourceLabel(p.source)}</span>
        <span className="tm">{clock(p.start_ms)}</span>
      </div>
      <div className="txt">
        <p className="orig">{p.text}</p>
        {p.translation && <p className="tr">{p.translation}</p>}
      </div>
    </article>
  );
}
