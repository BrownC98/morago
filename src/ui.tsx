import { useEffect, useId, useRef, type ReactNode } from "react";

const PATHS = {
  logo: "M4 7h10M4 12h7M4 17h12M17 9l3 3-3 3",
  chevron: "M6 9l6 6 6-6",
  search: "M20 20l-4-4M18 11a7 7 0 1 1-14 0 7 7 0 0 1 14 0",
  download: "M12 4v11M7 10l5 5 5-5M5 20h14",
  stop: "M7 6h10a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1z",
  play: "M8 5v14l11-7z",
  down: "M12 5v14M6 13l6 6 6-6",
  close: "M6 6l12 12M18 6L6 18",
} as const;

export function Icon({ name }: { name: keyof typeof PATHS }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d={PATHS[name]} />
    </svg>
  );
}

/** Switch with a real checkbox inside, so keyboard and screen readers work. */
export function Switch({
  checked,
  onChange,
  disabled,
  children,
  describedBy,
  title,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  children: ReactNode;
  describedBy?: string;
  title?: string;
}) {
  return (
    <label className="tog" data-disabled={disabled || undefined} title={title}>
      <input
        type="checkbox"
        role="switch"
        className="sr"
        checked={checked}
        disabled={disabled}
        aria-describedby={describedBy}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="sw" aria-hidden="true" />
      {children}
    </label>
  );
}

/** Native select in the product style. */
export function Select({
  label,
  value,
  onChange,
  options,
  width,
  inline,
  disabled,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  width?: number;
  /** Show the label inside the box (toolbar). Otherwise the label is for screen readers. */
  inline?: boolean;
  disabled?: boolean;
}) {
  return (
    <label className="sel" style={width ? { width } : undefined}>
      {inline ? <span className="lab">{label}</span> : <span className="sr">{label}</span>}
      <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      <Icon name="chevron" />
    </label>
  );
}

/** Radio group of pills. Native radios give arrow-key movement. */
export function Radios<T extends string>({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string }[];
}) {
  const name = useId();
  return (
    <div className="radio" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <label className="ro" key={o.value}>
          <input
            type="radio"
            className="sr"
            name={name}
            checked={value === o.value}
            onChange={() => onChange(o.value)}
          />
          {o.label}
        </label>
      ))}
    </div>
  );
}

/** Confirm dialog on the native <dialog>: focus is trapped, Esc cancels. */
export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  body: string;
  confirmLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog ref={ref} className="dlg" aria-labelledby="dlg-title" onCancel={(e) => { e.preventDefault(); onCancel(); }}>
      <h2 id="dlg-title">{title}</h2>
      <p>{body}</p>
      <div className="row2 end">
        <button type="button" className="ab" autoFocus onClick={onCancel}>
          취소
        </button>
        <button type="button" className="ab stop" onClick={onConfirm}>
          {confirmLabel}
        </button>
      </div>
    </dialog>
  );
}
