import { useEffect } from "react";
import type { ThemeSetting } from "./api";

const KEY = "morago-theme";
const query = () => window.matchMedia("(prefers-color-scheme: dark)");

function resolve(t: ThemeSetting): "light" | "dark" {
  return t === "system" ? (query().matches ? "dark" : "light") : t;
}

function apply(t: ThemeSetting) {
  document.documentElement.dataset.theme = resolve(t);
}

function cached(): ThemeSetting {
  const v = localStorage.getItem(KEY);
  return v === "light" || v === "dark" ? v : "system";
}

/** Sets the theme before the first render, from the last saved choice, so there is no flash. */
export function applyCachedTheme() {
  apply(cached());
}

/** Applies the theme setting. System follows the OS color scheme live. */
export function useTheme(saved: ThemeSetting | undefined) {
  const setting = saved ?? cached();
  useEffect(() => {
    if (saved) localStorage.setItem(KEY, setting);
    apply(setting);
    if (setting !== "system") return;
    const q = query();
    const on = () => apply("system");
    q.addEventListener("change", on);
    return () => q.removeEventListener("change", on);
  }, [setting, saved]);
}
