import { useEffect, useState } from "react";
import type { NotifyFn } from "./types";

/** Joins truthy class names: cx("btn", isActive && "active") → "btn active". */
export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

/** Copies text and reports the outcome through the app toast system. */
export async function copyToClipboard(text: string, notify: NotifyFn, label = "Conteúdo"): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
    notify("success", "Copiado!", `${label} copiado para a área de transferência.`);
  } catch {
    notify("warning", "Não foi possível copiar automaticamente", "Selecione o texto manualmente.");
  }
}

/** Splits "a, b ,c" into ["a", "b", "c"]. */
export function splitList(value: string): string[] {
  return value
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

/** Accent/case-insensitive normalisation for client-side search. */
export function normalizeSearch(value: string): string {
  return value
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase();
}

/** Safe localStorage access (private windows and locked-down webviews can throw). */
export const storage = {
  get(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, value: string): void {
    try {
      localStorage.setItem(key, value);
    } catch {
      // ignore — preference simply won't persist
    }
  },
};

/** Tracks a CSS media query (e.g. narrow windows) reactively. */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window !== "undefined" && "matchMedia" in window ? window.matchMedia(query).matches : false,
  );

  useEffect(() => {
    if (typeof window === "undefined" || !("matchMedia" in window)) return;
    const mql = window.matchMedia(query);
    const onChange = () => setMatches(mql.matches);
    onChange();
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}
