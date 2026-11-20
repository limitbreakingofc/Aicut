/** Terminal-style color schemes (CSS tokens live in styles/tokens.css). */
export type ThemeId = "dark" | "phosphor" | "amber" | "light";

export interface ThemeMeta {
  id: ThemeId;
  label: string;
  description: string;
  /** [background, accent, text] — used for the preview swatches. */
  swatch: [string, string, string];
}

export const THEMES: ThemeMeta[] = [
  { id: "dark", label: "Noite", description: "Escuro neutro com destaque ciano (padrão)", swatch: ["#0a0c10", "#56d4dd", "#d9dfea"] },
  { id: "phosphor", label: "Fósforo", description: "Verde de terminal CRT clássico", swatch: ["#040805", "#39ff88", "#c6f5d2"] },
  { id: "amber", label: "Âmbar", description: "Monitor âmbar retrô, quente e confortável", swatch: ["#0b0905", "#ffb000", "#ffe3b3"] },
  { id: "light", label: "Claro", description: "Fundo claro para ambientes iluminados", swatch: ["#f6f7f9", "#0a7b86", "#1b2230"] },
];

export function isThemeId(value: unknown): value is ThemeId {
  return THEMES.some((t) => t.id === value);
}

export function themeLabel(id: ThemeId): string {
  return THEMES.find((t) => t.id === id)?.label ?? id;
}
