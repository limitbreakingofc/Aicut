import { Keyboard, PanelLeftClose, PanelLeftOpen, Palette } from "lucide-react";
import { Kbd, useContextMenu } from "../components";
import type { RouteId } from "../lib/types";
import { BrandMark } from "./BrandMark";
import { APP_VERSION, routePath } from "./navigation";
import { THEMES, type ThemeId } from "./themes";

interface TopbarProps {
  route: RouteId;
  sidebarCollapsed: boolean;
  /** When the window is too narrow the sidebar is always an icon rail. */
  sidebarLocked: boolean;
  theme: ThemeId;
  onToggleSidebar: () => void;
  onOpenPalette: () => void;
  onOpenShortcuts: () => void;
  onThemeChange: (theme: ThemeId) => void;
}

export function Topbar({
  route,
  sidebarCollapsed,
  sidebarLocked,
  theme,
  onToggleSidebar,
  onOpenPalette,
  onOpenShortcuts,
  onThemeChange,
}: TopbarProps) {
  const menu = useContextMenu();
  const [group, page] = routePath(route).split("/");

  return (
    <header className="topbar" data-tauri-drag-region>
      <div className="topbar-left">
        {!sidebarLocked && (
          <button
            type="button"
            className="btn btn-ghost btn-icon btn-sm"
            onClick={onToggleSidebar}
            aria-label={sidebarCollapsed ? "Expandir menu lateral" : "Recolher menu lateral"}
            title={`${sidebarCollapsed ? "Expandir" : "Recolher"} menu (Ctrl+B)`}
          >
            {sidebarCollapsed ? <PanelLeftOpen aria-hidden /> : <PanelLeftClose aria-hidden />}
          </button>
        )}
        <div className="brand">
          <BrandMark />
          <span>aicut</span>
          <span className="brand-version">v{APP_VERSION}</span>
        </div>
        <nav className="topbar-path" aria-label="Você está em">
          <span className="path-parent">~/{group}/</span>
          <span className="path-current">{page}</span>
        </nav>
      </div>

      <button
        type="button"
        className="command-center"
        onClick={onOpenPalette}
        aria-label="Abrir paleta de comandos (Ctrl+K)"
        title="Paleta de comandos (Ctrl+K)"
      >
        <span className="palette-prompt" aria-hidden>
          ›
        </span>
        <span className="command-label">buscar páginas e comandos…</span>
        <Kbd>Ctrl K</Kbd>
      </button>

      <div className="topbar-right">
        <button
          type="button"
          className="btn btn-ghost btn-icon btn-sm"
          aria-label="Esquema de cores"
          aria-haspopup="menu"
          title="Esquema de cores"
          onClick={(e) =>
            menu.openAt(e.currentTarget, [
              { type: "label", label: "Esquema de cores" },
              ...THEMES.map((t) => ({
                label: t.label,
                checked: t.id === theme,
                onSelect: () => onThemeChange(t.id),
              })),
            ])
          }
        >
          <Palette aria-hidden />
        </button>
        <button
          type="button"
          className="btn btn-ghost btn-icon btn-sm"
          aria-label="Atalhos de teclado"
          title="Atalhos de teclado (?)"
          onClick={onOpenShortcuts}
        >
          <Keyboard aria-hidden />
        </button>
      </div>
    </header>
  );
}
