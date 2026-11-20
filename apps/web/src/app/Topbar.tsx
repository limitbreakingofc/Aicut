import { PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { Kbd } from "../components";
import { BrandMark } from "./BrandMark";
import { APP_VERSION, ROUTE_MAP, type PrimaryRouteId } from "./navigation";

interface TopbarProps {
  route: PrimaryRouteId;
  sidebarCollapsed: boolean;
  sidebarLocked: boolean;
  onToggleSidebar: () => void;
  onOpenPalette: () => void;
}

/** Minimalist topbar: brand, active workspace name, and single quick-command trigger (no redundant menus). */
export function Topbar({
  route,
  sidebarCollapsed,
  sidebarLocked,
  onToggleSidebar,
  onOpenPalette,
}: TopbarProps) {
  const meta = ROUTE_MAP[route];

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
        <span className="topbar-path" aria-label="Aba atual">
          <span className="path-parent">/</span>
          <span className="path-current">{meta.label}</span>
        </span>
      </div>

      <div className="topbar-right">
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
          <span className="command-label">comandos…</span>
          <Kbd>Ctrl K</Kbd>
        </button>
      </div>
    </header>
  );
}
