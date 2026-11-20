import { useContextMenu, type MenuItems } from "../components";
import { ROUTES, ROUTE_MAP, routeDigit, type PrimaryRouteId, type RouteMeta } from "./navigation";

export interface NavCount {
  value: number;
  tone?: "accent" | "warning";
  title?: string;
}

interface SidebarProps {
  route: PrimaryRouteId;
  collapsed: boolean;
  counts: Partial<Record<PrimaryRouteId, NavCount>>;
  onNavigate: (route: PrimaryRouteId) => void;
  menuFor?: (route: PrimaryRouteId) => MenuItems;
}

export function Sidebar({ route, collapsed, counts, onNavigate, menuFor }: SidebarProps) {
  const mainItems = ROUTES.filter((r) => !r.footer);
  const settings = ROUTE_MAP.settings;

  const renderItem = (meta: RouteMeta) => (
    <NavItem
      key={meta.id}
      meta={meta}
      active={route === meta.id}
      collapsed={collapsed}
      count={counts[meta.id]}
      onClick={() => onNavigate(meta.id)}
      menu={() => menuFor?.(meta.id) ?? []}
    />
  );

  return (
    <aside className="sidebar">
      <nav className="sidebar-scroll" aria-label="Navegação principal">
        <div className="nav-section">{mainItems.map(renderItem)}</div>
      </nav>
      <div className="sidebar-footer">{renderItem(settings)}</div>
    </aside>
  );
}

interface NavItemProps {
  meta: RouteMeta;
  active: boolean;
  collapsed: boolean;
  count?: NavCount;
  onClick: () => void;
  menu: () => MenuItems;
}

function NavItem({ meta, active, collapsed, count, onClick, menu }: NavItemProps) {
  const ctx = useContextMenu();
  const Icon = meta.icon;
  const digit = routeDigit(meta.id);
  const showCount = count !== undefined && count.value > 0;
  const tooltip = `${meta.label} (Alt+${digit})${showCount && count.title ? ` · ${count.title}` : ""}`;

  return (
    <button
      type="button"
      className="nav-item"
      aria-current={active ? "page" : undefined}
      aria-keyshortcuts={`Alt+${digit}`}
      title={collapsed ? tooltip : undefined}
      onClick={onClick}
      onContextMenu={(e) =>
        ctx.open(e, [
          { type: "label", label: meta.label },
          { label: "Abrir", icon: Icon, shortcut: `Alt+${digit}`, onSelect: onClick },
          ...menu(),
        ])
      }
    >
      <Icon aria-hidden />
      <span className="nav-label">{meta.label}</span>
      {showCount && (
        <span className="nav-count" data-tone={count.tone} title={count.title}>
          {count.value}
        </span>
      )}
      <span className="nav-key" aria-hidden>
        {digit}
      </span>
    </button>
  );
}
