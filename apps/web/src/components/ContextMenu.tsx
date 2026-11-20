import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { MoreHorizontal, type LucideIcon } from "lucide-react";
import { cx } from "../lib/ui";

/* ==========================================================================
   Context menus (right click) — one provider, used by every view.
   - open(event, items): at the pointer (right click / Shift+F10)
   - openAt(element, items): anchored below an element ("⋯" buttons)
   Keyboard: ↑ ↓ Home End to move, Enter/Space to run, Esc/Tab to close.
   ========================================================================== */

export interface MenuAction {
  type?: "item";
  label: string;
  icon?: LucideIcon;
  shortcut?: string;
  danger?: boolean;
  disabled?: boolean;
  /** Renders a radio-style marker (e.g. current color scheme). */
  checked?: boolean;
  onSelect: () => void;
}

export type MenuEntry = MenuAction | { type: "separator" } | { type: "label"; label: string };

/** Falsy entries are ignored, so items can be declared conditionally. */
export type MenuItems = Array<MenuEntry | false | null | undefined | "">;

interface PointerLike {
  clientX: number;
  clientY: number;
  target?: EventTarget | null;
  preventDefault?: () => void;
  stopPropagation?: () => void;
}

interface ContextMenuApi {
  open: (event: PointerLike, items: MenuItems) => void;
  openAt: (element: Element, items: MenuItems) => void;
  close: () => void;
}

interface MenuState {
  id: number;
  x: number;
  y: number;
  items: MenuEntry[];
  returnFocus: HTMLElement | null;
}

const ContextMenuContext = createContext<ContextMenuApi | null>(null);

export function useContextMenu(): ContextMenuApi {
  const api = useContext(ContextMenuContext);
  if (!api) throw new Error("useContextMenu() must be used inside <ContextMenuProvider>");
  return api;
}

const isAction = (entry: MenuEntry): entry is MenuAction =>
  entry.type === undefined || entry.type === "item";

/** Drops falsy entries plus leading, trailing and doubled separators. */
function normalize(items: MenuItems): MenuEntry[] {
  const out: MenuEntry[] = [];
  for (const entry of items) {
    if (!entry) continue;
    if (entry.type === "separator" && (out.length === 0 || out[out.length - 1].type === "separator")) continue;
    out.push(entry);
  }
  while (out.length && out[out.length - 1].type === "separator") out.pop();
  return out;
}

export function ContextMenuProvider({ children }: { children: ReactNode }) {
  const [menu, setMenu] = useState<MenuState | null>(null);

  const close = useCallback(() => setMenu(null), []);

  const show = useCallback((x: number, y: number, items: MenuItems) => {
    const list = normalize(items);
    if (!list.some(isAction)) return;
    const active = document.activeElement;
    setMenu({ id: Date.now() + Math.random(), x, y, items: list, returnFocus: active instanceof HTMLElement ? active : null });
  }, []);

  const open = useCallback(
    (event: PointerLike, items: MenuItems) => {
      event.preventDefault?.();
      event.stopPropagation?.();
      let { clientX: x, clientY: y } = event;
      // Keyboard-triggered context menus (Shift+F10 / Menu key) report 0,0
      if (x === 0 && y === 0 && event.target instanceof Element) {
        const rect = event.target.getBoundingClientRect();
        x = rect.left + 8;
        y = rect.bottom - 4;
      }
      show(x, y, items);
    },
    [show],
  );

  const openAt = useCallback(
    (element: Element, items: MenuItems) => {
      const rect = element.getBoundingClientRect();
      show(rect.left, rect.bottom + 4, items);
    },
    [show],
  );

  const api = useMemo(() => ({ open, openAt, close }), [open, openAt, close]);

  return (
    <ContextMenuContext.Provider value={api}>
      {children}
      {menu && <MenuView key={menu.id} menu={menu} onClose={close} />}
    </ContextMenuContext.Provider>
  );
}

function MenuView({ menu, onClose }: { menu: MenuState; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: menu.x, top: menu.y });
  const [active, setActive] = useState(-1);
  const hasChecks = menu.items.some((e) => isAction(e) && e.checked !== undefined);

  const enabled = useMemo(
    () => menu.items.map((e, i) => (isAction(e) && !e.disabled ? i : -1)).filter((i) => i >= 0),
    [menu.items],
  );

  // Keep the menu inside the viewport (flip/clamp) and take focus
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let left = menu.x;
    let top = menu.y;
    if (left + rect.width > vw - 8) left = Math.max(8, menu.x - rect.width);
    if (top + rect.height > vh - 8) top = Math.max(8, vh - rect.height - 8);
    setPos({ left, top });
    el.focus({ preventScroll: true });
  }, [menu]);

  // Close on outside interaction
  useEffect(() => {
    const inside = (target: EventTarget | null) => target instanceof Node && Boolean(ref.current?.contains(target));
    const onPointerDown = (e: MouseEvent) => {
      if (!inside(e.target)) onClose();
    };
    const onScroll = (e: Event) => {
      if (!inside(e.target)) onClose();
    };
    window.addEventListener("mousedown", onPointerDown, true);
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onClose);
    window.addEventListener("blur", onClose);
    return () => {
      window.removeEventListener("mousedown", onPointerDown, true);
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onClose);
      window.removeEventListener("blur", onClose);
    };
  }, [onClose]);

  // Give focus back to where the user was (e.g. the "⋯" button)
  useEffect(() => {
    const target = menu.returnFocus;
    return () => {
      if (target?.isConnected) target.focus({ preventScroll: true });
    };
  }, [menu.returnFocus]);

  const run = (entry: MenuEntry | undefined) => {
    if (!entry || !isAction(entry) || entry.disabled) return;
    onClose();
    entry.onSelect();
  };

  const move = (delta: 1 | -1) => {
    if (enabled.length === 0) return;
    const at = enabled.indexOf(active);
    const next = at === -1 ? (delta === 1 ? 0 : enabled.length - 1) : (at + delta + enabled.length) % enabled.length;
    setActive(enabled[next]);
  };

  const onKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    e.stopPropagation();
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        move(1);
        break;
      case "ArrowUp":
        e.preventDefault();
        move(-1);
        break;
      case "Home":
        e.preventDefault();
        setActive(enabled[0] ?? -1);
        break;
      case "End":
        e.preventDefault();
        setActive(enabled[enabled.length - 1] ?? -1);
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        run(menu.items[active]);
        break;
      case "Escape":
      case "Tab":
        e.preventDefault();
        onClose();
        break;
    }
  };

  return createPortal(
    <div
      ref={ref}
      className="ctx-menu"
      role="menu"
      tabIndex={-1}
      style={{ left: pos.left, top: pos.top }}
      onKeyDown={onKeyDown}
      onContextMenu={(e) => e.preventDefault()}
    >
      {menu.items.map((entry, i) => {
        if (entry.type === "separator") return <div key={i} className="ctx-sep" role="separator" />;
        if (entry.type === "label") {
          return (
            <div key={i} className="ctx-label" role="presentation">
              {entry.label}
            </div>
          );
        }
        const Icon = entry.icon;
        return (
          <button
            key={i}
            type="button"
            role={entry.checked !== undefined ? "menuitemradio" : "menuitem"}
            aria-checked={entry.checked}
            className="ctx-item"
            data-active={i === active || undefined}
            data-danger={entry.danger || undefined}
            disabled={entry.disabled}
            tabIndex={-1}
            onMouseEnter={() => setActive(entry.disabled ? -1 : i)}
            onMouseLeave={() => setActive(-1)}
            onClick={() => run(entry)}
          >
            {hasChecks && <span className="ctx-check">{entry.checked ? "●" : ""}</span>}
            {Icon ? <Icon aria-hidden /> : !hasChecks && <span style={{ width: 14 }} aria-hidden />}
            <span className="ctx-item-label">{entry.label}</span>
            {entry.shortcut && <span className="ctx-item-shortcut">{entry.shortcut}</span>}
          </button>
        );
      })}
    </div>,
    document.body,
  );
}

/** "⋯" button that opens the same menu as a right click (discoverability + keyboard). */
export function MoreButton({
  items,
  label = "Mais ações",
  className,
}: {
  items: () => MenuItems;
  label?: string;
  className?: string;
}) {
  const menu = useContextMenu();
  return (
    <button
      type="button"
      className={cx("btn btn-ghost btn-icon btn-xs", className)}
      aria-label={label}
      aria-haspopup="menu"
      title={`${label} (ou clique com o botão direito)`}
      onClick={(e) => {
        e.stopPropagation();
        menu.openAt(e.currentTarget, items());
      }}
    >
      <MoreHorizontal aria-hidden />
    </button>
  );
}
