import { Fragment, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import type { LucideIcon } from "lucide-react";
import { IconTile, Kbd } from "../components";
import { normalizeSearch } from "../lib/ui";

export interface PaletteCommand {
  id: string;
  label: string;
  description?: string;
  icon: LucideIcon;
  group: string;
  /** Keyboard shortcut shown as key caps, e.g. "Alt+2". */
  shortcut?: string;
  keywords?: string;
  run: () => void;
}

interface CommandPaletteProps {
  open: boolean;
  commands: PaletteCommand[];
  onClose: () => void;
}

/** Ctrl+K launcher with accent-insensitive search and full keyboard navigation. */
export function CommandPalette({ open, commands, onClose }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIndex(0);
    }
  }, [open]);

  const searching = query.trim().length > 0;

  // Rank: label prefix > label contains > description / keywords (stable within each rank)
  const filtered = useMemo(() => {
    const terms = normalizeSearch(query.trim()).split(/\s+/).filter(Boolean);
    if (terms.length === 0) return commands;
    const scored: Array<{ cmd: PaletteCommand; score: number; order: number }> = [];
    commands.forEach((cmd, order) => {
      const label = normalizeSearch(cmd.label);
      const haystack = `${label} ${normalizeSearch(`${cmd.description ?? ""} ${cmd.keywords ?? ""} ${cmd.group}`)}`;
      if (!terms.every((t) => haystack.includes(t))) return;
      const score = label.startsWith(terms[0]) ? 3 : terms.every((t) => label.includes(t)) ? 2 : 1;
      scored.push({ cmd, score, order });
    });
    scored.sort((a, b) => b.score - a.score || a.order - b.order);
    return scored.map((s) => s.cmd);
  }, [commands, query]);

  useEffect(() => {
    setActiveIndex(0);
  }, [query]);

  useEffect(() => {
    listRef.current
      ?.querySelector<HTMLElement>(`[data-index="${activeIndex}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);

  if (!open) return null;

  const run = (cmd: PaletteCommand | undefined) => {
    if (!cmd) return;
    onClose();
    cmd.run();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => (filtered.length ? (i + 1) % filtered.length : 0));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => (filtered.length ? (i - 1 + filtered.length) % filtered.length : 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      run(filtered[activeIndex]);
    }
  };

  // Browsing: grouped in registry order. Searching: one ranked list (keeps ↑/↓ in visual order).
  const groups: Array<{ name: string; items: Array<{ cmd: PaletteCommand; index: number }> }> = [];
  filtered.forEach((cmd, index) => {
    const name = searching ? "Resultados" : cmd.group;
    let group = groups.find((g) => g.name === name);
    if (!group) {
      group = { name, items: [] };
      groups.push(group);
    }
    group.items.push({ cmd, index });
  });

  const optionId = (index: number) => `palette-option-${index}`;

  return (
    <div className="palette-backdrop" onMouseDown={onClose}>
      <div
        className="palette"
        role="dialog"
        aria-modal="true"
        aria-label="Paleta de comandos"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="palette-search">
          <span className="palette-prompt" aria-hidden>
            ›
          </span>
          <input
            autoFocus
            role="combobox"
            aria-expanded="true"
            aria-controls="palette-list"
            aria-activedescendant={filtered.length ? optionId(activeIndex) : undefined}
            placeholder="ir para uma página ou executar um comando…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
          />
          <span className="muted text-xs tabular">
            {filtered.length}/{commands.length}
          </span>
        </div>

        <div className="palette-list" id="palette-list" role="listbox" ref={listRef}>
          {filtered.length === 0 ? (
            <div className="palette-empty">nenhum resultado para “{query}”</div>
          ) : (
            groups.map((group) => (
              <div key={group.name} role="group" aria-label={group.name}>
                <div className="palette-group-label">{group.name}</div>
                {group.items.map(({ cmd, index }) => (
                  <button
                    key={cmd.id}
                    id={optionId(index)}
                    type="button"
                    role="option"
                    tabIndex={-1}
                    className="palette-item"
                    data-index={index}
                    aria-selected={index === activeIndex}
                    onMouseMove={() => setActiveIndex(index)}
                    onClick={() => run(cmd)}
                  >
                    <IconTile icon={cmd.icon} size="sm" />
                    <span className="palette-item-text">
                      <span className="palette-item-label">{cmd.label}</span>
                      {cmd.description && <span className="palette-item-desc">{cmd.description}</span>}
                    </span>
                    {cmd.shortcut && (
                      <span className="palette-item-hint">
                        {cmd.shortcut.split("+").map((key, i) => (
                          <Fragment key={key}>
                            {i > 0 && "+"}
                            <Kbd>{key}</Kbd>
                          </Fragment>
                        ))}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            ))
          )}
        </div>

        <div className="palette-footer" aria-hidden>
          <span>
            <Kbd>↑</Kbd>
            <Kbd>↓</Kbd> navegar
          </span>
          <span>
            <Kbd>↵</Kbd> executar
          </span>
          <span>
            <Kbd>Esc</Kbd> fechar
          </span>
        </div>
      </div>
    </div>
  );
}
