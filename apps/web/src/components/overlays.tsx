import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Copy, X } from "lucide-react";
import { storage } from "../lib/ui";

/* ---------- Modal (centered dialog) ---------- */
interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

export function Modal({ open, onClose, title, children }: ModalProps) {
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeRef.current();
    };
    window.addEventListener("keydown", onKey);
    dialogRef.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  if (!open) return null;

  return createPortal(
    <div className="modal-backdrop" onMouseDown={() => closeRef.current()}>
      <div
        ref={dialogRef}
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <span className="modal-title">{title}</span>
          <button
            type="button"
            className="btn btn-ghost btn-icon btn-sm"
            onClick={() => closeRef.current()}
            aria-label="Fechar"
            title="Fechar (Esc)"
          >
            <X aria-hidden />
          </button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>,
    document.body,
  );
}

/* ---------- Code block with copy button ---------- */
interface CodeBlockProps {
  code: string;
  onCopy?: () => void;
  copyLabel?: string;
  minHeight?: number;
  maxHeight?: number;
}

export function CodeBlock({ code, onCopy, copyLabel = "Copiar", minHeight, maxHeight }: CodeBlockProps) {
  return (
    <div className="code-wrap">
      <pre className="code-block" style={{ minHeight, maxHeight }}>
        {code}
      </pre>
      {onCopy && (
        <button type="button" className="btn btn-xs code-copy" onClick={onCopy}>
          <Copy aria-hidden /> {copyLabel}
        </button>
      )}
    </div>
  );
}

/* ---------- Resizable panes ---------- */
/** Width state persisted per pane (clamped to [min, max]). */
export function usePersistentWidth(key: string, initial: number, min: number, max: number) {
  const [width, setWidth] = useState(() => {
    const saved = Number(storage.get(key));
    return Number.isFinite(saved) && saved >= min && saved <= max ? saved : initial;
  });

  useEffect(() => {
    storage.set(key, String(Math.round(width)));
  }, [key, width]);

  return [width, setWidth] as const;
}

interface ResizeHandleProps {
  width: number;
  min: number;
  max: number;
  onChange: (width: number) => void;
  /** Which edge of the pane the handle sits on. */
  edge?: "left" | "right";
  label?: string;
  resetTo?: number;
}

/** Drag (or ←/→ when focused) to resize a pane; double click restores the default width. */
export function ResizeHandle({ width, min, max, onChange, edge = "left", label = "Redimensionar painel", resetTo }: ResizeHandleProps) {
  const [dragging, setDragging] = useState(false);
  const clamp = (v: number) => Math.min(max, Math.max(min, v));

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = width;
    setDragging(true);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    const onMove = (ev: PointerEvent) => {
      const dx = ev.clientX - startX;
      onChange(clamp(edge === "left" ? startWidth - dx : startWidth + dx));
    };
    const onUp = () => {
      setDragging(false);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 64 : 16;
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      onChange(clamp(width + (edge === "left" ? step : -step)));
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      onChange(clamp(width + (edge === "left" ? -step : step)));
    }
  };

  return (
    <div
      className="resize-handle"
      style={edge === "right" ? { left: "auto", right: -5 } : undefined}
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={Math.round(width)}
      aria-valuemin={min}
      aria-valuemax={max}
      tabIndex={0}
      title={`${label} (duplo clique para restaurar)`}
      data-dragging={dragging || undefined}
      onPointerDown={onPointerDown}
      onKeyDown={onKeyDown}
      onDoubleClick={() => resetTo !== undefined && onChange(clamp(resetTo))}
    />
  );
}
