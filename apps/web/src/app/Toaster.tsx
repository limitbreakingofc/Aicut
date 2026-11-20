import { CircleCheck, CircleX, Info, TriangleAlert, X, type LucideIcon } from "lucide-react";
import type { NotifyLevel } from "../lib/types";

export interface ToastItem {
  id: string;
  level: NotifyLevel;
  title: string;
  message: string;
}

const ICONS: Record<NotifyLevel, LucideIcon> = {
  info: Info,
  success: CircleCheck,
  warning: TriangleAlert,
  error: CircleX,
};

export function Toaster({ toasts, onDismiss }: { toasts: ToastItem[]; onDismiss: (id: string) => void }) {
  // The live region stays mounted so screen readers announce new toasts.
  return (
    <div className="toast-stack" aria-live="polite" aria-relevant="additions">
      {toasts.map((t) => {
        const Icon = ICONS[t.level];
        return (
          <div key={t.id} className="toast" data-level={t.level} role={t.level === "error" ? "alert" : "status"}>
            <span className="toast-icon" aria-hidden>
              <Icon />
            </span>
            <div className="toast-text">
              <div className="toast-title">{t.title}</div>
              {t.message && <div className="toast-message">{t.message}</div>}
            </div>
            <button
              type="button"
              className="btn btn-ghost btn-icon btn-xs"
              onClick={() => onDismiss(t.id)}
              aria-label="Dispensar notificação"
            >
              <X aria-hidden />
            </button>
          </div>
        );
      })}
    </div>
  );
}
