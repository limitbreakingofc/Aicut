import React from "react";
import {
  ArrowRight,
  CircleCheck,
  CircleX,
  Info,
  Sparkles,
  TriangleAlert,
  type LucideIcon,
} from "lucide-react";
import { cx } from "../lib/ui";
import type { JobRecord } from "../lib/types";
import { Badge, Panel, Spinner, type Tone } from "./primitives";

/* ---------- Callout (inline banner) ---------- */
const CALLOUT_ICONS: Record<Tone, LucideIcon> = {
  neutral: Info,
  accent: Sparkles,
  success: CircleCheck,
  warning: TriangleAlert,
  danger: CircleX,
  info: Info,
  magenta: Sparkles,
};

interface CalloutProps {
  tone?: Tone;
  /** Custom icon; pass `null` to hide it. */
  icon?: LucideIcon | null;
  title?: React.ReactNode;
  children?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}

export function Callout({ tone = "neutral", icon, title, children, actions, className }: CalloutProps) {
  const Icon = icon === null ? null : (icon ?? CALLOUT_ICONS[tone]);
  return (
    <div className={cx("callout", className)} data-tone={tone === "neutral" ? undefined : tone}>
      {Icon && (
        <span className="callout-icon" aria-hidden>
          <Icon />
        </span>
      )}
      <div className="callout-body">
        {title && <div className="callout-title">{title}</div>}
        {children && <div>{children}</div>}
      </div>
      {actions && <div className="callout-actions">{actions}</div>}
    </div>
  );
}

/* ---------- Progress bar ---------- */
interface ProgressBarProps {
  value: number;
  tone?: "accent" | "success" | "warning" | "danger";
  /** Animated shimmer for work in progress. */
  active?: boolean;
  size?: "sm" | "md";
  label?: string;
}

export function ProgressBar({ value, tone = "accent", active, size = "md", label }: ProgressBarProps) {
  const pct = Math.min(100, Math.max(0, Number.isFinite(value) ? value : 0));
  return (
    <div
      className={cx("progress", size === "sm" && "progress-sm")}
      data-tone={tone === "accent" ? undefined : tone}
      data-active={active || undefined}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pct)}
      aria-label={label}
    >
      <div className="progress-fill" style={{ width: `${pct}%` }} />
    </div>
  );
}

/* ---------- Score meter ---------- */
export function Meter({ label, value }: { label: React.ReactNode; value: number }) {
  const pct = Math.min(100, Math.max(0, value));
  const tone = pct >= 85 ? "success" : pct >= 65 ? undefined : "warning";
  return (
    <div className="meter" data-tone={tone}>
      <div className="meter-head">
        <span>{label}</span>
        <span className="meter-value">{Math.round(pct)}</span>
      </div>
      <div className="meter-track">
        <div className="meter-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

/* ---------- Active jobs panel ---------- */
interface ActiveJobsPanelProps {
  jobs: JobRecord[];
  title?: React.ReactNode;
  onViewAll?: () => void;
  tone?: "accent";
}

/** Compact live progress list used by Ingest, Transcribe and Smart Clips. Renders nothing when idle. */
export function ActiveJobsPanel({ jobs, title = "Em andamento", onViewAll, tone }: ActiveJobsPanelProps) {
  if (jobs.length === 0) return null;
  return (
    <Panel
      tone={tone}
      title={
        <>
          <Spinner size={16} />
          {title}
          <Badge tone="accent">{jobs.length}</Badge>
        </>
      }
      actions={
        onViewAll && (
          <button type="button" className="btn btn-ghost btn-sm" onClick={onViewAll}>
            Ver fila <ArrowRight aria-hidden />
          </button>
        )
      }
    >
      <div className="stack">
        {jobs.map((job) => (
          <div key={job.id} className="job-mini">
            <div className="job-mini-head">
              <span className="job-mini-title" title={job.title}>
                {job.title}
              </span>
              <span className="job-mini-percent">{job.progress.percent.toFixed(0)}%</span>
            </div>
            <ProgressBar value={job.progress.percent} active size="sm" label={job.title} />
            <div className="job-mini-meta">{job.progress.message ?? job.progress.stage}</div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
