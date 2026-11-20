import React from "react";
import { LoaderCircle, type LucideIcon } from "lucide-react";
import { cx } from "../lib/ui";

export type Tone = "neutral" | "accent" | "success" | "warning" | "danger" | "info" | "magenta";

const toneAttr = (tone: Tone | undefined) => (tone && tone !== "neutral" ? tone : undefined);

/* ---------- Badge ---------- */
interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  tone?: Tone;
  icon?: LucideIcon;
  /** Renders a small status dot before the label. */
  dot?: boolean;
}

export function Badge({ tone, icon: Icon, dot, className, children, ...rest }: BadgeProps) {
  return (
    <span className={cx("badge", dot && "badge-dot", className)} data-tone={toneAttr(tone)} {...rest}>
      {Icon && <Icon aria-hidden />}
      {children}
    </span>
  );
}

/* ---------- Keyboard key ---------- */
export function Kbd({ children }: { children: React.ReactNode }) {
  return <kbd className="kbd">{children}</kbd>;
}

/* ---------- Icon tile ---------- */
interface IconTileProps {
  icon: LucideIcon;
  tone?: Tone;
  size?: "sm" | "md" | "lg";
  className?: string;
}

export function IconTile({ icon: Icon, tone, size = "md", className }: IconTileProps) {
  return (
    <span
      className={cx("icon-tile", size !== "md" && `icon-tile-${size}`, className)}
      data-tone={toneAttr(tone)}
      aria-hidden
    >
      <Icon />
    </span>
  );
}

/* ---------- Spinner ---------- */
export function Spinner({ size = 15 }: { size?: number }) {
  return <LoaderCircle size={size} className="spin" aria-hidden />;
}

/** Icon that turns into a spinner while `busy` is true (keeps button widths stable). */
export function BusyIcon({ busy, icon: Icon, size = 15 }: { busy: boolean; icon: LucideIcon; size?: number }) {
  return busy ? <Spinner size={size} /> : <Icon size={size} aria-hidden />;
}

/* ---------- Page header ---------- */
interface PageHeaderProps {
  icon?: LucideIcon;
  eyebrow?: React.ReactNode;
  title: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
}

export function PageHeader({ icon: Icon, eyebrow, title, description, actions }: PageHeaderProps) {
  return (
    <header className="page-header">
      <div className="page-heading">
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1 className="page-title">
          {Icon && <Icon aria-hidden />}
          {title}
        </h1>
        {description && <p className="page-subtitle">{description}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

/* ---------- Panel (pane) ---------- */
interface PanelProps extends Omit<React.HTMLAttributes<HTMLElement>, "title"> {
  title?: React.ReactNode;
  icon?: LucideIcon;
  /** Numbered workflow step shown before the title. */
  step?: number;
  /** Muted intro line rendered at the top of the body. */
  description?: React.ReactNode;
  actions?: React.ReactNode;
  footer?: React.ReactNode;
  tone?: "accent";
  /** Removes body padding (tables, terminal output). */
  flush?: boolean;
  bodyClassName?: string;
}

export function Panel({
  title,
  icon: Icon,
  step,
  description,
  actions,
  footer,
  tone,
  flush,
  className,
  bodyClassName,
  children,
  ...rest
}: PanelProps) {
  const hasHeader = Boolean(title || actions);
  const hasChildren = children !== undefined && children !== null && children !== false;
  const hasBody = hasChildren || Boolean(description);
  return (
    <section className={cx("panel", className)} data-tone={tone} {...rest}>
      {hasHeader && (
        <header className="panel-header">
          <div className="panel-heading">
            {step !== undefined && <span className="step-badge">{step}</span>}
            {title && (
              <h2 className="panel-title">
                {Icon && <Icon aria-hidden />}
                {title}
              </h2>
            )}
          </div>
          {actions && <div className="panel-actions">{actions}</div>}
        </header>
      )}
      {hasBody && (
        <div className={cx("panel-body", flush && "panel-body-flush", bodyClassName)}>
          {description && <p className="panel-description">{description}</p>}
          {children}
        </div>
      )}
      {footer && <footer className="panel-footer">{footer}</footer>}
    </section>
  );
}

/* ---------- Empty state ---------- */
interface EmptyStateProps {
  icon: LucideIcon;
  title: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  compact?: boolean;
}

export function EmptyState({ icon: Icon, title, description, actions, compact }: EmptyStateProps) {
  return (
    <div className={cx("empty-state", compact && "empty-state-compact")}>
      <div className="empty-state-icon" aria-hidden>
        <Icon />
      </div>
      <div className="empty-state-title">{title}</div>
      {description && <p className="empty-state-desc">{description}</p>}
      {actions && <div className="empty-state-actions">{actions}</div>}
    </div>
  );
}

/* ---------- Stat tile & key/value grid ---------- */
interface StatTileProps {
  icon?: LucideIcon;
  label: React.ReactNode;
  value: React.ReactNode;
  detail?: React.ReactNode;
}

export function StatTile({ icon: Icon, label, value, detail }: StatTileProps) {
  return (
    <div className="stat-tile">
      <div className="stat-tile-head">
        {Icon && <Icon aria-hidden />}
        {label}
      </div>
      <div className="stat-tile-value">{value}</div>
      {detail && <div className="stat-tile-detail">{detail}</div>}
    </div>
  );
}

export interface KeyValueItem {
  label: React.ReactNode;
  value: React.ReactNode;
}

export function KeyValueGrid({ items }: { items: Array<KeyValueItem | null | false | undefined | 0 | ""> }) {
  return (
    <dl className="kv">
      {items.filter(Boolean).map((item, i) => {
        const kv = item as KeyValueItem;
        return (
          <div key={i} className="kv-item">
            <dt className="kv-label">{kv.label}</dt>
            <dd className="kv-value">{kv.value}</dd>
          </div>
        );
      })}
    </dl>
  );
}
