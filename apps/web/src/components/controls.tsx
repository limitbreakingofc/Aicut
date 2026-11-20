import React, { useId, useState } from "react";
import { ChevronRight, type LucideIcon } from "lucide-react";
import { cx } from "../lib/ui";

/** Moves focus/selection with arrow keys inside a tablist or radiogroup. */
function handleRovingKeys<T extends string>(
  e: React.KeyboardEvent<HTMLElement>,
  ids: T[],
  current: T,
  onChange: (id: T) => void,
  selector: string,
) {
  const keys = ["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp", "Home", "End"];
  if (!keys.includes(e.key) || ids.length === 0) return;
  e.preventDefault();
  const idx = Math.max(0, ids.indexOf(current));
  let next = idx;
  if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (idx + 1) % ids.length;
  if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (idx - 1 + ids.length) % ids.length;
  if (e.key === "Home") next = 0;
  if (e.key === "End") next = ids.length - 1;
  onChange(ids[next]);
  const items = e.currentTarget.querySelectorAll<HTMLElement>(selector);
  items[next]?.focus();
}

/* ---------- Tabs (page sections) ---------- */
export interface TabItem<T extends string> {
  id: T;
  label: React.ReactNode;
  icon?: LucideIcon;
  count?: number;
}

interface TabsProps<T extends string> {
  items: TabItem<T>[];
  value: T;
  onChange: (id: T) => void;
  ariaLabel?: string;
}

export function Tabs<T extends string>({ items, value, onChange, ariaLabel }: TabsProps<T>) {
  return (
    <div
      className="tabs"
      role="tablist"
      aria-label={ariaLabel}
      onKeyDown={(e) =>
        handleRovingKeys(
          e,
          items.map((i) => i.id),
          value,
          onChange,
          '[role="tab"]',
        )
      }
    >
      {items.map((item) => {
        const Icon = item.icon;
        const selected = item.id === value;
        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            className="tab"
            aria-selected={selected}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(item.id)}
          >
            {Icon && <Icon aria-hidden />}
            {item.label}
            {item.count !== undefined && <span className="tab-count">{item.count}</span>}
          </button>
        );
      })}
    </div>
  );
}

/* ---------- Segmented control ---------- */
export interface SegmentedOption<T extends string> {
  id: T;
  label: React.ReactNode;
  icon?: LucideIcon;
  title?: string;
}

interface SegmentedControlProps<T extends string> {
  options: SegmentedOption<T>[];
  value: T;
  onChange: (id: T) => void;
  size?: "sm" | "md";
  block?: boolean;
  ariaLabel?: string;
}

export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  size = "md",
  block,
  ariaLabel,
}: SegmentedControlProps<T>) {
  return (
    <div
      className={cx("segmented", size === "sm" && "segmented-sm", block && "segmented-block")}
      role="radiogroup"
      aria-label={ariaLabel}
      onKeyDown={(e) =>
        handleRovingKeys(
          e,
          options.map((o) => o.id),
          value,
          onChange,
          '[role="radio"]',
        )
      }
    >
      {options.map((opt) => {
        const Icon = opt.icon;
        const checked = opt.id === value;
        return (
          <button
            key={opt.id}
            type="button"
            role="radio"
            aria-checked={checked}
            tabIndex={checked ? 0 : -1}
            className="segmented-btn"
            title={opt.title}
            onClick={() => onChange(opt.id)}
          >
            {Icon && <Icon aria-hidden />}
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

/* ---------- Choice cards (rich radio group) ---------- */
export interface ChoiceOption<T extends string> {
  id: T;
  title: React.ReactNode;
  description?: React.ReactNode;
  icon?: LucideIcon;
  /** Optional custom visual shown instead of the radio dot. */
  preview?: React.ReactNode;
}

interface ChoiceCardsProps<T extends string> {
  options: ChoiceOption<T>[];
  value: T;
  onChange: (id: T) => void;
  columns?: number;
  ariaLabel?: string;
  className?: string;
}

export function ChoiceCards<T extends string>({
  options,
  value,
  onChange,
  columns = 2,
  ariaLabel,
  className,
}: ChoiceCardsProps<T>) {
  return (
    <div
      className={cx("choice-grid", className)}
      role="radiogroup"
      aria-label={ariaLabel}
      style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
      onKeyDown={(e) =>
        handleRovingKeys(
          e,
          options.map((o) => o.id),
          value,
          onChange,
          '[role="radio"]',
        )
      }
    >
      {options.map((opt) => {
        const Icon = opt.icon;
        const checked = opt.id === value;
        return (
          <button
            key={opt.id}
            type="button"
            role="radio"
            aria-checked={checked}
            tabIndex={checked ? 0 : -1}
            className="choice"
            onClick={() => onChange(opt.id)}
          >
            {opt.preview ?? <span className="choice-radio" aria-hidden />}
            <span className="choice-body">
              <span className="choice-title">
                {Icon && <Icon aria-hidden />}
                {opt.title}
              </span>
              {opt.description && <span className="choice-desc">{opt.description}</span>}
            </span>
          </button>
        );
      })}
    </div>
  );
}

/* ---------- Disclosure (collapsible section) ---------- */
interface DisclosureProps {
  title: React.ReactNode;
  icon?: LucideIcon;
  /** Short state summary shown on the right while collapsed (e.g. "pt, en · SRT"). */
  summary?: React.ReactNode;
  /** Highlights the summary when the section holds non-default values. */
  summaryActive?: boolean;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: React.ReactNode;
}

export function Disclosure({
  title,
  icon: Icon,
  summary,
  summaryActive,
  open,
  defaultOpen = false,
  onOpenChange,
  children,
}: DisclosureProps) {
  const [innerOpen, setInnerOpen] = useState(defaultOpen);
  const isOpen = open ?? innerOpen;
  const bodyId = useId();

  const toggle = () => {
    const next = !isOpen;
    if (open === undefined) setInnerOpen(next);
    onOpenChange?.(next);
  };

  return (
    <div className="disclosure" data-open={isOpen || undefined}>
      <button
        type="button"
        className="disclosure-trigger"
        aria-expanded={isOpen}
        aria-controls={bodyId}
        onClick={toggle}
      >
        {Icon && <Icon className="disclosure-icon" size={14} aria-hidden />}
        <span className="disclosure-title">{title}</span>
        {summary && (
          <span className="disclosure-summary" data-active={summaryActive || undefined}>
            {summary}
          </span>
        )}
        <ChevronRight className="disclosure-chevron" size={16} aria-hidden />
      </button>
      {isOpen && (
        <div className="disclosure-body" id={bodyId}>
          {children}
        </div>
      )}
    </div>
  );
}

/* ---------- Switch ---------- */
interface SwitchProps {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
  disabled?: boolean;
}

export function Switch({ checked, onChange, label, disabled }: SwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      className="switch"
      aria-checked={checked}
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
    />
  );
}

/* ---------- Field & checkbox ---------- */
interface FieldProps {
  label?: React.ReactNode;
  htmlFor?: string;
  hint?: React.ReactNode;
  /** Extra content aligned to the right of the label (status, counters…). */
  aside?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}

export function Field({ label, htmlFor, hint, aside, className, children }: FieldProps) {
  return (
    <div className={cx("field", className)}>
      {(label || aside) && (
        <div className="field-label">
          {label && <label htmlFor={htmlFor}>{label}</label>}
          {aside}
        </div>
      )}
      {children}
      {hint && <div className="field-hint">{hint}</div>}
    </div>
  );
}

interface CheckboxProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: React.ReactNode;
  description?: React.ReactNode;
  disabled?: boolean;
}

export function Checkbox({ checked, onChange, label, description, disabled }: CheckboxProps) {
  return (
    <label className="check-row">
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="check-text">
        <span>{label}</span>
        {description && <span className="check-desc">{description}</span>}
      </span>
    </label>
  );
}
