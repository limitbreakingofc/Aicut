import React, { Fragment, useMemo, useState } from "react";
import {
  Activity,
  Ban,
  Bot,
  ChevronRight,
  CircleCheck,
  CircleX,
  Clock,
  Copy,
  RefreshCw,
  ScrollText,
  Trash2,
  User,
  type LucideIcon,
} from "lucide-react";
import { runAction } from "../../lib/client";
import { formatBytes, formatDuration, formatRelativeDate, formatSpeed } from "../../lib/format";
import type { JobRecord, JobStatus, NotifyFn } from "../../lib/types";
import { copyToClipboard } from "../../lib/ui";
import { pageMeta } from "../../app/navigation";
import {
  Callout,
  EmptyState,
  MoreButton,
  PageHeader,
  ProgressBar,
  Spinner,
  Tabs,
  useContextMenu,
  type MenuItems,
} from "../../components";
import "./core.css";

interface JobsViewProps {
  jobs: JobRecord[];
  onRefresh: () => void;
  onClearFinished: () => Promise<void>;
  onNotify: NotifyFn;
}

type Filter = "all" | "active" | "completed" | "failed" | "cancelled";

const STATUS_META: Record<JobStatus, { label: string; icon: LucideIcon; className: string }> = {
  queued: { label: "na fila", icon: Clock, className: "muted" },
  running: { label: "rodando", icon: Activity, className: "text-accent" },
  completed: { label: "ok", icon: CircleCheck, className: "text-success" },
  failed: { label: "falhou", icon: CircleX, className: "text-danger" },
  cancelled: { label: "cancelado", icon: Ban, className: "text-warning" },
};

const isActive = (j: JobRecord) => j.status === "queued" || j.status === "running";

export const JobsView: React.FC<JobsViewProps> = ({ jobs, onRefresh, onClearFinished, onNotify }) => {
  const menu = useContextMenu();
  const [expandedLogs, setExpandedLogs] = useState<Record<string, boolean>>({});
  const [filter, setFilter] = useState<Filter>("all");

  async function handleCancel(jobId: string) {
    try {
      await runAction("jobs.cancel", { job_id: jobId });
      onNotify("info", "Tarefa cancelada", `Job ${jobId.slice(0, 8)} interrompido.`);
      onRefresh();
    } catch (e) {
      onNotify("error", "Erro ao cancelar", (e as Error).message);
    }
  }

  const counts = useMemo(
    () => ({
      all: jobs.length,
      active: jobs.filter(isActive).length,
      completed: jobs.filter((j) => j.status === "completed").length,
      failed: jobs.filter((j) => j.status === "failed").length,
      cancelled: jobs.filter((j) => j.status === "cancelled").length,
    }),
    [jobs],
  );

  const visible = jobs.filter((j) =>
    filter === "all" ? true : filter === "active" ? isActive(j) : j.status === filter,
  );

  const toggleLogs = (id: string) => setExpandedLogs((p) => ({ ...p, [id]: !p[id] }));

  const jobMenu = (job: JobRecord): MenuItems => [
    { type: "label", label: `${job.kind} · ${job.id.slice(0, 8)}` },
    isActive(job) && { label: "Cancelar tarefa", icon: Ban, danger: true, onSelect: () => void handleCancel(job.id) },
    job.logs.length > 0 && {
      label: expandedLogs[job.id] ? "Ocultar logs" : `Ver logs (${job.logs.length})`,
      icon: ScrollText,
      onSelect: () => toggleLogs(job.id),
    },
    { type: "separator" },
    { label: "Copiar ID", icon: Copy, onSelect: () => void copyToClipboard(job.id, onNotify, "ID da tarefa") },
    job.logs.length > 0 && {
      label: "Copiar logs",
      icon: Copy,
      onSelect: () => void copyToClipboard(job.logs.join("\n"), onNotify, "Logs"),
    },
    job.error && {
      label: "Copiar erro",
      icon: Copy,
      onSelect: () =>
        void copyToClipboard(
          `${job.error?.code}: ${job.error?.message}${job.error?.hint ? `\nDica: ${job.error.hint}` : ""}`,
          onNotify,
          "Erro",
        ),
    },
    { type: "separator" },
    { label: "Limpar finalizadas", icon: Trash2, onSelect: () => void onClearFinished() },
  ];

  return (
    <>
      <PageHeader
        {...pageMeta("jobs")}
        actions={
          <>
            <button type="button" className="btn btn-sm" onClick={onRefresh}>
              <RefreshCw aria-hidden /> Atualizar
            </button>
            <button type="button" className="btn btn-sm" onClick={() => void onClearFinished()}>
              <Trash2 aria-hidden /> Limpar finalizadas
            </button>
          </>
        }
      />

      <Tabs
        ariaLabel="Filtrar tarefas"
        value={filter}
        onChange={setFilter}
        items={[
          { id: "all", label: "Todas", count: counts.all },
          { id: "active", label: "Em andamento", count: counts.active },
          { id: "completed", label: "Concluídas", count: counts.completed },
          { id: "failed", label: "Com falha", count: counts.failed },
          { id: "cancelled", label: "Canceladas", count: counts.cancelled },
        ]}
      />

      {visible.length === 0 ? (
        <div className="panel">
          <EmptyState
            icon={Activity}
            title={jobs.length === 0 ? "nenhuma tarefa na fila" : "nada neste filtro"}
            description="Downloads, importações, transcrições e renderizações — suas ou de qualquer IA — aparecem aqui em tempo real."
          />
        </div>
      ) : (
        <div className="table-wrap">
          <table className="table jobs-table">
            <thead>
              <tr>
                <th className="col-fit" aria-label="Expandir" />
                <th className="col-fit">status</th>
                <th>tarefa</th>
                <th className="col-fit">ator</th>
                <th className="progress-col">progresso</th>
                <th className="col-fit num c-xfer">transferido</th>
                <th className="col-fit num c-speed">velocidade</th>
                <th className="col-fit num">eta</th>
                <th className="col-fit c-created">criada</th>
                <th className="col-fit" aria-label="Ações" />
              </tr>
            </thead>
            <tbody>
              {visible.map((job) => {
                const status = STATUS_META[job.status];
                const open = Boolean(expandedLogs[job.id]);
                const hasDetails = job.logs.length > 0;
                const tone =
                  job.status === "completed"
                    ? "success"
                    : job.status === "failed"
                      ? "danger"
                      : job.status === "cancelled"
                        ? "warning"
                        : "accent";
                return (
                  <Fragment key={job.id}>
                    <tr
                      data-clickable={hasDetails || undefined}
                      aria-expanded={hasDetails ? open : undefined}
                      onClick={() => hasDetails && toggleLogs(job.id)}
                      onContextMenu={(e) => menu.open(e, jobMenu(job))}
                    >
                      <td className="col-fit">
                        {hasDetails && (
                          <ChevronRight size={14} className="expand-chevron" data-open={open || undefined} aria-hidden />
                        )}
                      </td>
                      <td className="col-fit">
                        <span className={`job-status ${status.className}`}>
                          {job.status === "running" ? <Spinner size={13} /> : <status.icon size={13} aria-hidden />}
                          {status.label}
                        </span>
                      </td>
                      <td className="cell-title" title={job.title}>
                        <div className="truncate">{job.title}</div>
                        <div className="muted text-xs truncate">
                          {job.kind} · {job.id.slice(0, 8)}
                        </div>
                      </td>
                      <td className="col-fit">
                        <span className={job.actor === "ai" ? "actor text-magenta" : "actor"}>
                          {job.actor === "ai" ? <Bot size={12} aria-hidden /> : <User size={12} aria-hidden />}
                          {job.actor === "ai" ? "ia" : "humano"}
                        </span>
                      </td>
                      <td className="progress-col">
                        <div className="job-progress">
                          <ProgressBar
                            value={job.progress.percent}
                            tone={tone}
                            active={job.status === "running"}
                            size="sm"
                            label={`Progresso de ${job.title}`}
                          />
                          <span className="job-percent">{job.progress.percent.toFixed(1)}%</span>
                        </div>
                        <div className="muted text-xs truncate" title={job.progress.message ?? job.progress.stage}>
                          {job.progress.stage}
                          {job.progress.message ? ` — ${job.progress.message}` : ""}
                        </div>
                      </td>
                      <td className="col-fit num c-xfer">
                        {job.progress.downloaded_bytes ? formatBytes(job.progress.downloaded_bytes) : "—"}
                        {job.progress.total_bytes ? <span className="muted"> / {formatBytes(job.progress.total_bytes)}</span> : null}
                      </td>
                      <td className="col-fit num c-speed">
                        {job.progress.speed_bytes_per_sec ? formatSpeed(job.progress.speed_bytes_per_sec) : "—"}
                      </td>
                      <td className="col-fit num">{job.progress.eta_seconds ? formatDuration(job.progress.eta_seconds) : "—"}</td>
                      <td className="col-fit muted c-created">{formatRelativeDate(job.created_at)}</td>
                      <td className="col-fit" onClick={(e) => e.stopPropagation()}>
                        <span className="row gap-xs">
                          {isActive(job) && (
                            <button type="button" className="btn btn-danger btn-xs" onClick={() => handleCancel(job.id)}>
                              Cancelar
                            </button>
                          )}
                          <MoreButton items={() => jobMenu(job)} />
                        </span>
                      </td>
                    </tr>
                    {(job.error || (open && hasDetails)) && (
                      <tr className="job-details">
                        <td colSpan={10}>
                          <div className="stack stack-sm">
                            {job.error && (
                              <Callout tone="danger" title={job.error.code}>
                                {job.error.message}
                                {job.error.hint && <div className="muted">dica: {job.error.hint}</div>}
                              </Callout>
                            )}
                            {open && hasDetails && (
                              <div className="term job-log">
                                {job.logs.map((line, i) => (
                                  <div key={i} className="term-line">
                                    <span className="term-time">{String(i + 1).padStart(3, "0")}</span>
                                    <span>{line}</span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
};
