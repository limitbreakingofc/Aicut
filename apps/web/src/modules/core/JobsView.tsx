import React, { useState } from "react";
import { Bot, CheckCircle2, Clock, Trash2, User, XCircle } from "lucide-react";
import { runAction } from "../../lib/client";
import { formatBytes, formatDuration, formatRelativeDate, formatSpeed } from "../../lib/format";
import type { JobRecord } from "../../lib/types";

interface JobsViewProps {
  jobs: JobRecord[];
  onRefresh: () => void;
  onNotify: (level: "info" | "success" | "warning" | "error", title: string, message: string) => void;
}

export const JobsView: React.FC<JobsViewProps> = ({ jobs, onRefresh, onNotify }) => {
  const [expandedLogs, setExpandedLogs] = useState<Record<string, boolean>>({});

  async function handleCancel(jobId: string) {
    try {
      await runAction("jobs.cancel", { job_id: jobId });
      onNotify("info", "Tarefa cancelada", `Job ${jobId.slice(0, 8)} interrompido.`);
      onRefresh();
    } catch (e) {
      onNotify("error", "Erro ao cancelar", (e as Error).message);
    }
  }

  async function handleClearFinished() {
    try {
      const res = await runAction<{ removed_count: number }>("jobs.clear_finished");
      onNotify("info", "Fila limpa", `${res.data.removed_count} tarefa(s) finalizada(s) removida(s).`);
      onRefresh();
    } catch (e) {
      onNotify("error", "Erro ao limpar fila", (e as Error).message);
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Fila de Tarefas</h1>
          <p className="page-subtitle">
            Acompanhe downloads, importações e atualizações disparados por você ou por qualquer chat de IA em tempo real.
          </p>
        </div>
        <button type="button" className="btn" onClick={handleClearFinished}>
          <Trash2 size={14} /> Limpar Finalizadas
        </button>
      </div>

      {jobs.length === 0 ? (
        <div className="panel" style={{ textAlign: "center", padding: "44px 20px", color: "var(--text-secondary)" }}>
          Nenhuma tarefa na fila no momento.
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {jobs.map((job) => {
            const isActive = job.status === "queued" || job.status === "running";
            const statusClass =
              job.status === "completed"
                ? "success"
                : job.status === "failed"
                  ? "danger"
                  : job.status === "cancelled"
                    ? "warning"
                    : "accent";
            const showLogs = Boolean(expandedLogs[job.id]);

            return (
              <div key={job.id} className="panel" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
                    {job.status === "completed" ? (
                      <CheckCircle2 size={17} color="var(--success)" />
                    ) : job.status === "failed" ? (
                      <XCircle size={17} color="var(--danger)" />
                    ) : (
                      <Clock size={17} color="var(--accent)" />
                    )}
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {job.title}
                      </div>
                      <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                        {job.kind} · {formatRelativeDate(job.created_at)}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span className="pill">
                      {job.actor === "ai" ? <Bot size={11} /> : <User size={11} />}
                      {job.actor === "ai" ? "IA" : "Humano"}
                    </span>
                    <span className={`pill ${statusClass}`}>{job.status}</span>
                    {isActive && (
                      <button
                        type="button"
                        className="btn btn-danger btn-sm"
                        onClick={() => handleCancel(job.id)}
                      >
                        Cancelar
                      </button>
                    )}
                  </div>
                </div>

                <div className="progress-track">
                  <div
                    className={`progress-fill ${job.status === "completed" ? "success" : job.status === "failed" ? "danger" : ""}`}
                    style={{ width: `${Math.min(100, Math.max(0, job.progress.percent))}%` }}
                  />
                </div>

                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    justifyContent: "space-between",
                    fontSize: 12.5,
                    color: "var(--text-secondary)",
                    gap: 8,
                  }}
                >
                  <span>
                    {job.progress.percent.toFixed(1)}% · {job.progress.stage}
                    {job.progress.message ? ` — ${job.progress.message}` : ""}
                  </span>
                  <span style={{ fontFamily: "var(--font-mono)" }}>
                    {job.progress.downloaded_bytes ? formatBytes(job.progress.downloaded_bytes) : ""}
                    {job.progress.total_bytes ? ` / ${formatBytes(job.progress.total_bytes)}` : ""}
                    {job.progress.speed_bytes_per_sec ? ` · ${formatSpeed(job.progress.speed_bytes_per_sec)}` : ""}
                    {job.progress.eta_seconds ? ` · ETA ${formatDuration(job.progress.eta_seconds)}` : ""}
                  </span>
                </div>

                {job.error && (
                  <div
                    style={{
                      padding: "8px 11px",
                      borderRadius: "var(--radius-sm)",
                      background: "var(--danger-soft)",
                      color: "var(--danger)",
                      fontSize: 12.5,
                    }}
                  >
                    <strong>{job.error.code}:</strong> {job.error.message}
                    {job.error.hint && <div style={{ marginTop: 3, opacity: 0.9 }}>Dica: {job.error.hint}</div>}
                  </div>
                )}

                {job.logs.length > 0 && (
                  <div>
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm"
                      onClick={() => setExpandedLogs((p) => ({ ...p, [job.id]: !p[job.id] }))}
                    >
                      {showLogs ? "Ocultar logs" : `Ver logs (${job.logs.length})`}
                    </button>
                    {showLogs && (
                      <pre className="code-block" style={{ marginTop: 8, maxHeight: 180 }}>
                        {job.logs.join("\n")}
                      </pre>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
