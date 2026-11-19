export function formatBytes(bytes?: number | null): string {
  if (!bytes || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let idx = 0;
  let val = bytes;
  while (val >= 1024 && idx < units.length - 1) {
    val /= 1024;
    idx++;
  }
  return `${val.toFixed(val >= 100 || idx === 0 ? 0 : 1)} ${units[idx]}`;
}

export function formatDuration(seconds?: number | null): string {
  if (seconds === undefined || seconds === null || !Number.isFinite(seconds)) return "--:--";
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

export function formatSpeed(bps?: number | null): string {
  if (!bps || bps <= 0) return "";
  return `${formatBytes(bps)}/s`;
}

export function formatRelativeDate(iso: string): string {
  try {
    return new Intl.DateTimeFormat("pt-BR", {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}
