import { useEffect, useState } from "react";
import { runAction } from "../../lib/client";
import type { JobRecord, ToolStatus, WorkspaceConfig, WhisperModelStatus, VisionModelStatus } from "../../lib/types";

type Model = (WhisperModelStatus | VisionModelStatus) & { module: "clips" | "transcribe" };
type Props = { jobs: JobRecord[]; tools: ToolStatus[]; workspace: WorkspaceConfig | null; refreshTools: () => Promise<void> };

export function DownloadsView({ jobs, tools, workspace, refreshTools }: Props) {
  const [group, setGroup] = useState("Ferramentas");
  const [installedOnly, setInstalledOnly] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const [face, setFace] = useState(false);
  const [noise, setNoise] = useState(false);
  const [tags, setTags] = useState<Record<string, string[]>>({});
  const [versions, setVersions] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("");

  async function reload() {
    const [speech, vision, audio, detector] = await Promise.all([
      runAction<{ models: WhisperModelStatus[] }>("transcribe.models_list"),
      runAction<{ models: VisionModelStatus[] }>("clips.models_list"),
      runAction<{ rnnoise_installed: boolean }>("audio.config.get"),
      runAction<boolean>("clips.face_status"),
      refreshTools(),
    ]);
    setModels([...speech.data.models.map(m => ({ ...m, module: "transcribe" as const })),
      ...vision.data.models.map(m => ({ ...m, module: "clips" as const }))]);
    setFace(detector.data);
    setNoise(audio.data.rnnoise_installed);
  }
  useEffect(() => { void reload().catch(e => setMessage(e.message)); }, []);

  async function execute(action: string, input: Record<string, unknown> = {}) {
    setBusy(action);
    setMessage("");
    try {
      const result = await runAction(action, input);
      if (action === "transcribe.model_delete") {
        try {
          if (localStorage.getItem("aicut_whisper_model") === input.model_id) localStorage.removeItem("aicut_whisper_model");
        } catch { /* Storage may be disabled in the webview. */ }
      }
      setMessage(result.summary);
    } catch (e) {
      setMessage((e as Error).message);
    } finally {
      try { await reload(); } catch (e) { setMessage((e as Error).message); }
      setBusy("");
    }
  }
  const visibleModels = models.filter(m => (group === "Voz" ? m.module === "transcribe" : m.module === "clips") && (!installedOnly || m.installed));
  const visibleTools = tools.filter(t => t.id !== "ffprobe" && (!installedOnly || t.installed));
  return <section className="panel stack">
    <div className="row wrap gap-sm">
      {["Ferramentas", "Voz", "Visão", "Áudio"].map(label => <button key={label} aria-pressed={group === label}
        className={group === label ? "btn btn-primary" : "btn"} onClick={() => setGroup(label)}>{label}</button>)}
      <label className="row gap-xs"><input type="checkbox" checked={installedOnly} onChange={e => setInstalledOnly(e.target.checked)} />Somente instalados</label>
    </div>
    <div className="row wrap gap-sm"><small className="secondary break-all">{group === "Ferramentas" ? workspace?.bin_dir : workspace?.models_dir}</small>
      {window.__TAURI_INTERNALS__?.invoke && <button className="btn btn-sm" onClick={() => window.__TAURI_INTERNALS__?.invoke?.("open_download_folder", {tools:group === "Ferramentas"}).catch(e => setMessage(String(e)))}>Abrir pasta</button>}</div>
    {group === "Ferramentas" && <button className="btn btn-sm" disabled={!!busy} onClick={() => execute("tools.check_updates")}>Verificar atualizações</button>}
    {group === "Ferramentas" && visibleTools.map(t => <div key={t.id} className="inset row row-between wrap gap-sm">
      <div><strong>{t.display_name}</strong><div className="secondary text-xs">{t.installed ? [t.managed_by_aicut ? "Aicut" : "Sistema", t.installed_version, t.binary_path].filter(Boolean).join(" · ") : "Disponível"}{t.update_available && ' · Atualização disponível'}</div><small>{t.recommended_build}</small></div>
      <div className="row wrap gap-xs"><datalist id={"versions-"+t.id}>{tags[t.id]?.map(tag => <option key={tag} value={tag} />)}</datalist><input list={"versions-"+t.id} onFocus={async () => {
        if (tags[t.id]) return;
        try {
          const res = await runAction<string[]>("tools.versions", {tool_id:t.id});
          setTags(prev => ({...prev, [t.id]:res.data}));
        } catch (e) { setMessage((e as Error).message); }
      }} className="input" style={{width:"12em"}} aria-label={'Tag de versão de '+t.display_name} placeholder="Tag (padrão: última)" value={versions[t.id] ?? ""} onChange={e => setVersions({...versions,[t.id]:e.target.value})}/>
      <button className="btn btn-sm" disabled={!!busy} onClick={() => execute("tools.install",{tool_id:t.id, version:versions[t.id]?.trim() || null})}>{t.managed_by_aicut ? "Atualizar / instalar versão" : "Instalar"}</button>
      {t.managed_by_aicut && <button className="btn btn-sm" disabled={!!busy} onClick={() => execute("tools.uninstall",{tool_id:t.id})}>Desinstalar</button>}</div>
    </div>)}
    {(group === "Voz" || group === "Visão") && visibleModels.map(m => <div key={m.id} className="inset row row-between wrap gap-sm">
      <div><strong>{m.display_name}</strong><div className="secondary text-xs">{[m.version_tag, m.size_mb+" MB", m.installed ? "Instalado" : "Disponível", m.recommended && "Recomendado para sua GPU"].filter(Boolean).join(" · ")}</div></div>
      <div className="row gap-xs">{m.installed && <button className="btn btn-sm" disabled={!!busy} onClick={() => execute(m.module+".model_download", {model_id:m.id})}>Atualizar</button>}
      <button className="btn btn-sm" disabled={!!busy} onClick={() => execute(m.module+(m.installed ? ".model_delete" : ".model_download"),{model_id:m.id,wait:true})}>{m.installed ? "Desinstalar" : "Instalar"}</button></div>
    </div>)}
    {group === "Visão" && (!installedOnly || face) && <div className="inset row row-between"><strong>UltraFace · ONNX</strong><button className="btn btn-sm" disabled={!!busy} onClick={() => execute(face ? "clips.face_remove" : "clips.face_install")}>{face ? "Desinstalar" : "Instalar"}</button></div>}
    {group === "Áudio" && (!installedOnly || noise) && <div className="inset row row-between"><strong>RNNoise · bd.rnnn</strong><button className="btn btn-sm" disabled={!!busy} onClick={() => execute(noise ? "audio.remove_rnnoise" : "audio.install_rnnoise")}>{noise ? "Desinstalar" : "Instalar"}</button></div>}
    {installedOnly && (group === "Áudio" ? !noise : group === "Ferramentas" ? !visibleTools.length : !visibleModels.length && !(group === "Visão" && face)) && <p>Nenhum item instalado nesta categoria.</p>}
    {jobs.filter(j => (j.kind === "tool.install" || j.kind.endsWith(".model_download")) && j.status === "running").map(j =>
      <div key={j.id} className="stack"><small>{j.title} · {Math.round(j.progress.percent || 0)}%</small><progress max={100} value={j.progress.percent || 0} /></div>)}
    <div role="status">{busy ? "Processando…" : message}</div>
  </section>;
}
