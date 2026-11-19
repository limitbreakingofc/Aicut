import React, { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Award,
  Check,
  CheckCircle2,
  Cloud,
  Cpu,
  Download,
  Eye,
  Film,
  HardDrive,
  KeyRound,
  Loader2,
  Play,
  Scissors,
  Sparkles,
  Subtitles,
  Trash2,
} from "lucide-react";
import { AicutActionError, runAction } from "../../lib/client";
import { formatBytes, formatDuration, formatRelativeDate } from "../../lib/format";
import type {
  ClipsConfig,
  HardwareProfile,
  JobRecord,
  MediaAsset,
  RouteId,
  SmartClipRun,
  ToolStatus,
  TranscriptRecord,
  VisionModelStatus,
} from "../../lib/types";

interface SmartClipsViewProps {
  media: MediaAsset[];
  tools: ToolStatus[];
  jobs: JobRecord[];
  hardware: HardwareProfile | null;
  preselectedMediaId?: string | null;
  onNavigate: (route: RouteId) => void;
  onSelectMedia: (mediaId: string) => void;
  onNotify: (level: "info" | "success" | "warning" | "error", title: string, message: string) => void;
  onRefreshTools: () => Promise<void>;
  onRefreshMedia: () => Promise<void>;
}

const DESCRIPTION_PRESETS = [
  {
    label: "Melhores momentos (Geral)",
    text: "Encontre os momentos mais envolventes, visualmente marcantes e surpreendentes do vídeo ou live, com gancho forte nos primeiros 5 segundos e fechamento completo.",
  },
  {
    label: "Jogadas / Clutches / Ação",
    text: "Procure jogadas épicas, clutches, viradas inesperadas, falhas engraçadas ou reações intensas na gameplay, mostrando a preparação, a jogada e a reação final.",
  },
  {
    label: "Explicações & Dicas Completas",
    text: "Encontre explicações didáticas, demonstrações na tela ou dicas valiosas que comecem com uma pergunta/problema claro e terminem com a solução completa.",
  },
  {
    label: "Humor / Reações / Cortes Virais",
    text: "Encontre momentos engraçados, piadas, sustos, rage divertido ou interações memoráveis com expressões faciais fortes e punchline fechada.",
  },
];

const SHORTS_LAYOUTS = [
  {
    id: "template_split",
    label: "Moldura Shorts (SMOKING33) · Câmera em Cima + Conteúdo em Baixo",
    desc: "Aplica sua moldura 9:16 com borda dourada: webcam na janela superior e conteúdo/gameplay na janela inferior.",
  },
  {
    id: "gaming_split",
    label: "Split 9:16 Sem Moldura · Câmera em Cima + Conteúdo em Baixo",
    desc: "Webcam cheia na parte superior e gameplay/conteúdo na parte inferior, sem moldura.",
  },
  {
    id: "fit_blur",
    label: "Shorts 9:16 · Tela Inteira + Blur",
    desc: "Mantém o vídeo 16:9 sem cortar bordas sobre fundo desfocado (ideal para tutoriais e tela única).",
  },
  {
    id: "center_crop",
    label: "Shorts 9:16 · Foco Central",
    desc: "Enquadramento vertical cheio focado no centro (ideal para vlogs, podcasts e câmera principal).",
  },
  {
    id: "original",
    label: "Original 16:9 (Horizontal)",
    desc: "Mantém a proporção original do vídeo sem crop vertical.",
  },
] as const;

export const SmartClipsView: React.FC<SmartClipsViewProps> = ({
  media,
  tools,
  jobs,
  hardware,
  preselectedMediaId,
  onNavigate,
  onSelectMedia,
  onNotify,
  onRefreshTools,
  onRefreshMedia,
}) => {
  const videoMedia = useMemo(
    () => media.filter((m) => m.kind === "video" || m.kind === "video_only"),
    [media],
  );

  const [selectedMediaId, setSelectedMediaId] = useState<string>(
    preselectedMediaId ?? videoMedia[0]?.id ?? "",
  );
  const [provider, setProvider] = useState<"gemini" | "local_free">("gemini");
  const [description, setDescription] = useState<string>(DESCRIPTION_PRESETS[0].text);
  const [maxDurationSec, setMaxDurationSec] = useState<number>(180);
  const [maxClips, setMaxClips] = useState<number>(5);
  const [allowJumpCuts, setAllowJumpCuts] = useState<boolean>(false);

  // Provider configuration
  const [config, setConfig] = useState<ClipsConfig | null>(null);
  const [geminiKeyInput, setGeminiKeyInput] = useState<string>("");
  const [geminiDirectorModel, setGeminiDirectorModel] = useState<string>("gemini-2.5-pro");
  const [savingConfig, setSavingConfig] = useState<boolean>(false);

  // Local Free vision models
  const [visionModels, setVisionModels] = useState<VisionModelStatus[]>([]);
  const [totalVisionBytes, setTotalVisionBytes] = useState<number>(0);
  const [selectedLocalModel, setSelectedLocalModel] = useState<string>("");
  const [installingLlama, setInstallingLlama] = useState<boolean>(false);
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [uninstallingModel, setUninstallingModel] = useState<string | null>(null);

  // Runs & transcripts
  const [runs, setRuns] = useState<SmartClipRun[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [transcripts, setTranscripts] = useState<TranscriptRecord[]>([]);

  // Shorts rendering options
  const [layout, setLayout] = useState<
    "template_split" | "gaming_split" | "fit_blur" | "center_crop" | "original"
  >("template_split");
  const [webcamCorner, setWebcamCorner] = useState<
    "bottom_right" | "bottom_left" | "top_right" | "top_left" | "center"
  >("bottom_right");
  const [contentFit, setContentFit] = useState<"fill" | "fit_blur">("fill");
  const [customTemplatePath, setCustomTemplatePath] = useState<string>("");
  const [burnSubtitles, setBurnSubtitles] = useState<boolean>(true);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [renderingKey, setRenderingKey] = useState<string | null>(null);

  const ffmpegTool = tools.find((t) => t.id === "ffmpeg");
  const llamaTool = tools.find((t) => t.id === "llama-server");

  useEffect(() => {
    if (preselectedMediaId) {
      setSelectedMediaId(preselectedMediaId);
    } else if (!selectedMediaId && videoMedia.length > 0) {
      setSelectedMediaId(videoMedia[0].id);
    }
  }, [preselectedMediaId, videoMedia.length]);

  const loadAll = async () => {
    try {
      const [cfgRes, modelsRes, runsRes, trRes] = await Promise.all([
        runAction<ClipsConfig>("clips.config_get"),
        runAction<{
          models: VisionModelStatus[];
          recommended_model_id: string;
          total_installed_bytes: number;
        }>("clips.models_list"),
        runAction<{ runs: SmartClipRun[] }>("clips.runs_list"),
        runAction<{ transcripts: TranscriptRecord[] }>("transcribe.list"),
      ]);

      setConfig(cfgRes.data);
      setCustomTemplatePath(cfgRes.data.template_path ?? "");
      if (cfgRes.data.default_provider === "local_free") {
        setProvider("local_free");
      } else {
        setProvider("gemini");
      }
      setGeminiDirectorModel(cfgRes.data.gemini_director_model || "gemini-2.5-pro");

      const mList = modelsRes.data.models ?? [];
      setVisionModels(mList);
      setTotalVisionBytes(modelsRes.data.total_installed_bytes ?? 0);
      setSelectedLocalModel((curr) => {
        if (curr && mList.some((m) => m.id === curr && m.installed)) return curr;
        const firstInst = mList.find((m) => m.installed);
        if (firstInst) return firstInst.id;
        return cfgRes.data.local_vision_model || modelsRes.data.recommended_model_id || "";
      });

      const rList = runsRes.data.runs ?? [];
      setRuns(rList);
      if (rList.length > 0 && !activeRunId) {
        setActiveRunId(rList[0].id);
      }

      setTranscripts(trRes.data.transcripts ?? []);
    } catch {
      // ignore if backend offline
    }
  };

  useEffect(() => {
    void loadAll();
  }, []);

  const activeClipJobs = jobs.filter(
    (j) =>
      (j.kind === "clips.analyze" ||
        j.kind === "clips.render" ||
        j.kind === "clips.model_download") &&
      (j.status === "queued" || j.status === "running"),
  );
  const completedClipJobsCount = jobs.filter(
    (j) =>
      (j.kind === "clips.analyze" ||
        j.kind === "clips.render" ||
        j.kind === "clips.model_download") &&
      j.status === "completed",
  ).length;

  useEffect(() => {
    if (completedClipJobsCount > 0) {
      void loadAll();
      void onRefreshMedia();
    }
  }, [completedClipJobsCount]);

  const selectedTranscript = useMemo(
    () => transcripts.find((t) => t.media_id === selectedMediaId) ?? null,
    [transcripts, selectedMediaId],
  );

  const activeRun = useMemo(
    () => runs.find((r) => r.id === activeRunId) ?? runs[0] ?? null,
    [runs, activeRunId],
  );

  const saveGeminiSettings = async () => {
    setSavingConfig(true);
    try {
      const payload: Record<string, unknown> = {
        default_provider: provider,
        gemini_director_model: geminiDirectorModel,
      };
      if (geminiKeyInput.trim()) {
        payload.gemini_api_key = geminiKeyInput.trim();
      }
      const res = await runAction<ClipsConfig>("clips.config_set", payload);
      setConfig(res.data);
      setGeminiKeyInput("");
      onNotify("success", "Configuração salva", "Preferências da Gemini API atualizadas.");
    } catch (e) {
      onNotify(
        "error",
        "Erro ao salvar",
        e instanceof AicutActionError ? e.message : "Falha ao salvar configuração.",
      );
    } finally {
      setSavingConfig(false);
    }
  };

  const installLlamaServer = async () => {
    setInstallingLlama(true);
    try {
      await runAction("tools.install", { tool_id: "llama-server" });
      await onRefreshTools();
      onNotify(
        "success",
        "llama.cpp instalado",
        "Runtime visual local pronto na melhor build para sua GPU/hardware.",
      );
    } catch (e) {
      onNotify(
        "error",
        "Erro ao instalar llama.cpp",
        e instanceof AicutActionError ? e.message : "Falha na instalação.",
      );
    } finally {
      setInstallingLlama(false);
    }
  };

  const handleDownloadModel = async (modelId: string) => {
    setDownloadingModel(modelId);
    try {
      await runAction("clips.model_download", { model_id: modelId });
      setSelectedLocalModel(modelId);
      await runAction("clips.config_set", { local_vision_model: modelId }).catch(() => {});
      await loadAll();
      onNotify("success", "Modelo visual pronto", `Modelo ${modelId} baixado com sucesso.`);
    } catch (e) {
      onNotify(
        "error",
        "Erro no download",
        e instanceof AicutActionError ? e.message : "Falha ao baixar modelo visual.",
      );
    } finally {
      setDownloadingModel(null);
    }
  };

  const handleDeleteModel = async (modelId: string) => {
    setUninstallingModel(modelId);
    try {
      const res = await runAction<{ freed_bytes: number }>("clips.model_delete", {
        model_id: modelId,
      });
      await loadAll();
      onNotify(
        "info",
        "Modelo desinstalado",
        `Espaço liberado: ${formatBytes(res.data.freed_bytes ?? 0)}.`,
      );
    } catch (e) {
      onNotify(
        "error",
        "Erro ao desinstalar",
        e instanceof AicutActionError ? e.message : "Falha ao remover modelo.",
      );
    } finally {
      setUninstallingModel(null);
    }
  };

  const handleStartAnalyze = async () => {
    if (!selectedMediaId) {
      onNotify("warning", "Selecione um vídeo", "Escolha um vídeo ou live da Biblioteca primeiro.");
      return;
    }
    if (!description.trim()) {
      onNotify(
        "warning",
        "Descreva o que procurar",
        "Informe uma descrição editorial para guiar a IA visual.",
      );
      return;
    }
    setAnalyzing(true);
    try {
      await runAction("clips.config_set", {
        default_provider: provider,
        gemini_director_model: geminiDirectorModel,
        local_vision_model: selectedLocalModel,
      }).catch(() => {});

      await runAction("clips.analyze", {
        media_id: selectedMediaId,
        description: description.trim(),
        provider,
        model_id: provider === "local_free" ? selectedLocalModel : geminiDirectorModel,
        max_duration_sec: maxDurationSec,
        min_duration_sec: 25,
        max_clips: maxClips,
        allow_jump_cuts: allowJumpCuts,
        wait: false,
      });
      onNotify(
        "info",
        "Análise visual iniciada",
        "A IA está assistindo ao vídeo para encontrar cortes coerentes de até 3 min. Acompanhe o progresso abaixo.",
      );
    } catch (e) {
      onNotify(
        "error",
        "Não foi possível iniciar",
        e instanceof AicutActionError ? e.message : "Verifique os pré-requisitos do motor escolhido.",
      );
    } finally {
      setAnalyzing(false);
    }
  };

  const handleRender = async (runId: string, clipId?: string) => {
    const key = clipId ?? `all:${runId}`;
    setRenderingKey(key);
    try {
      await runAction("clips.render", {
        run_id: runId,
        clip_id: clipId,
        layout,
        webcam_corner: webcamCorner,
        content_fit: contentFit,
        template_path: customTemplatePath.trim() ? customTemplatePath.trim() : undefined,
        burn_subtitles: burnSubtitles,
        wait: false,
      });
      onNotify(
        "info",
        "Renderização Shorts iniciada",
        "Os cortes serão salvos automaticamente na Biblioteca assim que o FFmpeg concluir.",
      );
    } catch (e) {
      onNotify(
        "error",
        "Erro ao renderizar",
        e instanceof AicutActionError ? e.message : "Falha ao iniciar renderização.",
      );
    } finally {
      setRenderingKey(null);
    }
  };

  const handleDeleteRun = async (runId: string) => {
    try {
      await runAction("clips.run_delete", { id: runId });
      if (activeRunId === runId) setActiveRunId(null);
      await loadAll();
    } catch {
      // ignore
    }
  };

  const primaryGpu = hardware?.gpus?.[0];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Scissors size={20} style={{ color: "var(--accent)" }} />
            Cortes Shorts IA (Visual + Narrativa até 3 min)
          </h1>
          <p className="page-subtitle">
            IA visual que assiste vídeos e lives, segue sua descrição e cria cortes coerentes com{" "}
            <strong>Início, Meio e Fim</strong> em até 3 minutos — via <strong>Gemini API</strong> ou{" "}
            <strong>100% Local Grátis (Qwen2.5-VL)</strong> na sua GPU.
          </p>
        </div>

        <button type="button" className="btn btn-sm" onClick={() => onNavigate("ingest")}>
          <Download size={14} /> Baixar YouTube / Twitch / PC
        </button>
      </div>

      {/* Hardware & Dependency banner */}
      {!ffmpegTool?.installed && (
        <div
          className="card"
          style={{
            borderColor: "var(--warning)",
            background: "var(--warning-soft)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <AlertTriangle size={18} style={{ color: "var(--warning)" }} />
            <span>
              O <strong>FFmpeg</strong> precisa estar instalado para extrair frames e renderizar Shorts.
            </span>
          </div>
          <button type="button" className="btn btn-primary btn-sm" onClick={() => onNavigate("settings")}>
            Instalar em Ferramentas
          </button>
        </div>
      )}

      {/* Active Progress Jobs */}
      {activeClipJobs.length > 0 && (
        <div className="card" style={{ borderColor: "var(--accent-border)", background: "var(--accent-soft)" }}>
          <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 8, display: "flex", alignItems: "center", gap: 8 }}>
            <Loader2 size={15} className="spin" /> Tarefas de Cortes IA em andamento
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {activeClipJobs.map((j) => (
              <div key={j.id} style={{ fontSize: 12.5 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span>{j.title}</span>
                  <strong>{Math.round(j.progress.percent)}%</strong>
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${Math.min(100, j.progress.percent)}%` }} />
                </div>
                {j.progress.message && (
                  <div style={{ fontSize: 11.5, color: "var(--text-secondary)", marginTop: 3 }}>
                    {j.progress.message}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main 2-column Studio Configuration */}
      <div style={{ display: "grid", gridTemplateColumns: "1.15fr 1fr", gap: 16, alignItems: "start" }}>
        {/* Left: Video + Editorial Description + Run Button */}
        <div className="card" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ fontWeight: 600, fontSize: 14, display: "flex", alignItems: "center", gap: 8 }}>
            <Film size={16} style={{ color: "var(--accent)" }} /> 1. Vídeo ou Live & Descrição Editorial
          </div>

          <div className="field">
            <label>Vídeo / Live da Biblioteca</label>
            {videoMedia.length === 0 ? (
              <div style={{ fontSize: 12.5, color: "var(--text-secondary)", padding: 10, background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)" }}>
                Nenhum vídeo na Biblioteca ainda. Baixe do YouTube/Twitch ou importe um arquivo local na aba{" "}
                <button type="button" className="btn btn-ghost btn-sm" onClick={() => onNavigate("ingest")}>
                  Baixar & Importar
                </button>
              </div>
            ) : (
              <select
                className="select"
                value={selectedMediaId}
                onChange={(e) => setSelectedMediaId(e.target.value)}
              >
                {videoMedia.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.title}
                    {m.metadata.duration_sec ? ` (${formatDuration(m.metadata.duration_sec)})` : ""}
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Transcript anchor status */}
          {selectedMediaId && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 10,
                padding: "8px 10px",
                borderRadius: "var(--radius-sm)",
                background: selectedTranscript ? "var(--success-soft)" : "var(--bg-elevated)",
                fontSize: 12,
              }}
            >
              <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <Subtitles size={14} style={{ color: selectedTranscript ? "var(--success)" : "var(--text-muted)" }} />
                {selectedTranscript
                  ? `Transcrição vinculada (${selectedTranscript.segments.length} falas): cortes alinhados por palavra e legendas Karaokê prontas.`
                  : "Dica: transcreva este vídeo antes na aba Transcrever IA para alinhar cortes exatos por palavra e gerar legendas Karaokê."}
              </span>
              {!selectedTranscript && (
                <button type="button" className="btn btn-sm" onClick={() => onNavigate("transcribe")}>
                  Transcrever
                </button>
              )}
            </div>
          )}

          <div className="field">
            <label>O que a IA visual deve procurar nos cortes?</label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 6 }}>
              {DESCRIPTION_PRESETS.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  className="btn btn-sm"
                  style={{ fontSize: 11.5 }}
                  onClick={() => setDescription(p.text)}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <textarea
              className="textarea"
              rows={4}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Ex.: Ache os momentos mais engraçados da live, jogadas absurdas com reação na webcam ou explicações completas que façam sentido do início ao fim..."
            />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <div className="field">
              <label>Duração máxima (Shorts até 3 min)</label>
              <select
                className="select"
                value={maxDurationSec}
                onChange={(e) => setMaxDurationSec(Number(e.target.value))}
              >
                <option value={60}>Até 1 minuto (60s)</option>
                <option value={90}>Até 1m30s (90s)</option>
                <option value={120}>Até 2 minutos (120s)</option>
                <option value={180}>Até 3 minutos (180s · Recomendado)</option>
              </select>
            </div>
            <div className="field">
              <label>Quantidade máxima de cortes</label>
              <select
                className="select"
                value={maxClips}
                onChange={(e) => setMaxClips(Number(e.target.value))}
              >
                <option value={3}>Até 3 cortes</option>
                <option value={5}>Até 5 cortes</option>
                <option value={8}>Até 8 cortes</option>
                <option value={10}>Até 10 cortes</option>
              </select>
            </div>
          </div>

          <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={allowJumpCuts}
              onChange={(e) => setAllowJumpCuts(e.target.checked)}
            />
            Permitir jump-cuts internos leves (pular pausas longas/tempos mortos mantendo início–meio–fim em até {maxDurationSec}s)
          </label>

          <button
            type="button"
            className="btn btn-primary"
            disabled={analyzing || !selectedMediaId}
            onClick={handleStartAnalyze}
            style={{ justifyContent: "center", padding: "10px 16px" }}
          >
            {analyzing ? <Loader2 size={16} className="spin" /> : <Sparkles size={16} />}
            Analisar Vídeo Visualmente e Gerar Planos de Cortes
          </button>
        </div>

        {/* Right: Dual-Engine Visual AI Selector (Gemini Cloud vs Local Free) */}
        <div className="card" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ fontWeight: 600, fontSize: 14, display: "flex", alignItems: "center", gap: 8 }}>
            <Eye size={16} style={{ color: "var(--accent)" }} /> 2. Motor de IA Visual
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            <button
              type="button"
              className={`btn ${provider === "gemini" ? "btn-primary" : ""}`}
              style={{ justifyContent: "center" }}
              onClick={() => setProvider("gemini")}
            >
              <Cloud size={15} /> Gemini API (Nativo Vídeo)
            </button>
            <button
              type="button"
              className={`btn ${provider === "local_free" ? "btn-primary" : ""}`}
              style={{ justifyContent: "center" }}
              onClick={() => setProvider("local_free")}
            >
              <Cpu size={15} /> Local Free (Offline GPU)
            </button>
          </div>

          {provider === "gemini" ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div
                style={{
                  fontSize: 12.5,
                  color: "var(--text-secondary)",
                  background: "var(--bg-elevated)",
                  padding: 10,
                  borderRadius: "var(--radius-sm)",
                }}
              >
                <strong>Visão + Áudio Nativos:</strong> o Gemini assiste diretamente aos blocos de vídeo (25 min com 3 min de sobreposição) usando <strong>Gemini 2.5 Flash</strong> na varredura e <strong>{geminiDirectorModel}</strong> na direção final. 0 MB de download local.
              </div>

              <div className="field">
                <label style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span>Chave Gemini API</span>
                  {config?.gemini_configured && (
                    <span style={{ color: "var(--success)", fontSize: 11.5, display: "inline-flex", alignItems: "center", gap: 4 }}>
                      <CheckCircle2 size={12} /> Configurada ({config.gemini_key_preview})
                    </span>
                  )}
                </label>
                <div style={{ display: "flex", gap: 8 }}>
                  <input
                    type="password"
                    className="input"
                    value={geminiKeyInput}
                    onChange={(e) => setGeminiKeyInput(e.target.value)}
                    placeholder={
                      config?.gemini_configured
                        ? "Cole uma nova chave apenas se quiser trocar..."
                        : "AIza... (obtenha grátis no Google AI Studio)"
                    }
                  />
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    disabled={savingConfig}
                    onClick={saveGeminiSettings}
                  >
                    <KeyRound size={14} /> Salvar
                  </button>
                </div>
              </div>

              <div className="field">
                <label>Modelo Diretor (Passagem B — Coerência Início–Meio–Fim)</label>
                <select
                  className="select"
                  value={geminiDirectorModel}
                  onChange={(e) => setGeminiDirectorModel(e.target.value)}
                >
                  <option value="gemini-2.5-pro">Gemini 2.5 Pro (Máxima inteligência visual e narrativa)</option>
                  <option value="gemini-2.5-flash">Gemini 2.5 Flash (Mais rápido e econômico)</option>
                </select>
              </div>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {/* Hardware summary */}
              <div
                style={{
                  fontSize: 12,
                  color: "var(--text-secondary)",
                  background: "var(--bg-elevated)",
                  padding: 10,
                  borderRadius: "var(--radius-sm)",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <span>
                  <strong>Hardware:</strong>{" "}
                  {primaryGpu
                    ? `${primaryGpu.name}${primaryGpu.vram_mb ? ` (${Math.round(primaryGpu.vram_mb / 1024)} GB VRAM)` : ""}`
                    : "Modo CPU"}
                  {hardware?.cuda_version ? ` · CUDA ${hardware.cuda_version}` : ""}
                </span>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  <HardDrive size={12} /> {formatBytes(totalVisionBytes)} em disco
                </span>
              </div>

              {/* Embedded llama-server status */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 10,
                  padding: 10,
                  borderRadius: "var(--radius-sm)",
                  border: "1px solid var(--border-default)",
                  background: llamaTool?.installed ? "var(--success-soft)" : "var(--bg-input)",
                }}
              >
                <div style={{ fontSize: 12 }}>
                  <div style={{ fontWeight: 600 }}>
                    Motor llama.cpp Server {llamaTool?.installed ? "· Pronto" : "· Não instalado"}
                  </div>
                  <div style={{ color: "var(--text-secondary)", marginTop: 2 }}>
                    {llamaTool?.recommended_build ?? "Detecta automaticamente CUDA / Vulkan / CPU"}
                  </div>
                </div>
                <button
                  type="button"
                  className={`btn btn-sm ${llamaTool?.installed ? "" : "btn-primary"}`}
                  disabled={installingLlama}
                  onClick={installLlamaServer}
                >
                  {installingLlama ? <Loader2 size={13} className="spin" /> : <Download size={13} />}
                  {llamaTool?.installed ? "Atualizar" : "Instalar Auto"}
                </button>
              </div>

              {/* Curated Local Vision Models */}
              <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 280, overflowY: "auto" }}>
                {visionModels.map((m) => {
                  const isActive = selectedLocalModel === m.id;
                  return (
                    <div
                      key={m.id}
                      style={{
                        padding: 10,
                        borderRadius: "var(--radius-sm)",
                        border: `1px solid ${isActive ? "var(--accent)" : "var(--border-subtle)"}`,
                        background: isActive ? "var(--accent-soft)" : "var(--bg-elevated)",
                        display: "flex",
                        flexDirection: "column",
                        gap: 6,
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
                        <div style={{ fontWeight: 600, fontSize: 12.5, display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                          <span>{m.display_name}</span>
                          {m.recommended && (
                            <span className="nav-badge accent" style={{ fontSize: 10.5 }}>
                              <Award size={10} /> Recomendado p/ sua GPU
                            </span>
                          )}
                          <span className="nav-badge" style={{ fontSize: 10.5 }}>{m.quality_tier}</span>
                        </div>
                        <span style={{ fontSize: 11.5, color: "var(--text-muted)" }}>
                          {(m.size_mb / 1024).toFixed(1)} GB
                        </span>
                      </div>

                      <div style={{ fontSize: 11.5, color: "var(--text-secondary)" }}>{m.quality_notes}</div>
                      <div style={{ fontSize: 11, color: m.compatible ? "var(--success)" : "var(--warning)" }}>
                        {m.compatibility_reason}
                      </div>

                      <div style={{ display: "flex", gap: 6, marginTop: 2 }}>
                        {m.installed ? (
                          <>
                            <button
                              type="button"
                              className={`btn btn-sm ${isActive ? "btn-primary" : ""}`}
                              onClick={() => setSelectedLocalModel(m.id)}
                            >
                              {isActive ? <Check size={13} /> : <Play size={13} />}
                              {isActive ? "Ativo" : "Usar este"}
                            </button>
                            <button
                              type="button"
                              className="btn btn-sm"
                              disabled={uninstallingModel === m.id}
                              onClick={() => handleDeleteModel(m.id)}
                            >
                              {uninstallingModel === m.id ? (
                                <Loader2 size={13} className="spin" />
                              ) : (
                                <Trash2 size={13} />
                              )}
                              Desinstalar
                            </button>
                          </>
                        ) : (
                          <button
                            type="button"
                            className="btn btn-primary btn-sm"
                            disabled={downloadingModel !== null}
                            onClick={() => handleDownloadModel(m.id)}
                          >
                            {downloadingModel === m.id ? (
                              <Loader2 size={13} className="spin" />
                            ) : (
                              <Download size={13} />
                            )}
                            Baixar ({(m.size_mb / 1024).toFixed(1)} GB)
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Results & Shorts 9:16 Render Panel */}
      <div className="card" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
          <div style={{ fontWeight: 600, fontSize: 14, display: "flex", alignItems: "center", gap: 8 }}>
            <Sparkles size={16} style={{ color: "var(--accent)" }} />
            3. Cortes Coerentes Encontrados & Renderização Shorts (9:16)
          </div>

          {runs.length > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <select
                className="select"
                style={{ width: "auto", minWidth: 260 }}
                value={activeRun?.id ?? ""}
                onChange={(e) => setActiveRunId(e.target.value)}
              >
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.media_title} · {r.clips.length} corte(s) · {r.provider} ({formatRelativeDate(r.created_at)})
                  </option>
                ))}
              </select>
              {activeRun && (
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => handleDeleteRun(activeRun.id)}
                  title="Excluir esta análise"
                >
                  <Trash2 size={14} />
                </button>
              )}
            </div>
          )}
        </div>

        {/* Shorts render controls */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 10,
            padding: 12,
            borderRadius: "var(--radius-md)",
            background: "var(--bg-elevated)",
          }}
        >
          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                layout === "template_split" || layout === "gaming_split"
                  ? "1.4fr 1fr 1fr auto"
                  : "1.6fr auto auto",
              gap: 10,
              alignItems: "end",
            }}
          >
            <div className="field" style={{ margin: 0 }}>
              <label>Enquadramento Shorts</label>
              <select
                className="select"
                value={layout}
                onChange={(e) =>
                  setLayout(
                    e.target.value as
                      | "template_split"
                      | "gaming_split"
                      | "fit_blur"
                      | "center_crop"
                      | "original",
                  )
                }
              >
                {SHORTS_LAYOUTS.map((l) => (
                  <option key={l.id} value={l.id}>
                    {l.label}
                  </option>
                ))}
              </select>
            </div>

            {(layout === "template_split" || layout === "gaming_split") && (
              <>
                <div className="field" style={{ margin: 0 }}>
                  <label>Origem da Câmera (Janela de Cima)</label>
                  <select
                    className="select"
                    value={webcamCorner}
                    onChange={(e) =>
                      setWebcamCorner(
                        e.target.value as
                          | "bottom_right"
                          | "bottom_left"
                          | "top_right"
                          | "top_left"
                          | "center",
                      )
                    }
                  >
                    <option value="bottom_right">Canto inferior direito</option>
                    <option value="bottom_left">Canto inferior esquerdo</option>
                    <option value="top_right">Canto superior direito</option>
                    <option value="top_left">Canto superior esquerdo</option>
                    <option value="center">Centro (vídeo sem webcam no canto)</option>
                  </select>
                </div>

                <div className="field" style={{ margin: 0 }}>
                  <label>Conteúdo (Janela de Baixo)</label>
                  <select
                    className="select"
                    value={contentFit}
                    onChange={(e) => setContentFit(e.target.value as "fill" | "fit_blur")}
                  >
                    <option value="fill">Preencher janela (sem bordas)</option>
                    <option value="fit_blur">Tela 16:9 inteira + Blur</option>
                  </select>
                </div>
              </>
            )}

            {activeRun && activeRun.clips.length > 0 && (
              <button
                type="button"
                className="btn btn-primary"
                disabled={renderingKey !== null}
                onClick={() => handleRender(activeRun.id)}
              >
                {renderingKey === `all:${activeRun.id}` ? (
                  <Loader2 size={14} className="spin" />
                ) : (
                  <Scissors size={14} />
                )}
                Renderizar Todos ({activeRun.clips.length})
              </button>
            )}
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap",
              gap: 10,
            }}
          >
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12.5, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={burnSubtitles}
                onChange={(e) => setBurnSubtitles(e.target.checked)}
              />
              Queimar legendas Karaokê sincronizadas (se transcrito)
            </label>

            {layout === "template_split" && (
              <div style={{ display: "flex", alignItems: "center", gap: 6, flex: 1, minWidth: 280, maxWidth: 520 }}>
                <input
                  type="text"
                  className="input"
                  style={{ fontSize: 12, padding: "5px 8px" }}
                  value={customTemplatePath}
                  onChange={(e) => setCustomTemplatePath(e.target.value)}
                  placeholder="Moldura padrão SMOKING33 ativa (ou cole caminho de outro PNG 9:16 transparente)"
                />
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => {
                    void runAction("clips.config_set", {
                      template_path: customTemplatePath.trim(),
                    })
                      .then(() =>
                        onNotify(
                          "success",
                          "Moldura salva",
                          customTemplatePath.trim()
                            ? "Moldura PNG personalizada salva como padrão."
                            : "Moldura padrão SMOKING33 restaurada.",
                        ),
                      )
                      .catch(() => {});
                  }}
                >
                  Salvar PNG
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Clip Cards */}
        {!activeRun || activeRun.clips.length === 0 ? (
          <div style={{ padding: 24, textAlign: "center", color: "var(--text-secondary)", fontSize: 13 }}>
            Nenhuma análise concluída ainda. Selecione um vídeo acima, descreva o que procurar e clique em{" "}
            <strong>Analisar Vídeo Visualmente</strong>.
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {activeRun.clips.map((clip, i) => (
              <div
                key={clip.id}
                style={{
                  padding: 14,
                  borderRadius: "var(--radius-md)",
                  border: "1px solid var(--border-default)",
                  background: "var(--bg-input)",
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 14.5, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                      <span>
                        #{i + 1} · {clip.title}
                      </span>
                      <span className="nav-badge accent">{formatDuration(clip.total_duration_sec)}</span>
                      <span className="nav-badge">Aderência {clip.scores.match_description}%</span>
                      <span className="nav-badge">Visual {clip.scores.visual_interest}%</span>
                      <span className="nav-badge">Coerência {clip.scores.narrative_coherence}%</span>
                    </div>
                    <div style={{ fontSize: 12, fontFamily: "var(--font-mono)", color: "var(--text-muted)", marginTop: 4 }}>
                      {clip.segments
                        .map((s) => `${formatDuration(s.start_sec)} → ${formatDuration(s.end_sec)}`)
                        .join("  +  ")}
                    </div>
                  </div>

                  <div style={{ display: "flex", gap: 8 }}>
                    {clip.rendered_media_id && (
                      <button
                        type="button"
                        className="btn btn-sm"
                        onClick={() => onSelectMedia(clip.rendered_media_id!)}
                      >
                        <Play size={13} /> Abrir na Biblioteca
                      </button>
                    )}
                    <button
                      type="button"
                      className="btn btn-primary btn-sm"
                      disabled={renderingKey !== null}
                      onClick={() => handleRender(activeRun.id, clip.id)}
                    >
                      {renderingKey === clip.id ? (
                        <Loader2 size={13} className="spin" />
                      ) : (
                        <Scissors size={13} />
                      )}
                      {clip.rendered_media_id ? "Re-renderizar Short" : "Renderizar Short"}
                    </button>
                  </div>
                </div>

                {clip.visual_evidence && (
                  <div style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
                    <strong>👁️ O que a IA viu na tela:</strong> {clip.visual_evidence}
                  </div>
                )}

                {clip.hook_first_5s && (
                  <div style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
                    <strong>⚡ Gancho (5s):</strong> {clip.hook_first_5s}
                  </div>
                )}

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(3, 1fr)",
                    gap: 8,
                    fontSize: 12,
                  }}
                >
                  <div style={{ padding: 8, borderRadius: "var(--radius-sm)", background: "var(--bg-elevated)" }}>
                    <div style={{ fontWeight: 600, color: "var(--success)", marginBottom: 3 }}>1. Início</div>
                    <div style={{ color: "var(--text-secondary)" }}>{clip.narrative_arc.beginning || "—"}</div>
                  </div>
                  <div style={{ padding: 8, borderRadius: "var(--radius-sm)", background: "var(--bg-elevated)" }}>
                    <div style={{ fontWeight: 600, color: "var(--warning)", marginBottom: 3 }}>2. Meio</div>
                    <div style={{ color: "var(--text-secondary)" }}>{clip.narrative_arc.middle || "—"}</div>
                  </div>
                  <div style={{ padding: 8, borderRadius: "var(--radius-sm)", background: "var(--bg-elevated)" }}>
                    <div style={{ fontWeight: 600, color: "var(--accent)", marginBottom: 3 }}>3. Fim</div>
                    <div style={{ color: "var(--text-secondary)" }}>{clip.narrative_arc.end || "—"}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
