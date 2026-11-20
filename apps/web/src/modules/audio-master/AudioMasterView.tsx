import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  CheckCircle2,
  Download,
  Play,
  SlidersHorizontal,
  Sparkles,
  Volume2,
  Waves,
} from "lucide-react";
import { AicutActionError, runAction } from "../../lib/client";
import { formatDuration } from "../../lib/format";
import type {
  AudioConfigOutput,
  AudioEnhanceResult,
  AudioMasterConfig,
  JobRecord,
  MediaAsset,
  NotifyFn,
  RouteId,
  ToolStatus,
} from "../../lib/types";
import {
  ActiveJobsPanel,
  Badge,
  BusyIcon,
  Callout,
  Checkbox,
  ChoiceCards,
  EmptyState,
  Field,
  Panel,
} from "../../components";

interface AudioMasterViewProps {
  media: MediaAsset[];
  tools: ToolStatus[];
  jobs: JobRecord[];
  onNavigate: (route: RouteId) => void;
  onSelectMedia: (mediaId: string) => void;
  onNotify: NotifyFn;
  onRefreshMedia: () => Promise<void>;
}

const PRESET_OPTIONS: {
  id: AudioMasterConfig["preset"];
  title: string;
  description: string;
}[] = [
  {
    id: "youtube_auto",
    title: "YouTube / Shorts (-14 LUFS)",
    description: "Padrão oficial YouTube (-14 LUFS · -1.0 dBTP), corte de ruído, clareza vocal e compressão automática.",
  },
  {
    id: "reels_tiktok",
    title: "Reels / TikTok (-14 LUFS Impacto)",
    description: "Dinâmica mais compacta (LRA 7) para voz sempre forte e presente em alto-falantes de celular.",
  },
  {
    id: "podcast_voice",
    title: "Podcast / Conversa (-16 LUFS)",
    description: "Compressão natural suave (-16 LUFS · -1.5 dBTP) ideal para cortes longos e entrevistas.",
  },
  {
    id: "off",
    title: "Desligado (Áudio Original)",
    description: "Mantém o áudio exatamente como gravado, sem normalização nem compressão.",
  },
];

export const AudioMasterView: React.FC<AudioMasterViewProps> = ({
  media,
  tools,
  jobs,
  onNavigate,
  onSelectMedia,
  onNotify,
  onRefreshMedia,
}) => {
  const playableMedia = useMemo(
    () => media.filter((m) => m.kind === "video" || m.kind === "audio" || m.kind === "video_only"),
    [media],
  );

  const [audioInfo, setAudioInfo] = useState<AudioConfigOutput | null>(null);
  const [saving, setSaving] = useState(false);
  const [installingModel, setInstallingModel] = useState(false);
  const [selectedMediaId, setSelectedMediaId] = useState<string>(playableMedia[0]?.id ?? "");
  const [enhancing, setEnhancing] = useState(false);
  const [lastResult, setLastResult] = useState<AudioEnhanceResult | null>(null);

  const ffmpegTool = tools.find((t) => t.id === "ffmpeg");

  useEffect(() => {
    if (!selectedMediaId && playableMedia.length > 0) {
      setSelectedMediaId(playableMedia[0].id);
    }
  }, [playableMedia, selectedMediaId]);

  const loadConfig = useCallback(async () => {
    try {
      const res = await runAction<AudioConfigOutput>("audio.config.get");
      setAudioInfo(res.data);
    } catch {
      // offline
    }
  }, []);

  useEffect(() => {
    void loadConfig();
  }, [loadConfig]);

  const activeAudioJobs = jobs.filter(
    (j) => j.kind === "audio.enhance" && (j.status === "queued" || j.status === "running"),
  );

  const updateConfig = async (patch: Partial<AudioMasterConfig>) => {
    setSaving(true);
    try {
      const res = await runAction<AudioConfigOutput>("audio.config.set", patch);
      setAudioInfo(res.data);
      onNotify(
        "success",
        "Áudio IA atualizado",
        res.data.config.preset === "off"
          ? "Tratamento automático de áudio desativado nas exportações."
          : `Padrão ${res.data.config.preset} (${res.data.config.target_lufs.toFixed(1)} LUFS) ativo para Cortes Shorts e Motion Studio.`,
      );
    } catch (e) {
      const msg = e instanceof AicutActionError ? e.message : (e as Error).message;
      onNotify("error", "Erro ao salvar configuração de áudio", msg);
    } finally {
      setSaving(false);
    }
  };

  const handleInstallRnnoise = async () => {
    setInstallingModel(true);
    try {
      const res = await runAction<AudioConfigOutput>("audio.install_rnnoise");
      setAudioInfo(res.data);
      onNotify(
        "success",
        "RNNoise Neural instalado",
        "Modelo gratuito de redução neural de ruído ativo automaticamente em todas as exportações.",
      );
    } catch (e) {
      const msg = e instanceof AicutActionError ? e.message : (e as Error).message;
      onNotify("error", "Falha ao instalar RNNoise", msg);
    } finally {
      setInstallingModel(false);
    }
  };

  const handleEnhanceMedia = async () => {
    if (!selectedMediaId) return;
    setEnhancing(true);
    try {
      const res = await runAction<{ job_id: string; status: string; result?: AudioEnhanceResult }>(
        "audio.enhance",
        { media_id: selectedMediaId, wait: true },
      );
      if (res.data.result) {
        setLastResult(res.data.result);
      }
      await onRefreshMedia();
      onNotify(
        "success",
        "Áudio masterizado para YouTube",
        `Nova versão salva na Biblioteca em ${res.data.result?.target_lufs.toFixed(1) ?? "-14.0"} LUFS (vídeo copiado sem perda).`,
      );
    } catch (e) {
      const msg = e instanceof AicutActionError ? e.message : (e as Error).message;
      onNotify("error", "Falha ao masterizar áudio", msg);
    } finally {
      setEnhancing(false);
    }
  };

  const cfg = audioInfo?.config;

  return (
    <div className="stack">
      {!ffmpegTool?.installed && (
        <Callout
          tone="warning"
          title="FFmpeg necessário"
          actions={
            <button type="button" className="btn btn-primary btn-sm" onClick={() => onNavigate("settings")}>
              Instalar FFmpeg
            </button>
          }
        >
          Instale o FFmpeg em Configurações → Ferramentas (ou via <code>sudo pacman -S ffmpeg</code>) para habilitar a masterização automática de áudio.
        </Callout>
      )}

      <ActiveJobsPanel jobs={activeAudioJobs} title="Masterizando áudio" tone="accent" onViewAll={() => onNavigate("jobs")} />

      <Panel
        title="Masterização Automática (Cortes Shorts + Motion Studio)"
        icon={SlidersHorizontal}
        description="Aplicada automaticamente em 2 passos (EBU R128 real) em todo vídeo exportado nos Cortes Shorts e no Motion Studio."
        actions={
          cfg && (
            <Badge tone={cfg.preset === "off" ? "warning" : "success"} dot>
              {cfg.preset === "off"
                ? "Desativado"
                : `${cfg.target_lufs.toFixed(0)} LUFS · ${cfg.true_peak_db.toFixed(1)} dBTP`}
            </Badge>
          )
        }
      >
        <ChoiceCards
          ariaLabel="Preset de plataforma"
          columns={4}
          value={cfg?.preset ?? "youtube_auto"}
          onChange={(preset) => void updateConfig({ preset })}
          options={PRESET_OPTIONS.map((p) => ({
            id: p.id,
            title: p.title,
            description: p.description,
          }))}
        />

        {cfg && cfg.preset !== "off" && (
          <div className="stack">
            <div className="grid-2">
              <Field
                label="Redução de ruído automática"
                htmlFor="am-denoise"
                hint={
                  audioInfo?.rnnoise_installed
                    ? "RNNoise Neural (IA) instalado e pronto para uso."
                    : "Usa denoiser espectral adaptativo ou instale o modelo RNNoise Neural (85 KB grátis)."
                }
              >
                <select
                  id="am-denoise"
                  className="select"
                  value={cfg.denoise}
                  disabled={saving}
                  onChange={(e) =>
                    void updateConfig({
                      denoise: e.target.value as AudioMasterConfig["denoise"],
                    })
                  }
                >
                  <option value="auto">Automático (RNNoise Neural quando instalado, senão Espectral)</option>
                  <option value="neural_rnnoise">Priorizar RNNoise Neural (IA)</option>
                  <option value="spectral">Espectral adaptativo (FFmpeg afftdn)</option>
                  <option value="off">Sem redução de ruído</option>
                </select>
              </Field>

              <Field
                label="Modelo Neural RNNoise (grátis · 85 KB)"
                hint="Rede neural open-source treinada para limpar voz de microfone sem robotizar."
              >
                <div className="row wrap gap-sm">
                  {audioInfo?.rnnoise_installed ? (
                    <Badge tone="success">
                      <CheckCircle2 size={12} aria-hidden /> RNNoise Neural ativo
                    </Badge>
                  ) : (
                    <button
                      type="button"
                      className="btn btn-sm"
                      disabled={installingModel}
                      onClick={() => void handleInstallRnnoise()}
                    >
                      <BusyIcon busy={installingModel} icon={Download} size={13} />
                      Baixar modelo RNNoise (85 KB)
                    </button>
                  )}
                </div>
              </Field>
            </div>

            <div className="row wrap gap-md">
              <Checkbox
                checked={cfg.voice_eq}
                disabled={saving}
                onChange={(v) => void updateConfig({ voice_eq: v })}
                label="Equalização de voz automática (corta grave abafado + realce de clareza)"
              />
              <Checkbox
                checked={cfg.auto_compressor}
                disabled={saving}
                onChange={(v) => void updateConfig({ auto_compressor: v })}
                label="Compressão vocal + limiter (nivela sussurros e gritos automaticamente)"
              />
            </div>
          </div>
        )}
      </Panel>

      <Panel
        title="Aprimorar Áudio de Mídia da Biblioteca (1 Clique)"
        icon={Volume2}
        description="Masteriza qualquer vídeo ou áudio já salvo na Biblioteca mantendo o vídeo original intacto (cópia sem re-encodar)."
      >
        {playableMedia.length === 0 ? (
          <EmptyState
            compact
            icon={Waves}
            title="nenhuma mídia na Biblioteca"
            description="Importe ou baixe um vídeo/áudio para aplicar o tratamento automático de áudio."
            actions={
              <button type="button" className="btn btn-primary btn-sm" onClick={() => onNavigate("ingest")}>
                Ir para Importar
              </button>
            }
          />
        ) : (
          <div className="stack">
            <div className="row wrap gap-sm">
              <select
                className="select grow"
                aria-label="Selecionar mídia para masterizar"
                value={selectedMediaId}
                onChange={(e) => setSelectedMediaId(e.target.value)}
              >
                {playableMedia.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.title}
                    {m.metadata.duration_sec ? ` (${formatDuration(m.metadata.duration_sec)})` : ""}
                  </option>
                ))}
              </select>

              <button
                type="button"
                className="btn btn-primary"
                disabled={!selectedMediaId || enhancing || !ffmpegTool?.installed}
                onClick={() => void handleEnhanceMedia()}
              >
                <BusyIcon busy={enhancing} icon={Sparkles} size={14} />
                Masterizar Áudio Agora
              </button>
            </div>

            {lastResult && (
              <Callout
                tone="success"
                title={`Masterizado: ${lastResult.asset.title}`}
                actions={
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={() => onSelectMedia(lastResult.asset.id)}
                  >
                    <Play size={13} aria-hidden /> Abrir na Biblioteca
                  </button>
                }
              >
                {lastResult.before_lufs !== undefined
                  ? `Volume original: ${lastResult.before_lufs.toFixed(1)} LUFS (pico ${lastResult.before_true_peak_db?.toFixed(1) ?? "?"} dBTP) → Normalizado para ${lastResult.target_lufs.toFixed(1)} LUFS / ${lastResult.target_true_peak_db.toFixed(1)} dBTP`
                  : `Normalizado para ${lastResult.target_lufs.toFixed(1)} LUFS / ${lastResult.target_true_peak_db.toFixed(1)} dBTP`}
                {lastResult.rnnoise_used ? " com RNNoise Neural." : " com denoiser espectral adaptativo."}
              </Callout>
            )}
          </div>
        )}
      </Panel>
    </div>
  );
};
