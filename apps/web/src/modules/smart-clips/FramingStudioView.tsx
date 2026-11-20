import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Camera,
  CheckCircle2,
  Crop,
  Eye,
  Film,
  Maximize2,
  Minimize2,
  RefreshCw,
  Save,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Trash2,
  Wand2,
} from "lucide-react";
import { AicutActionError, runAction } from "../../lib/client";
import { formatDuration } from "../../lib/format";
import type {
  ContentRegionMode,
  FramingGetOutput,
  FramingPreviewOutput,
  FramingProfile,
  FramingReviewOutput,
  MediaAsset,
  NormalizedRect,
  NotifyFn,
  RouteId,
  ToolStatus,
} from "../../lib/types";
import { Badge, BusyIcon, Callout, Checkbox, EmptyState, Field, Panel } from "../../components";
import "./framing-studio.css";

interface FramingStudioViewProps {
  media: MediaAsset[];
  tools: ToolStatus[];
  preselectedMediaId?: string | null;
  onNavigate: (route: RouteId) => void;
  onNotify: NotifyFn;
}

type HandleDir = "move" | "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";
type ActiveLayer = "webcam" | "content";

const HANDLES: Array<Exclude<HandleDir, "move">> = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
const DEFAULT_WEBCAM_RECT: NormalizedRect = { x: 0.70, y: 0.64, w: 0.28, h: 0.33 };
const DEFAULT_CONTENT_RECT: NormalizedRect = { x: 0.0, y: 0.0, w: 1.0, h: 1.0 };
const TOP_SLOT_ASPECT = 1020 / 582; // SMOKING33 top window exact aspect ratio (~1.7525)

function clampRect(r: NormalizedRect): NormalizedRect {
  const w = Math.max(0.06, Math.min(1.0, r.w));
  const h = Math.max(0.06, Math.min(1.0, r.h));
  const x = Math.max(0.0, Math.min(1.0 - w, r.x));
  const y = Math.max(0.0, Math.min(1.0 - h, r.y));
  return {
    x: Math.round(x * 10000) / 10000,
    y: Math.round(y * 10000) / 10000,
    w: Math.round(w * 10000) / 10000,
    h: Math.round(h * 10000) / 10000,
  };
}

function drawRoundedRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawCroppedRegion(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  rect: NormalizedRect,
  dx: number,
  dy: number,
  dw: number,
  dh: number,
  fitBlur: boolean,
) {
  const sx = rect.x * img.naturalWidth;
  const sy = rect.y * img.naturalHeight;
  const sw = Math.max(4, rect.w * img.naturalWidth);
  const sh = Math.max(4, rect.h * img.naturalHeight);
  const targetAspect = dw / dh;
  const srcAspect = sw / sh;

  if (fitBlur) {
    ctx.save();
    ctx.filter = "blur(10px)";
    const bgScale = Math.max(dw / sw, dh / sh);
    const bgW = sw * bgScale;
    const bgH = sh * bgScale;
    ctx.drawImage(img, sx, sy, sw, sh, dx + (dw - bgW) / 2, dy + (dh - bgH) / 2, bgW, bgH);
    ctx.restore();
    const fgScale = Math.min(dw / sw, dh / sh);
    const fgW = sw * fgScale;
    const fgH = sh * fgScale;
    ctx.drawImage(img, sx, sy, sw, sh, dx + (dw - fgW) / 2, dy + (dh - fgH) / 2, fgW, fgH);
    return;
  }

  let cw = sw;
  let ch = sh;
  if (srcAspect > targetAspect) {
    cw = sh * targetAspect;
  } else {
    ch = sw / targetAspect;
  }
  const cx = sx + (sw - cw) * 0.5;
  const cy = sy + (sh - ch) * 0.5;
  ctx.drawImage(img, cx, cy, cw, ch, dx, dy, dw, dh);
}

export const FramingStudioView: React.FC<FramingStudioViewProps> = ({
  media,
  tools,
  preselectedMediaId,
  onNavigate,
  onNotify,
}) => {
  const videoMedia = useMemo(() => media.filter((m) => m.kind === "video" || m.kind === "video_only"), [media]);
  const ffmpegTool = tools.find((t) => t.id === "ffmpeg");

  const [selectedMediaId, setSelectedMediaId] = useState<string>(preselectedMediaId ?? videoMedia[0]?.id ?? "");
  const [preview, setPreview] = useState<FramingPreviewOutput | null>(null);
  const [timestampSec, setTimestampSec] = useState<number>(15);
  const [loadingFrame, setLoadingFrame] = useState<boolean>(false);
  const [autoDetecting, setAutoDetecting] = useState<boolean>(false);

  // Saved framing store
  const [framingState, setFramingState] = useState<FramingGetOutput | null>(null);
  const [saving, setSaving] = useState<boolean>(false);
  const [presetName, setPresetName] = useState<string>("");

  // Editable boxes & options
  const [activeLayer, setActiveLayer] = useState<ActiveLayer>("webcam");
  const [useCustomWebcam, setUseCustomWebcam] = useState<boolean>(true);
  const [webcamRect, setWebcamRect] = useState<NormalizedRect>(DEFAULT_WEBCAM_RECT);
  const [webcamLockStatic, setWebcamLockStatic] = useState<boolean>(true);

  const [useCustomContent, setUseCustomContent] = useState<boolean>(false);
  const [contentRect, setContentRect] = useState<NormalizedRect>(DEFAULT_CONTENT_RECT);
  const [contentMode, setContentMode] = useState<ContentRegionMode>("ai_director_in_region");

  // AI Reviewer state
  const [reviewing, setReviewing] = useState<boolean>(false);
  const [review, setReview] = useState<FramingReviewOutput | null>(null);

  const stageRef = useRef<HTMLDivElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);
  const loadedImageRef = useRef<HTMLImageElement | null>(null);
  const dragRef = useRef<{
    layer: ActiveLayer;
    dir: HandleDir;
    startClientX: number;
    startClientY: number;
    startRect: NormalizedRect;
    stageW: number;
    stageH: number;
  } | null>(null);

  useEffect(() => {
    if (preselectedMediaId) {
      setSelectedMediaId(preselectedMediaId);
    } else if (!selectedMediaId && videoMedia.length > 0) {
      setSelectedMediaId(videoMedia[0].id);
    }
  }, [preselectedMediaId, videoMedia]);

  const applyProfileToEditor = useCallback((p: FramingProfile) => {
    if (p.webcam_rect) {
      setUseCustomWebcam(true);
      setWebcamRect(clampRect(p.webcam_rect));
    } else {
      setUseCustomWebcam(false);
    }
    setWebcamLockStatic(p.webcam_lock_static ?? true);
    if (p.content_rect) {
      setUseCustomContent(true);
      setContentRect(clampRect(p.content_rect));
    } else {
      setUseCustomContent(false);
      setContentRect(DEFAULT_CONTENT_RECT);
    }
    setContentMode(p.content_mode ?? "ai_director_in_region");
  }, []);

  const loadSavedFraming = useCallback(
    async (mid: string) => {
      try {
        const res = await runAction<FramingGetOutput>("clips.framing_get", {
          media_id: mid || undefined,
        });
        setFramingState(res.data);
        applyProfileToEditor(res.data.active_profile);
      } catch {
        // ignore if offline
      }
    },
    [applyProfileToEditor],
  );

  const loadFrame = useCallback(
    async (mid: string, ts?: number, autoDetect = false) => {
      if (!mid) return;
      if (autoDetect) {
        setAutoDetecting(true);
      } else {
        setLoadingFrame(true);
      }
      setReview(null);
      try {
        const res = await runAction<FramingPreviewOutput>("clips.framing_preview_frame", {
          media_id: mid,
          timestamp_sec: ts,
          auto_detect: autoDetect,
          border_trim_pct: 0.045,
        });
        setPreview(res.data);
        setTimestampSec(Math.round(res.data.timestamp_sec));
        if (autoDetect && res.data.detected_webcam_rect) {
          setUseCustomWebcam(true);
          setWebcamRect(clampRect(res.data.detected_webcam_rect));
          onNotify(
            "success",
            "Webcam detectada pela IA",
            "A caixa verde foi posicionada no interior limpo da webcam (sem bordas). Ajuste se quiser e salve!",
          );
        } else if (autoDetect && !res.data.detected_webcam_rect) {
          onNotify(
            "warning",
            "Webcam não detectada automaticamente",
            "Arraste a caixa verde manualmente sobre a área interna da sua webcam e use 'Revisar com IA'.",
          );
        }
      } catch (e) {
        const msg = e instanceof AicutActionError ? e.message : (e as Error).message;
        onNotify("error", "Erro ao carregar frame", msg);
      } finally {
        setLoadingFrame(false);
        setAutoDetecting(false);
      }
    },
    [onNotify],
  );

  useEffect(() => {
    if (!selectedMediaId) return;
    void loadSavedFraming(selectedMediaId);
    void loadFrame(selectedMediaId, undefined, false);
  }, [selectedMediaId, loadSavedFraming, loadFrame]);

  // Render live 9:16 Short preview onto canvas whenever image or boxes change
  const redrawShortPreview = useCallback(() => {
    const canvas = previewCanvasRef.current;
    const img = loadedImageRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Subtle dark gradient representing the SMOKING33 template background
    const bg = ctx.createLinearGradient(0, 0, 0, H);
    bg.addColorStop(0, "#1d0f2e");
    bg.addColorStop(0.5, "#110c1c");
    bg.addColorStop(1, "#1c1224");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // Header label
    ctx.fillStyle = "#e8bd52";
    ctx.font = "700 11px monospace";
    ctx.textAlign = "center";
    ctx.fillText("SMOKING33 · 9:16 PREVIEW", W / 2, 22);

    // Exact 1/4 scale of 1080x1920 SMOKING33 slots:
    // Top camera: x=30, y=150, w=1020, h=582 -> /4 = (7.5, 37.5, 255, 145.5)
    // Bottom content: x=30, y=765, w=1020, h=889 -> /4 = (7.5, 191.25, 255, 222.25)
    const sx = W / 1080;
    const sy = H / 1920;
    const topX = 30 * sx;
    const topY = 150 * sy;
    const topW = 1020 * sx;
    const topH = 582 * sy;

    const botX = 30 * sx;
    const botY = 765 * sy;
    const botW = 1020 * sx;
    const botH = 889 * sy;

    const camR = useCustomWebcam ? webcamRect : DEFAULT_WEBCAM_RECT;
    const cntR = useCustomContent ? contentRect : DEFAULT_CONTENT_RECT;

    // Top window (Webcam)
    ctx.save();
    drawRoundedRect(ctx, topX, topY, topW, topH, 8);
    ctx.clip();
    ctx.fillStyle = "#08090c";
    ctx.fillRect(topX, topY, topW, topH);
    if (img && img.naturalWidth > 0) {
      drawCroppedRegion(ctx, img, camR, topX, topY, topW, topH, false);
    }
    ctx.restore();
    drawRoundedRect(ctx, topX, topY, topW, topH, 8);
    ctx.strokeStyle = "#eab308";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Bottom window (Content)
    ctx.save();
    drawRoundedRect(ctx, botX, botY, botW, botH, 8);
    ctx.clip();
    ctx.fillStyle = "#08090c";
    ctx.fillRect(botX, botY, botW, botH);
    if (img && img.naturalWidth > 0) {
      drawCroppedRegion(ctx, img, cntR, botX, botY, botW, botH, contentMode === "fit_blur_region");
    }
    ctx.restore();
    drawRoundedRect(ctx, botX, botY, botW, botH, 8);
    ctx.strokeStyle = "#eab308";
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [webcamRect, contentRect, useCustomWebcam, useCustomContent, contentMode]);

  useEffect(() => {
    if (!preview?.image_data_url) {
      loadedImageRef.current = null;
      redrawShortPreview();
      return;
    }
    const img = new Image();
    img.onload = () => {
      loadedImageRef.current = img;
      redrawShortPreview();
    };
    img.src = preview.image_data_url;
  }, [preview?.image_data_url, redrawShortPreview]);

  useEffect(() => {
    redrawShortPreview();
  }, [redrawShortPreview]);

  // Pointer drag / 8-handle resize handlers
  const beginDrag = (e: React.PointerEvent, layer: ActiveLayer, dir: HandleDir) => {
    e.stopPropagation();
    const stage = stageRef.current;
    if (!stage) return;
    const bounds = stage.getBoundingClientRect();
    if (bounds.width < 10 || bounds.height < 10) return;

    setActiveLayer(layer);
    if (layer === "webcam") setUseCustomWebcam(true);
    if (layer === "content") setUseCustomContent(true);

    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dragRef.current = {
      layer,
      dir,
      startClientX: e.clientX,
      startClientY: e.clientY,
      startRect: layer === "webcam" ? { ...webcamRect } : { ...contentRect },
      stageW: bounds.width,
      stageH: bounds.height,
    };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    const dx = (e.clientX - d.startClientX) / d.stageW;
    const dy = (e.clientY - d.startClientY) / d.stageH;
    const s = d.startRect;
    let x1 = s.x;
    let y1 = s.y;
    let x2 = s.x + s.w;
    let y2 = s.y + s.h;

    if (d.dir === "move") {
      const nx = Math.max(0, Math.min(1 - s.w, s.x + dx));
      const ny = Math.max(0, Math.min(1 - s.h, s.y + dy));
      const next = clampRect({ x: nx, y: ny, w: s.w, h: s.h });
      if (d.layer === "webcam") setWebcamRect(next);
      else setContentRect(next);
      return;
    }

    if (d.dir.includes("w")) x1 = Math.max(0, Math.min(x2 - 0.06, s.x + dx));
    if (d.dir.includes("e")) x2 = Math.min(1, Math.max(x1 + 0.06, s.x + s.w + dx));
    if (d.dir.includes("n")) y1 = Math.max(0, Math.min(y2 - 0.06, s.y + dy));
    if (d.dir.includes("s")) y2 = Math.min(1, Math.max(y1 + 0.06, s.y + s.h + dy));

    const next = clampRect({ x: x1, y: y1, w: x2 - x1, h: y2 - y1 });
    if (d.layer === "webcam") setWebcamRect(next);
    else setContentRect(next);
  };

  const endDrag = () => {
    dragRef.current = null;
  };

  // Surgical quick helpers for webcam box
  const nudgeWebcam = (deltaPct: number) => {
    // Positive delta shrinks inward (removes borders); negative expands outward
    setUseCustomWebcam(true);
    setWebcamRect((prev) => {
      const dx = prev.w * deltaPct;
      const dy = prev.h * deltaPct;
      return clampRect({
        x: prev.x + dx,
        y: prev.y + dy,
        w: prev.w - dx * 2,
        h: prev.h - dy * 2,
      });
    });
  };

  const matchTopSlotAspect = () => {
    const srcW = preview?.width ?? 1920;
    const srcH = preview?.height ?? 1080;
    const srcAspect = srcW / Math.max(1, srcH);
    // In normalized coords: (w * srcW) / (h * srcH) = TOP_SLOT_ASPECT => w / h = TOP_SLOT_ASPECT / srcAspect
    const normRatio = TOP_SLOT_ASPECT / srcAspect;
    setUseCustomWebcam(true);
    setWebcamRect((prev) => {
      const cx = prev.x + prev.w * 0.5;
      const cy = prev.y + prev.h * 0.5;
      let nw = prev.w;
      let nh = nw / normRatio;
      if (nh > prev.h) {
        nh = prev.h;
        nw = nh * normRatio;
      }
      return clampRect({ x: cx - nw * 0.5, y: cy - nh * 0.5, w: nw, h: nh });
    });
  };

  const buildProfilePayload = (name: string, id = "default"): FramingProfile => ({
    id,
    name,
    webcam_rect: useCustomWebcam ? clampRect(webcamRect) : null,
    content_rect: useCustomContent ? clampRect(contentRect) : null,
    webcam_lock_static: webcamLockStatic,
    content_mode: contentMode,
    border_trim_pct: 0.045,
    face_zoom: 1.0,
    updated_at: new Date().toISOString(),
  });

  const handleSave = async (scope: "default" | "media" | "preset") => {
    if (scope === "media" && !selectedMediaId) return;
    if (scope === "preset" && !presetName.trim()) {
      onNotify("warning", "Nome do preset", "Digite um nome para salvar o preset (ex.: Live GTA VI).");
      return;
    }
    setSaving(true);
    try {
      const label =
        scope === "preset"
          ? presetName.trim()
          : scope === "media"
            ? videoMedia.find((m) => m.id === selectedMediaId)?.title ?? "Este vídeo"
            : "Padrão";
      const res = await runAction<FramingGetOutput>("clips.framing_save", {
        scope,
        media_id: selectedMediaId || undefined,
        profile: buildProfilePayload(label, scope === "preset" ? "" : "default"),
      });
      setFramingState(res.data);
      if (scope === "preset") setPresetName("");
      onNotify(
        "success",
        "Enquadramento salvo",
        scope === "default"
          ? "Área da webcam e do conteúdo salva como padrão para todos os Shorts!"
          : scope === "media"
            ? "Enquadramento salvo especificamente para este vídeo!"
            : `Preset "${label}" salvo com sucesso.`,
      );
    } catch (e) {
      const msg = e instanceof AicutActionError ? e.message : (e as Error).message;
      onNotify("error", "Erro ao salvar enquadramento", msg);
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteEntry = async (opts: { preset_id?: string; media_id?: string }) => {
    try {
      const res = await runAction<FramingGetOutput>("clips.framing_delete", opts);
      setFramingState(res.data);
      applyProfileToEditor(res.data.active_profile);
      onNotify("info", "Removido", "Configuração de enquadramento removida.");
    } catch (e) {
      onNotify("error", "Erro ao remover", (e as Error).message);
    }
  };

  const handleAiReview = async () => {
    if (!selectedMediaId) return;
    setReviewing(true);
    try {
      const res = await runAction<FramingReviewOutput>("clips.framing_review", {
        media_id: selectedMediaId,
        timestamp_sec: timestampSec,
        webcam_rect: clampRect(webcamRect),
        content_rect: useCustomContent ? clampRect(contentRect) : undefined,
        use_vlm: true,
      });
      setReview(res.data);
      onNotify(
        res.data.approved ? "success" : "warning",
        `Revisão da IA · ${res.data.score}/100`,
        res.data.summary,
      );
    } catch (e) {
      const msg = e instanceof AicutActionError ? e.message : (e as Error).message;
      onNotify("error", "Erro na revisão por IA", msg);
    } finally {
      setReviewing(false);
    }
  };

  const activeRect = activeLayer === "webcam" ? webcamRect : contentRect;
  const setActiveRect = (next: NormalizedRect) => {
    if (activeLayer === "webcam") {
      setUseCustomWebcam(true);
      setWebcamRect(clampRect(next));
    } else {
      setUseCustomContent(true);
      setContentRect(clampRect(next));
    }
  };

  const duration = preview?.duration_sec ?? videoMedia.find((m) => m.id === selectedMediaId)?.metadata.duration_sec ?? 60;
  const reviewTone = !review ? "neutral" : review.approved ? "success" : review.score >= 55 ? "warning" : "danger";

  return (
    <div className="stack">
      {!ffmpegTool?.installed && (
        <Callout
          tone="warning"
          title="FFmpeg necessário"
          actions={
            <button type="button" className="btn btn-primary btn-sm" onClick={() => onNavigate("settings")}>
              Instalar em Ferramentas
            </button>
          }
        >
          Instale o FFmpeg em Configurações → Ferramentas para carregar o preview do vídeo.
        </Callout>
      )}

      {videoMedia.length === 0 ? (
        <EmptyState
          icon={Film}
          title="nenhum vídeo na Biblioteca"
          description="Baixe ou importe um vídeo para calibrar com precisão a área da webcam e do conteúdo."
          actions={
            <button type="button" className="btn btn-primary" onClick={() => onNavigate("ingest")}>
              Baixar & Importar
            </button>
          }
        />
      ) : (
        <div className="stack">
          {/* Top toolbar: Video + Timestamp + Presets */}
          <Panel title="Vídeo de referência & momento do frame" icon={Film}>
            <div className="grid-2">
              <Field label="Vídeo / live da Biblioteca" htmlFor="fr-media">
                <select
                  id="fr-media"
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
              </Field>

              <Field
                label={`Instante do frame (${formatDuration(timestampSec)} / ${formatDuration(duration)})`}
                htmlFor="fr-ts"
              >
                <div className="row gap-xs">
                  <input
                    id="fr-ts"
                    type="range"
                    min={0}
                    max={Math.max(1, Math.floor(duration - 1))}
                    value={Math.min(timestampSec, Math.max(1, Math.floor(duration - 1)))}
                    onChange={(e) => setTimestampSec(Number(e.target.value))}
                    className="grow"
                  />
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={loadingFrame}
                    onClick={() => void loadFrame(selectedMediaId, timestampSec, false)}
                  >
                    <BusyIcon busy={loadingFrame} icon={RefreshCw} size={13} />
                    Carregar
                  </button>
                </div>
                <div className="row wrap gap-xs" style={{ marginTop: 4 }}>
                  {[0.1, 0.3, 0.5, 0.75].map((frac) => (
                    <button
                      key={frac}
                      type="button"
                      className="chip chip-sm"
                      onClick={() => {
                        const t = Math.round(duration * frac);
                        setTimestampSec(t);
                        void loadFrame(selectedMediaId, t, false);
                      }}
                    >
                      {Math.round(frac * 100)}%
                    </button>
                  ))}
                </div>
              </Field>
            </div>

            {framingState?.has_media_override && (
              <Callout
                tone="info"
                title="Este vídeo tem um enquadramento próprio salvo"
                actions={
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={() => void handleDeleteEntry({ media_id: selectedMediaId })}
                  >
                    Usar padrão global
                  </button>
                }
              >
                Ao renderizar Shorts deste vídeo, o Aicut aplicará exatamente esta área salva.
              </Callout>
            )}
          </Panel>

          {/* Main 2-column Visual Editor + Live 9:16 Preview */}
          <div className="framing-grid">
            <Panel
              title="Editor visual de precisão (arraste e redimensione nas bordas)"
              icon={Crop}
              actions={
                <div className="row gap-xs">
                  <button
                    type="button"
                    className={`btn btn-sm ${activeLayer === "webcam" ? "btn-primary" : ""}`}
                    onClick={() => {
                      setActiveLayer("webcam");
                      setUseCustomWebcam(true);
                    }}
                  >
                    <Camera size={13} aria-hidden /> Webcam (topo)
                  </button>
                  <button
                    type="button"
                    className={`btn btn-sm ${activeLayer === "content" ? "btn-primary" : ""}`}
                    onClick={() => {
                      setActiveLayer("content");
                      setUseCustomContent(true);
                    }}
                  >
                    <Maximize2 size={13} aria-hidden /> Conteúdo (baixo)
                  </button>
                </div>
              }
            >
              <div
                ref={stageRef}
                className="framing-stage-wrap"
                onPointerMove={onPointerMove}
                onPointerUp={endDrag}
                onPointerCancel={endDrag}
              >
                {preview?.image_data_url ? (
                  <img src={preview.image_data_url} alt="Frame do vídeo" className="framing-stage-img" draggable={false} />
                ) : (
                  <div className="framing-stage-empty">
                    <BusyIcon busy={loadingFrame} icon={Film} size={22} />
                    <span>{loadingFrame ? "Extraindo frame real do vídeo..." : "Clique em Carregar frame"}</span>
                  </div>
                )}

                {/* Content precision box (Cyan) */}
                {useCustomContent && preview?.image_data_url && (
                  <div
                    className="framing-box framing-box-content"
                    data-active={activeLayer === "content"}
                    style={{
                      left: `${contentRect.x * 100}%`,
                      top: `${contentRect.y * 100}%`,
                      width: `${contentRect.w * 100}%`,
                      height: `${contentRect.h * 100}%`,
                    }}
                    onPointerDown={(e) => beginDrag(e, "content", "move")}
                  >
                    <span className="framing-box-label">Conteúdo (baixo)</span>
                    {HANDLES.map((dir) => (
                      <span
                        key={dir}
                        className="framing-handle"
                        data-dir={dir}
                        onPointerDown={(e) => beginDrag(e, "content", dir)}
                      />
                    ))}
                  </div>
                )}

                {/* Webcam precision box (Green) */}
                {useCustomWebcam && preview?.image_data_url && (
                  <div
                    className="framing-box framing-box-webcam"
                    data-active={activeLayer === "webcam"}
                    style={{
                      left: `${webcamRect.x * 100}%`,
                      top: `${webcamRect.y * 100}%`,
                      width: `${webcamRect.w * 100}%`,
                      height: `${webcamRect.h * 100}%`,
                    }}
                    onPointerDown={(e) => beginDrag(e, "webcam", "move")}
                  >
                    <span className="framing-box-label">Webcam limpa (topo)</span>
                    {HANDLES.map((dir) => (
                      <span
                        key={dir}
                        className="framing-handle"
                        data-dir={dir}
                        onPointerDown={(e) => beginDrag(e, "webcam", dir)}
                      />
                    ))}
                  </div>
                )}
              </div>

              {/* Quick surgical adjustment buttons */}
              <div className="row wrap gap-xs">
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  disabled={autoDetecting || !selectedMediaId}
                  onClick={() => void loadFrame(selectedMediaId, timestampSec, true)}
                >
                  <BusyIcon busy={autoDetecting} icon={Wand2} size={13} />
                  Detectar webcam (IA)
                </button>
                <button type="button" className="btn btn-sm" onClick={() => nudgeWebcam(0.012)}>
                  <Minimize2 size={13} aria-hidden /> Encolher bordas (-1.2%)
                </button>
                <button type="button" className="btn btn-sm" onClick={() => nudgeWebcam(-0.025)}>
                  <Maximize2 size={13} aria-hidden /> Abrir webcam (+2.5%)
                </button>
                <button type="button" className="btn btn-sm" onClick={matchTopSlotAspect}>
                  <Crop size={13} aria-hidden /> Proporção exata SMOKING33 (1.75:1)
                </button>
              </div>

              {/* Exact numeric coordinates for active layer */}
              <div className="inset stack">
                <div className="row row-between">
                  <span className="section-label">
                    Coordenadas exatas · {activeLayer === "webcam" ? "Webcam (topo)" : "Conteúdo (baixo)"}
                  </span>
                  <span className="muted text-xs">
                    {preview
                      ? `${Math.round(activeRect.w * preview.width)}×${Math.round(activeRect.h * preview.height)} px no preview`
                      : "em % da tela"}
                  </span>
                </div>
                <div className="framing-coords">
                  <div className="framing-coord-cell">
                    <label htmlFor="fr-x">X (%)</label>
                    <input
                      id="fr-x"
                      type="number"
                      step={0.2}
                      min={0}
                      max={94}
                      className="input input-sm"
                      value={Number((activeRect.x * 100).toFixed(1))}
                      onChange={(e) => setActiveRect({ ...activeRect, x: Number(e.target.value) / 100 })}
                    />
                  </div>
                  <div className="framing-coord-cell">
                    <label htmlFor="fr-y">Y (%)</label>
                    <input
                      id="fr-y"
                      type="number"
                      step={0.2}
                      min={0}
                      max={94}
                      className="input input-sm"
                      value={Number((activeRect.y * 100).toFixed(1))}
                      onChange={(e) => setActiveRect({ ...activeRect, y: Number(e.target.value) / 100 })}
                    />
                  </div>
                  <div className="framing-coord-cell">
                    <label htmlFor="fr-w">Largura (%)</label>
                    <input
                      id="fr-w"
                      type="number"
                      step={0.2}
                      min={6}
                      max={100}
                      className="input input-sm"
                      value={Number((activeRect.w * 100).toFixed(1))}
                      onChange={(e) => setActiveRect({ ...activeRect, w: Number(e.target.value) / 100 })}
                    />
                  </div>
                  <div className="framing-coord-cell">
                    <label htmlFor="fr-h">Altura (%)</label>
                    <input
                      id="fr-h"
                      type="number"
                      step={0.2}
                      min={6}
                      max={100}
                      className="input input-sm"
                      value={Number((activeRect.h * 100).toFixed(1))}
                      onChange={(e) => setActiveRect({ ...activeRect, h: Number(e.target.value) / 100 })}
                    />
                  </div>
                </div>
              </div>
            </Panel>

            {/* Right panel: Live 9:16 Short preview + Save controls */}
            <Panel title="Preview ao vivo do Short 9:16" icon={Eye}>
              <div className="framing-short-preview">
                <canvas ref={previewCanvasRef} width={270} height={480} className="framing-short-canvas" />
                <span className="muted text-xs" style={{ textAlign: "center" }}>
                  Atualiza em tempo real — confira que a webcam aparece natural (sem zoom no rosto e sem nenhuma borda).
                </span>
              </div>

              <div className="stack">
                <Checkbox
                  checked={useCustomWebcam}
                  onChange={setUseCustomWebcam}
                  label="Usar área exata da webcam salva acima"
                  description="Desmarque para deixar a detecção automática da IA."
                />
                <Checkbox
                  checked={webcamLockStatic}
                  onChange={setWebcamLockStatic}
                  label="Travar webcam inteira sem zoom no rosto (Recomendado)"
                  description="Exibe 100% da área verde salva como uma webcam normal, sem aproximar do rosto."
                />
                <Checkbox
                  checked={useCustomContent}
                  onChange={setUseCustomContent}
                  label="Delimitar área exata do conteúdo (caixa azul)"
                  description="Opcional: deixe desmarcado para a IA Diretor usar a tela inteira (excluindo sua webcam)."
                />

                <Field label="Comportamento da área de conteúdo (baixo)" htmlFor="fr-cmode">
                  <select
                    id="fr-cmode"
                    className="select select-sm"
                    value={contentMode}
                    onChange={(e) => setContentMode(e.target.value as ContentRegionMode)}
                  >
                    <option value="ai_director_in_region">IA Diretor · Segue quem fala ou ação dentro da área</option>
                    <option value="fixed_region">Área fixa exata salva (sem movimento)</option>
                    <option value="fit_blur_region">Área inteira + fundo blur</option>
                  </select>
                </Field>

                <div className="row wrap gap-xs">
                  <button
                    type="button"
                    className="btn btn-primary btn-sm grow"
                    disabled={saving}
                    onClick={() => void handleSave("default")}
                  >
                    <BusyIcon busy={saving} icon={Save} size={13} />
                    Salvar como Padrão
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm grow"
                    disabled={saving || !selectedMediaId}
                    onClick={() => void handleSave("media")}
                  >
                    <Save size={13} aria-hidden />
                    Salvar p/ Este Vídeo
                  </button>
                </div>

                {/* Named Presets */}
                <div className="inset stack">
                  <span className="section-label">Presets salvos (ex.: Live GTA, React, Podcast)</span>
                  <div className="row gap-xs">
                    <input
                      type="text"
                      className="input input-sm grow"
                      placeholder="Nome do preset…"
                      value={presetName}
                      onChange={(e) => setPresetName(e.target.value)}
                    />
                    <button
                      type="button"
                      className="btn btn-sm"
                      disabled={saving || !presetName.trim()}
                      onClick={() => void handleSave("preset")}
                    >
                      Salvar preset
                    </button>
                  </div>

                  {framingState && framingState.presets.length > 0 && (
                    <div className="stack" style={{ gap: 6 }}>
                      {framingState.presets.map((p) => (
                        <div key={p.id} className="framing-preset-row">
                          <span className="text-xs strong">{p.name}</span>
                          <div className="row gap-xs">
                            <button
                              type="button"
                              className="btn btn-sm"
                              onClick={() => {
                                applyProfileToEditor(p);
                                onNotify("info", "Preset carregado", `Preset "${p.name}" aplicado no editor.`);
                              }}
                            >
                              Carregar
                            </button>
                            <button
                              type="button"
                              className="btn btn-ghost btn-icon btn-sm"
                              aria-label="Excluir preset"
                              onClick={() => void handleDeleteEntry({ preset_id: p.id })}
                            >
                              <Trash2 size={13} aria-hidden />
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </Panel>
          </div>

          {/* Dedicated AI Webcam Framing Reviewer */}
          <Panel
            title="Revisão da IA — A webcam está bem enquadrada e sem bordas?"
            icon={Sparkles}
            actions={
              <button
                type="button"
                className="btn btn-primary btn-sm"
                disabled={reviewing || !selectedMediaId}
                onClick={() => void handleAiReview()}
              >
                <BusyIcon busy={reviewing} icon={Sparkles} size={13} />
                {reviewing ? "IA revisando webcam..." : "Revisar enquadramento com IA"}
              </button>
            }
          >
            {!review ? (
              <Callout tone="info" icon={Sparkles} title="Peça para a IA conferir antes de renderizar">
                A IA recorta exatamente a sua janela da webcam em 3 momentos do vídeo, analisa se sobrou qualquer borda do
                OBS, verifica se o zoom está natural (sem ficar colado no rosto) e se a cabeça/ombros estão bem
                enquadrados — sugerindo correção em 1 clique caso encontre algum defeito.
              </Callout>
            ) : (
              <div className="framing-review-layout">
                <div className="stack" style={{ gap: 6 }}>
                  <img
                    src={review.cropped_webcam_preview_data_url}
                    alt="Recorte exato analisado pela IA"
                    className="framing-review-thumb"
                  />
                  <span className="muted text-xs">Recorte exato inspecionado pela IA</span>
                </div>

                <div className="stack">
                  <div className="row wrap gap-sm" style={{ alignItems: "center" }}>
                    <span className="framing-review-score" data-tone={reviewTone}>
                      {review.score}/100
                    </span>
                    <Badge
                      tone={review.approved ? "success" : review.score >= 55 ? "warning" : "danger"}
                      icon={review.approved ? ShieldCheck : ShieldAlert}
                    >
                      {review.approved
                        ? "Aprovado · Enquadramento ótimo"
                        : review.verdict === "ajustar"
                          ? "Ajuste recomendado"
                          : "Problema detectado"}
                    </Badge>
                    <Badge>{review.reviewer_engine}</Badge>
                  </div>

                  <p className="text-sm">{review.summary}</p>

                  <div className="row wrap gap-xs">
                    <Badge tone={review.has_border_leak ? "danger" : "success"}>
                      Bordas OBS:{" "}
                      {review.has_border_leak
                        ? `detectada em ${review.border_sides.join(", ") || "borda"}`
                        : "limpo (0 bordas)"}
                    </Badge>
                    <Badge tone={review.zoom_assessment === "ideal" ? "success" : "warning"}>
                      Zoom:{" "}
                      {review.zoom_assessment === "ideal"
                        ? "natural (webcam inteira)"
                        : review.zoom_assessment === "muito_perto"
                          ? "muito perto / colado no rosto"
                          : review.zoom_assessment === "muito_longe"
                            ? "muito longe"
                            : "rosto não detectado"}
                      {review.face_height_pct ? ` · rosto ${review.face_height_pct.toFixed(0)}%` : ""}
                    </Badge>
                    <Badge tone={review.head_or_face_clipped ? "warning" : "success"}>
                      Cabeça/rosto: {review.head_or_face_clipped ? "próximo/cortado na borda" : "bem centralizado"}
                    </Badge>
                  </div>

                  {review.suggestions.length > 0 && (
                    <ul className="stack text-xs secondary" style={{ paddingLeft: 18, margin: 0, gap: 4 }}>
                      {review.suggestions.map((s, idx) => (
                        <li key={idx}>{s}</li>
                      ))}
                    </ul>
                  )}

                  <div className="row wrap gap-xs">
                    {review.suggested_webcam_rect && (
                      <button
                        type="button"
                        className="btn btn-primary btn-sm"
                        onClick={() => {
                          setUseCustomWebcam(true);
                          setWebcamRect(clampRect(review.suggested_webcam_rect!));
                          onNotify(
                            "success",
                            "Ajuste da IA aplicado",
                            "Confira no preview 9:16 ao lado e clique em 'Revisar novamente' ou 'Salvar como Padrão'!",
                          );
                        }}
                      >
                        <CheckCircle2 size={13} aria-hidden />
                        Aplicar ajuste sugerido pela IA
                      </button>
                    )}
                    <button
                      type="button"
                      className="btn btn-sm"
                      disabled={reviewing}
                      onClick={() => void handleAiReview()}
                    >
                      <RefreshCw size={13} aria-hidden />
                      Revisar novamente
                    </button>
                  </div>
                </div>
              </div>
            )}
          </Panel>
        </div>
      )}
    </div>
  );
};
