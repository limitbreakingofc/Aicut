import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bot,
  Code2,
  Copy,
  Crop,
  Film,
  Layers,
  Pause,
  Play,
  Plus,
  Save,
  Scissors,
  Send,
  Sparkles,
  Trash2,
  Wand2,
} from "lucide-react";
import { AicutActionError, mediaStreamUrl, runAction } from "../../lib/client";
import { formatDuration } from "../../lib/format";
import type {
  AicutComposition,
  JobRecord,
  MediaAsset,
  MotionExportCodeOutput,
  MotionScene,
  NormalizedRect,
  NotifyFn,
  PipCorner,
  RouteId,
  SceneLayoutPreset,
  ToolStatus,
} from "../../lib/types";
import { copyToClipboard } from "../../lib/ui";
import { pageMeta } from "../../app/navigation";
import {
  ActiveJobsPanel,
  Badge,
  BusyIcon,
  Callout,
  Checkbox,
  EmptyState,
  Field,
  PageHeader,
  Panel,
  SegmentedControl,
} from "../../components";
import "./motion-studio.css";

export interface MotionSeed {
  mediaId: string;
  startSec?: number;
  endSec?: number;
  title?: string;
}

interface MotionStudioViewProps {
  media: MediaAsset[];
  tools: ToolStatus[];
  jobs: JobRecord[];
  seed?: MotionSeed | null;
  onNavigate: (route: RouteId) => void;
  onSelectMedia: (mediaId: string) => void;
  onNotify: NotifyFn;
  onRefreshMedia: () => Promise<void>;
}

const LAYOUT_LABELS: Record<SceneLayoutPreset, string> = {
  split_branded: "SMOKING33 · Moldura",
  split_clean: "Split sem moldura",
  content_hero_pip: "Tela cheia + Webcam PiP",
  content_fullscreen: "100% Conteúdo",
  camera_focus: "Foco na Câmera",
  camera_hero_pip: "Câmera grande (68%)",
  callout_zoom: "Zoom Destaque + Callout",
};

const QUICK_CHAT_PROMPTS = [
  "Nos primeiros 4s coloca foco na câmera com título 'ISSO MUDOU TUDO!'",
  "Entre 0:10 e 0:18 destaca o conteúdo em tela cheia com minha webcam no canto superior direito",
  "Entre 0:15 e 0:21 dá zoom no detalhe com aviso 'OLHA ISSO'",
  "Deixa o vídeo inteiro no padrão SMOKING33",
];

function drawRoundedRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawVideoCrop(
  ctx: CanvasRenderingContext2D,
  video: HTMLVideoElement,
  rect: NormalizedRect,
  dx: number,
  dy: number,
  dw: number,
  dh: number,
  fitBlur = false,
) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (vw < 16 || vh < 16) return;

  const sx = rect.x * vw;
  const sy = rect.y * vh;
  const sw = Math.max(8, rect.w * vw);
  const sh = Math.max(8, rect.h * vh);
  const targetAspect = dw / dh;
  const srcAspect = sw / sh;

  if (fitBlur) {
    ctx.save();
    ctx.filter = "blur(10px)";
    const bgScale = Math.max(dw / sw, dh / sh);
    const bgW = sw * bgScale;
    const bgH = sh * bgScale;
    ctx.drawImage(video, sx, sy, sw, sh, dx + (dw - bgW) / 2, dy + (dh - bgH) / 2, bgW, bgH);
    ctx.restore();
    const fgScale = Math.min(dw / sw, dh / sh);
    const fgW = sw * fgScale;
    const fgH = sh * fgScale;
    ctx.drawImage(video, sx, sy, sw, sh, dx + (dw - fgW) / 2, dy + (dh - fgH) / 2, fgW, fgH);
    return;
  }

  let cw = sw;
  let ch = sh;
  if (srcAspect > targetAspect) cw = sh * targetAspect;
  else ch = sw / targetAspect;
  const cx = sx + (sw - cw) * 0.5;
  const cy = sy + (sh - ch) * 0.5;
  ctx.drawImage(video, cx, cy, cw, ch, dx, dy, dw, dh);
}

function computeZoomedContentRect(base: NormalizedRect | null | undefined, sc: MotionScene): NormalizedRect {
  const b = base ?? { x: 0, y: 0, w: 1, h: 1 };
  const z = Math.max(1, Math.min(3, sc.content_focus.zoom || 1));
  if (z <= 1.02) return b;
  const zw = Math.max(0.15, b.w / z);
  const zh = Math.max(0.15, b.h / z);
  const fx = b.x + b.w * sc.content_focus.focus_x;
  const fy = b.y + b.h * sc.content_focus.focus_y;
  return {
    x: Math.max(b.x, Math.min(b.x + b.w - zw, fx - zw * 0.5)),
    y: Math.max(b.y, Math.min(b.y + b.h - zh, fy - zh * 0.5)),
    w: zw,
    h: zh,
  };
}

export const MotionStudioView: React.FC<MotionStudioViewProps> = ({
  media,
  tools,
  jobs,
  seed,
  onNavigate,
  onSelectMedia,
  onNotify,
  onRefreshMedia,
}) => {
  const videoMedia = useMemo(() => media.filter((m) => m.kind === "video" || m.kind === "video_only"), [media]);
  const ffmpegTool = tools.find((t) => t.id === "ffmpeg");

  const [selectedMediaId, setSelectedMediaId] = useState<string>(seed?.mediaId ?? videoMedia[0]?.id ?? "");
  const [clipStartInput, setClipStartInput] = useState<number>(seed?.startSec ?? 0);
  const [clipEndInput, setClipEndInput] = useState<number>(seed?.endSec ?? 45);
  const [goalInput, setGoalInput] = useState<string>("");

  const [compositions, setCompositions] = useState<AicutComposition[]>([]);
  const [comp, setComp] = useState<AicutComposition | null>(null);
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);

  const [planning, setPlanning] = useState(false);
  const [saving, setSaving] = useState(false);
  const [rendering, setRendering] = useState(false);

  // Interactive AI Chat
  const [chatInput, setChatInput] = useState("");
  const [chatSending, setChatSending] = useState(false);

  // Exported Motion Code viewer (HyperFrames HTML / Remotion JSON / Scene Graph)
  const [showCode, setShowCode] = useState(false);
  const [codeTarget, setCodeTarget] = useState<"hyperframes" | "remotion" | "json">("hyperframes");
  const [exportedCode, setExportedCode] = useState<string>("");

  // Live 9:16 playback
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [playing, setPlaying] = useState(false);
  const [relTime, setRelTime] = useState(0);

  const activeMotionJobs = jobs.filter(
    (j) => j.kind === "motion.render" && (j.status === "queued" || j.status === "running"),
  );

  // Sync seed when navigating directly from a Smart Clips card
  useEffect(() => {
    if (!seed?.mediaId) return;
    setSelectedMediaId(seed.mediaId);
    if (seed.startSec !== undefined) setClipStartInput(Math.floor(seed.startSec));
    if (seed.endSec !== undefined) setClipEndInput(Math.ceil(seed.endSec));
  }, [seed]);

  useEffect(() => {
    if (!selectedMediaId && videoMedia.length > 0) {
      setSelectedMediaId(videoMedia[0].id);
    }
  }, [selectedMediaId, videoMedia]);

  const loadCompositions = useCallback(async () => {
    try {
      const res = await runAction<{ compositions: AicutComposition[] }>("motion.list");
      const list = res.data.compositions ?? [];
      setCompositions(list);
      if (!comp && list.length > 0) {
        setComp(list[0]);
        setSelectedSceneId(list[0].scenes[0]?.id ?? null);
      }
    } catch {
      // ignore if offline
    }
  }, [comp]);

  useEffect(() => {
    void loadCompositions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refresh exported HyperFrames / Remotion code whenever drawer is open or comp changes
  useEffect(() => {
    if (!showCode || !comp) return;
    void runAction<MotionExportCodeOutput>("motion.export_code", {
      id: comp.id,
      target: codeTarget,
    })
      .then((res) => setExportedCode(res.data.code))
      .catch(() => {});
  }, [showCode, codeTarget, comp]);

  const activeScene = useMemo(() => {
    if (!comp || comp.scenes.length === 0) return null;
    return (
      comp.scenes.find((s) => relTime >= s.start_sec && relTime < s.end_sec) ??
      comp.scenes[comp.scenes.length - 1]
    );
  }, [comp, relTime]);

  const selectedScene = useMemo(() => {
    if (!comp) return null;
    return comp.scenes.find((s) => s.id === selectedSceneId) ?? activeScene ?? comp.scenes[0] ?? null;
  }, [comp, selectedSceneId, activeScene]);

  // 60fps Live 9:16 Dynamic Motion Canvas Renderer
  useEffect(() => {
    let raf = 0;
    const renderFrame = () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (canvas && comp) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          const curRel = video ? Math.max(0, Math.min(comp.duration_sec, video.currentTime - comp.clip_start_sec)) : relTime;
          if (video && !video.paused) {
            if (video.currentTime >= comp.clip_end_sec) {
              video.currentTime = comp.clip_start_sec;
            }
            setRelTime(curRel);
          }

          const sc =
            comp.scenes.find((s) => curRel >= s.start_sec && curRel < s.end_sec) ??
            comp.scenes[comp.scenes.length - 1];

          const W = canvas.width;
          const H = canvas.height;
          ctx.fillStyle = "#08090d";
          ctx.fillRect(0, 0, W, H);

          if (sc && video && video.readyState >= 2) {
            const camR = comp.webcam_rect;
            const cntR = computeZoomedContentRect(comp.content_rect, sc);
            const sx = W / 1080;
            const sy = H / 1920;

            if (sc.layout === "split_branded") {
              const bg = ctx.createLinearGradient(0, 0, 0, H);
              bg.addColorStop(0, "#1d0f2e");
              bg.addColorStop(1, "#181022");
              ctx.fillStyle = bg;
              ctx.fillRect(0, 0, W, H);
              ctx.fillStyle = "#e8bd52";
              ctx.font = "700 11px monospace";
              ctx.textAlign = "center";
              ctx.fillText("SMOKING33", W / 2, 23);

              const tx = 30 * sx, ty = 150 * sy, tw = 1020 * sx, th = 582 * sy;
              const bx = 30 * sx, by = 765 * sy, bw = 1020 * sx, bh = 889 * sy;

              ctx.save();
              drawRoundedRect(ctx, tx, ty, tw, th, 8);
              ctx.clip();
              drawVideoCrop(ctx, video, camR, tx, ty, tw, th, false);
              ctx.restore();
              drawRoundedRect(ctx, tx, ty, tw, th, 8);
              ctx.strokeStyle = "#eab308";
              ctx.lineWidth = 2;
              ctx.stroke();

              ctx.save();
              drawRoundedRect(ctx, bx, by, bw, bh, 8);
              ctx.clip();
              drawVideoCrop(ctx, video, cntR, bx, by, bw, bh, sc.content_focus.fit_blur);
              ctx.restore();
              drawRoundedRect(ctx, bx, by, bw, bh, 8);
              ctx.strokeStyle = "#eab308";
              ctx.lineWidth = 2;
              ctx.stroke();
            } else if (sc.layout === "split_clean") {
              const th = Math.round(H * (730 / 1920));
              drawVideoCrop(ctx, video, camR, 0, 0, W, th, false);
              drawVideoCrop(ctx, video, cntR, 0, th, W, H - th, sc.content_focus.fit_blur);
            } else if (sc.layout === "camera_hero_pip") {
              const th = Math.round(H * (1260 / 1920));
              drawVideoCrop(ctx, video, camR, 0, 0, W, th, false);
              drawVideoCrop(ctx, video, cntR, 0, th, W, H - th, sc.content_focus.fit_blur);
            } else if (sc.layout === "camera_focus") {
              drawVideoCrop(ctx, video, camR, 0, 0, W, H, false);
            } else if (sc.layout === "content_fullscreen") {
              drawVideoCrop(ctx, video, cntR, 0, 0, W, H, sc.content_focus.fit_blur);
            } else {
              // content_hero_pip or callout_zoom
              drawVideoCrop(ctx, video, cntR, 0, 0, W, H, sc.content_focus.fit_blur);
              if (sc.layout === "callout_zoom") {
                ctx.strokeStyle = sc.overlay.accent_hex || "#eab308";
                ctx.lineWidth = 3;
                drawRoundedRect(ctx, W * 0.24, H * 0.34, W * 0.52, H * 0.25, 10);
                ctx.stroke();
              }
              if (sc.pip.visible) {
                const pw = W * Math.max(0.24, Math.min(0.48, sc.pip.width_pct));
                const ph = pw / 1.75;
                const pad = 10;
                const px = sc.pip.corner.includes("right") ? W - pw - pad : pad;
                const py = sc.pip.corner.includes("bottom") ? H - ph - 66 : 28;
                ctx.save();
                drawRoundedRect(ctx, px, py, pw, ph, 8);
                ctx.clip();
                drawVideoCrop(ctx, video, camR, px, py, pw, ph, false);
                ctx.restore();
                drawRoundedRect(ctx, px, py, pw, ph, 8);
                ctx.strokeStyle = sc.pip.border_hex || "#eab308";
                ctx.lineWidth = 2.5;
                ctx.stroke();
              }
            }

            // Kinetic Headline / Badge / Callout live preview
            if (sc.overlay.badge) {
              ctx.fillStyle = "rgba(10,11,16,0.82)";
              drawRoundedRect(ctx, 10, 10, 88, 20, 4);
              ctx.fill();
              ctx.fillStyle = sc.overlay.accent_hex || "#eab308";
              ctx.font = "700 9px monospace";
              ctx.textAlign = "left";
              ctx.fillText(sc.overlay.badge.toUpperCase().slice(0, 14), 16, 23);
            }
            if (sc.overlay.headline) {
              ctx.fillStyle = "rgba(8,9,13,0.84)";
              drawRoundedRect(ctx, 16, 36, W - 32, 32, 8);
              ctx.fill();
              ctx.strokeStyle = sc.overlay.accent_hex || "#eab308";
              ctx.lineWidth = 1.5;
              ctx.stroke();
              ctx.fillStyle = "#ffffff";
              ctx.font = "800 13px sans-serif";
              ctx.textAlign = "center";
              ctx.fillText(sc.overlay.headline.toUpperCase().slice(0, 26), W / 2, 57);
            }
            if (sc.overlay.callout_text) {
              ctx.fillStyle = "rgba(8,9,13,0.88)";
              drawRoundedRect(ctx, W * 0.2, H * 0.27, W * 0.6, 26, 6);
              ctx.fill();
              ctx.fillStyle = sc.overlay.accent_hex || "#eab308";
              ctx.font = "700 11px sans-serif";
              ctx.textAlign = "center";
              ctx.fillText(sc.overlay.callout_text.slice(0, 22), W / 2, H * 0.27 + 17);
            }
          }
        }
      }
      raf = requestAnimationFrame(renderFrame);
    };
    raf = requestAnimationFrame(renderFrame);
    return () => cancelAnimationFrame(raf);
  }, [comp, relTime]);

  const seekTo = (nextRel: number) => {
    if (!comp) return;
    const clamped = Math.max(0, Math.min(comp.duration_sec, nextRel));
    setRelTime(clamped);
    if (videoRef.current) {
      videoRef.current.currentTime = comp.clip_start_sec + clamped;
    }
  };

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v || !comp) return;
    if (v.paused) {
      if (v.currentTime < comp.clip_start_sec || v.currentTime >= comp.clip_end_sec) {
        v.currentTime = comp.clip_start_sec;
      }
      void v.play().then(() => setPlaying(true)).catch(() => {});
    } else {
      v.pause();
      setPlaying(false);
    }
  };

  const handlePlanWithAi = async () => {
    if (!selectedMediaId) return;
    setPlanning(true);
    try {
      const res = await runAction<{ composition: AicutComposition; assistant_reply: string }>("motion.plan", {
        media_id: selectedMediaId,
        clip_start_sec: clipStartInput,
        clip_end_sec: clipEndInput > clipStartInput + 1 ? clipEndInput : undefined,
        title: seed?.title,
        goal: goalInput.trim() || undefined,
      });
      setComp(res.data.composition);
      setSelectedSceneId(res.data.composition.scenes[0]?.id ?? null);
      setRelTime(0);
      await loadCompositions();
      onNotify("success", "Roteiro dinâmico criado", res.data.assistant_reply);
    } catch (e) {
      const msg = e instanceof AicutActionError ? e.message : (e as Error).message;
      onNotify("error", "Erro ao planejar cenas", msg);
    } finally {
      setPlanning(false);
    }
  };

  const handleChatSend = async (textOverride?: string) => {
    const msg = (textOverride ?? chatInput).trim();
    if (!comp || !msg) return;
    setChatSending(true);
    if (!textOverride) setChatInput("");
    try {
      const res = await runAction<{ composition: AicutComposition; assistant_reply: string }>("motion.chat_edit", {
        id: comp.id,
        message: msg,
      });
      setComp(res.data.composition);
      await loadCompositions();
    } catch (e) {
      const err = e instanceof AicutActionError ? e.message : (e as Error).message;
      onNotify("error", "Erro no chat de edição", err);
    } finally {
      setChatSending(false);
    }
  };

  const updateSelectedScene = (patch: Partial<MotionScene>) => {
    if (!comp || !selectedScene) return;
    const nextScenes = comp.scenes.map((s) => (s.id === selectedScene.id ? { ...s, ...patch } : s));
    setComp({ ...comp, scenes: nextScenes });
  };

  const handleSplitAtPlayhead = () => {
    if (!comp || !activeScene) return;
    if (relTime <= activeScene.start_sec + 0.8 || relTime >= activeScene.end_sec - 0.8) {
      onNotify("info", "Posicione a agulha", "Mova o player para o meio de uma cena para dividir.");
      return;
    }
    const splitT = Math.round(relTime * 100) / 100;
    const second: MotionScene = {
      ...activeScene,
      id: `sc-${Math.random().toString(36).slice(2, 8)}`,
      title: `${activeScene.title} (parte 2)`,
      start_sec: splitT,
      end_sec: activeScene.end_sec,
    };
    const nextScenes = comp.scenes.flatMap((s) =>
      s.id === activeScene.id ? [{ ...s, end_sec: splitT }, second] : [s],
    );
    setComp({ ...comp, scenes: nextScenes });
    setSelectedSceneId(second.id);
  };

  const handleDeleteSelectedScene = () => {
    if (!comp || !selectedScene || comp.scenes.length <= 1) return;
    const nextScenes = comp.scenes.filter((s) => s.id !== selectedScene.id);
    setComp({ ...comp, scenes: nextScenes });
    setSelectedSceneId(nextScenes[0]?.id ?? null);
  };

  const handleSaveManual = async () => {
    if (!comp) return;
    setSaving(true);
    try {
      const res = await runAction<AicutComposition>("motion.save", { composition: comp });
      setComp(res.data);
      await loadCompositions();
      onNotify("success", "Composição salva", `${res.data.scenes.length} cena(s) sincronizadas.`);
    } catch (e) {
      onNotify("error", "Erro ao salvar", (e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const handleRender = async () => {
    if (!comp) return;
    setRendering(true);
    try {
      // Save latest tweaks before rendering
      await runAction("motion.save", { composition: comp });
      const res = await runAction<{ job_id: string }>("motion.render", { id: comp.id, wait: false });
      onNotify(
        "success",
        "Renderização Motion iniciada",
        `Job ${res.data.job_id.slice(0, 8)} em GPU NVENC — o Short dinâmico irá direto para a Biblioteca!`,
      );
      void onRefreshMedia();
    } catch (e) {
      const msg = e instanceof AicutActionError ? e.message : (e as Error).message;
      onNotify("error", "Erro ao renderizar", msg);
    } finally {
      setRendering(false);
    }
  };

  return (
    <>
      <PageHeader
        {...pageMeta("motion-studio")}
        actions={
          <div className="row gap-xs">
            <button type="button" className="btn btn-sm" onClick={() => onNavigate("framing")}>
              <Crop size={13} aria-hidden /> Enquadramento
            </button>
            <button type="button" className="btn btn-sm" onClick={() => onNavigate("smart-clips")}>
              <Scissors size={13} aria-hidden /> Cortes IA
            </button>
          </div>
        }
      />

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
          Instale o FFmpeg em Configurações → Ferramentas para renderizar composições Motion.
        </Callout>
      )}

      <ActiveJobsPanel jobs={activeMotionJobs} title="Renderizações Motion em andamento" tone="accent" onViewAll={() => onNavigate("jobs")} />

      {videoMedia.length === 0 ? (
        <EmptyState
          icon={Film}
          title="nenhum vídeo na Biblioteca"
          description="Baixe ou importe um vídeo para dirigir layouts dinâmicos em código com a IA."
          actions={
            <button type="button" className="btn btn-primary" onClick={() => onNavigate("ingest")}>
              Baixar & Importar
            </button>
          }
        />
      ) : (
        <div className="stack">
          {/* Compact setup / generator bar */}
          <Panel
            title="Nova composição dinâmica ou projeto salvo"
            icon={Wand2}
            actions={
              compositions.length > 0 && (
                <select
                  className="select select-sm select-auto"
                  aria-label="Composições salvas"
                  value={comp?.id ?? ""}
                  onChange={(e) => {
                    const found = compositions.find((c) => c.id === e.target.value) ?? null;
                    setComp(found);
                    setSelectedSceneId(found?.scenes[0]?.id ?? null);
                    setRelTime(0);
                  }}
                >
                  {compositions.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.title} ({c.scenes.length} cenas · {formatDuration(c.duration_sec)})
                    </option>
                  ))}
                </select>
              )
            }
          >
            <div className="grid-3">
              <Field label="Vídeo da Biblioteca" htmlFor="ms-media">
                <select
                  id="ms-media"
                  className="select select-sm"
                  value={selectedMediaId}
                  onChange={(e) => {
                    const mid = e.target.value;
                    setSelectedMediaId(mid);
                    const m = videoMedia.find((x) => x.id === mid);
                    if (m?.metadata.duration_sec) {
                      setClipStartInput(0);
                      setClipEndInput(Math.min(60, Math.floor(m.metadata.duration_sec)));
                    }
                  }}
                >
                  {videoMedia.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.title}
                      {m.metadata.duration_sec ? ` (${formatDuration(m.metadata.duration_sec)})` : ""}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Trecho (Início → Fim em seg.)" htmlFor="ms-start">
                <div className="row gap-xs">
                  <input
                    id="ms-start"
                    type="number"
                    min={0}
                    className="input input-sm"
                    value={clipStartInput}
                    onChange={(e) => setClipStartInput(Math.max(0, Number(e.target.value)))}
                  />
                  <span className="muted text-xs">→</span>
                  <input
                    type="number"
                    min={1}
                    className="input input-sm"
                    value={clipEndInput}
                    onChange={(e) => setClipEndInput(Math.max(1, Number(e.target.value)))}
                  />
                </div>
              </Field>
              <Field label="Direção inicial para a IA (opcional)" htmlFor="ms-goal">
                <div className="row gap-xs">
                  <input
                    id="ms-goal"
                    type="text"
                    className="input input-sm grow"
                    placeholder="Ex.: foco tela cheia nas jogadas, gancho na câmera…"
                    value={goalInput}
                    onChange={(e) => setGoalInput(e.target.value)}
                  />
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    disabled={planning || !selectedMediaId}
                    onClick={() => void handlePlanWithAi()}
                  >
                    <BusyIcon busy={planning} icon={Sparkles} size={13} />
                    Gerar IA
                  </button>
                </div>
              </Field>
            </div>
          </Panel>

          {/* Main 2-column Studio: Left = Live 9:16 Player + Timeline; Right = AI Chat + Inspector */}
          {comp ? (
            <div className="motion-workspace">
              {/* LEFT: Live 9:16 Preview + Interactive Scene Timeline */}
              <Panel
                title={`Preview ao vivo · ${comp.title}`}
                icon={Layers}
                actions={
                  <div className="row gap-xs">
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => setShowCode((v) => !v)}
                    >
                      <Code2 size={13} aria-hidden /> {showCode ? "Ocultar código" : "Código Motion"}
                    </button>
                    {comp.rendered_media_id && (
                      <button
                        type="button"
                        className="btn btn-sm"
                        onClick={() => onSelectMedia(comp.rendered_media_id!)}
                      >
                        <Play size={13} aria-hidden /> Abrir MP4
                      </button>
                    )}
                    <button
                      type="button"
                      className="btn btn-primary btn-sm"
                      disabled={rendering}
                      onClick={() => void handleRender()}
                    >
                      <BusyIcon busy={rendering} icon={Scissors} size={13} />
                      Renderizar (NVENC)
                    </button>
                  </div>
                }
              >
                {/* Hidden synced video decoder feeding the 60fps 9:16 canvas */}
                <video
                  ref={videoRef}
                  src={mediaStreamUrl(comp.media_id)}
                  crossOrigin="anonymous"
                  playsInline
                  preload="auto"
                  style={{ display: "none" }}
                  onLoadedMetadata={() => {
                    if (videoRef.current) videoRef.current.currentTime = comp.clip_start_sec;
                  }}
                  onPause={() => setPlaying(false)}
                  onPlay={() => setPlaying(true)}
                />

                <div className="motion-player-stage">
                  <div className="motion-canvas-shell">
                    <canvas ref={canvasRef} width={270} height={480} className="motion-canvas" />
                  </div>

                  {/* Playback scrubber */}
                  <div className="motion-playback-bar">
                    <button type="button" className="btn btn-sm btn-primary" onClick={togglePlay}>
                      {playing ? <Pause size={13} aria-hidden /> : <Play size={13} aria-hidden />}
                    </button>
                    <span className="motion-timecode">
                      {formatDuration(relTime)} / {formatDuration(comp.duration_sec)}
                    </span>
                    <input
                      type="range"
                      min={0}
                      max={comp.duration_sec}
                      step={0.1}
                      value={relTime}
                      onChange={(e) => seekTo(Number(e.target.value))}
                      className="grow"
                    />
                    {activeScene && <Badge tone="accent">{LAYOUT_LABELS[activeScene.layout]}</Badge>}
                  </div>

                  {/* Multi-scene interactive timeline */}
                  <div className="stack" style={{ width: "100%", gap: 6 }}>
                    <div className="row row-between">
                      <span className="section-label">Timeline de cenas dinâmicas ({comp.scenes.length})</span>
                      <div className="row gap-xs">
                        <button type="button" className="btn btn-sm" onClick={handleSplitAtPlayhead}>
                          <Plus size={12} aria-hidden /> Dividir na agulha
                        </button>
                        <button
                          type="button"
                          className="btn btn-sm"
                          disabled={comp.scenes.length <= 1 || !selectedScene}
                          onClick={handleDeleteSelectedScene}
                        >
                          <Trash2 size={12} aria-hidden /> Excluir cena
                        </button>
                      </div>
                    </div>

                    <div className="motion-timeline">
                      {comp.scenes.map((sc) => {
                        const weight = Math.max(0.5, sc.end_sec - sc.start_sec);
                        const isSel = selectedScene?.id === sc.id;
                        const isPlayingNow = activeScene?.id === sc.id;
                        return (
                          <button
                            key={sc.id}
                            type="button"
                            className="motion-scene-block"
                            data-layout={sc.layout}
                            data-selected={isSel}
                            data-playing={isPlayingNow}
                            style={{ flex: weight }}
                            onClick={() => {
                              setSelectedSceneId(sc.id);
                              seekTo(sc.start_sec + 0.05);
                            }}
                          >
                            <span className="motion-scene-title">{sc.title}</span>
                            <span className="motion-scene-meta">
                              {LAYOUT_LABELS[sc.layout]} · {formatDuration(sc.start_sec)}–{formatDuration(sc.end_sec)}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Live HyperFrames HTML / Remotion Props / Scene JSON viewer */}
                  {showCode && (
                    <div className="inset stack" style={{ width: "100%" }}>
                      <div className="row row-between wrap gap-xs">
                        <SegmentedControl<"hyperframes" | "remotion" | "json">
                          size="sm"
                          ariaLabel="Formato de código Motion"
                          value={codeTarget}
                          onChange={setCodeTarget}
                          options={[
                            { id: "hyperframes", label: "HyperFrames HTML/GSAP" },
                            { id: "remotion", label: "Remotion Props" },
                            { id: "json", label: "Scene Graph JSON" },
                          ]}
                        />
                        <button
                          type="button"
                          className="btn btn-sm"
                          onClick={() => void copyToClipboard(exportedCode, onNotify, "Código Motion")}
                        >
                          <Copy size={12} aria-hidden /> Copiar código
                        </button>
                      </div>
                      <pre className="motion-code-pre">{exportedCode || "// carregando código motion..."}</pre>
                    </div>
                  )}
                </div>
              </Panel>

              {/* RIGHT: AI Interactive Chat Director + Selected Scene Quick Inspector */}
              <div className="stack">
                <Panel title="Chat com a IA Diretora de Layout" icon={Bot}>
                  <div className="motion-chat-feed">
                    {comp.chat_history.length === 0 ? (
                      <span className="muted text-xs">
                        Converse com a IA para alterar qualquer trecho: trocar layout, destacar a tela cheia, mover a webcam PiP, adicionar títulos ou dar zoom em detalhes.
                      </span>
                    ) : (
                      comp.chat_history.map((m, idx) => (
                        <div key={idx} className="motion-msg" data-role={m.role}>
                          {m.text}
                        </div>
                      ))
                    )}
                  </div>

                  <div className="row wrap gap-xs">
                    {QUICK_CHAT_PROMPTS.map((q) => (
                      <button
                        key={q}
                        type="button"
                        className="chip chip-sm"
                        disabled={chatSending}
                        onClick={() => void handleChatSend(q)}
                      >
                        {q}
                      </button>
                    ))}
                  </div>

                  <form
                    className="motion-chat-input-row"
                    onSubmit={(e) => {
                      e.preventDefault();
                      void handleChatSend();
                    }}
                  >
                    <input
                      type="text"
                      className="input grow"
                      placeholder="Peça à IA: ex. 'de 0:12 a 0:20 deixa tela cheia com minha webcam no canto superior direito'…"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      disabled={chatSending}
                    />
                    <button type="submit" className="btn btn-primary" disabled={chatSending || !chatInput.trim()}>
                      <BusyIcon busy={chatSending} icon={Send} size={14} />
                      Enviar
                    </button>
                  </form>
                </Panel>

                {/* Compact manual inspector for selected scene */}
                {selectedScene && (
                  <Panel
                    title={`Ajuste fino · ${selectedScene.title}`}
                    icon={Sparkles}
                    actions={
                      <button type="button" className="btn btn-sm btn-primary" disabled={saving} onClick={() => void handleSaveManual()}>
                        <BusyIcon busy={saving} icon={Save} size={12} />
                        Salvar
                      </button>
                    }
                  >
                    <div className="grid-2">
                      <Field label="Layout desta cena" htmlFor="ms-lay">
                        <select
                          id="ms-lay"
                          className="select select-sm"
                          value={selectedScene.layout}
                          onChange={(e) => updateSelectedScene({ layout: e.target.value as SceneLayoutPreset })}
                        >
                          {(Object.keys(LAYOUT_LABELS) as SceneLayoutPreset[]).map((k) => (
                            <option key={k} value={k}>
                              {LAYOUT_LABELS[k]}
                            </option>
                          ))}
                        </select>
                      </Field>

                      <Field label="Canto da webcam flutuante (PiP)" htmlFor="ms-pip">
                        <select
                          id="ms-pip"
                          className="select select-sm"
                          value={selectedScene.pip.corner}
                          onChange={(e) =>
                            updateSelectedScene({
                              pip: { ...selectedScene.pip, corner: e.target.value as PipCorner },
                            })
                          }
                        >
                          <option value="top_right">Superior direito</option>
                          <option value="top_left">Superior esquerdo</option>
                          <option value="bottom_right">Inferior direito</option>
                          <option value="bottom_left">Inferior esquerdo</option>
                        </select>
                      </Field>
                    </div>

                    <div className="grid-2">
                      <Field label="Título animado no topo (opcional)" htmlFor="ms-hl">
                        <input
                          id="ms-hl"
                          type="text"
                          className="input input-sm"
                          placeholder="Ex.: OLHA O QUE ACONTECEU!"
                          value={selectedScene.overlay.headline ?? ""}
                          onChange={(e) =>
                            updateSelectedScene({
                              overlay: { ...selectedScene.overlay, headline: e.target.value || null },
                            })
                          }
                        />
                      </Field>
                      <Field label="Zoom no conteúdo (1.0× – 2.5×)" htmlFor="ms-zoom">
                        <input
                          id="ms-zoom"
                          type="number"
                          min={1.0}
                          max={2.6}
                          step={0.1}
                          className="input input-sm"
                          value={selectedScene.content_focus.zoom}
                          onChange={(e) =>
                            updateSelectedScene({
                              content_focus: {
                                ...selectedScene.content_focus,
                                zoom: Math.max(1, Math.min(2.6, Number(e.target.value) || 1)),
                              },
                            })
                          }
                        />
                      </Field>
                    </div>

                    <div className="row wrap gap-md">
                      <Checkbox
                        checked={selectedScene.content_focus.fit_blur}
                        onChange={(v) =>
                          updateSelectedScene({
                            content_focus: { ...selectedScene.content_focus, fit_blur: v },
                          })
                        }
                        label="Mostrar 16:9 inteiro + blur (sem cortar laterais)"
                      />
                      <Checkbox
                        checked={comp.burn_subtitles}
                        onChange={(v) => setComp({ ...comp, burn_subtitles: v })}
                        label="Queimar legendas karaokê"
                      />
                    </div>
                  </Panel>
                )}
              </div>
            </div>
          ) : (
            <EmptyState
              icon={Layers}
              title="pronto para dirigir seu primeiro Short dinâmico"
              description="Escolha um vídeo acima e clique em 'Gerar IA' — a IA cria a timeline de cenas dinâmicas usando suas áreas salvas de Enquadramento, e você pode conversar com ela no chat para personalizar cada momento."
              actions={
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={planning || !selectedMediaId}
                  onClick={() => void handlePlanWithAi()}
                >
                  <BusyIcon busy={planning} icon={Sparkles} size={14} />
                  Gerar Roteiro Dinâmico com IA
                </button>
              }
            />
          )}
        </div>
      )}
    </>
  );
};
