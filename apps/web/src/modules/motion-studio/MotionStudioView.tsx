import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bot,
  Code2,
  Copy,
  Download,
  Film,
  Pause,
  Play,
  Send,
  Sparkles,
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
  RouteId,
  ToolStatus,
} from "../../lib/types";
import { copyToClipboard } from "../../lib/ui";
import { ActiveJobsPanel, Badge, BusyIcon, Callout, EmptyState } from "../../components";
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
  extraZoom = 1.0,
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
  const z = Math.max(1, Math.min(1.4, extraZoom));
  cw /= z;
  ch /= z;
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
  const [clipStart, setClipStart] = useState<number>(seed?.startSec ?? 0);
  const [clipEnd, setClipEnd] = useState<number>(seed?.endSec ?? 45);

  const [comp, setComp] = useState<AicutComposition | null>(null);
  const [liveHtmlDoc, setLiveHtmlDoc] = useState<string>("");
  const [showCode, setShowCode] = useState<boolean>(false);

  const [planning, setPlanning] = useState<boolean>(false);
  const [chatInput, setChatInput] = useState<string>("");
  const [chatSending, setChatSending] = useState<boolean>(false);
  const [rendering, setRendering] = useState<boolean>(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const [playing, setPlaying] = useState<boolean>(false);
  const [relTime, setRelTime] = useState<number>(0);

  const activeMotionJobs = jobs.filter(
    (j) => j.kind === "motion.render" && (j.status === "queued" || j.status === "running"),
  );

  // Sync seed when coming from Smart Clips
  useEffect(() => {
    if (!seed?.mediaId) return;
    setSelectedMediaId(seed.mediaId);
    if (seed.startSec !== undefined) setClipStart(Math.floor(seed.startSec));
    if (seed.endSec !== undefined) setClipEnd(Math.ceil(seed.endSec));
  }, [seed]);

  useEffect(() => {
    if (!selectedMediaId && videoMedia.length > 0) {
      setSelectedMediaId(videoMedia[0].id);
      const dur = videoMedia[0].metadata.duration_sec;
      if (dur) {
        setClipStart(0);
        setClipEnd(Math.min(60, Math.floor(dur)));
      }
    }
  }, [selectedMediaId, videoMedia]);

  // Load most recent composition for the selected video if one exists
  useEffect(() => {
    if (!selectedMediaId) return;
    void runAction<{ compositions: AicutComposition[] }>("motion.list", { media_id: selectedMediaId })
      .then((res) => {
        const list = res.data.compositions ?? [];
        if (list.length > 0) {
          setComp(list[0]);
          setClipStart(Math.floor(list[0].clip_start_sec));
          setClipEnd(Math.ceil(list[0].clip_end_sec));
          setRelTime(0);
        } else {
          setComp(null);
          setLiveHtmlDoc("");
        }
      })
      .catch(() => {});
  }, [selectedMediaId]);

  // Fetch wrapped live HyperFrames HTML document whenever composition updates
  useEffect(() => {
    if (!comp) {
      setLiveHtmlDoc("");
      return;
    }
    void runAction<MotionExportCodeOutput>("motion.export_code", {
      id: comp.id,
      target: "hyperframes",
    })
      .then((res) => setLiveHtmlDoc(res.data.code))
      .catch(() => {});
  }, [comp]);

  // Auto-scroll chat to latest AI reply
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [comp?.chat_history.length, chatSending]);

  const postSeekToHyperframes = useCallback((t: number) => {
    iframeRef.current?.contentWindow?.postMessage({ type: "aicut_seek", time: t }, "*");
  }, []);

  // 60fps Live Video Choreography + Synchronized HyperFrames Overlay loop
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (canvas && comp) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          const curRel = video
            ? Math.max(0, Math.min(comp.duration_sec, video.currentTime - comp.clip_start_sec))
            : relTime;
          if (video && !video.paused) {
            if (video.currentTime >= comp.clip_end_sec) {
              video.currentTime = comp.clip_start_sec;
            }
            setRelTime(curRel);
          }

          // Synchronize bespoke HyperFrames HTML/CSS/SVG motion iframe to exact video time
          postSeekToHyperframes(curRel);

          const sc =
            comp.scenes.find((s) => curRel >= s.start_sec && curRel < s.end_sec) ??
            comp.scenes[comp.scenes.length - 1];

          const W = canvas.width;
          const H = canvas.height;
          ctx.save();
          ctx.fillStyle = "#06070a";
          ctx.fillRect(0, 0, W, H);

          if (sc && video && video.readyState >= 2) {
            const scDur = Math.max(0.2, sc.end_sec - sc.start_sec);
            const scProg = Math.max(0, Math.min(1, (curRel - sc.start_sec) / scDur));
            const fx = sc.effects;
            const dynZoom = fx && fx.camera_zoom_end > 1.0 ? 1 + (fx.camera_zoom_end - 1) * scProg : 1.0;

            // Optional brief impact shake at scene start
            if (fx && fx.impact_shake > 0.05 && curRel - sc.start_sec < 0.35) {
              const decay = 1 - (curRel - sc.start_sec) / 0.35;
              const amp = fx.impact_shake * 6 * decay;
              ctx.translate(Math.sin(curRel * 70) * amp, Math.cos(curRel * 85) * amp);
            }

            // Color mood filter
            if (fx?.color_mood === "vibrant") ctx.filter = "contrast(1.06) saturate(1.22)";
            else if (fx?.color_mood === "dramatic") ctx.filter = "contrast(1.12) brightness(0.95) saturate(0.88)";
            else if (fx?.color_mood === "cyber") ctx.filter = "contrast(1.08) saturate(1.25)";

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
              ctx.fillText("SMOKING33", W / 2, 24);

              const tx = 30 * sx, ty = 150 * sy, tw = 1020 * sx, th = 582 * sy;
              const bx = 30 * sx, by = 765 * sy, bw = 1020 * sx, bh = 889 * sy;

              ctx.save();
              drawRoundedRect(ctx, tx, ty, tw, th, 8);
              ctx.clip();
              drawVideoCrop(ctx, video, camR, tx, ty, tw, th, false, 1.0);
              ctx.restore();
              drawRoundedRect(ctx, tx, ty, tw, th, 8);
              ctx.strokeStyle = "#eab308";
              ctx.lineWidth = 2;
              ctx.stroke();

              ctx.save();
              drawRoundedRect(ctx, bx, by, bw, bh, 8);
              ctx.clip();
              drawVideoCrop(ctx, video, cntR, bx, by, bw, bh, sc.content_focus.fit_blur, dynZoom);
              ctx.restore();
              drawRoundedRect(ctx, bx, by, bw, bh, 8);
              ctx.strokeStyle = "#eab308";
              ctx.lineWidth = 2;
              ctx.stroke();
            } else if (sc.layout === "split_clean") {
              const th = Math.round(H * (730 / 1920));
              drawVideoCrop(ctx, video, camR, 0, 0, W, th, false, 1.0);
              drawVideoCrop(ctx, video, cntR, 0, th, W, H - th, sc.content_focus.fit_blur, dynZoom);
            } else if (sc.layout === "camera_hero_pip") {
              const th = Math.round(H * (1260 / 1920));
              drawVideoCrop(ctx, video, camR, 0, 0, W, th, false, dynZoom);
              drawVideoCrop(ctx, video, cntR, 0, th, W, H - th, sc.content_focus.fit_blur, 1.0);
            } else if (sc.layout === "camera_focus") {
              drawVideoCrop(ctx, video, camR, 0, 0, W, H, false, dynZoom);
            } else if (sc.layout === "content_fullscreen") {
              drawVideoCrop(ctx, video, cntR, 0, 0, W, H, sc.content_focus.fit_blur, dynZoom);
            } else {
              drawVideoCrop(ctx, video, cntR, 0, 0, W, H, sc.content_focus.fit_blur, dynZoom);
              if (sc.pip.visible) {
                const pw = W * Math.max(0.24, Math.min(0.46, sc.pip.width_pct));
                const ph = pw / 1.75;
                const pad = 10;
                const px = sc.pip.corner.includes("right") ? W - pw - pad : pad;
                const py = sc.pip.corner.includes("bottom") ? H - ph - 72 : 30;
                ctx.save();
                drawRoundedRect(ctx, px, py, pw, ph, 9);
                ctx.clip();
                drawVideoCrop(ctx, video, camR, px, py, pw, ph, false, 1.0);
                ctx.restore();
                drawRoundedRect(ctx, px, py, pw, ph, 9);
                ctx.strokeStyle = sc.pip.border_hex || "#eab308";
                ctx.lineWidth = 2.5;
                ctx.stroke();
              }
            }

            // Optional cinematic vignette
            if (fx?.vignette) {
              ctx.filter = "none";
              const vig = ctx.createRadialGradient(W / 2, H / 2, W * 0.3, W / 2, H / 2, H * 0.65);
              vig.addColorStop(0, "rgba(0,0,0,0)");
              vig.addColorStop(1, "rgba(0,0,0,0.52)");
              ctx.fillStyle = vig;
              ctx.fillRect(0, 0, W, H);
            }
          }
          ctx.restore();
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [comp, relTime, postSeekToHyperframes]);

  const seekTo = (nextRel: number) => {
    if (!comp) return;
    const clamped = Math.max(0, Math.min(comp.duration_sec, nextRel));
    setRelTime(clamped);
    if (videoRef.current) videoRef.current.currentTime = comp.clip_start_sec + clamped;
    postSeekToHyperframes(clamped);
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

  const handleAutoEditWithAi = async (customGoal?: string) => {
    if (!selectedMediaId) return;
    setPlanning(true);
    try {
      const res = await runAction<{ composition: AicutComposition; assistant_reply: string }>("motion.plan", {
        media_id: selectedMediaId,
        clip_start_sec: clipStart,
        clip_end_sec: clipEnd > clipStart + 1 ? clipEnd : undefined,
        title: seed?.title,
        goal: customGoal,
      });
      setComp(res.data.composition);
      setRelTime(0);
    } catch (e) {
      const msg = e instanceof AicutActionError ? e.message : (e as Error).message;
      onNotify("error", "Erro ao analisar vídeo", msg);
    } finally {
      setPlanning(false);
    }
  };

  const handleSendChat = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const msg = chatInput.trim();
    if (!msg) return;
    setChatInput("");

    // If user starts chatting directly before generating an initial edit, let AI analyze & plan with their prompt!
    if (!comp) {
      await handleAutoEditWithAi(msg);
      return;
    }

    setChatSending(true);
    try {
      const res = await runAction<{ composition: AicutComposition; assistant_reply: string }>("motion.chat_edit", {
        id: comp.id,
        message: msg,
      });
      setComp(res.data.composition);
    } catch (err) {
      const em = err instanceof AicutActionError ? err.message : (err as Error).message;
      onNotify("error", "Erro na conversa com a IA", em);
    } finally {
      setChatSending(false);
    }
  };

  const handleRender = async () => {
    if (!comp) return;
    setRendering(true);
    try {
      const res = await runAction<{ job_id: string }>("motion.render", { id: comp.id, wait: false });
      onNotify(
        "success",
        "Renderização iniciada",
        `Aplicando HyperFrames + GPU NVENC (${res.data.job_id.slice(0, 8)}) — o vídeo final irá para a Biblioteca.`,
      );
      void onRefreshMedia();
    } catch (err) {
      const em = err instanceof AicutActionError ? err.message : (err as Error).message;
      onNotify("error", "Erro ao renderizar", em);
    } finally {
      setRendering(false);
    }
  };

  const activeScene = useMemo(() => {
    if (!comp || comp.scenes.length === 0) return null;
    return (
      comp.scenes.find((s) => relTime >= s.start_sec && relTime < s.end_sec) ??
      comp.scenes[comp.scenes.length - 1]
    );
  }, [comp, relTime]);

  if (videoMedia.length === 0) {
    return (
      <EmptyState
        icon={Film}
        title="nenhum vídeo na Biblioteca"
        description="Baixe ou importe um vídeo para conversar com a IA e criar uma edição personalizada."
        actions={
          <button type="button" className="btn btn-primary" onClick={() => onNavigate("ingest")}>
            Baixar & Importar
          </button>
        }
      />
    );
  }

  return (
    <div className="ms-shell">
      {/* Ultra-minimalist single top bar */}
      <header className="ms-topbar">
        <div className="ms-topbar-left">
          <select
            className="select select-sm ms-media-select"
            aria-label="Vídeo para editar"
            value={selectedMediaId}
            onChange={(e) => {
              const mid = e.target.value;
              setSelectedMediaId(mid);
              const m = videoMedia.find((x) => x.id === mid);
              if (m?.metadata.duration_sec) {
                setClipStart(0);
                setClipEnd(Math.min(60, Math.floor(m.metadata.duration_sec)));
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

          <span className="ms-range-pill" title="Trecho em segundos">
            <input
              type="number"
              min={0}
              className="ms-range-input"
              value={clipStart}
              onChange={(e) => setClipStart(Math.max(0, Number(e.target.value)))}
              aria-label="Início (s)"
            />
            <span>s →</span>
            <input
              type="number"
              min={1}
              className="ms-range-input"
              value={clipEnd}
              onChange={(e) => setClipEnd(Math.max(1, Number(e.target.value)))}
              aria-label="Fim (s)"
            />
            <span>s</span>
          </span>
        </div>

        <div className="row gap-xs">
          {comp && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              title="Ver código HyperFrames gerado pela IA"
              onClick={() => setShowCode((v) => !v)}
            >
              <Code2 size={14} aria-hidden />
            </button>
          )}
          {comp?.rendered_media_id && (
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => onSelectMedia(comp.rendered_media_id!)}
            >
              <Play size={13} aria-hidden /> Ver MP4
            </button>
          )}
          <button
            type="button"
            className="btn btn-primary btn-sm"
            disabled={!comp || rendering}
            onClick={() => void handleRender()}
          >
            <BusyIcon busy={rendering} icon={Download} size={13} />
            Renderizar MP4
          </button>
        </div>
      </header>

      {!ffmpegTool?.installed && (
        <Callout
          tone="warning"
          title="FFmpeg necessário"
          actions={
            <button type="button" className="btn btn-primary btn-sm" onClick={() => onNavigate("settings")}>
              Instalar
            </button>
          }
        >
          Instale o FFmpeg em Configurações → Ferramentas para que a IA assista e renderize o vídeo.
        </Callout>
      )}

      <ActiveJobsPanel jobs={activeMotionJobs} title="Renderizando edição IA" tone="accent" onViewAll={() => onNavigate("jobs")} />

      {/* Minimalist 2-Zone Studio: Live 9:16 Preview + Conversational AI Editor */}
      <div className="ms-grid">
        {/* LEFT: Clean 9:16 Live Preview (Video Choreography + Live HyperFrames HTML/CSS/SVG Overlay) */}
        <section className="ms-stage-col" aria-label="Preview ao vivo">
          {comp && (
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
          )}

          <div className="ms-phone-frame">
            <canvas ref={canvasRef} width={288} height={512} className="ms-video-canvas" />
            {liveHtmlDoc && (
              <iframe
                ref={iframeRef}
                title="HyperFrames Motion Layer"
                srcDoc={liveHtmlDoc}
                sandbox="allow-scripts"
                className="ms-hyperframes-iframe"
                onLoad={() => postSeekToHyperframes(relTime)}
              />
            )}
            {!comp && (
              <div className="ms-empty-stage">
                <Wand2 size={28} className="text-accent" aria-hidden />
                <span>Converse com a IA ao lado ou peça para ela assistir e editar este vídeo.</span>
              </div>
            )}
          </div>

          {comp && (
            <div className="ms-player-controls">
              <div className="ms-scrubber-row">
                <button type="button" className="btn btn-sm btn-primary" onClick={togglePlay} aria-label={playing ? "Pausar" : "Reproduzir"}>
                  {playing ? <Pause size={13} aria-hidden /> : <Play size={13} aria-hidden />}
                </button>
                <input
                  type="range"
                  min={0}
                  max={comp.duration_sec}
                  step={0.05}
                  value={relTime}
                  onChange={(e) => seekTo(Number(e.target.value))}
                  className="grow"
                />
                <span className="ms-time">
                  {formatDuration(relTime)} / {formatDuration(comp.duration_sec)}
                </span>
              </div>

              <div className="ms-scene-bar" title="Momentos da edição">
                {comp.scenes.map((sc) => (
                  <button
                    key={sc.id}
                    type="button"
                    className="ms-scene-seg"
                    style={{ flex: Math.max(0.5, sc.end_sec - sc.start_sec) }}
                    data-active={activeScene?.id === sc.id}
                    title={`${sc.title} (${formatDuration(sc.start_sec)}–${formatDuration(sc.end_sec)})`}
                    onClick={() => seekTo(sc.start_sec + 0.05)}
                  />
                ))}
              </div>
            </div>
          )}
        </section>

        {/* RIGHT: Pure Conversational AI Editor Chat */}
        <section className="ms-chat-col" aria-label="Chat com IA Editora">
          <div className="ms-chat-head">
            <div className="ms-chat-head-meta">
              <div className="ms-chat-head-title">
                <Bot size={15} className="text-accent" aria-hidden />
                <span>Editora IA · HyperFrames</span>
                {comp && <Badge tone="accent">{comp.scenes.length} cenas</Badge>}
              </div>
              {comp?.creative_direction && (
                <span className="ms-chat-direction" title={comp.creative_direction}>
                  {comp.creative_direction}
                </span>
              )}
            </div>
            <button
              type="button"
              className="btn btn-sm"
              disabled={planning || !selectedMediaId}
              onClick={() => void handleAutoEditWithAi()}
            >
              <BusyIcon busy={planning} icon={Sparkles} size={13} />
              {comp ? "Reanalisar vídeo" : "Editar com IA"}
            </button>
          </div>

          <div className="ms-chat-messages">
            {!comp || comp.chat_history.length === 0 ? (
              <div className="ms-chat-welcome">
                <div className="ms-chat-welcome-icon">
                  <Sparkles size={24} aria-hidden />
                </div>
                <strong className="text-sm">Sua editora de vídeo em código motion</strong>
                <p className="secondary text-xs">
                  Eu assisto aos frames e leio o que você fala neste vídeo para criar técnicas de edição e animações
                  HyperFrames sob medida — ou podemos planejar juntos por aqui antes de aplicar.
                </p>
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={planning || !selectedMediaId}
                  onClick={() => void handleAutoEditWithAi()}
                >
                  <BusyIcon busy={planning} icon={Wand2} size={14} />
                  {planning ? "Assistindo ao vídeo e criando edição…" : "Assistir vídeo e criar edição automática"}
                </button>
              </div>
            ) : (
              <>
                {comp.chat_history.map((m, idx) => (
                  <div key={idx} className="ms-bubble" data-role={m.role}>
                    {m.text}
                  </div>
                ))}
                {(chatSending || planning) && (
                  <div className="ms-bubble" data-role="assistant">
                    <span className="muted">Assistindo ao vídeo e preparando a edição…</span>
                  </div>
                )}
                <div ref={chatEndRef} />
              </>
            )}
          </div>

          {/* Optional hidden-by-default code peek */}
          {showCode && liveHtmlDoc && (
            <div className="ms-code-drawer">
              <div className="row row-between">
                <span className="section-label">Código HyperFrames (HTML + CSS + SVG)</span>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => void copyToClipboard(liveHtmlDoc, onNotify, "Código HyperFrames")}
                >
                  <Copy size={12} aria-hidden /> Copiar
                </button>
              </div>
              <pre className="ms-code-pre">{liveHtmlDoc}</pre>
            </div>
          )}

          <footer className="ms-chat-footer">
            <form className="ms-chat-form" onSubmit={(e) => void handleSendChat(e)}>
              <input
                type="text"
                className="ms-chat-input"
                placeholder="Converse com a IA: planeje ideias, peça animações específicas ou melhore qualquer momento…"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                disabled={chatSending || planning}
              />
              <button
                type="submit"
                className="btn btn-primary"
                disabled={chatSending || planning || !chatInput.trim()}
              >
                <BusyIcon busy={chatSending || planning} icon={Send} size={14} />
              </button>
            </form>
          </footer>
        </section>
      </div>
    </div>
  );
};
