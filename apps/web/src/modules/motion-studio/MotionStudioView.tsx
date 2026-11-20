import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bot,
  Code2,
  Copy,
  Download,
  Film,
  Loader2,
  Pause,
  Play,
  Scissors,
  Send,
  Sparkles,
  Wand2,
} from "lucide-react";
import { AicutActionError, mediaPreviewUrl, mediaStreamUrl, runAction } from "../../lib/client";
import { formatDuration } from "../../lib/format";
import type {
  AicutComposition,
  FramingGetOutput,
  JobRecord,
  MediaAsset,
  MotionExportCodeOutput,
  MotionScene,
  NormalizedRect,
  NotifyFn,
  RouteId,
  SmartClipPlan,
  SmartClipRun,
  ToolStatus,
} from "../../lib/types";
import { copyToClipboard } from "../../lib/ui";
import { ActiveJobsPanel, Badge, BusyIcon, Callout, EmptyState } from "../../components";
import "./motion-studio.css";

export interface MotionSeed {
  runId?: string;
  clipId?: string;
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

interface PlanClipOption {
  key: string;
  runId: string;
  runTitle: string;
  mediaId: string;
  clip: SmartClipPlan;
  index: number;
  startSec: number;
  endSec: number;
  durationSec: number;
}

const DEFAULT_WEBCAM_RECT: NormalizedRect = { x: 0.7, y: 0.64, w: 0.28, h: 0.33 };

const LAYOUT_SHORT_LABELS: Record<string, string> = {
  split_branded: "SMOKING33",
  split_clean: "Split Limpo",
  content_hero_pip: "Conteúdo + PiP",
  content_fullscreen: "Tela Cheia",
  camera_focus: "Foco Câmera",
  camera_hero_pip: "Câmera Destaque",
  callout_zoom: "Zoom Detalhe",
};

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

function buildPlanClipGoal(opt: PlanClipOption, userPrompt?: string): string {
  const parts: string[] = [
    `Corte selecionado do Plano de Cortes: "${opt.clip.title}" (${formatDuration(opt.startSec)} → ${formatDuration(opt.endSec)}).`,
  ];
  if (opt.clip.hook_first_5s) parts.push(`Gancho inicial (5s): ${opt.clip.hook_first_5s}`);
  if (opt.clip.visual_evidence) parts.push(`O que acontece visualmente: ${opt.clip.visual_evidence}`);
  if (opt.clip.narrative_arc.beginning || opt.clip.narrative_arc.middle || opt.clip.narrative_arc.end) {
    parts.push(
      `Arco narrativo: Início="${opt.clip.narrative_arc.beginning}" | Meio="${opt.clip.narrative_arc.middle}" | Fim="${opt.clip.narrative_arc.end}".`,
    );
  }
  if (userPrompt && userPrompt.trim()) {
    parts.push(`Pedido do criador: ${userPrompt.trim()}`);
  }
  return parts.join("\n");
}

type VideoStatus = "loading" | "proxy" | "ready" | "error";

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
  const ffmpegTool = tools.find((t) => t.id === "ffmpeg");

  // Clip plans loaded directly from Cortes Shorts (Plano de Cortes)
  const [runs, setRuns] = useState<SmartClipRun[]>([]);
  const [loadingRuns, setLoadingRuns] = useState<boolean>(true);
  const [selectedClipKey, setSelectedClipKey] = useState<string>("");

  const [comp, setComp] = useState<AicutComposition | null>(null);
  const [savedFraming, setSavedFraming] = useState<{ webcam: NormalizedRect; content: NormalizedRect | null }>({
    webcam: DEFAULT_WEBCAM_RECT,
    content: null,
  });
  const [liveHtmlDoc, setLiveHtmlDoc] = useState<string>("");
  const [showCode, setShowCode] = useState<boolean>(false);

  const [planning, setPlanning] = useState<boolean>(false);
  const [chatInput, setChatInput] = useState<string>("");
  const [chatSending, setChatSending] = useState<boolean>(false);
  const [rendering, setRendering] = useState<boolean>(false);

  const [useProxyVideo, setUseProxyVideo] = useState<boolean>(false);
  const [videoStatus, setVideoStatus] = useState<VideoStatus>("loading");
  const [playing, setPlaying] = useState<boolean>(false);
  const [relTime, setRelTime] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const compRef = useRef<AicutComposition | null>(null);
  const framingRef = useRef(savedFraming);
  const clipBoundsRef = useRef({ start: 0, duration: 30 });
  const relTimeRef = useRef<number>(0);
  const lastUiSyncRef = useRef<number>(-1);
  const lastIframeSeekRef = useRef<number>(-1);

  const activeMotionJobs = jobs.filter(
    (j) => j.kind === "motion.render" && (j.status === "queued" || j.status === "running"),
  );

  // Load all Plano de Cortes runs
  useEffect(() => {
    let alive = true;
    setLoadingRuns(true);
    void runAction<{ runs: SmartClipRun[] }>("clips.runs_list")
      .then((res) => {
        if (!alive) return;
        setRuns(res.data.runs ?? []);
      })
      .catch(() => {
        if (alive) setRuns([]);
      })
      .finally(() => {
        if (alive) setLoadingRuns(false);
      });
    return () => {
      alive = false;
    };
  }, [media.length, seed?.runId, seed?.clipId]);

  const planClips = useMemo<PlanClipOption[]>(() => {
    const out: PlanClipOption[] = [];
    for (const run of runs) {
      run.clips.forEach((clip, idx) => {
        if (!clip.segments || clip.segments.length === 0) return;
        const startSec = clip.segments[0].start_sec;
        const endSec = clip.segments[clip.segments.length - 1].end_sec;
        const durationSec = Math.max(1, endSec - startSec);
        out.push({
          key: `${run.id}:${clip.id}`,
          runId: run.id,
          runTitle: run.media_title,
          mediaId: run.media_id,
          clip,
          index: idx + 1,
          startSec,
          endSec,
          durationSec,
        });
      });
    }
    return out;
  }, [runs]);

  // Select clip from seed (when arriving from Cortes Shorts -> Motion IA) or default to first clip
  useEffect(() => {
    if (planClips.length === 0) {
      setSelectedClipKey("");
      return;
    }
    if (seed?.runId && seed?.clipId) {
      const exactKey = `${seed.runId}:${seed.clipId}`;
      if (planClips.some((c) => c.key === exactKey)) {
        setSelectedClipKey(exactKey);
        return;
      }
    }
    if (seed?.mediaId && seed.startSec !== undefined) {
      const byRange = planClips.find(
        (c) => c.mediaId === seed.mediaId && Math.abs(c.startSec - seed.startSec!) < 1.5,
      );
      if (byRange) {
        setSelectedClipKey(byRange.key);
        return;
      }
    }
    setSelectedClipKey((curr) => (curr && planClips.some((c) => c.key === curr) ? curr : planClips[0].key));
  }, [planClips, seed]);

  const activePlanClip = useMemo(
    () => planClips.find((c) => c.key === selectedClipKey) ?? planClips[0] ?? null,
    [planClips, selectedClipKey],
  );

  const activeMediaId = activePlanClip?.mediaId ?? "";
  const effectiveStart = comp ? comp.clip_start_sec : (activePlanClip?.startSec ?? 0);
  const effectiveDuration = comp ? comp.duration_sec : (activePlanClip?.durationSec ?? 30);

  compRef.current = comp;
  framingRef.current = savedFraming;
  clipBoundsRef.current = { start: effectiveStart, duration: effectiveDuration };

  // Reset proxy fallback & load saved framing when source video changes
  useEffect(() => {
    if (!activeMediaId) return;
    setUseProxyVideo(false);
    setVideoStatus("loading");
    setPlaying(false);

    void runAction<FramingGetOutput>("clips.framing_get", { media_id: activeMediaId })
      .then((res) => {
        const p = res.data.active_profile;
        setSavedFraming({
          webcam: p.webcam_rect ?? DEFAULT_WEBCAM_RECT,
          content: p.content_rect ?? null,
        });
      })
      .catch(() => {});
  }, [activeMediaId]);

  // Load composition matching THIS EXACT clip from the Plano de Cortes (never hijack with another clip!)
  useEffect(() => {
    if (!activePlanClip) {
      setComp(null);
      setLiveHtmlDoc("");
      return;
    }
    relTimeRef.current = 0;
    lastUiSyncRef.current = 0;
    setRelTime(0);
    if (videoRef.current && videoRef.current.readyState >= 1) {
      try {
        videoRef.current.pause();
        videoRef.current.currentTime = activePlanClip.startSec;
      } catch {
        // ignore
      }
    }

    let alive = true;
    void runAction<{ compositions: AicutComposition[] }>("motion.list", { media_id: activePlanClip.mediaId })
      .then((res) => {
        if (!alive) return;
        const list = res.data.compositions ?? [];
        const matched = list.find(
          (c) =>
            Math.abs(c.clip_start_sec - activePlanClip.startSec) < 1.5 &&
            Math.abs(c.clip_end_sec - activePlanClip.endSec) < 1.5,
        );
        setComp(matched ?? null);
        if (!matched) setLiveHtmlDoc("");
      })
      .catch(() => {
        if (alive) {
          setComp(null);
          setLiveHtmlDoc("");
        }
      });
    return () => {
      alive = false;
    };
  }, [activePlanClip?.key]);

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

  const postSeekToHyperframes = useCallback((t: number, force = false) => {
    if (!force && Math.abs(t - lastIframeSeekRef.current) < 0.012) return;
    lastIframeSeekRef.current = t;
    iframeRef.current?.contentWindow?.postMessage({ type: "aicut_seek", time: t }, "*");
  }, []);

  // Seek video cleanly whenever active clip start changes
  useEffect(() => {
    const v = videoRef.current;
    if (!v || v.readyState < 1) return;
    if (v.paused) {
      try {
        v.currentTime = effectiveStart + relTimeRef.current;
      } catch {
        // ignore
      }
    }
  }, [effectiveStart]);

  // Single stable 60fps Live Video Choreography + Synchronized HyperFrames Overlay loop
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const curComp = compRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          const { start: startSec, duration: durSec } = clipBoundsRef.current;
          const endSec = startSec + durSec;

          let curRel = relTimeRef.current;
          if (video && video.readyState >= 1 && !video.paused) {
            if (video.currentTime < startSec || video.currentTime >= endSec) {
              video.currentTime = startSec;
            }
            curRel = Math.max(0, Math.min(durSec, video.currentTime - startSec));
            relTimeRef.current = curRel;
            if (Math.abs(curRel - lastUiSyncRef.current) >= 0.06) {
              lastUiSyncRef.current = curRel;
              setRelTime(curRel);
            }
          }

          if (curComp) {
            postSeekToHyperframes(curRel, false);
          }

          const W = canvas.width;
          const H = canvas.height;
          ctx.save();
          ctx.fillStyle = "#06070a";
          ctx.fillRect(0, 0, W, H);

          const hasVideoFrame = Boolean(video && video.readyState >= 2 && video.videoWidth > 0);
          if (hasVideoFrame && video) {
            const sx = W / 1080;
            const sy = H / 1920;

            if (curComp && curComp.scenes.length > 0) {
              const sc =
                curComp.scenes.find((s) => curRel >= s.start_sec && curRel < s.end_sec) ??
                curComp.scenes[curComp.scenes.length - 1];
              const scDur = Math.max(0.2, sc.end_sec - sc.start_sec);
              const scProg = Math.max(0, Math.min(1, (curRel - sc.start_sec) / scDur));
              const fx = sc.effects;
              const dynZoom = fx && fx.camera_zoom_end > 1.0 ? 1 + (fx.camera_zoom_end - 1) * scProg : 1.0;

              if (fx && fx.impact_shake > 0.05 && curRel - sc.start_sec < 0.35) {
                const decay = 1 - (curRel - sc.start_sec) / 0.35;
                const amp = fx.impact_shake * 6 * decay;
                ctx.translate(Math.sin(curRel * 70) * amp, Math.cos(curRel * 85) * amp);
              }

              if (fx?.color_mood === "vibrant") ctx.filter = "contrast(1.06) saturate(1.22)";
              else if (fx?.color_mood === "dramatic") ctx.filter = "contrast(1.12) brightness(0.95) saturate(0.88)";
              else if (fx?.color_mood === "cyber") ctx.filter = "contrast(1.08) saturate(1.25)";

              const camR = curComp.webcam_rect;
              const cntR = computeZoomedContentRect(curComp.content_rect, sc);

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

              if (fx?.vignette) {
                ctx.filter = "none";
                const vig = ctx.createRadialGradient(W / 2, H / 2, W * 0.3, W / 2, H / 2, H * 0.65);
                vig.addColorStop(0, "rgba(0,0,0,0)");
                vig.addColorStop(1, "rgba(0,0,0,0.52)");
                ctx.fillStyle = vig;
                ctx.fillRect(0, 0, W, H);
              }
            } else {
              // Instant live preview of the selected Plano de Cortes cut before AI edits it
              const bg = ctx.createLinearGradient(0, 0, 0, H);
              bg.addColorStop(0, "#1d0f2e");
              bg.addColorStop(1, "#181022");
              ctx.fillStyle = bg;
              ctx.fillRect(0, 0, W, H);
              ctx.fillStyle = "#e8bd52";
              ctx.font = "700 11px monospace";
              ctx.textAlign = "center";
              ctx.fillText("CORTE DO PLANO", W / 2, 24);

              const tx = 30 * sx, ty = 150 * sy, tw = 1020 * sx, th = 582 * sy;
              const bx = 30 * sx, by = 765 * sy, bw = 1020 * sx, bh = 889 * sy;
              const camR = framingRef.current.webcam;
              const cntR = framingRef.current.content ?? { x: 0, y: 0, w: 1, h: 1 };

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
              drawVideoCrop(ctx, video, cntR, bx, by, bw, bh, false, 1.0);
              ctx.restore();
              drawRoundedRect(ctx, bx, by, bw, bh, 8);
              ctx.strokeStyle = "#eab308";
              ctx.lineWidth = 2;
              ctx.stroke();
            }
          }
          ctx.restore();
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [postSeekToHyperframes]);

  const seekTo = (nextRel: number) => {
    const clamped = Math.max(0, Math.min(effectiveDuration, nextRel));
    relTimeRef.current = clamped;
    lastUiSyncRef.current = clamped;
    setRelTime(clamped);
    if (videoRef.current && videoRef.current.readyState >= 1) {
      videoRef.current.currentTime = effectiveStart + clamped;
    }
    if (comp) {
      postSeekToHyperframes(clamped, true);
    }
  };

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      const endSec = effectiveStart + effectiveDuration;
      if (v.currentTime < effectiveStart || v.currentTime >= endSec - 0.05) {
        v.currentTime = effectiveStart;
        relTimeRef.current = 0;
        setRelTime(0);
      }
      void v
        .play()
        .then(() => setPlaying(true))
        .catch(() => {
          v.muted = true;
          void v.play().then(() => setPlaying(true)).catch(() => {});
        });
    } else {
      v.pause();
      setPlaying(false);
    }
  };

  const handleVideoError = () => {
    if (!useProxyVideo && activeMediaId) {
      setUseProxyVideo(true);
      setVideoStatus("proxy");
    } else {
      setVideoStatus("error");
    }
  };

  const handleAutoEditWithAi = async (customGoal?: string) => {
    if (!activePlanClip) return;
    setPlanning(true);
    try {
      const res = await runAction<{ composition: AicutComposition; assistant_reply: string }>("motion.plan", {
        media_id: activePlanClip.mediaId,
        clip_start_sec: activePlanClip.startSec,
        clip_end_sec: activePlanClip.endSec,
        title: activePlanClip.clip.title,
        goal: buildPlanClipGoal(activePlanClip, customGoal),
      });
      setComp(res.data.composition);
      relTimeRef.current = 0;
      setRelTime(0);
      if (videoRef.current && videoRef.current.readyState >= 1) {
        videoRef.current.currentTime = res.data.composition.clip_start_sec;
      }
    } catch (e) {
      const msg = e instanceof AicutActionError ? e.message : (e as Error).message;
      onNotify("error", "Erro ao analisar corte", msg);
    } finally {
      setPlanning(false);
    }
  };

  const handleSendChat = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const msg = chatInput.trim();
    if (!msg) return;
    setChatInput("");

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

  const activeSceneIndex = useMemo(() => {
    if (!comp || comp.scenes.length === 0) return -1;
    const idx = comp.scenes.findIndex((s) => relTime >= s.start_sec && relTime < s.end_sec);
    return idx >= 0 ? idx : comp.scenes.length - 1;
  }, [comp, relTime]);

  const activeScene = activeSceneIndex >= 0 && comp ? comp.scenes[activeSceneIndex] : null;

  if (!loadingRuns && planClips.length === 0) {
    return (
      <EmptyState
        icon={Scissors}
        title="nenhum corte no Plano de Cortes"
        description="O Motion Studio edita a partir de um corte gerado no Plano de Cortes. Gere um plano em Cortes Shorts e selecione o corte que deseja editar."
        actions={
          <button type="button" className="btn btn-primary" onClick={() => onNavigate("smart-clips")}>
            Ir para Cortes Shorts
          </button>
        }
      />
    );
  }

  const videoSrc = activeMediaId
    ? useProxyVideo
      ? mediaPreviewUrl(activeMediaId)
      : mediaStreamUrl(activeMediaId)
    : "";

  return (
    <div className="ms-shell">
      {/* Ultra-minimalist top bar: select cut straight from Plano de Cortes */}
      <header className="ms-topbar">
        <div className="ms-topbar-left">
          <select
            className="select select-sm ms-media-select"
            aria-label="Corte do Plano de Cortes"
            value={activePlanClip?.key ?? ""}
            onChange={(e) => setSelectedClipKey(e.target.value)}
          >
            {runs.map((run) => {
              const validClips = run.clips
                .map((clip, idx) => ({ clip, idx }))
                .filter(({ clip }) => clip.segments && clip.segments.length > 0);
              if (validClips.length === 0) return null;
              return (
                <optgroup key={run.id} label={`Plano: ${run.media_title}`}>
                  {validClips.map(({ clip, idx }) => (
                    <option key={`${run.id}:${clip.id}`} value={`${run.id}:${clip.id}`}>
                      #{String(idx + 1).padStart(2, "0")} {clip.title} ({formatDuration(clip.total_duration_sec)})
                    </option>
                  ))}
                </optgroup>
              );
            })}
          </select>

          {activePlanClip && (
            <span className="ms-range-pill" title="Trecho exato do Plano de Cortes">
              {formatDuration(activePlanClip.startSec)} → {formatDuration(activePlanClip.endSec)} ·{" "}
              {formatDuration(activePlanClip.durationSec)}
            </span>
          )}
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
          Instale o FFmpeg em Configurações → Ferramentas para que a IA assista e renderize o corte.
        </Callout>
      )}

      <ActiveJobsPanel jobs={activeMotionJobs} title="Renderizando edição IA" tone="accent" onViewAll={() => onNavigate("jobs")} />

      {/* Minimalist 2-Zone Studio: Live 9:16 Preview + Conversational AI Editor */}
      <div className="ms-grid">
        {/* LEFT: Clean 9:16 Live Preview */}
        <section className="ms-stage-col" aria-label="Preview ao vivo">
          {videoSrc && (
            <video
              key={`${activeMediaId}-${useProxyVideo ? "proxy" : "direct"}`}
              ref={videoRef}
              src={videoSrc}
              playsInline
              preload="auto"
              className="ms-hidden-video"
              onLoadedMetadata={() => {
                if (videoRef.current) {
                  try {
                    videoRef.current.currentTime = effectiveStart + relTimeRef.current;
                  } catch {
                    // ignore
                  }
                }
              }}
              onLoadedData={() => setVideoStatus("ready")}
              onCanPlay={() => setVideoStatus("ready")}
              onError={handleVideoError}
              onPause={() => setPlaying(false)}
              onPlay={() => setPlaying(true)}
            />
          )}

          <div
            className="ms-phone-frame"
            onClick={togglePlay}
            role="button"
            tabIndex={0}
            aria-label={playing ? "Pausar preview" : "Reproduzir preview"}
          >
            <canvas ref={canvasRef} width={288} height={512} className="ms-video-canvas" />
            {comp && liveHtmlDoc && (
              <iframe
                ref={iframeRef}
                title="HyperFrames Motion Layer"
                srcDoc={liveHtmlDoc}
                sandbox="allow-scripts"
                className="ms-hyperframes-iframe"
                onLoad={() => postSeekToHyperframes(relTimeRef.current, true)}
              />
            )}
            {(videoStatus === "loading" || videoStatus === "proxy") && (
              <div className="ms-stage-status">
                <Loader2 size={22} className="spin text-accent" aria-hidden />
                <span>
                  {videoStatus === "proxy"
                    ? "Convertendo preview compatível (MKV/HEVC)…"
                    : "Carregando corte…"}
                </span>
              </div>
            )}
            {videoStatus === "error" && (
              <div className="ms-stage-status">
                <Film size={22} className="text-warning" aria-hidden />
                <span>Não foi possível decodificar este vídeo no preview.</span>
              </div>
            )}
          </div>

          <div className="ms-player-controls">
            <div className="ms-scrubber-row">
              <button
                type="button"
                className="btn btn-sm btn-primary"
                onClick={togglePlay}
                disabled={videoStatus !== "ready"}
                aria-label={playing ? "Pausar" : "Reproduzir"}
              >
                {playing ? <Pause size={13} aria-hidden /> : <Play size={13} aria-hidden />}
              </button>
              <input
                type="range"
                min={0}
                max={effectiveDuration}
                step={0.05}
                value={Math.min(effectiveDuration, relTime)}
                onChange={(e) => seekTo(Number(e.target.value))}
                className="grow"
              />
              <span className="ms-time">
                {formatDuration(relTime)} / {formatDuration(effectiveDuration)}
              </span>
            </div>

            {comp && comp.scenes.length > 0 && (
              <>
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
                {activeScene && (
                  <div className="ms-active-scene-caption">
                    <span>
                      Cena {activeSceneIndex + 1}/{comp.scenes.length} ·{" "}
                      {LAYOUT_SHORT_LABELS[activeScene.layout] ?? activeScene.layout}
                      {(activeScene.effects?.camera_zoom_end ?? 1) > 1.02
                        ? ` · Zoom ${(activeScene.effects?.camera_zoom_end ?? 1).toFixed(2)}x`
                        : ""}
                    </span>
                    {comp.motion_html && <span className="ms-hf-dot">HyperFrames</span>}
                  </div>
                )}
              </>
            )}
          </div>
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
              {comp?.creative_direction ? (
                <span className="ms-chat-direction" title={comp.creative_direction}>
                  {comp.creative_direction}
                </span>
              ) : (
                activePlanClip && (
                  <span className="ms-chat-direction" title={activePlanClip.clip.hook_first_5s || activePlanClip.clip.title}>
                    {activePlanClip.clip.title}
                  </span>
                )
              )}
            </div>
            <button
              type="button"
              className="btn btn-sm"
              disabled={planning || !activePlanClip}
              onClick={() => void handleAutoEditWithAi()}
            >
              <BusyIcon busy={planning} icon={Sparkles} size={13} />
              {comp ? "Reanalisar corte" : "Editar com IA"}
            </button>
          </div>

          <div className="ms-chat-messages">
            {!comp || comp.chat_history.length === 0 ? (
              <div className="ms-chat-welcome">
                <div className="ms-chat-welcome-icon">
                  <Sparkles size={24} aria-hidden />
                </div>
                <strong className="text-sm">{activePlanClip?.clip.title ?? "Editora IA em código motion"}</strong>
                <p className="secondary text-xs">
                  {activePlanClip?.clip.hook_first_5s
                    ? `Gancho do plano: "${activePlanClip.clip.hook_first_5s}" — Eu assisto aos frames deste corte e crio técnicas de edição e animações HyperFrames sob medida, ou podemos planejar juntos por aqui.`
                    : "Eu assisto aos frames e leio o que você fala neste corte do Plano de Cortes para aplicar técnicas de edição e animações HyperFrames sob medida."}
                </p>
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={planning || !activePlanClip}
                  onClick={() => void handleAutoEditWithAi()}
                >
                  <BusyIcon busy={planning} icon={Wand2} size={14} />
                  {planning ? "Assistindo ao corte e criando edição…" : "Assistir corte e criar edição automática"}
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
                    <span className="muted">Assistindo ao corte e preparando a edição…</span>
                  </div>
                )}
                <div ref={chatEndRef} />
              </>
            )}
          </div>

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
                placeholder="Converse com a IA sobre este corte: planeje ideias ou peça animações específicas…"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                disabled={chatSending || planning || !activePlanClip}
              />
              <button
                type="submit"
                className="btn btn-primary"
                disabled={chatSending || planning || !chatInput.trim() || !activePlanClip}
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
