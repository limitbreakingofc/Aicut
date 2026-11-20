use aicut_core::{configure_tokio_command, AicutError, AicutKernel, AicutResult, TranscriptRecord};
use aicut_module_smart_clips::engine::{
    resolve_gemini_api_key, SETTING_DEFAULT_PROVIDER, SETTING_GEMINI_DIRECTOR_MODEL,
    SETTING_TEMPLATE_PATH,
};
use aicut_module_smart_clips::framing::{resolve_effective_framing, NormalizedRect};
use aicut_module_smart_clips::frames::probe_duration_sec;
use aicut_module_smart_clips::gemini::{
    generate_json_text, generate_with_inline_image, list_ranked_gemini_models,
};
use aicut_module_smart_clips::local_vlm::generate_with_local_vlm_once;
use aicut_module_smart_clips::prompts::extract_json_object;
use base64::Engine as _;
use chrono::Utc;
use serde_json::Value;
use std::path::Path;
use tokio::process::Command;
use uuid::Uuid;

use crate::schema::{
    save_composition, AicutComposition, ContentFocusConfig, MotionChatMessage, MotionScene,
    PipConfig, PipCorner, SceneEffects, SceneLayoutPreset, SceneOverlay, SceneTransition,
};

#[derive(Debug)]
struct AiDirectorResponse {
    /// "apply" (update scenes & motion_html) | "discuss" (brainstorm / plan with user without overwriting edit)
    mode: String,
    title: Option<String>,
    creative_direction: Option<String>,
    assistant_reply: Option<String>,
    scenes: Vec<MotionScene>,
    /// Bespoke HyperFrames HTML/CSS/SVG/JS motion graphics written specifically for THIS video!
    motion_html: Option<String>,
    burn_subtitles: Option<bool>,
}

/// Extracts a crisp 4x2 (8-frame) visual storyboard JPEG of the exact clip so the AI
/// literally WATCHES the video frames and tailors editing techniques & HyperFrames motion
/// graphics to what is actually happening on screen.
async fn extract_clip_storyboard_b64(
    ffmpeg_bin: &Path,
    video_path: &Path,
    clip_start: f64,
    duration_sec: f64,
) -> Option<String> {
    let tmp_path = std::env::temp_dir().join(format!("aicut-motion-sb-{}.jpg", Uuid::new_v4()));
    let dur = duration_sec.max(1.0);
    let sample_fps = (8.0 / dur).clamp(0.05, 4.0);
    let vf = format!(
        "fps={sample_fps:.4},scale=320:180:force_original_aspect_ratio=decrease,pad=320:180:(ow-iw)/2:(oh-ih)/2:color=black,tile=4x2:margin=4:padding=4:color=0x111111"
    );

    let mut cmd = Command::new(ffmpeg_bin);
    configure_tokio_command(&mut cmd);
    let ok = cmd
        .args([
            "-y",
            "-ss",
            &format!("{clip_start:.3}"),
            "-t",
            &format!("{dur:.3}"),
            "-i",
        ])
        .arg(video_path)
        .args(["-vf", &vf, "-frames:v", "1", "-q:v", "5"])
        .arg(&tmp_path)
        .output()
        .await
        .map(|o| o.status.success() && tmp_path.is_file())
        .unwrap_or(false);

    if !ok {
        let _ = std::fs::remove_file(&tmp_path);
        return None;
    }
    let bytes = std::fs::read(&tmp_path).ok();
    let _ = std::fs::remove_file(&tmp_path);
    bytes.map(|b| base64::engine::general_purpose::STANDARD.encode(b))
}

/// Watches the video frames + reads the transcript and automatically crafts a bespoke,
/// video-specific editing plan + dynamic camera choreography + custom HyperFrames motion code!
pub async fn plan_composition_with_ai(
    kernel: &AicutKernel,
    media_id: &str,
    clip_start_sec: Option<f64>,
    clip_end_sec: Option<f64>,
    title_override: Option<&str>,
    user_goal: Option<&str>,
) -> AicutResult<(AicutComposition, String)> {
    let source = kernel
        .repo
        .get_media(media_id)?
        .ok_or_else(|| AicutError::not_found(format!("Media '{media_id}' not found in Library")))?;

    let ffprobe_bin = kernel.tools.resolve_tool("ffprobe").ok();
    let ffmpeg_bin = kernel.tools.resolve_tool("ffmpeg").ok();
    let total_dur = match source.metadata.duration_sec {
        Some(d) if d > 0.5 => d,
        _ => match ffprobe_bin {
            Some(ref p) => probe_duration_sec(p, Path::new(&source.file_path))
                .await
                .unwrap_or(60.0),
            None => 60.0,
        },
    };

    let start = clip_start_sec
        .unwrap_or(0.0)
        .clamp(0.0, (total_dur - 1.0).max(0.0));
    let default_end = (start + 60.0).min(total_dur);
    let end = clip_end_sec
        .unwrap_or(default_end)
        .clamp((start + 1.0).min(total_dur), total_dur);
    let duration_sec = (end - start).max(1.0);

    let (framing, _) = resolve_effective_framing(kernel, Some(&source.id));
    let webcam_rect = framing.webcam_rect.unwrap_or(NormalizedRect {
        x: 0.70,
        y: 0.64,
        w: 0.28,
        h: 0.33,
    });
    let template_path = kernel
        .repo
        .get_setting(SETTING_TEMPLATE_PATH)
        .ok()
        .flatten()
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .filter(|s| !s.trim().is_empty());

    let transcript = kernel
        .repo
        .find_best_transcript_for_media(&source.id)
        .ok()
        .flatten()
        .map(|rt| rt.transcript);
    let transcript_excerpt = format_relative_transcript(transcript.as_ref(), start, end);

    let storyboard_b64 = match ffmpeg_bin {
        Some(ref bin) => {
            extract_clip_storyboard_b64(bin, Path::new(&source.file_path), start, duration_sec)
                .await
        }
        None => None,
    };

    let mut comp = AicutComposition {
        id: Uuid::new_v4().to_string(),
        title: title_override
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{} · Edição IA", source.title)),
        media_id: source.id.clone(),
        clip_start_sec: start,
        clip_end_sec: end,
        duration_sec,
        width: 1080,
        height: 1920,
        fps: 30,
        webcam_rect,
        content_rect: framing.content_rect,
        burn_subtitles: true,
        template_path,
        engine_preference: "hybrid_nvenc".to_string(),
        motion_html: None,
        creative_direction: String::new(),
        scenes: Vec::new(),
        chat_history: Vec::new(),
        rendered_media_id: None,
        updated_at: Utc::now().to_rfc3339(),
    };

    let goal_str = user_goal
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("Assista ao corte sem padronização fixa: crie uma edição única para este vídeo, com ilustrações e animações visuais sob medida sempre que nada relevante estiver acontecendo na área de conteúdo.");

    let prompt = build_context_aware_plan_prompt(
        &comp.title,
        duration_sec,
        goal_str,
        &transcript_excerpt,
        storyboard_b64.is_some(),
    );

    let mut reply = String::new();
    let mut used_ai = false;
    let mut ai_error_note: Option<String> = None;

    match call_director_ai(kernel, storyboard_b64.as_deref(), &prompt).await {
        Ok((parsed, engine_used)) if !parsed.scenes.is_empty() => {
            if let Some(t) = parsed.title.filter(|s| !s.trim().is_empty()) {
                if title_override.is_none() {
                    comp.title = t;
                }
            }
            comp.scenes = parsed.scenes;
            if let Some(html) = parsed.motion_html.filter(|s| !s.trim().is_empty()) {
                comp.motion_html = Some(html);
            }
            if let Some(cd) = parsed.creative_direction.filter(|s| !s.trim().is_empty()) {
                comp.creative_direction = format!("{cd} · {engine_used}");
            } else {
                comp.creative_direction = format!("Edição IA ({engine_used}) · Motion sob medida");
            }
            reply = parsed.assistant_reply.unwrap_or_else(|| {
                format!(
                    "Assisti ao corte ({engine_used}) e criei uma edição sob medida com {} momentos + ilustrações e animações próprias para este vídeo! Dá play ao lado e me diz no chat o que você quer evoluir.",
                    comp.scenes.len()
                )
            });
            used_ai = true;
        }
        Ok(_) => {
            ai_error_note = Some("A IA respondeu sem cenas válidas para o corte.".to_string());
        }
        Err(err) => {
            ai_error_note = Some(err);
        }
    }

    if !used_ai {
        comp.scenes = build_smart_default_scenes(duration_sec);
        comp.motion_html = Some(build_default_hyperframes_motion(duration_sec, &comp.title));
        comp.creative_direction = "Edição base local (fallback)".to_string();
        reply = if let Some(err) = ai_error_note {
            format!(
                "Não consegui usar a IA agora ({err}). Apliquei uma base inicial para o preview — configure o Gemini ou baixe um modelo local (Qwen2.5-VL + llama-server) em Configurações → IA & Chaves e tente novamente!"
            )
        } else {
            "Configure sua Gemini API Key ou ative a IA Local Grátis (Qwen2.5-VL) em Configurações → IA & Chaves para eu assistir aos frames e criar edições e ilustrações animadas 100% sob medida para este corte.".to_string()
        };
    }

    comp.normalize();
    comp.chat_history.push(MotionChatMessage {
        role: "assistant".to_string(),
        text: reply.clone(),
        created_at: Utc::now().to_rfc3339(),
    });
    save_composition(kernel, &comp)?;
    Ok((comp, reply))
}

/// Conversational AI Video Editor:
/// - Can brainstorm & PLAN ideas with the user (`mode: "discuss"`) based on the real video frames + transcript
/// - Can APPLY / IMPROVE bespoke camera choreography & HyperFrames HTML/CSS/SVG motion graphics (`mode: "apply"`)
pub async fn edit_composition_via_chat(
    kernel: &AicutKernel,
    mut comp: AicutComposition,
    user_message: &str,
) -> AicutResult<(AicutComposition, String)> {
    let trimmed = user_message.trim();
    if trimmed.is_empty() {
        return Err(AicutError::invalid_input(
            "Digite o que você quer planejar ou mudar na edição.",
        ));
    }

    comp.chat_history.push(MotionChatMessage {
        role: "user".to_string(),
        text: trimmed.to_string(),
        created_at: Utc::now().to_rfc3339(),
    });

    let source = kernel.repo.get_media(&comp.media_id).ok().flatten();
    let ffmpeg_bin = kernel.tools.resolve_tool("ffmpeg").ok();
    let storyboard_b64 = match (source.as_ref(), ffmpeg_bin.as_ref()) {
        (Some(src), Some(bin)) => {
            extract_clip_storyboard_b64(
                bin,
                Path::new(&src.file_path),
                comp.clip_start_sec,
                comp.duration_sec,
            )
            .await
        }
        _ => None,
    };

    let transcript = kernel
        .repo
        .find_best_transcript_for_media(&comp.media_id)
        .ok()
        .flatten()
        .map(|rt| rt.transcript);
    let transcript_excerpt =
        format_relative_transcript(transcript.as_ref(), comp.clip_start_sec, comp.clip_end_sec);

    let mut assistant_reply: Option<String> = None;
    let mut ai_err_note: Option<String> = None;

    let prompt = build_conversational_editor_prompt(
        &comp,
        &transcript_excerpt,
        trimmed,
        storyboard_b64.is_some(),
    );

    match call_director_ai(kernel, storyboard_b64.as_deref(), &prompt).await {
        Ok((parsed, engine_used)) => {
            let is_discuss = parsed.mode.eq_ignore_ascii_case("discuss");
            if !is_discuss {
                if !parsed.scenes.is_empty() {
                    comp.scenes = parsed.scenes;
                }
                if let Some(html) = parsed.motion_html {
                    let cleaned = html.trim().to_string();
                    comp.motion_html = if cleaned.is_empty() {
                        None
                    } else {
                        Some(cleaned)
                    };
                }
                if let Some(cd) = parsed.creative_direction.filter(|s| !s.trim().is_empty()) {
                    comp.creative_direction = format!("{cd} · {engine_used}");
                }
                if let Some(b) = parsed.burn_subtitles {
                    comp.burn_subtitles = b;
                }
            }
            assistant_reply = Some(parsed.assistant_reply.unwrap_or_else(|| {
                if is_discuss {
                    format!("({engine_used}) Essas são minhas ideias para este corte — me diga qual você curtiu que eu já aplico!")
                } else {
                    format!("Edição atualizada via {engine_used}! Dá play no preview ao lado para ver as novas ilustrações, animações e cortes.")
                }
            }));
        }
        Err(err) => {
            ai_err_note = Some(err);
        }
    }

    let final_reply = match assistant_reply {
        Some(r) => r,
        None => {
            let local_reply = apply_local_ptbr_chat_edit(&mut comp, trimmed);
            if let Some(err) = ai_err_note {
                format!("(Aviso: IA indisponível agora — {err})\n\n{local_reply}")
            } else {
                local_reply
            }
        }
    };

    comp.normalize();
    comp.chat_history.push(MotionChatMessage {
        role: "assistant".to_string(),
        text: final_reply.clone(),
        created_at: Utc::now().to_rfc3339(),
    });
    if comp.chat_history.len() > 40 {
        let drain_n = comp.chat_history.len() - 40;
        comp.chat_history.drain(0..drain_n);
    }
    save_composition(kernel, &comp)?;
    Ok((comp, final_reply))
}

/// Offline PT-BR fallback command handler when no Gemini API key is configured.
pub fn apply_local_ptbr_chat_edit(comp: &mut AicutComposition, msg: &str) -> String {
    let lower = msg.to_lowercase();
    let dur = comp.duration_sec;
    let extracted_quote = extract_quoted_text(msg);

    let requested_layout =
        if lower.contains("zoom") || lower.contains("detalhe") || lower.contains("seta") {
            Some(SceneLayoutPreset::CalloutZoom)
        } else if (lower.contains("tela cheia") || lower.contains("conteudo cheio"))
            && (lower.contains("sem webcam") || lower.contains("só conteúdo") || lower.contains("so conteudo"))
        {
            Some(SceneLayoutPreset::ContentFullscreen)
        } else if lower.contains("tela cheia") || lower.contains("pip") || lower.contains("canto") {
            Some(SceneLayoutPreset::ContentHeroPip)
        } else if lower.contains("camera grande") || lower.contains("câmera grande") {
            Some(SceneLayoutPreset::CameraHeroPip)
        } else if lower.contains("foco na camera") || lower.contains("foco na câmera") || lower.contains("gancho") || lower.contains("reação") || lower.contains("reacao") {
            Some(SceneLayoutPreset::CameraFocus)
        } else if lower.contains("smoking") || lower.contains("padrao") || lower.contains("padrão") {
            Some(SceneLayoutPreset::SplitBranded)
        } else {
            None
        };

    let requested_corner = if lower.contains("superior esquerdo") {
        Some(PipCorner::TopLeft)
    } else if lower.contains("superior direito") {
        Some(PipCorner::TopRight)
    } else if lower.contains("inferior esquerdo") {
        Some(PipCorner::BottomLeft)
    } else if lower.contains("inferior direito") {
        Some(PipCorner::BottomRight)
    } else {
        None
    };

    let time_range = parse_time_range_ptbr(&lower, dur);

    if let Some((t0, t1)) = time_range {
        let lay = requested_layout.unwrap_or(SceneLayoutPreset::ContentHeroPip);
        let new_scene = MotionScene {
            id: format!("sc-{}", &Uuid::new_v4().to_string()[..6]),
            title: extracted_quote
                .clone()
                .unwrap_or_else(|| format!("Destaque {:.0}s–{:.0}s", t0, t1)),
            start_sec: t0,
            end_sec: t1,
            layout: lay,
            transition_in: SceneTransition::Fade,
            pip: PipConfig {
                corner: requested_corner.unwrap_or_default(),
                ..Default::default()
            },
            content_focus: ContentFocusConfig {
                zoom: if lay == SceneLayoutPreset::CalloutZoom { 1.65 } else { 1.0 },
                ..Default::default()
            },
            overlay: SceneOverlay {
                headline: extracted_quote.clone(),
                ..Default::default()
            },
            effects: SceneEffects {
                camera_zoom_end: if lay == SceneLayoutPreset::CameraFocus { 1.10 } else { 1.04 },
                impact_shake: if lower.contains("trem") || lower.contains("impacto") { 0.5 } else { 0.0 },
                vignette: lower.contains("tensão") || lower.contains("tensao") || lower.contains("dramático"),
                color_mood: "natural".into(),
            },
            rationale: format!("Ajustado via chat: {msg}"),
            custom_code: None,
        };
        insert_scene_into_timeline(&mut comp.scenes, new_scene, dur);
        if let Some(q) = extracted_quote {
            let mut html = comp.motion_html.clone().unwrap_or_default();
            html.push_str(&format!(
                "\n<div class=\"hf-headline\" data-start=\"{t0:.2}\" data-end=\"{t1:.2}\" data-anim=\"pop\">{}</div>",
                q.replace('<', "&lt;").replace('>', "&gt;")
            ));
            comp.motion_html = Some(html);
        }
        return format!("Pronto! Apliquei esse ajuste entre {t0:.1}s e {t1:.1}s. Dá play ao lado para conferir!");
    }

    if lower.contains("sem legenda") {
        comp.burn_subtitles = false;
        return "Legendas desativadas.".to_string();
    }
    if lower.contains("com legenda") {
        comp.burn_subtitles = true;
        return "Legendas karaokê ativadas.".to_string();
    }

    "Para eu assistir visualmente ao corte e criar ilustrações e animações sob medida pelo chat, configure sua Gemini API Key ou baixe um modelo da IA Local Grátis (Qwen2.5-VL + llama-server) em Configurações → IA & Chaves. No modo offline você ainda pode pedir cortes por tempo (ex.: 'entre 0:08 e 0:16 tela cheia com título \"OLHA ISSO\"').".to_string()
}

fn insert_scene_into_timeline(scenes: &mut Vec<MotionScene>, inserted: MotionScene, total_dur: f64) {
    let mut next: Vec<MotionScene> = Vec::new();
    for s in scenes.drain(..) {
        if s.end_sec <= inserted.start_sec + 0.1 || s.start_sec >= inserted.end_sec - 0.1 {
            next.push(s);
        } else {
            if s.start_sec + 0.5 < inserted.start_sec {
                let mut left = s.clone();
                left.id = format!("sc-{}", &Uuid::new_v4().to_string()[..6]);
                left.end_sec = inserted.start_sec;
                next.push(left);
            }
            if s.end_sec > inserted.end_sec + 0.5 {
                let mut right = s.clone();
                right.id = format!("sc-{}", &Uuid::new_v4().to_string()[..6]);
                right.start_sec = inserted.end_sec;
                next.push(right);
            }
        }
    }
    next.push(inserted);
    next.sort_by(|a, b| a.start_sec.partial_cmp(&b.start_sec).unwrap_or(std::cmp::Ordering::Equal));
    if next.is_empty() {
        next.push(MotionScene {
            id: format!("sc-{}", &Uuid::new_v4().to_string()[..6]),
            title: "Principal".into(),
            start_sec: 0.0,
            end_sec: total_dur,
            layout: SceneLayoutPreset::SplitBranded,
            transition_in: SceneTransition::Cut,
            pip: PipConfig::default(),
            content_focus: ContentFocusConfig::default(),
            overlay: SceneOverlay::default(),
            effects: SceneEffects::default(),
            rationale: String::new(),
            custom_code: None,
        });
    }
    *scenes = next;
}

fn build_default_hyperframes_motion(dur: f64, title: &str) -> String {
    let hook_end = (dur * 0.14).clamp(2.8, 4.2);
    let safe_title: String = title
        .chars()
        .take(28)
        .collect::<String>()
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    format!(
        r#"<div class="hf-headline" data-start="0.15" data-end="{hook_end:.2}" data-anim="pop">{safe_title}</div>
<div style="position:absolute;bottom:24px;left:48px;right:48px;height:6px;border-radius:99px;background:rgba(255,255,255,0.16);overflow:hidden;" data-start="0" data-end="{dur:.2}" data-anim="fade">
  <div id="aicut-prog" style="height:100%;width:0%;background:linear-gradient(90deg,#eab308,#f97316);border-radius:99px;"></div>
</div>
<script>
window.__aicutCustomSeek = function(t) {{
  var p = document.getElementById('aicut-prog');
  if (p) p.style.width = Math.min(100, Math.max(0, (t / {dur:.2}) * 100)).toFixed(2) + '%';
}};
</script>"#
    )
}

fn build_smart_default_scenes(dur: f64) -> Vec<MotionScene> {
    if dur <= 10.0 {
        return vec![MotionScene {
            id: "sc-main".into(),
            title: "SMOKING33".into(),
            start_sec: 0.0,
            end_sec: dur,
            layout: SceneLayoutPreset::SplitBranded,
            transition_in: SceneTransition::Cut,
            pip: PipConfig::default(),
            content_focus: ContentFocusConfig::default(),
            overlay: SceneOverlay::default(),
            effects: SceneEffects { camera_zoom_end: 1.04, ..Default::default() },
            rationale: "Clipe curto em SMOKING33.".into(),
            custom_code: None,
        }];
    }
    let hook_end = (dur * 0.14).clamp(3.0, 4.5);
    let mid_start = (dur * 0.42).max(hook_end + 3.0);
    let mid_end = (dur * 0.72).min(dur - 3.0).max(mid_start + 3.0);
    vec![
        MotionScene {
            id: "sc-hook".into(),
            title: "Gancho".into(),
            start_sec: 0.0,
            end_sec: hook_end,
            layout: SceneLayoutPreset::CameraFocus,
            transition_in: SceneTransition::Cut,
            pip: PipConfig::default(),
            content_focus: ContentFocusConfig::default(),
            overlay: SceneOverlay::default(),
            effects: SceneEffects { camera_zoom_end: 1.08, ..Default::default() },
            rationale: "Punch-in natural na webcam para prender atenção no gancho.".into(),
            custom_code: None,
        },
        MotionScene {
            id: "sc-split-1".into(),
            title: "Desenvolvimento".into(),
            start_sec: hook_end,
            end_sec: mid_start,
            layout: SceneLayoutPreset::SplitBranded,
            transition_in: SceneTransition::Fade,
            pip: PipConfig::default(),
            content_focus: ContentFocusConfig::default(),
            overlay: SceneOverlay::default(),
            effects: SceneEffects::default(),
            rationale: "SMOKING33 natural equilibrando webcam e conteúdo.".into(),
            custom_code: None,
        },
        MotionScene {
            id: "sc-hero".into(),
            title: "Destaque na tela".into(),
            start_sec: mid_start,
            end_sec: mid_end,
            layout: SceneLayoutPreset::ContentHeroPip,
            transition_in: SceneTransition::Fade,
            pip: PipConfig { corner: PipCorner::TopRight, width_pct: 0.34, ..Default::default() },
            content_focus: ContentFocusConfig::default(),
            overlay: SceneOverlay::default(),
            effects: SceneEffects { camera_zoom_end: 1.06, ..Default::default() },
            rationale: "Conteúdo em tela cheia no ponto principal com reação em PiP.".into(),
            custom_code: None,
        },
        MotionScene {
            id: "sc-close".into(),
            title: "Fechamento".into(),
            start_sec: mid_end,
            end_sec: dur,
            layout: SceneLayoutPreset::SplitBranded,
            transition_in: SceneTransition::Fade,
            pip: PipConfig::default(),
            content_focus: ContentFocusConfig::default(),
            overlay: SceneOverlay::default(),
            effects: SceneEffects::default(),
            rationale: "Retorno limpo ao SMOKING33 para encerramento.".into(),
            custom_code: None,
        },
    ]
}

/// Unified dual-engine AI dispatcher for Motion Studio:
/// - Respects the user's chosen engine in `Configurações → IA & Chaves` (`gemini` or `local_free`).
/// - Gemini path: automatic 503/429 retry + full ranked multi-model failover across active Gemini models.
/// - Local path: `llama-server` + installed `Qwen2.5-VL` on GPU (with automatic VRAM offload fallback).
/// - Cross-provider failover: if the primary engine is down/overloaded, automatically tries the other engine if configured!
async fn call_director_ai(
    kernel: &AicutKernel,
    storyboard_b64: Option<&str>,
    prompt: &str,
) -> Result<(AiDirectorResponse, String), String> {
    let default_provider = read_setting_str(kernel, SETTING_DEFAULT_PROVIDER, "gemini");
    let prefer_local = default_provider.eq_ignore_ascii_case("local_free");
    let gemini_key = resolve_gemini_api_key(kernel, None);
    let mut errors: Vec<String> = Vec::new();

    if prefer_local {
        match call_local_vlm_director(kernel, storyboard_b64, prompt).await {
            Ok(ok) => return Ok(ok),
            Err(e) => errors.push(format!("IA Local: {e}")),
        }
        if let Some(ref api_key) = gemini_key {
            match call_gemini_models_chain(kernel, api_key, storyboard_b64, prompt).await {
                Ok(ok) => return Ok(ok),
                Err(e) => errors.push(format!("Gemini (fallback): {e}")),
            }
        }
    } else {
        if let Some(ref api_key) = gemini_key {
            match call_gemini_models_chain(kernel, api_key, storyboard_b64, prompt).await {
                Ok(ok) => return Ok(ok),
                Err(e) => errors.push(format!("Gemini: {e}")),
            }
        }
        // Automatic fallback to Local GPU AI whenever Gemini is unavailable / 503 / no key
        if kernel.tools.resolve_tool("llama-server").is_ok() {
            match call_local_vlm_director(kernel, storyboard_b64, prompt).await {
                Ok(ok) => return Ok(ok),
                Err(e) => errors.push(format!("IA Local: {e}")),
            }
        } else if gemini_key.is_none() {
            return Err(
                "nenhuma IA configurada — salve sua Gemini API Key ou instale a IA Local (llama-server + Qwen2.5-VL) em Configurações → IA & Chaves".to_string(),
            );
        }
    }

    Err(errors.join(" | "))
}

async fn call_gemini_models_chain(
    kernel: &AicutKernel,
    api_key: &str,
    storyboard_b64: Option<&str>,
    prompt: &str,
) -> Result<(AiDirectorResponse, String), String> {
    let http = kernel.tools.http_client();
    let requested_dir = read_setting_str(kernel, SETTING_GEMINI_DIRECTOR_MODEL, "auto");
    let candidates = list_ranked_gemini_models(http, api_key, &requested_dir).await;

    let mut last_err = String::from("nenhum modelo Gemini disponível respondeu");
    for model in candidates {
        let raw_res = if let Some(b64) = storyboard_b64 {
            generate_with_inline_image(http, api_key, &model, b64, prompt).await
        } else {
            generate_json_text(http, api_key, &model, prompt).await
        };
        match raw_res {
            Ok(raw) => {
                if let Some(parsed) = parse_ai_director_response(&raw) {
                    return Ok((parsed, model));
                }
                last_err = format!("resposta JSON inválida do modelo {model}");
            }
            Err(e) => {
                let lower = e.message.to_lowercase();
                last_err = format!("{model}: {}", compact_api_err(&e.message));
                if lower.contains("api_key_invalid")
                    || lower.contains("http 401")
                    || lower.contains("http 403")
                {
                    break;
                }
            }
        }
    }
    Err(last_err)
}

async fn call_local_vlm_director(
    kernel: &AicutKernel,
    storyboard_b64: Option<&str>,
    prompt: &str,
) -> Result<(AiDirectorResponse, String), String> {
    let sys = "Você é um Editor de Vídeo e Diretor de Motion Graphics criativo (sem padronização fixa). Assista aos frames, crie ilustrações e animações SVG/HTML sob medida quando a área de conteúdo estiver parada e responda SEMPRE com um único objeto JSON válido.";
    let (raw, model_id) =
        generate_with_local_vlm_once(kernel, storyboard_b64, sys, prompt, 3600)
            .await
            .map_err(|e| e.message)?;
    parse_ai_director_response(&raw)
        .map(|parsed| (parsed, format!("Local · {model_id}")))
        .ok_or_else(|| format!("modelo local {model_id} retornou JSON incompleto ou inválido"))
}

fn compact_api_err(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.chars().count() <= 220 {
        return trimmed.to_string();
    }
    let head: String = trimmed.chars().take(210).collect();
    format!("{head}…")
}

/// Resilient parser for Gemini JSON outputs: tolerates explicit `null` nested objects,
/// numeric strings, and creative enum aliases so real AI responses NEVER silently get dropped.
fn parse_ai_director_response(raw: &str) -> Option<AiDirectorResponse> {
    let json_str = extract_json_object(raw).unwrap_or(raw);
    let val: Value = serde_json::from_str(json_str).ok()?;
    let obj = val.as_object()?;

    let mode = obj
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("apply")
        .to_string();
    let title = obj
        .get("title")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let creative_direction = obj
        .get("creative_direction")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let assistant_reply = obj
        .get("assistant_reply")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let motion_html = obj
        .get("motion_html")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let burn_subtitles = obj.get("burn_subtitles").and_then(|v| v.as_bool());

    let mut scenes = Vec::new();
    if let Some(arr) = obj.get("scenes").and_then(|v| v.as_array()) {
        for (idx, item) in arr.iter().enumerate() {
            if let Some(sc) = parse_lenient_scene(item, idx) {
                scenes.push(sc);
            }
        }
    }

    Some(AiDirectorResponse {
        mode,
        title,
        creative_direction,
        assistant_reply,
        scenes,
        motion_html,
        burn_subtitles,
    })
}

fn val_f64(v: Option<&Value>) -> Option<f64> {
    let v = v?;
    v.as_f64()
        .or_else(|| v.as_i64().map(|n| n as f64))
        .or_else(|| v.as_str().and_then(|s| s.trim().trim_end_matches('s').parse::<f64>().ok()))
}

fn parse_lenient_scene(v: &Value, idx: usize) -> Option<MotionScene> {
    let obj = v.as_object()?;
    let start_sec = val_f64(obj.get("start_sec")).unwrap_or(0.0).max(0.0);
    let end_sec = val_f64(obj.get("end_sec")).unwrap_or(start_sec + 4.0);
    if end_sec <= start_sec + 0.05 {
        return None;
    }

    let id = obj
        .get("id")
        .and_then(|x| x.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("sc-{}", idx + 1));
    let title = obj
        .get("title")
        .and_then(|x| x.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("Cena {}", idx + 1));

    let layout = match obj
        .get("layout")
        .and_then(|x| x.as_str())
        .unwrap_or("split_branded")
        .to_lowercase()
        .as_str()
    {
        "split_clean" => SceneLayoutPreset::SplitClean,
        "content_hero_pip" | "pip" | "content_pip" => SceneLayoutPreset::ContentHeroPip,
        "content_fullscreen" | "fullscreen" | "full_content" => SceneLayoutPreset::ContentFullscreen,
        "camera_focus" | "camera" | "webcam" | "webcam_focus" => SceneLayoutPreset::CameraFocus,
        "camera_hero_pip" | "camera_hero" => SceneLayoutPreset::CameraHeroPip,
        "callout_zoom" | "zoom" | "callout" => SceneLayoutPreset::CalloutZoom,
        _ => SceneLayoutPreset::SplitBranded,
    };

    let transition_in = match obj
        .get("transition_in")
        .and_then(|x| x.as_str())
        .unwrap_or("cut")
        .to_lowercase()
        .as_str()
    {
        "fade" | "dissolve" | "crossfade" => SceneTransition::Fade,
        "slide_up" | "slide" => SceneTransition::SlideUp,
        "wipe_left" | "wipe" => SceneTransition::WipeLeft,
        _ => SceneTransition::Cut,
    };

    let mut pip = PipConfig::default();
    if let Some(po) = obj.get("pip").and_then(|x| x.as_object()) {
        pip.corner = match po
            .get("corner")
            .and_then(|x| x.as_str())
            .unwrap_or("top_right")
            .to_lowercase()
            .as_str()
        {
            "top_left" => PipCorner::TopLeft,
            "bottom_right" => PipCorner::BottomRight,
            "bottom_left" => PipCorner::BottomLeft,
            _ => PipCorner::TopRight,
        };
        if let Some(w) = val_f64(po.get("width_pct")) {
            pip.width_pct = w.clamp(0.24, 0.48);
        }
        if let Some(hex) = po.get("border_hex").and_then(|x| x.as_str()) {
            if !hex.trim().is_empty() {
                pip.border_hex = hex.trim().to_string();
            }
        }
        if let Some(vis) = po.get("visible").and_then(|x| x.as_bool()) {
            pip.visible = vis;
        }
    }

    let mut content_focus = ContentFocusConfig::default();
    if let Some(cf) = obj.get("content_focus").and_then(|x| x.as_object()) {
        if let Some(fx) = val_f64(cf.get("focus_x")) {
            content_focus.focus_x = fx.clamp(0.05, 0.95);
        }
        if let Some(fy) = val_f64(cf.get("focus_y")) {
            content_focus.focus_y = fy.clamp(0.05, 0.95);
        }
        if let Some(z) = val_f64(cf.get("zoom")) {
            content_focus.zoom = z.clamp(1.0, 3.0);
        }
        if let Some(fb) = cf.get("fit_blur").and_then(|x| x.as_bool()) {
            content_focus.fit_blur = fb;
        }
    }

    let mut overlay = SceneOverlay::default();
    if let Some(ov) = obj.get("overlay").and_then(|x| x.as_object()) {
        overlay.headline = ov
            .get("headline")
            .and_then(|x| x.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        overlay.badge = ov
            .get("badge")
            .and_then(|x| x.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        overlay.callout_text = ov
            .get("callout_text")
            .and_then(|x| x.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        if let Some(hex) = ov.get("accent_hex").and_then(|x| x.as_str()) {
            if !hex.trim().is_empty() {
                overlay.accent_hex = hex.trim().to_string();
            }
        }
    }

    let mut effects = SceneEffects::default();
    if let Some(ef) = obj.get("effects").and_then(|x| x.as_object()) {
        if let Some(ze) = val_f64(ef.get("camera_zoom_end")) {
            effects.camera_zoom_end = ze.clamp(1.0, 1.3);
        }
        if let Some(sh) = val_f64(ef.get("impact_shake")) {
            effects.impact_shake = sh.clamp(0.0, 1.0);
        }
        if let Some(vg) = ef.get("vignette").and_then(|x| x.as_bool()) {
            effects.vignette = vg;
        }
        if let Some(m) = ef.get("color_mood").and_then(|x| x.as_str()) {
            if !m.trim().is_empty() {
                effects.color_mood = m.trim().to_lowercase();
            }
        }
    }

    let rationale = obj
        .get("rationale")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();

    Some(MotionScene {
        id,
        title,
        start_sec,
        end_sec,
        layout,
        transition_in,
        pip,
        content_focus,
        overlay,
        effects,
        rationale,
        custom_code: None,
    })
}

fn build_context_aware_plan_prompt(
    title: &str,
    dur: f64,
    goal: &str,
    transcript: &str,
    has_storyboard: bool,
) -> String {
    let vis_note = if has_storyboard {
        "VOCÊ ESTÁ VENDO O STORYBOARD VISUAL REAL (grade 4x2 de frames do corte anexada)! Analise frame a frame: em quais momentos há ação visual real na tela de conteúdo e em quais momentos a área de conteúdo está parada/vazia/monótona enquanto a pessoa fala."
    } else {
        "Baseie-se profundamente na transcrição e no assunto exato deste corte."
    };
    format!(
        r##"Você é um Editor de Vídeo & Diretor Criativo de Motion Graphics / Ilustrações Animadas (estilo Remotion + HyperFrames HTML/CSS/SVG/JS).
{vis_note}

PRINCÍPIO CENTRAL — ZERO PADRONIZAÇÃO:
- NUNCA use uma fórmula fixa, template repetitivo ou apenas uma barrinha genérica de título!
- Cada vídeo merece uma identidade visual, paleta de cores, tipografia, ritmo de cortes e animações **100% criados do zero para o assunto exato deste corte** de {dur:.1}s ("{title}").
- **REGRA DE OURO PARA A ÁREA DE CONTEÚDO PARADA:** Sempre que você perceber (pelos frames ou pela fala) que **não há nada visualmente relevante acontecendo na área de conteúdo embaixo** (ex.: tela parada, lobby, menu, chat estático, pessoa apenas contando história, explicando um conceito, citando valores ou comparando ideias), **CRIE ILUSTRAÇÕES E ANIMAÇÕES VETORIAIS COMPLETAS NO `motion_html`** cobrindo a área de conteúdo (`class="hf-content-stage"` — `left:30px; top:765px; width:1020px; height:889px;` ou estilo próprio) enquanto a webcam continua 100% limpa e visível em cima!
  * Exemplos de ilustrações/animações criativas na área de conteúdo quando estiver ociosa:
    - Falou de números/dinheiro/crescimento → desenhe um **gráfico SVG animado / barras crescendo (`data-bar-to="82"`) / contador (`data-count-from`/`data-count-to`)** com cores do tema!
    - Explicou uma história, processo ou ideia → desenhe uma **cena vetorial SVG ilustrativa, diagrama animado, linha do tempo com `data-anim="draw"`, ícones/personagens/objetos geométricos com movimento (`float`, `pop`, `slide-up`) e tipografia cinética** representando visualmente o que está sendo dito!
    - Comparou duas coisas (A vs B) → anime um **duelo visual / cards lado a lado com spring physics**!
  * E quando **HOUVER ação visual importante na tela** (jogada, vídeo reagido, detalhe visual), **NÃO tampe a ação**: destaque a tela real (`content_hero_pip`, `callout_zoom`, `split_clean` ou `split_branded`) usando apenas anotações leves (setas/círculos `data-anim="draw"` ou badges sutis)!

Contexto do corte & pedido do criador:
{goal}

Você controla DUAS camadas sincronizadas:
1. **Câmera & Ritmo (`scenes[]`, 0.0 até {dur:.1}s sem lacunas):**
   - `layout`: escolha livremente conforme cada segundo do vídeo: "split_branded", "split_clean", "content_hero_pip", "content_fullscreen", "camera_focus", "camera_hero_pip", "callout_zoom".
   - `effects`: `camera_zoom_end` (1.0..1.20), `impact_shake` (0.0..0.8), `vignette` (bool), `color_mood` ("natural"|"vibrant"|"dramatic"|"cyber").

2. **Ilustrações & Motion Graphics Customizados (`motion_html` — canvas transparente 1080x1920, 100% offline e determinístico):**
   - Você tem liberdade total para escrever `<style>`, HTML rico, `<svg viewBox="...">` com ilustrações detalhadas e `<script>window.__aicutCustomSeek = function(t) {{ ... }};</script>` usando os utilitários embutidos estilo Remotion: `window.__aicut.interpolate(t, inS, inE, outS, outE, 'out'|'back'|'inout')` e `window.__aicut.spring(t, start, stiffness, damping)`.
   - Todo elemento animado por atributo usa `data-start="SEG"`, `data-end="SEG"` e `data-anim="pop|slide-up|slide-down|slide-left|slide-right|fade|scale|pulse|float|spin|draw|flash"` (opcional `data-count-from`/`data-count-to`/`data-prefix`/`data-suffix` ou `data-bar-to="80"`).
   - NÃO use URLs externas nem `<img>` externos nem CSS `@keyframes` solto (tudo deve ser guiado pelo tempo `t`!).
   - Preserve o rosto na webcam (topo) limpo e deixe a faixa inferior das legendas (`y > 1660px`) legível.

Responda APENAS JSON válido:
{{
  "mode": "apply",
  "title": "Título curto do corte",
  "creative_direction": "Resumo em 1 frase da direção visual e ilustrações criadas para este vídeo.",
  "assistant_reply": "Explicação natural em PT-BR contando o que você viu no vídeo, onde criou ilustrações/animações na área de conteúdo e quais técnicas aplicou.",
  "burn_subtitles": true,
  "scenes": [ ... ],
  "motion_html": "<style>...</style><div ...>...</div>"
}}

Transcrição com tempos relativos (0.0s .. {dur:.1}s):
{transcript}"##
    )
}

fn build_conversational_editor_prompt(
    comp: &AicutComposition,
    transcript: &str,
    user_msg: &str,
    has_storyboard: bool,
) -> String {
    let scenes_json = serde_json::to_string_pretty(&comp.scenes).unwrap_or_else(|_| "[]".into());
    let cur_html = comp.motion_html.as_deref().unwrap_or("(nenhum motion customizado ainda)");
    let recent_chat: String = comp
        .chat_history
        .iter()
        .rev()
        .take(8)
        .rev()
        .map(|m| format!("{}: {}", m.role, m.text))
        .collect::<Vec<_>>()
        .join("\n");
    let vis_note = if has_storyboard {
        "O storyboard visual (4x2 frames reais do corte) está anexado — observe o que acontece na tela e onde a área de conteúdo está parada para criar ilustrações animadas."
    } else {
        ""
    };

    format!(
        r##"Você é o Editor de Vídeo & Diretor de Ilustrações/Motion Graphics pessoal do criador no Aicut (sem padronização fixa — cada vídeo tem sua própria linguagem visual, ilustrações SVG e animações sob medida).
{vis_note}

IMPORTANTE — Escolha `mode` conforme a intenção da mensagem do criador:
- Use `"mode": "discuss"` quando o criador estiver **conversando, perguntando sua opinião, pedindo ideias ou planejando** ANTES de aplicar. Responda de forma criativa e específica para este vídeo sem precisar reescrever `scenes`/`motion_html`.
- Use `"mode": "apply"` quando o criador pedir para **editar, criar ilustrações/animações, aplicar um plano, melhorar, trocar estilo ou remover algo**. Entregue `scenes` completos (0.0 .. {dur:.1}s sem lacunas) e o `motion_html` completo atualizado!

Regras criativas para `motion_html` (canvas 1080x1920 transparente, zero padronização):
- Crie ilustrações vetoriais SVG, diagramas, gráficos, cenários animados e tipografia cinética únicos para o tema do vídeo — especialmente preenchendo a área de conteúdo (`class="hf-content-stage"`, `left:30px; top:765px; width:1020px; height:889px`) quando nada relevante estiver ocorrendo no vídeo embaixo!
- Use `<style>`, HTML/SVG inline, atributos `data-start`, `data-end`, `data-anim="pop|slide-up|slide-down|slide-left|slide-right|fade|scale|pulse|float|spin|draw|flash"`, `data-count-from`/`data-count-to`, `data-bar-to`, e opcional `<script>window.__aicutCustomSeek = function(t){{...}};</script>` com `window.__aicut.interpolate(...)` e `window.__aicut.spring(...)`.
- Sem URLs externas; mantenha a webcam limpa e a faixa das legendas (`y > 1660px`) legível.

Layouts válidos em `scenes[].layout`: "split_branded", "split_clean", "content_hero_pip", "content_fullscreen", "camera_focus", "camera_hero_pip", "callout_zoom".
Effects válidos em `scenes[].effects`: `camera_zoom_end` (1.0..1.25), `impact_shake` (0.0..1.0), `vignette` (bool), `color_mood` ("natural"|"vibrant"|"dramatic"|"cyber").

Direção criativa atual: {cd}
Cenas atuais (0.0s .. {dur:.1}s):
{scenes_json}

Motion HTML atual:
{cur_html}

Transcrição do corte:
{transcript}

Conversa recente:
{recent_chat}

Nova mensagem do criador:
"{user_msg}"

Responda APENAS JSON válido:
{{
  "mode": "apply",
  "creative_direction": "Resumo atualizado da direção visual/ilustrações",
  "assistant_reply": "Resposta natural e colaborativa em PT-BR.",
  "burn_subtitles": {subs},
  "scenes": [ ... ],
  "motion_html": "..."
}}"##,
        dur = comp.duration_sec,
        cd = comp.creative_direction,
        subs = comp.burn_subtitles,
    )
}

fn format_relative_transcript(tr: Option<&TranscriptRecord>, clip_start: f64, clip_end: f64) -> String {
    let Some(t) = tr else {
        return "(sem transcrição disponível)".to_string();
    };
    let mut lines = Vec::new();
    for seg in &t.segments {
        let s = seg.start_ms as f64 / 1000.0;
        let e = seg.end_ms as f64 / 1000.0;
        if e <= clip_start || s >= clip_end {
            continue;
        }
        let rs = (s - clip_start).max(0.0);
        let re = (e - clip_start).min(clip_end - clip_start);
        lines.push(format!("[{rs:.1}s–{re:.1}s] {}", seg.text.trim()));
        if lines.len() >= 80 {
            break;
        }
    }
    if lines.is_empty() {
        "(sem falas neste intervalo)".to_string()
    } else {
        lines.join("\n")
    }
}

fn read_setting_str(kernel: &AicutKernel, key: &str, default: &str) -> String {
    kernel
        .repo
        .get_setting(key)
        .ok()
        .flatten()
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn extract_quoted_text(s: &str) -> Option<String> {
    for quote in ['"', '\''] {
        if let Some(i) = s.find(quote) {
            if let Some(j) = s[i + 1..].find(quote) {
                let inner = s[i + 1..i + 1 + j].trim();
                if !inner.is_empty() {
                    return Some(inner.to_string());
                }
            }
        }
    }
    None
}

fn parse_time_token(tok: &str) -> Option<f64> {
    let clean = tok.trim().trim_end_matches('s');
    if let Some((m, s)) = clean.split_once(':') {
        let mv = m.parse::<f64>().ok()?;
        let sv = s.parse::<f64>().ok()?;
        Some(mv * 60.0 + sv)
    } else {
        clean.parse::<f64>().ok()
    }
}

fn parse_time_range_ptbr(lower: &str, dur: f64) -> Option<(f64, f64)> {
    if lower.contains("primeiros ") || lower.contains("inicio") || lower.contains("início") || lower.contains("abertura") {
        let secs = lower
            .split_whitespace()
            .find_map(|w| {
                let c = w.trim_matches(|ch: char| !ch.is_ascii_digit() && ch != '.');
                c.parse::<f64>().ok()
            })
            .unwrap_or(4.0)
            .clamp(1.5, dur * 0.5);
        return Some((0.0, secs));
    }
    if lower.contains("final") || lower.contains("ultimos") || lower.contains("últimos") {
        let secs = lower
            .split_whitespace()
            .find_map(|w| {
                let c = w.trim_matches(|ch: char| !ch.is_ascii_digit() && ch != '.');
                c.parse::<f64>().ok()
            })
            .unwrap_or(5.0)
            .clamp(1.5, dur * 0.5);
        return Some(((dur - secs).max(0.0), dur));
    }

    let mut found = Vec::new();
    for raw in lower.split(|c: char| c.is_whitespace() || c == ',' || c == '-' || c == '–' || c == '→') {
        let w = raw.trim_matches(|c: char| !c.is_ascii_digit() && c != ':' && c != 's' && c != '.');
        if w.contains(':') || (w.ends_with('s') && w.len() >= 2 && w.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)) {
            if let Some(t) = parse_time_token(w) {
                found.push(t);
            }
        }
    }
    if found.len() >= 2 {
        let a = found[0].clamp(0.0, dur);
        let b = found[1].clamp(0.0, dur);
        if (b - a).abs() >= 0.8 {
            return Some((a.min(b), a.max(b)));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_ptbr_chat_inserts_scene_and_hyperframes_headline() {
        let mut comp = AicutComposition {
            id: "c-test".into(),
            title: "Teste".into(),
            media_id: "m1".into(),
            clip_start_sec: 0.0,
            clip_end_sec: 30.0,
            duration_sec: 30.0,
            width: 1080,
            height: 1920,
            fps: 30,
            webcam_rect: NormalizedRect { x: 0.7, y: 0.65, w: 0.28, h: 0.33 },
            content_rect: None,
            burn_subtitles: true,
            template_path: None,
            engine_preference: "hybrid_nvenc".into(),
            motion_html: None,
            creative_direction: String::new(),
            scenes: build_smart_default_scenes(30.0),
            chat_history: vec![],
            rendered_media_id: None,
            updated_at: Utc::now().to_rfc3339(),
        };
        let reply = apply_local_ptbr_chat_edit(
            &mut comp,
            "Entre 0:10 e 0:18 coloca tela cheia com webcam no canto superior esquerdo e título 'OLHA O PLACAR!'",
        );
        comp.normalize();
        assert!(reply.contains("10.0s"));
        assert!(comp.motion_html.as_deref().unwrap_or("").contains("OLHA O PLACAR!"));
    }

    #[test]
    fn lenient_parser_handles_nulls_and_creative_enum_aliases_from_gemini() {
        let messy = r##"{
            "mode": "apply",
            "title": "Jogada Insana",
            "creative_direction": "Ritmo rápido",
            "assistant_reply": "Editei tudo sob medida!",
            "motion_html": "<div class=\"hf-pill\" data-start=\"0\" data-end=\"3\">AO VIVO</div>",
            "scenes": [
                {
                    "id": "s1",
                    "start_sec": 0,
                    "end_sec": 4.5,
                    "layout": "camera",
                    "transition_in": "flash",
                    "pip": null,
                    "overlay": null,
                    "effects": { "camera_zoom_end": 1.12, "impact_shake": 0.4, "vignette": true, "color_mood": "dramatic" }
                },
                {
                    "start_sec": "4.5",
                    "end_sec": "12",
                    "layout": "fullscreen",
                    "effects": null
                }
            ]
        }"##;
        let parsed = parse_ai_director_response(messy).expect("lenient parser must succeed");
        assert_eq!(parsed.scenes.len(), 2);
        assert_eq!(parsed.scenes[0].layout, SceneLayoutPreset::CameraFocus);
        assert_eq!(parsed.scenes[1].layout, SceneLayoutPreset::ContentFullscreen);
        assert!(parsed.scenes[0].effects.vignette);
        assert!(parsed.motion_html.unwrap().contains("AO VIVO"));
    }
}
