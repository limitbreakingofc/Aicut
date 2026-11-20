use aicut_core::{AicutError, AicutKernel, AicutResult, TranscriptRecord};
use aicut_module_smart_clips::engine::{
    resolve_gemini_api_key, SETTING_GEMINI_DIRECTOR_MODEL, SETTING_GEMINI_SCAN_MODEL,
    SETTING_TEMPLATE_PATH,
};
use aicut_module_smart_clips::framing::resolve_effective_framing;
use aicut_module_smart_clips::frames::probe_duration_sec;
use aicut_module_smart_clips::gemini::{generate_json_text, resolve_active_gemini_models};
use aicut_module_smart_clips::prompts::extract_json_object;
use chrono::Utc;
use serde::Deserialize;
use std::path::Path;
use uuid::Uuid;

use crate::schema::{
    save_composition, AicutComposition, ContentFocusConfig, MotionChatMessage, MotionScene,
    PipConfig, PipCorner, SceneLayoutPreset, SceneOverlay, SceneTransition,
};

#[derive(Debug, Deserialize)]
struct AiPlanResponse {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    assistant_reply: Option<String>,
    #[serde(default)]
    scenes: Vec<MotionScene>,
    #[serde(default)]
    burn_subtitles: Option<bool>,
}

/// Plans a dynamic multi-scene `AicutComposition` for a video/clip using Gemini (when configured)
/// or an intelligent multi-scene editorial template (when offline), automatically inheriting the
/// user's saved webcam & content areas from the Enquadramento tab.
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
    let total_dur = match source.metadata.duration_sec {
        Some(d) if d > 0.5 => d,
        _ => match ffprobe_bin {
            Some(ref p) => probe_duration_sec(p, Path::new(&source.file_path))
                .await
                .unwrap_or(60.0),
            None => 60.0,
        },
    };

    let start = clip_start_sec.unwrap_or(0.0).clamp(0.0, (total_dur - 1.0).max(0.0));
    let default_end = (start + 60.0).min(total_dur);
    let end = clip_end_sec
        .unwrap_or(default_end)
        .clamp((start + 1.0).min(total_dur), total_dur);
    let duration_sec = (end - start).max(1.0);

    let (framing, _) = resolve_effective_framing(kernel, Some(&source.id));
    let webcam_rect = framing.webcam_rect.unwrap_or(aicut_module_smart_clips::framing::NormalizedRect {
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

    let mut comp = AicutComposition {
        id: Uuid::new_v4().to_string(),
        title: title_override
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{} · Motion", source.title)),
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
        scenes: Vec::new(),
        chat_history: Vec::new(),
        rendered_media_id: None,
        updated_at: Utc::now().to_rfc3339(),
    };

    let goal_str = user_goal
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("Criar um Short dinâmico e profissional alternando entre SMOKING33, destaque em tela cheia nos momentos importantes e foco na câmera no gancho/reação.");

    let mut reply = String::new();
    let mut used_ai = false;

    if let Some(api_key) = resolve_gemini_api_key(kernel, None) {
        let http = kernel.tools.http_client();
        let (scan_m, dir_m) = resolve_active_gemini_models(
            http,
            &api_key,
            &read_setting_str(kernel, SETTING_GEMINI_SCAN_MODEL, "auto"),
            &read_setting_str(kernel, SETTING_GEMINI_DIRECTOR_MODEL, "auto"),
        )
        .await;
        let model = if dir_m.is_empty() { scan_m } else { dir_m };
        let prompt = build_plan_prompt(&comp.title, duration_sec, goal_str, &transcript_excerpt);
        if let Ok(raw) = generate_json_text(http, &api_key, &model, &prompt).await {
            if let Some(parsed) = parse_ai_plan_response(&raw) {
                if !parsed.scenes.is_empty() {
                    if let Some(t) = parsed.title.filter(|s| !s.trim().is_empty()) {
                        if title_override.is_none() {
                            comp.title = t;
                        }
                    }
                    comp.scenes = parsed.scenes;
                    reply = parsed.assistant_reply.unwrap_or_else(|| {
                        format!(
                            "Criei um roteiro dinâmico com {} cenas usando {model}. Você pode testar no preview ao lado e me pedir qualquer alteração pelo chat!",
                            comp.scenes.len()
                        )
                    });
                    used_ai = true;
                }
            }
        }
    }

    if !used_ai {
        comp.scenes = build_smart_default_scenes(duration_sec);
        reply = format!(
            "Montei uma estrutura dinâmica inicial com {} cenas (Gancho → SMOKING33 → Destaque Tela Cheia + Webcam PiP → Fechamento). Me diga no chat qualquer ajuste que quiser fazer!",
            comp.scenes.len()
        );
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

/// Interactively edits an existing `AicutComposition` from a natural-language chat message
/// (using Gemini when configured, with a responsive offline PT-BR command fallback!).
pub async fn edit_composition_via_chat(
    kernel: &AicutKernel,
    mut comp: AicutComposition,
    user_message: &str,
) -> AicutResult<(AicutComposition, String)> {
    let trimmed = user_message.trim();
    if trimmed.is_empty() {
        return Err(AicutError::invalid_input("Digite uma instrução para a IA ajustar o layout."));
    }

    comp.chat_history.push(MotionChatMessage {
        role: "user".to_string(),
        text: trimmed.to_string(),
        created_at: Utc::now().to_rfc3339(),
    });

    let transcript = kernel
        .repo
        .find_best_transcript_for_media(&comp.media_id)
        .ok()
        .flatten()
        .map(|rt| rt.transcript);
    let transcript_excerpt =
        format_relative_transcript(transcript.as_ref(), comp.clip_start_sec, comp.clip_end_sec);

    let mut assistant_reply: Option<String> = None;

    if let Some(api_key) = resolve_gemini_api_key(kernel, None) {
        let http = kernel.tools.http_client();
        let (scan_m, dir_m) = resolve_active_gemini_models(
            http,
            &api_key,
            &read_setting_str(kernel, SETTING_GEMINI_SCAN_MODEL, "auto"),
            &read_setting_str(kernel, SETTING_GEMINI_DIRECTOR_MODEL, "auto"),
        )
        .await;
        let model = if dir_m.is_empty() { scan_m } else { dir_m };
        let prompt = build_chat_edit_prompt(&comp, &transcript_excerpt, trimmed);
        if let Ok(raw) = generate_json_text(http, &api_key, &model, &prompt).await {
            if let Some(parsed) = parse_ai_plan_response(&raw) {
                if !parsed.scenes.is_empty() {
                    comp.scenes = parsed.scenes;
                }
                if let Some(b) = parsed.burn_subtitles {
                    comp.burn_subtitles = b;
                }
                assistant_reply = Some(parsed.assistant_reply.unwrap_or_else(|| {
                    "Layout atualizado! Confira as novas cenas no preview ao vivo e me diga se quer refinar mais algum detalhe.".to_string()
                }));
            }
        }
    }

    let final_reply = match assistant_reply {
        Some(r) => r,
        None => apply_local_ptbr_chat_edit(&mut comp, trimmed),
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

/// Deterministic offline PT-BR command interpreter so the user can still adjust scenes via chat
/// instantaneously even without an internet connection or Gemini API key.
pub fn apply_local_ptbr_chat_edit(comp: &mut AicutComposition, msg: &str) -> String {
    let lower = msg.to_lowercase();
    let dur = comp.duration_sec;

    // 1. Parse optional quoted title/headline text
    let extracted_quote = extract_quoted_text(msg);

    // 2. Determine requested layout preset
    let requested_layout = if lower.contains("zoom") || lower.contains("detalhe") || lower.contains("destaque com seta") {
        Some(SceneLayoutPreset::CalloutZoom)
    } else if (lower.contains("tela cheia") || lower.contains("conteudo cheio") || lower.contains("conteúdo cheio"))
        && (lower.contains("sem webcam") || lower.contains("so conteudo") || lower.contains("só conteúdo") || lower.contains("esconde"))
    {
        Some(SceneLayoutPreset::ContentFullscreen)
    } else if lower.contains("tela cheia") || lower.contains("pip") || lower.contains("webcam pequena") || lower.contains("canto") {
        Some(SceneLayoutPreset::ContentHeroPip)
    } else if lower.contains("camera grande") || lower.contains("câmera grande") || lower.contains("webcam grande") {
        Some(SceneLayoutPreset::CameraHeroPip)
    } else if lower.contains("so camera") || lower.contains("só câmera") || lower.contains("foco na camera") || lower.contains("foco na câmera") || lower.contains("gancho") {
        Some(SceneLayoutPreset::CameraFocus)
    } else if lower.contains("sem moldura") || lower.contains("split limpo") {
        Some(SceneLayoutPreset::SplitClean)
    } else if lower.contains("smoking") || lower.contains("moldura") || lower.contains("padrao") || lower.contains("padrão") {
        Some(SceneLayoutPreset::SplitBranded)
    } else {
        None
    };

    // 3. Determine optional PiP corner
    let requested_corner = if lower.contains("superior esquerdo") || lower.contains("cima esquerda") {
        Some(PipCorner::TopLeft)
    } else if lower.contains("superior direito") || lower.contains("cima direita") {
        Some(PipCorner::TopRight)
    } else if lower.contains("inferior esquerdo") || lower.contains("baixo esquerda") {
        Some(PipCorner::BottomLeft)
    } else if lower.contains("inferior direito") || lower.contains("baixo direita") {
        Some(PipCorner::BottomRight)
    } else {
        None
    };

    // 4. Determine target time range
    let time_range = parse_time_range_ptbr(&lower, dur);

    if lower.contains("tudo") || lower.contains("video inteiro") || lower.contains("vídeo inteiro") {
        let lay = requested_layout.unwrap_or(SceneLayoutPreset::SplitBranded);
        comp.scenes = vec![MotionScene {
            id: format!("sc-{}", &Uuid::new_v4().to_string()[..6]),
            title: "Vídeo inteiro".to_string(),
            start_sec: 0.0,
            end_sec: dur,
            layout: lay,
            transition_in: SceneTransition::Cut,
            pip: PipConfig {
                corner: requested_corner.unwrap_or_default(),
                ..Default::default()
            },
            content_focus: ContentFocusConfig::default(),
            overlay: SceneOverlay {
                headline: extracted_quote.clone(),
                ..Default::default()
            },
            rationale: format!("Aplicado via chat: {msg}"),
            custom_code: None,
        }];
        return format!("Pronto! Apliquei o layout {:?} no vídeo inteiro (0:00 → {:.1}s).", lay, dur);
    }

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
                callout_text: if lay == SceneLayoutPreset::CalloutZoom {
                    extracted_quote.or_else(|| Some("DESTAQUE".to_string()))
                } else {
                    None
                },
                ..Default::default()
            },
            rationale: format!("Ajustado pelo usuário: {msg}"),
            custom_code: None,
        };
        insert_scene_into_timeline(&mut comp.scenes, new_scene, dur);
        return format!(
            "Feito! Inseri a cena {:?} entre {:.1}s e {:.1}s. Confira no preview ao vivo!",
            lay, t0, t1
        );
    }

    // Apply subtitle toggle if requested
    if lower.contains("sem legenda") || lower.contains("tira legenda") || lower.contains("desativa legenda") {
        comp.burn_subtitles = false;
        return "Legendas desativadas nesta composição.".to_string();
    }
    if lower.contains("com legenda") || lower.contains("ativa legenda") {
        comp.burn_subtitles = true;
        return "Legendas karaokê ativadas nesta composição.".to_string();
    }

    // Apply headline or corner change to the first/selected scene
    if let Some(first) = comp.scenes.first_mut() {
        if let Some(q) = extracted_quote {
            first.overlay.headline = Some(q.clone());
            return format!("Atualizei o título da cena de abertura para \"{q}\"!");
        }
        if let Some(c) = requested_corner {
            for s in &mut comp.scenes {
                s.pip.corner = c;
            }
            return format!("Reposicionei a webcam flutuante (PiP) para {:?} nas cenas aplicáveis.", c);
        }
        if let Some(lay) = requested_layout {
            first.layout = lay;
            return format!("Alterei a primeira cena para {:?}. Para mudar um trecho específico, diga o tempo (ex.: 'de 0:10 a 0:18 tela cheia').", lay);
        }
    }

    "Não identifiquei um trecho exato no modo offline — tente algo como: 'de 0:08 a 0:18 tela cheia com webcam no canto superior direito' ou configure sua Gemini API Key para direção completa por IA!".to_string()
}

fn insert_scene_into_timeline(scenes: &mut Vec<MotionScene>, inserted: MotionScene, total_dur: f64) {
    let mut next: Vec<MotionScene> = Vec::new();
    for s in scenes.drain(..) {
        if s.end_sec <= inserted.start_sec + 0.1 || s.start_sec >= inserted.end_sec - 0.1 {
            next.push(s);
        } else {
            // Left slice before inserted scene
            if s.start_sec + 0.5 < inserted.start_sec {
                let mut left = s.clone();
                left.id = format!("sc-{}", &Uuid::new_v4().to_string()[..6]);
                left.end_sec = inserted.start_sec;
                next.push(left);
            }
            // Right slice after inserted scene
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
            title: "Cena principal".into(),
            start_sec: 0.0,
            end_sec: total_dur,
            layout: SceneLayoutPreset::SplitBranded,
            transition_in: SceneTransition::Cut,
            pip: PipConfig::default(),
            content_focus: ContentFocusConfig::default(),
            overlay: SceneOverlay::default(),
            rationale: String::new(),
            custom_code: None,
        });
    }
    *scenes = next;
}

fn build_smart_default_scenes(dur: f64) -> Vec<MotionScene> {
    if dur <= 10.0 {
        return vec![MotionScene {
            id: "sc-main".into(),
            title: "SMOKING33 Principal".into(),
            start_sec: 0.0,
            end_sec: dur,
            layout: SceneLayoutPreset::SplitBranded,
            transition_in: SceneTransition::Cut,
            pip: PipConfig::default(),
            content_focus: ContentFocusConfig::default(),
            overlay: SceneOverlay::default(),
            rationale: "Clipe curto mantido em SMOKING33 para leitura direta.".into(),
            custom_code: None,
        }];
    }
    let hook_end = (dur * 0.14).clamp(3.0, 4.5);
    let mid_start = (dur * 0.42).max(hook_end + 3.0);
    let mid_end = (dur * 0.72).min(dur - 3.0).max(mid_start + 3.0);
    vec![
        MotionScene {
            id: "sc-hook".into(),
            title: "Gancho na câmera".into(),
            start_sec: 0.0,
            end_sec: hook_end,
            layout: SceneLayoutPreset::CameraFocus,
            transition_in: SceneTransition::Cut,
            pip: PipConfig::default(),
            content_focus: ContentFocusConfig::default(),
            overlay: SceneOverlay {
                badge: Some("GANCHO".into()),
                ..Default::default()
            },
            rationale: "Abertura direta na webcam para prender atenção nos primeiros segundos.".into(),
            custom_code: None,
        },
        MotionScene {
            id: "sc-split-1".into(),
            title: "Contexto · SMOKING33".into(),
            start_sec: hook_end,
            end_sec: mid_start,
            layout: SceneLayoutPreset::SplitBranded,
            transition_in: SceneTransition::Fade,
            pip: PipConfig::default(),
            content_focus: ContentFocusConfig::default(),
            overlay: SceneOverlay::default(),
            rationale: "Layout padrão SMOKING33 equilibrando webcam natural e conteúdo.".into(),
            custom_code: None,
        },
        MotionScene {
            id: "sc-hero".into(),
            title: "Momento principal · Tela cheia + PiP".into(),
            start_sec: mid_start,
            end_sec: mid_end,
            layout: SceneLayoutPreset::ContentHeroPip,
            transition_in: SceneTransition::Fade,
            pip: PipConfig {
                corner: PipCorner::TopRight,
                width_pct: 0.34,
                ..Default::default()
            },
            content_focus: ContentFocusConfig::default(),
            overlay: SceneOverlay {
                badge: Some("DESTAQUE".into()),
                ..Default::default()
            },
            rationale: "Amplia o conteúdo para tela cheia no clímax mantendo a reação em PiP.".into(),
            custom_code: None,
        },
        MotionScene {
            id: "sc-close".into(),
            title: "Fechamento · SMOKING33".into(),
            start_sec: mid_end,
            end_sec: dur,
            layout: SceneLayoutPreset::SplitBranded,
            transition_in: SceneTransition::Fade,
            pip: PipConfig::default(),
            content_focus: ContentFocusConfig::default(),
            overlay: SceneOverlay::default(),
            rationale: "Retorna ao layout SMOKING33 para o fechamento com reação completa.".into(),
            custom_code: None,
        },
    ]
}

fn parse_ai_plan_response(raw: &str) -> Option<AiPlanResponse> {
    let json_str = extract_json_object(raw).unwrap_or(raw);
    serde_json::from_str::<AiPlanResponse>(json_str).ok()
}

fn build_plan_prompt(title: &str, dur: f64, goal: &str, transcript: &str) -> String {
    format!(
        r##"Você é o Diretor de Motion & Layout Dinâmico do Aicut (editor de Shorts 9:16 estilo motion code).
Crie um roteiro de cenas dinâmicas para um Short de {dur:.1} segundos ("{title}").
Objetivo editorial do usuário: {goal}

Layouts disponíveis para `layout` (escolha dinamicamente o melhor para cada momento!):
- "split_branded": Moldura SMOKING33 (webcam natural limpa em cima + conteúdo em baixo)
- "split_clean": Split 9:16 sem moldura
- "content_hero_pip": Conteúdo em TELA CHEIA + webcam limpa flutuante no canto (ideal quando algo visualmente importante acontece na tela!)
- "content_fullscreen": 100% conteúdo sem webcam
- "camera_focus": Webcam limpa em tela cheia (ideal para gancho 3–4s ou reação épica!)
- "camera_hero_pip": Webcam grande (68%) + conteúdo compacto (32%)
- "callout_zoom": Zoom focal em (focus_x, focus_y, zoom 1.5..2.2) + callout_text destacando um detalhe específico na tela + webcam PiP

Regras:
- Crie entre 3 e 6 cenas cobrindo exatamente 0.0 até {dur:.1}s sem lacunas nem sobreposições.
- Trocas naturais (cada cena com pelo menos 2.5s). Use `overlay.headline` curto (<= 5 palavras) só em 1–2 momentos de impacto.
- Responda APENAS JSON válido no formato:
{{
  "title": "Título curto",
  "assistant_reply": "Explicação amigável em PT-BR das escolhas de layout.",
  "scenes": [
    {{
      "id": "sc-1",
      "title": "Gancho inicial",
      "start_sec": 0.0,
      "end_sec": 3.8,
      "layout": "camera_focus",
      "transition_in": "cut",
      "pip": {{ "corner": "top_right", "width_pct": 0.34, "border_hex": "#EAB308", "visible": true }},
      "content_focus": {{ "focus_x": 0.5, "focus_y": 0.5, "zoom": 1.0, "fit_blur": false }},
      "overlay": {{ "headline": "OLHA ISSO!", "badge": null, "callout_text": null, "accent_hex": "#EAB308" }},
      "rationale": "Prende atenção imediatamente na expressão do apresentador."
    }}
  ]
}}

Transcrição relativa (0.0s .. {dur:.1}s):
{transcript}"##
    )
}

fn build_chat_edit_prompt(comp: &AicutComposition, transcript: &str, user_msg: &str) -> String {
    let scenes_json = serde_json::to_string_pretty(&comp.scenes).unwrap_or_else(|_| "[]".into());
    let recent_chat: String = comp
        .chat_history
        .iter()
        .rev()
        .take(6)
        .rev()
        .map(|m| format!("{}: {}", m.role, m.text))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"Você é o Diretor de Motion & Layout do Aicut editando uma composição de Short 9:16 de {dur:.1}s.
Aplique EXATAMENTE o pedido do usuário preservando o restante das cenas que ele não pediu para mudar.

Layouts válidos: "split_branded", "split_clean", "content_hero_pip", "content_fullscreen", "camera_focus", "camera_hero_pip", "callout_zoom".
Cantos PiP válidos: "top_right", "top_left", "bottom_right", "bottom_left".
Transições válidas: "cut", "fade", "slide_up", "wipe_left".

Cenas atuais (JSON):
{scenes_json}

Transcrição do clipe (0.0s .. {dur:.1}s):
{transcript}

Histórico recente:
{recent_chat}

Novo pedido do usuário:
"{user_msg}"

Responda APENAS JSON válido:
{{
  "assistant_reply": "Resposta direta em PT-BR explicando exatamente o que você mudou.",
  "burn_subtitles": {subs},
  "scenes": [ ...array completo atualizado de 0.0 até {dur:.1}s sem lacunas... ]
}}"#,
        dur = comp.duration_sec,
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

    // Look for two time tokens like "0:12" and "0:20", or "10s" and "22s"
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
    fn offline_ptbr_chat_inserts_fullscreen_pip_scene_at_exact_timecode() {
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
            webcam_rect: aicut_module_smart_clips::framing::NormalizedRect { x: 0.7, y: 0.65, w: 0.28, h: 0.33 },
            content_rect: None,
            burn_subtitles: true,
            template_path: None,
            engine_preference: "hybrid_nvenc".into(),
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
        let target = comp
            .scenes
            .iter()
            .find(|s| (s.start_sec - 10.0).abs() < 0.2 && (s.end_sec - 18.0).abs() < 0.2)
            .expect("expected inserted scene 10..18s");
        assert_eq!(target.layout, SceneLayoutPreset::ContentHeroPip);
        assert_eq!(target.pip.corner, PipCorner::TopLeft);
        assert_eq!(target.overlay.headline.as_deref(), Some("OLHA O PLACAR!"));
    }
}
