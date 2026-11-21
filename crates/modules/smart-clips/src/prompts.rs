use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct CandidateWindow {
    pub title: String,
    pub start_sec: f64,
    pub end_sec: f64,
    #[serde(default)]
    pub hook_reason: String,
    #[serde(default)]
    pub visual_evidence: String,
    #[serde(default)]
    pub score: u8,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PassAResponse {
    #[serde(default)]
    pub candidates: Vec<CandidateWindow>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawSegment {
    pub start_sec: f64,
    pub end_sec: f64,
    #[serde(default)]
    pub role: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RawArc {
    #[serde(default)]
    pub beginning: String,
    #[serde(default)]
    pub middle: String,
    #[serde(default)]
    pub end: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RawScores {
    #[serde(default)]
    pub match_description: u8,
    #[serde(default)]
    pub visual_interest: u8,
    #[serde(default)]
    pub narrative_coherence: u8,
    #[serde(default)]
    pub hook_strength: u8,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawClipPlan {
    pub title: String,
    #[serde(default)]
    pub hook_first_5s: String,
    #[serde(default)]
    pub visual_evidence: String,
    #[serde(default)]
    pub why_interesting: String,
    #[serde(default)]
    pub narrative_arc: RawArc,
    #[serde(default)]
    pub segments: Vec<RawSegment>,
    #[serde(default)]
    pub scores: RawScores,
}

fn default_approved() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct PassBResponse {
    #[serde(default = "default_approved")]
    pub approved: bool,
    #[serde(default)]
    pub rejection_reason: String,
    #[serde(default)]
    pub clips: Vec<RawClipPlan>,
}

/// Validates a Pass B full-analysis response: keeps only clips explicitly approved by the
/// Director that also meet minimum editorial/visual quality thresholds; otherwise discards them.
pub fn validate_pass_b_response(resp: &PassBResponse) -> Result<Vec<RawClipPlan>, String> {
    if !resp.approved {
        let reason = if resp.rejection_reason.trim().is_empty() {
            "Reprovado na analise completa de qualidade audiovisual.".to_string()
        } else {
            resp.rejection_reason.trim().to_string()
        };
        return Err(reason);
    }

    if resp.clips.is_empty() {
        let reason = if resp.rejection_reason.trim().is_empty() {
            "Sem corte coerente de inicio-meio-fim apos revisao completa.".to_string()
        } else {
            resp.rejection_reason.trim().to_string()
        };
        return Err(reason);
    }

    let mut validated = Vec::new();
    let mut last_discard_reason = String::new();

    for clip in &resp.clips {
        let s = &clip.scores;
        let non_zero: Vec<u32> = [
            s.match_description,
            s.visual_interest,
            s.narrative_coherence,
            s.hook_strength,
        ]
        .into_iter()
        .filter(|&v| v > 0)
        .map(u32::from)
        .collect();

        let avg = if non_zero.is_empty() {
            75
        } else {
            non_zero.iter().sum::<u32>() / non_zero.len() as u32
        };

        if s.match_description > 0 && s.match_description < 60 {
            last_discard_reason = format!(
                "Aderencia a descricao baixa ({}/100).",
                s.match_description
            );
            continue;
        }
        if s.narrative_coherence > 0 && s.narrative_coherence < 60 {
            last_discard_reason = format!(
                "Coerencia narrativa insuficiente ({}/100).",
                s.narrative_coherence
            );
            continue;
        }
        if avg < 65 {
            last_discard_reason = format!("Qualidade geral abaixo do minimo ({avg}/100 < 65).");
            continue;
        }
        if clip.segments.is_empty() {
            last_discard_reason = "Corte sem segmentos validos.".to_string();
            continue;
        }

        validated.push(clip.clone());
    }

    if validated.is_empty() {
        return Err(if last_discard_reason.is_empty() {
            "Descartado na validacao de qualidade.".to_string()
        } else {
            last_discard_reason
        });
    }

    Ok(validated)
}

/// Fast Stage-1 prompt: discovers the best candidate clip intervals directly from the
/// synchronized transcript (zero full-video re-encoding/upload). Every candidate will be
/// thoroughly validated with full video+audio analysis in Stage 2 before generation.
#[allow(clippy::too_many_arguments)]
pub fn build_transcript_pass_a_prompt(
    user_description: &str,
    window_start_sec: f64,
    window_end_sec: f64,
    max_clip_sec: f64,
    min_clip_sec: f64,
    max_candidates: usize,
    transcript_excerpt: &str,
) -> String {
    format!(
        r#"Voce e um estrategista senior de cortes virais (YouTube Shorts / Reels / TikTok ate 3 minutos).
ETAPA 1 (DESCOBERTA RAPIDA PELA TRANSCRICAO):
Leia a transcricao sincronizada abaixo e identifique os melhores trechos candidatos para cortes.
Na Etapa 2, cada candidato escolhido aqui passara por uma ANALISE COMPLETA (video + audio real) que decidira se o corte sera APROVADO para geracao ou DESCARTADO.

OBJETIVO DO USUARIO (DESCRICAO EDITORIAL):
"{user_description}"

JANELA DA TRANSCRICAO (TEMPOS ABSOLUTOS NO VIDEO ORIGINAL):
- Inicio: {window_start_sec:.1}s
- Fim: {window_end_sec:.1}s
- Duracao desejada de cada corte: entre {min_clip_sec:.0}s e {max_clip_sec:.0}s (NUNCA passe de {max_clip_sec:.0}s)

TRANSCRICAO SINCRONIZADA (TEMPOS ABSOLUTOS EM SEGUNDOS):
{transcript_excerpt}

REGRAS CRITICAS DA ETAPA 1:
1. Selecione ate {max_candidates} candidatos DISTINTOS (sem sobreposicao) com maior potencial de retencao e aderencia ao pedido do usuario.
2. Cada candidato ja deve ter arco narrativo completo identificavel na fala: INICIO (gancho/premissa), MEIO (desenvolvimento/tensao) e FIM (punchline/conclusao).
3. Use `start_sec` e `end_sec` EXATOS em SEGUNDOS ABSOLUTOS do video original (entre {window_start_sec:.1} e {window_end_sec:.1}), comecando no inicio de uma frase e terminando no final de uma conclusao.
4. Prefira trechos entre {min_clip_sec:.0}s e {max_clip_sec:.0}s. Ordene do melhor (`score` mais alto) para o menor.
5. Se nenhum trecho atender ao pedido, retorne `"candidates": []`.

RESPONDA EXCLUSIVAMENTE COM JSON VALIDO NESTE FORMATO (sem texto fora do JSON):
{{
  "candidates": [
    {{
      "title": "Titulo curto em portugues descrevendo o corte",
      "start_sec": 120.0,
      "end_sec": 195.0,
      "hook_reason": "Por que a abertura prende atencao nos primeiros 5 segundos",
      "visual_evidence": "Contexto esperado na tela segundo a fala (sera verificado na Etapa 2)",
      "score": 92
    }}
  ]
}}"#
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_pass_a_prompt(
    user_description: &str,
    window_start_sec: f64,
    window_end_sec: f64,
    max_clip_sec: f64,
    max_candidates: usize,
    transcript_excerpt: &str,
    is_storyboard: bool,
) -> String {
    let visual_note = if is_storyboard {
        "Voce esta vendo grades visuais (storyboards 3x3) extraidas sequencialmente deste trecho do video, em ordem cronologica (esquerda para direita, cima para baixo), sincronizadas com a transcricao abaixo."
    } else {
        "Voce esta assistindo diretamente a este trecho do video (frames visuais + audio real) sincronizado com a transcricao abaixo."
    };

    format!(
        r#"Voce e um diretor de edicao e estrategista de YouTube Shorts/Reels/TikTok de altissimo nivel.
{visual_note}

OBJETIVO DO USUARIO (DESCRICAO EDITORIAL):
"{user_description}"

JANELA ANALISADA DO VIDEO ORIGINAL:
- Inicio da janela: {window_start_sec:.1}s
- Fim da janela: {window_end_sec:.1}s
- Duracao maxima de cada corte final: {max_clip_sec:.0} segundos (ate 3 minutos)

TRANSCRICAO SINCRONIZADA DESTA JANELA (tempos ABSOLUTOS no video original):
{transcript_excerpt}

TAREFA (PASSAGEM A — DESCOBERTA VISUAL + NARRATIVA):
Encontre ate {max_candidates} momentos candidatos mais interessantes e aderentes a descricao do usuario.
REGRAS CRITICAS:
1. USE O QUE VOCE VE NA TELA (expressoes faciais, jogadas, acao, mudancas visuais, demonstracoes, humor visual) junto com o que e falado.
2. Cada candidato precisa ter potencial para uma historia COMPLETA com INICIO (gancho + contexto compreensivel), MEIO (desenvolvimento) e FIM (desfecho/conclusao/punchline).
3. Retorne `start_sec` e `end_sec` em SEGUNDOS ABSOLUTOS do video original (entre {window_start_sec:.1} e {window_end_sec:.1}), incluindo uma pequena folga de contexto (5–15s) se necessario.
4. Prefira candidatos entre 25s e {max_clip_sec:.0}s.

RESPONDA EXCLUSIVAMENTE COM JSON VALIDO NESTE FORMATO (sem texto fora do JSON):
{{
  "candidates": [
    {{
      "title": "Titulo curto em portugues descrevendo o momento",
      "start_sec": 120.0,
      "end_sec": 215.0,
      "hook_reason": "Por que prende atencao nos primeiros 5 segundos",
      "visual_evidence": "O que voce viu acontecendo visualmente na tela",
      "score": 92
    }}
  ]
}}"#
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_pass_b_prompt(
    user_description: &str,
    clip_window_start_sec: f64,
    clip_window_end_sec: f64,
    max_clip_sec: f64,
    min_clip_sec: f64,
    allow_jump_cuts: bool,
    candidate_hint: &str,
    transcript_detailed: &str,
) -> String {
    let jump_cut_rule = if allow_jump_cuts {
        "Voce PODE usar de 1 a 3 segmentos em `segments` para pular pausas longas ou trechos mortos no meio da historia, mantendo transicao natural e soma TOTAL <= {max_clip_sec:.0}s."
    } else {
        "Use EXATAMENTE 1 segmento continuo em `segments` (sem cortes internos) com (end_sec - start_sec) <= {max_clip_sec:.0}s."
    };
    let jump_cut_rule = jump_cut_rule.replace("{max_clip_sec:.0}", &format!("{max_clip_sec:.0}"));

    format!(
        r#"Voce e um Diretor Senior e Juiz de Qualidade de Cortes Shorts (YouTube Shorts / Reels / TikTok ate 3 minutos).
ETAPA 2 (VALIDACAO COMPLETA — VIDEO + AUDIO + TRANSCRICAO):
Este candidato foi pre-selecionado pela transcricao. Agora voce esta ASSISTINDO ao trecho real (video/frames + audio + transcricao) para AVALIAR A QUALIDADE REAL e decidir:
- SE FOR VALIDADO (qualidade forte de gancho, visual, ritmo e historia completa): retorne `"approved": true` com o corte refinado cirurgicamente em `"clips"`.
- SE NAO TIVER QUALIDADE SUFICIENTE (gancho fraco, historia incompleta, fora do pedido, visual morto/confuso ou corte quebrado): DESCARTE retornando `"approved": false`, `"rejection_reason": "motivo claro"` e `"clips": []`.

DESCRICAO DO USUARIO:
"{user_description}"

CANDIDATO PRE-SELECIONADO PELA TRANSCRICAO:
{candidate_hint}

TRECHO DISPONIVEL (TEMPOS ABSOLUTOS NO VIDEO ORIGINAL):
- De {clip_window_start_sec:.1}s ate {clip_window_end_sec:.1}s
- Duracao MINIMA: {min_clip_sec:.0}s
- Duracao MAXIMA RIGIDA: {max_clip_sec:.0}s (NUNCA ultrapasse {max_clip_sec:.0} segundos no total!)

TRANSCRICAO DETALHADA (TEMPOS ABSOLUTOS EM SEGUNDOS):
{transcript_detailed}

CRITERIOS RIGOROSOS DE APROVACAO OU DESCARTE:
1. INICIO (Gancho + Contexto): o corte deve comecar numa frase ou acao visual forte que faca sentido SOZINHA para quem nao assistiu ao resto do video. Nunca comece no meio de uma frase.
2. MEIO (Desenvolvimento): mantenha a tensao, demonstracao ou raciocinio fluindo com ritmo de Shorts. {jump_cut_rule}
3. FIM (Fechamento / Payoff): termine numa conclusao clara, punchline, resultado ou frase fechada. Nunca corte no meio de uma explicacao.
4. VALIDACAO AUDIOVISUAL REAL: confirme no video/audio real se o momento realmente entrega valor visual, expressao/entonacao e clareza — seja rigoroso e descarte candidatos fracos!
5. TODOS os `start_sec` e `end_sec` devem estar em SEGUNDOS ABSOLUTOS do video original (dentro de [{clip_window_start_sec:.1}, {clip_window_end_sec:.1}]).

RESPONDA EXCLUSIVAMENTE COM JSON VALIDO NESTE FORMATO:
{{
  "approved": true,
  "rejection_reason": "",
  "clips": [
    {{
      "title": "Titulo chamativo para Short em portugues",
      "hook_first_5s": "O que prende nos primeiros 5 segundos (visual + fala)",
      "visual_evidence": "O que acontece visualmente na tela neste corte",
      "why_interesting": "Por que este corte foi validado e atende perfeitamente ao pedido do usuario",
      "narrative_arc": {{
        "beginning": "Como a premissa/gancho abre de forma autossuficiente",
        "middle": "Como a situacao/explicacao se desenvolve",
        "end": "Qual e o desfecho ou punchline que fecha o corte"
      }},
      "segments": [
        {{ "start_sec": 124.5, "end_sec": 218.2, "role": "complete_story" }}
      ],
      "scores": {{
        "match_description": 95,
        "visual_interest": 90,
        "narrative_coherence": 94,
        "hook_strength": 91
      }}
    }}
  ]
}}"#
    )
}

/// Extracts the first valid JSON object from an LLM response (stripping markdown ```json fences or prose).
pub fn extract_json_object(raw: &str) -> Option<&str> {
    let trimmed = raw.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return Some(trimmed);
    }
    // Strip markdown fences if present
    let without_fence = if let Some(idx) = trimmed.find("```json") {
        let after = &trimmed[idx + 7..];
        after.split("```").next().unwrap_or(after).trim()
    } else if let Some(idx) = trimmed.find("```") {
        let after = &trimmed[idx + 3..];
        after.split("```").next().unwrap_or(after).trim()
    } else {
        trimmed
    };

    let start = without_fence.find('{')?;
    let end = without_fence.rfind('}')?;
    if end > start {
        Some(&without_fence[start..=end])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_json_from_markdown_fence() {
        let raw = "Aqui esta o resultado:\n```json\n{\"candidates\": [{\"title\":\"A\",\"start_sec\":10,\"end_sec\":50,\"score\":90}]}\n```\nPronto!";
        let json = extract_json_object(raw).unwrap();
        let parsed: PassAResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.candidates.len(), 1);
        assert_eq!(parsed.candidates[0].title, "A");
    }

    #[test]
    fn validates_and_discards_pass_b_quality_gate() {
        let rejected: PassBResponse = serde_json::from_str(
            r#"{"approved": false, "rejection_reason": "Gancho fraco e historia incompleta", "clips": []}"#,
        )
        .unwrap();
        let err = validate_pass_b_response(&rejected).unwrap_err();
        assert!(err.contains("Gancho fraco"));

        let low_score: PassBResponse = serde_json::from_str(
            r#"{
              "approved": true,
              "clips": [{
                "title": "Fraco",
                "segments": [{"start_sec": 10.0, "end_sec": 50.0, "role": "complete"}],
                "scores": {"match_description": 45, "visual_interest": 50, "narrative_coherence": 55, "hook_strength": 50}
              }]
            }"#,
        )
        .unwrap();
        assert!(validate_pass_b_response(&low_score).is_err());

        let approved: PassBResponse = serde_json::from_str(
            r#"{
              "approved": true,
              "clips": [{
                "title": "Excelente corte",
                "segments": [{"start_sec": 10.0, "end_sec": 75.0, "role": "complete"}],
                "scores": {"match_description": 92, "visual_interest": 88, "narrative_coherence": 94, "hook_strength": 90}
              }]
            }"#,
        )
        .unwrap();
        let ok = validate_pass_b_response(&approved).unwrap();
        assert_eq!(ok.len(), 1);
        assert_eq!(ok[0].title, "Excelente corte");
    }
}

