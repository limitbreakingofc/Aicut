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

#[derive(Debug, Clone, Deserialize)]
pub struct PassBResponse {
    #[serde(default)]
    pub clips: Vec<RawClipPlan>,
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
        r#"Voce e um Diretor Senior de Cortes Shorts (YouTube Shorts / Reels / TikTok ate 3 minutos).
Voce esta revisando visualmente um trecho candidato (frames em sequencia + transcricao detalhada) para definir o corte cirurgico FINAL.

DESCRICAO DO USUARIO:
"{user_description}"

PISTA DO CANDIDATO:
{candidate_hint}

TRECHO DISPONIVEL (TEMPOS ABSOLUTOS NO VIDEO ORIGINAL):
- De {clip_window_start_sec:.1}s ate {clip_window_end_sec:.1}s
- Duracao MINIMA: {min_clip_sec:.0}s
- Duracao MAXIMA RIGIDA: {max_clip_sec:.0}s (NUNCA ultrapasse {max_clip_sec:.0} segundos no total!)

TRANSCRICAO DETALHADA (TEMPOS ABSOLUTOS EM SEGUNDOS):
{transcript_detailed}

REGRAS OBRIGATORIAS DE COERENCIA (INICIO, MEIO E FIM):
1. INICIO (Gancho + Contexto): o corte deve comecar numa frase ou acao visual forte que faca sentido SOZINHA para quem nao assistiu ao resto do video/live. Nunca comece no meio de uma frase ou com referencias perdidas sem contexto.
2. MEIO (Desenvolvimento): mantenha a tensao, demonstracao ou raciocinio fluindo com ritmo de Shorts. {jump_cut_rule}
3. FIM (Fechamento / Payoff): termine numa conclusao clara, punchline, resultado da jogada ou frase fechada. Nunca corte no meio de uma explicacao.
4. EVIDENCIA VISUAL: descreva objetivamente o que aparece visualmente na tela.
5. TODOS os `start_sec` e `end_sec` devem estar em SEGUNDOS ABSOLUTOS do video original (dentro de [{clip_window_start_sec:.1}, {clip_window_end_sec:.1}]).
6. Se o trecho NAO tiver coerencia real de inicio-meio-fim ou nao combinar com a descricao do usuario, retorne `"clips": []`.

RESPONDA EXCLUSIVAMENTE COM JSON VALIDO NESTE FORMATO:
{{
  "clips": [
    {{
      "title": "Titulo chamativo para Short em portugues",
      "hook_first_5s": "O que prende nos primeiros 5 segundos (visual + fala)",
      "visual_evidence": "O que acontece visualmente na tela neste corte",
      "why_interesting": "Por que este corte atende perfeitamente ao pedido do usuario",
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
}
