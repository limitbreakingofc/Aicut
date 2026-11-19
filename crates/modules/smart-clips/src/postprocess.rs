use aicut_core::{
    ClipNarrativeArc, ClipScores, SmartClipPlan, SmartClipSegment, TranscriptRecord,
};
use uuid::Uuid;

use crate::prompts::RawClipPlan;

/// Normalizes, snaps to transcript word boundaries, strictly enforces `<= max_duration_sec` (e.g. 180s),
/// deduplicates overlapping candidates (IoU > 0.30), and ranks clips by combined score.
pub fn validate_and_finalize_clips(
    raw_clips: Vec<RawClipPlan>,
    transcript: Option<&TranscriptRecord>,
    video_duration_sec: f64,
    min_duration_sec: f64,
    max_duration_sec: f64,
    max_clips: usize,
    allow_jump_cuts: bool,
) -> Vec<SmartClipPlan> {
    let max_dur = max_duration_sec.clamp(15.0, 180.0);
    let min_dur = min_duration_sec.clamp(5.0, max_dur);

    let mut validated: Vec<SmartClipPlan> = Vec::new();

    for raw in raw_clips {
        let mut segs: Vec<SmartClipSegment> = raw
            .segments
            .into_iter()
            .filter_map(|s| {
                let start = s.start_sec.clamp(0.0, video_duration_sec);
                let end = s.end_sec.clamp(0.0, video_duration_sec);
                if end - start >= 2.0 {
                    Some(SmartClipSegment {
                        start_sec: start,
                        end_sec: end,
                        role: if s.role.trim().is_empty() {
                            "story".to_string()
                        } else {
                            s.role.trim().to_string()
                        },
                    })
                } else {
                    None
                }
            })
            .collect();

        if segs.is_empty() {
            continue;
        }

        // Sort chronologically and merge overlapping or near-adjacent segments (< 0.5s gap)
        segs.sort_by(|a, b| {
            a.start_sec
                .partial_cmp(&b.start_sec)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut merged: Vec<SmartClipSegment> = Vec::new();
        for seg in segs {
            if let Some(last) = merged.last_mut() {
                if seg.start_sec <= last.end_sec + 0.5 {
                    last.end_sec = last.end_sec.max(seg.end_sec);
                    continue;
                }
            }
            merged.push(seg);
        }

        // If continuous mode is requested, collapse into a single continuous span
        if !allow_jump_cuts && merged.len() > 1 {
            let first_start = merged.first().unwrap().start_sec;
            let last_end = merged.last().unwrap().end_sec;
            merged = vec![SmartClipSegment {
                start_sec: first_start,
                end_sec: last_end,
                role: "continuous_story".to_string(),
            }];
        }

        // Surgical word-boundary snapping using Whisper word/segment timestamps
        if let Some(tr) = transcript {
            for seg in &mut merged {
                snap_segment_to_transcript(seg, tr, video_duration_sec);
            }
        }

        // Enforce strict <= max_duration_sec cap, trimming cleanly at transcript boundaries if needed
        enforce_max_duration(&mut merged, max_dur, transcript);

        let total_dur: f64 = merged.iter().map(|s| (s.end_sec - s.start_sec).max(0.0)).sum();
        if total_dur < min_dur || merged.is_empty() {
            continue;
        }

        let scores = ClipScores {
            match_description: raw.scores.match_description.min(100),
            visual_interest: raw.scores.visual_interest.min(100),
            narrative_coherence: raw.scores.narrative_coherence.min(100),
            hook_strength: raw.scores.hook_strength.min(100),
        };

        validated.push(SmartClipPlan {
            id: Uuid::new_v4().to_string(),
            title: if raw.title.trim().is_empty() {
                "Corte Destaque".to_string()
            } else {
                raw.title.trim().to_string()
            },
            hook_first_5s: raw.hook_first_5s.trim().to_string(),
            visual_evidence: raw.visual_evidence.trim().to_string(),
            why_interesting: raw.why_interesting.trim().to_string(),
            narrative_arc: ClipNarrativeArc {
                beginning: raw.narrative_arc.beginning.trim().to_string(),
                middle: raw.narrative_arc.middle.trim().to_string(),
                end: raw.narrative_arc.end.trim().to_string(),
            },
            segments: merged,
            total_duration_sec: (total_dur * 10.0).round() / 10.0,
            scores,
            rendered_media_id: None,
        });
    }

    // Sort highest score first before deduplication
    validated.sort_by(|a, b| {
        b.scores
            .overall()
            .partial_cmp(&a.scores.overall())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Deduplicate clips that overlap heavily (> 30% temporal IoU)
    let mut deduped: Vec<SmartClipPlan> = Vec::new();
    for clip in validated {
        let overlaps = deduped.iter().any(|existing| clip_iou(existing, &clip) > 0.30);
        if !overlaps {
            deduped.push(clip);
            if deduped.len() >= max_clips {
                break;
            }
        }
    }

    deduped
}

/// Snaps `seg.start_sec` and `seg.end_sec` within +/- 2.0s to the nearest clean speech boundary
/// so clips never begin or end mid-word.
pub fn snap_segment_to_transcript(
    seg: &mut SmartClipSegment,
    transcript: &TranscriptRecord,
    video_duration_sec: f64,
) {
    let mut word_bounds: Vec<(f64, f64)> = Vec::new();
    for tseg in &transcript.segments {
        if !tseg.words.is_empty() {
            for w in &tseg.words {
                let s = w.start_ms as f64 / 1000.0;
                let e = w.end_ms as f64 / 1000.0;
                if e > s {
                    word_bounds.push((s, e));
                }
            }
        } else {
            let s = tseg.start_ms as f64 / 1000.0;
            let e = tseg.end_ms as f64 / 1000.0;
            if e > s {
                word_bounds.push((s, e));
            }
        }
    }

    if word_bounds.is_empty() {
        return;
    }

    // Snap start_sec: if inside a word, pull back to word start (-0.15s breathing room).
    // Otherwise snap to nearest word start within 1.5s.
    if let Some((ws, _)) = word_bounds
        .iter()
        .find(|(s, e)| seg.start_sec > *s && seg.start_sec < *e)
    {
        seg.start_sec = (*ws - 0.15).max(0.0);
    } else if let Some((ws, _)) = word_bounds
        .iter()
        .min_by(|a, b| {
            (a.0 - seg.start_sec)
                .abs()
                .partial_cmp(&(b.0 - seg.start_sec).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .filter(|(s, _)| (s - seg.start_sec).abs() <= 1.5)
    {
        seg.start_sec = (*ws - 0.15).max(0.0);
    }

    // Snap end_sec: if inside a word, extend to word end (+0.25s breathing room).
    // Otherwise snap to nearest word end within 1.5s.
    if let Some((_, we)) = word_bounds
        .iter()
        .find(|(s, e)| seg.end_sec > *s && seg.end_sec < *e)
    {
        seg.end_sec = (*we + 0.25).min(video_duration_sec);
    } else if let Some((_, we)) = word_bounds
        .iter()
        .min_by(|a, b| {
            (a.1 - seg.end_sec)
                .abs()
                .partial_cmp(&(b.1 - seg.end_sec).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .filter(|(_, e)| (e - seg.end_sec).abs() <= 1.5)
    {
        seg.end_sec = (*we + 0.25).min(video_duration_sec);
    }

    if seg.end_sec <= seg.start_sec + 1.0 {
        seg.end_sec = (seg.start_sec + 2.0).min(video_duration_sec);
    }
}

/// Strictly guarantees total summed segment duration <= `max_dur`.
pub fn enforce_max_duration(
    segments: &mut Vec<SmartClipSegment>,
    max_dur: f64,
    transcript: Option<&TranscriptRecord>,
) {
    let mut acc = 0.0;
    let mut kept = Vec::new();
    for mut seg in segments.drain(..) {
        let len = (seg.end_sec - seg.start_sec).max(0.0);
        if acc + len <= max_dur {
            acc += len;
            kept.push(seg);
        } else {
            let remaining = max_dur - acc;
            if remaining >= 4.0 {
                let mut target_end = seg.start_sec + remaining;
                // Prefer ending cleanly at the latest transcript segment ending <= target_end
                if let Some(tr) = transcript {
                    if let Some(clean_end) = tr
                        .segments
                        .iter()
                        .map(|ts| ts.end_ms as f64 / 1000.0)
                        .filter(|&e| e > seg.start_sec + 4.0 && e <= target_end)
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    {
                        target_end = (clean_end + 0.20).min(seg.start_sec + remaining);
                    }
                }
                seg.end_sec = target_end;
                kept.push(seg);
            }
            break;
        }
    }
    *segments = kept;
}

fn clip_span(clip: &SmartClipPlan) -> (f64, f64) {
    let start = clip
        .segments
        .first()
        .map(|s| s.start_sec)
        .unwrap_or(0.0);
    let end = clip.segments.last().map(|s| s.end_sec).unwrap_or(start);
    (start, end)
}

pub fn clip_iou(a: &SmartClipPlan, b: &SmartClipPlan) -> f64 {
    let (a0, a1) = clip_span(a);
    let (b0, b1) = clip_span(b);
    let inter = (a1.min(b1) - a0.max(b0)).max(0.0);
    let union = (a1.max(b1) - a0.min(b0)).max(0.001);
    inter / union
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompts::{RawArc, RawScores, RawSegment};
    use aicut_core::{TranscriptSegment, TranscriptWord};
    use chrono::Utc;

    fn sample_transcript() -> TranscriptRecord {
        TranscriptRecord {
            id: "t1".into(),
            media_id: "m1".into(),
            media_title: "Live".into(),
            model_id: "large-v3-turbo".into(),
            language: "pt".into(),
            duration_sec: Some(600.0),
            full_text: "Olha essa jogada absurda no final.".into(),
            segments: vec![
                TranscriptSegment {
                    index: 0,
                    start_ms: 10_000,
                    end_ms: 14_000,
                    text: "Olha essa jogada".into(),
                    words: vec![
                        TranscriptWord { start_ms: 10_000, end_ms: 11_200, text: "Olha".into(), probability: None },
                        TranscriptWord { start_ms: 11_200, end_ms: 12_500, text: "essa".into(), probability: None },
                        TranscriptWord { start_ms: 12_500, end_ms: 14_000, text: "jogada".into(), probability: None },
                    ],
                },
                TranscriptSegment {
                    index: 1,
                    start_ms: 170_000,
                    end_ms: 185_000,
                    text: "fechamento perfeito".into(),
                    words: vec![],
                },
            ],
            created_at: Utc::now(),
        }
    }

    #[test]
    fn enforces_strict_180s_cap_and_snaps_mid_word() {
        let tr = sample_transcript();
        let raw = vec![RawClipPlan {
            title: "Jogada Epica".into(),
            hook_first_5s: "Hook".into(),
            visual_evidence: "Clutch na tela".into(),
            why_interesting: "Epico".into(),
            narrative_arc: RawArc {
                beginning: "Inicio".into(),
                middle: "Meio".into(),
                end: "Fim".into(),
            },
            // Starts mid-word (10.6s is inside "Olha" 10.0-11.2) and spans 220s (> 180s!)
            segments: vec![RawSegment {
                start_sec: 10.6,
                end_sec: 230.6,
                role: "story".into(),
            }],
            scores: RawScores {
                match_description: 95,
                visual_interest: 92,
                narrative_coherence: 90,
                hook_strength: 88,
            },
        }];

        let out = validate_and_finalize_clips(raw, Some(&tr), 600.0, 20.0, 180.0, 5, false);
        assert_eq!(out.len(), 1);
        assert!(out[0].total_duration_sec <= 180.0);
        // Snapped before word "Olha" at 10.0 - 0.15 = 9.85s
        assert!((out[0].segments[0].start_sec - 9.85).abs() < 0.05);
    }

    #[test]
    fn deduplicates_overlapping_candidates_keeping_highest_score() {
        let mk = |title: &str, start: f64, end: f64, score: u8| RawClipPlan {
            title: title.into(),
            hook_first_5s: String::new(),
            visual_evidence: String::new(),
            why_interesting: String::new(),
            narrative_arc: RawArc::default(),
            segments: vec![RawSegment { start_sec: start, end_sec: end, role: "story".into() }],
            scores: RawScores {
                match_description: score,
                visual_interest: score,
                narrative_coherence: score,
                hook_strength: score,
            },
        };

        let out = validate_and_finalize_clips(
            vec![
                mk("Lower", 100.0, 200.0, 75),
                mk("Higher", 105.0, 205.0, 96),
                mk("Distinct", 350.0, 440.0, 88),
            ],
            None,
            600.0,
            20.0,
            180.0,
            5,
            false,
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].title, "Higher");
        assert_eq!(out[1].title, "Distinct");
    }
}
