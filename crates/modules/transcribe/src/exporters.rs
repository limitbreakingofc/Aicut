use aicut_core::{
    AicutEvent, AicutKernel, AicutResult, MediaAsset, MediaKind, MediaMetadata, MediaSourceType,
    TranscriptRecord, TranscriptSegment,
};
use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    /// SubRip subtitle (.srt) — universal standard
    Srt,
    /// WebVTT subtitle (.vtt) — web & YouTube
    Vtt,
    /// Advanced SubStation Alpha (.ass) — styled subtitles
    Ass,
    /// ASS with word-by-word karaoke highlight (\kf) for Reels/Shorts/TikTok
    AssKaraoke,
    /// Synchronized lyrics (.lrc)
    Lrc,
    /// Plain clean text (.txt)
    Txt,
    /// Text with [HH:MM:SS] timecodes (.txt)
    TxtTimestamps,
    /// Structured word-level JSON (.json) — ideal for AI agents
    Json,
    /// FCPXML 1.10 (.fcpxml) — DaVinci Resolve, Final Cut Pro, CapCut
    Fcpxml,
    /// Adobe Premiere Pro XMEML v4 (.xml)
    PremiereXml,
    /// CMX 3600 EDL (.edl) — DaVinci Resolve, Premiere, Avid
    Edl,
    /// OpenTimelineIO (.otio) — open industry timeline standard
    Otio,
    /// Timeline Markers CSV (.csv) — DaVinci Resolve & Premiere markers
    CsvMarkers,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExportFormatInfo {
    pub id: ExportFormat,
    pub name: String,
    pub extension: String,
    pub category: String,
    pub description: String,
}

impl ExportFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Ass | Self::AssKaraoke => "ass",
            Self::Lrc => "lrc",
            Self::Txt | Self::TxtTimestamps => "txt",
            Self::Json => "json",
            Self::Fcpxml => "fcpxml",
            Self::PremiereXml => "xml",
            Self::Edl => "edl",
            Self::Otio => "otio",
            Self::CsvMarkers => "csv",
        }
    }

    pub fn suffix(self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Ass => "ass",
            Self::AssKaraoke => "karaoke.ass",
            Self::Lrc => "lrc",
            Self::Txt => "txt",
            Self::TxtTimestamps => "timestamps.txt",
            Self::Json => "transcript.json",
            Self::Fcpxml => "fcpxml",
            Self::PremiereXml => "premiere.xml",
            Self::Edl => "edl",
            Self::Otio => "otio",
            Self::CsvMarkers => "markers.csv",
        }
    }

    pub fn media_kind(self) -> MediaKind {
        match self {
            Self::Srt | Self::Vtt | Self::Ass | Self::AssKaraoke | Self::Lrc => MediaKind::Subtitle,
            Self::Fcpxml | Self::PremiereXml | Self::Edl | Self::Otio | Self::CsvMarkers => {
                MediaKind::Timeline
            }
            Self::Txt | Self::TxtTimestamps | Self::Json => MediaKind::Transcript,
        }
    }

    pub fn display_label(self) -> &'static str {
        match self {
            Self::Srt => "Legenda SRT",
            Self::Vtt => "Legenda WebVTT",
            Self::Ass => "Legenda ASS Estilizada",
            Self::AssKaraoke => "Legenda ASS Karaokê (Palavra a Palavra)",
            Self::Lrc => "Legenda LRC Sincronizada",
            Self::Txt => "Transcrição TXT Limpa",
            Self::TxtTimestamps => "Transcrição TXT com Timecodes",
            Self::Json => "Transcrição JSON (Palavra a Palavra)",
            Self::Fcpxml => "Timeline FCPXML (DaVinci / CapCut / FCP)",
            Self::PremiereXml => "Timeline Premiere Pro XML",
            Self::Edl => "Timeline EDL (CMX 3600)",
            Self::Otio => "Timeline OpenTimelineIO (OTIO)",
            Self::CsvMarkers => "Marcadores CSV (DaVinci / Premiere)",
        }
    }

    pub fn all_info() -> Vec<ExportFormatInfo> {
        vec![
            ExportFormatInfo {
                id: Self::Srt,
                name: "SRT (SubRip)".into(),
                extension: "srt".into(),
                category: "subtitle".into(),
                description: "Formato universal de legendas compatível com qualquer player e editor.".into(),
            },
            ExportFormatInfo {
                id: Self::Vtt,
                name: "WebVTT (.vtt)".into(),
                extension: "vtt".into(),
                category: "subtitle".into(),
                description: "Legendas para web, YouTube e players HTML5.".into(),
            },
            ExportFormatInfo {
                id: Self::Ass,
                name: "ASS Estilizada (.ass)".into(),
                extension: "ass".into(),
                category: "subtitle".into(),
                description: "Legendas com estilo tipográfico avançado (fonte, contorno e sombra).".into(),
            },
            ExportFormatInfo {
                id: Self::AssKaraoke,
                name: "ASS Karaokê (.ass)".into(),
                extension: "ass".into(),
                category: "subtitle".into(),
                description: "Destaque palavra por palavra (estilo Shorts/Reels/TikTok).".into(),
            },
            ExportFormatInfo {
                id: Self::Lrc,
                name: "LRC (.lrc)".into(),
                extension: "lrc".into(),
                category: "subtitle".into(),
                description: "Formato sincronizado para músicas, podcasts e players de áudio.".into(),
            },
            ExportFormatInfo {
                id: Self::Fcpxml,
                name: "FCPXML 1.10 (.fcpxml)".into(),
                extension: "fcpxml".into(),
                category: "timeline".into(),
                description: "Timeline pronta para DaVinci Resolve, Final Cut Pro e CapCut com marcadores sincronizados.".into(),
            },
            ExportFormatInfo {
                id: Self::PremiereXml,
                name: "Premiere Pro XML (.xml)".into(),
                extension: "xml".into(),
                category: "timeline".into(),
                description: "Sequência XMEML para importar diretamente no Adobe Premiere Pro.".into(),
            },
            ExportFormatInfo {
                id: Self::Edl,
                name: "EDL CMX 3600 (.edl)".into(),
                extension: "edl".into(),
                category: "timeline".into(),
                description: "Edit Decision List universal para DaVinci Resolve, Premiere e Avid.".into(),
            },
            ExportFormatInfo {
                id: Self::Otio,
                name: "OpenTimelineIO (.otio)".into(),
                extension: "otio".into(),
                category: "timeline".into(),
                description: "Formato aberto de timeline da Academy Software Foundation.".into(),
            },
            ExportFormatInfo {
                id: Self::CsvMarkers,
                name: "Marcadores CSV (.csv)".into(),
                extension: "csv".into(),
                category: "timeline".into(),
                description: "Lista de marcadores importável no DaVinci Resolve Timeline Markers.".into(),
            },
            ExportFormatInfo {
                id: Self::Txt,
                name: "Texto Limpo (.txt)".into(),
                extension: "txt".into(),
                category: "transcript".into(),
                description: "Texto corrido sem marcas de tempo, ideal para leitura ou resumo.".into(),
            },
            ExportFormatInfo {
                id: Self::TxtTimestamps,
                name: "Texto com Timecodes (.txt)".into(),
                extension: "txt".into(),
                category: "transcript".into(),
                description: "Linhas com marcações [HH:MM:SS] para revisão e roteiro.".into(),
            },
            ExportFormatInfo {
                id: Self::Json,
                name: "JSON Estruturado (.json)".into(),
                extension: "json".into(),
                category: "transcript".into(),
                description: "Dados completos de segmentos e palavras em milissegundos para agentes de IA.".into(),
            },
        ]
    }
}

/// Renders a canonical `TranscriptRecord` into any supported subtitle, transcript, or pro timeline format.
pub fn render_format(
    rec: &TranscriptRecord,
    source_media: Option<&MediaAsset>,
    format: ExportFormat,
    fps_override: Option<f64>,
) -> String {
    let fps = fps_override
        .or_else(|| source_media.and_then(|m| m.metadata.fps))
        .filter(|&f| f > 1.0 && f <= 240.0)
        .unwrap_or(30.0);

    match format {
        ExportFormat::Srt => render_srt(&rec.segments),
        ExportFormat::Vtt => render_vtt(&rec.segments),
        ExportFormat::Ass => render_ass(&rec.media_title, &rec.segments, false),
        ExportFormat::AssKaraoke => render_ass(&rec.media_title, &rec.segments, true),
        ExportFormat::Lrc => render_lrc(&rec.segments),
        ExportFormat::Txt => rec.full_text.clone(),
        ExportFormat::TxtTimestamps => render_txt_timestamps(&rec.segments),
        ExportFormat::Json => serde_json::to_string_pretty(rec).unwrap_or_else(|_| "{}".into()),
        ExportFormat::Fcpxml => render_fcpxml(rec, source_media, fps),
        ExportFormat::PremiereXml => render_premiere_xml(rec, source_media, fps),
        ExportFormat::Edl => render_edl(rec, fps),
        ExportFormat::Otio => render_otio(rec, source_media, fps),
        ExportFormat::CsvMarkers => render_csv_markers(rec, fps),
    }
}

/// Renders the requested format, writes it to `<workspace>/Exports/`, and registers it in the Library!
pub fn export_and_save_to_library(
    kernel: &Arc<AicutKernel>,
    rec: &TranscriptRecord,
    source_media: Option<&MediaAsset>,
    format: ExportFormat,
    fps_override: Option<f64>,
) -> AicutResult<MediaAsset> {
    std::fs::create_dir_all(&kernel.workspace.exports_dir)?;
    let content = render_format(rec, source_media, format, fps_override);

    let safe_base = sanitize_filename(&rec.media_title);
    let short_id = &Uuid::new_v4().to_string()[..6];
    let file_name = format!("{safe_base}_{short_id}.{}", format.suffix());
    let file_path = kernel.workspace.exports_dir.join(&file_name);
    std::fs::write(&file_path, content.as_bytes())?;

    let id = Uuid::new_v4().to_string();
    let now = Utc::now();
    let asset = MediaAsset {
        id,
        title: format!("{} — {}", rec.media_title, format.display_label()),
        kind: format.media_kind(),
        source_type: MediaSourceType::Generated,
        source_url: source_media.and_then(|m| m.source_url.clone()),
        parent_media_id: Some(rec.media_id.clone()),
        file_path: file_path.to_string_lossy().to_string(),
        file_name,
        extension: format.extension().to_string(),
        file_size_bytes: content.len() as u64,
        fingerprint: None,
        thumbnail_path: source_media.and_then(|m| m.thumbnail_path.clone()),
        metadata: MediaMetadata {
            duration_sec: rec.duration_sec,
            fps: fps_override.or_else(|| source_media.and_then(|m| m.metadata.fps)),
            format_name: Some(format.display_label().to_string()),
            language: Some(rec.language.clone()),
            model_id: Some(rec.model_id.clone()),
            transcript_id: Some(rec.id.clone()),
            ..Default::default()
        },
        tags: vec![
            "whisper".to_string(),
            format.extension().to_string(),
            rec.language.clone(),
            format!("video_id:{}", rec.media_id),
        ],
        notes: format!(
            "Gerado localmente por IA ({}) a partir de '{}'.",
            rec.model_id, rec.media_title
        ),
        created_at: now,
        updated_at: now,
    };

    kernel.repo.insert_media(&asset)?;
    kernel.events.publish(AicutEvent::MediaCreated(asset.clone()));
    Ok(asset)
}

// ============================================================================
// Format Renderers
// ============================================================================

fn render_srt(segments: &[TranscriptSegment]) -> String {
    let mut out = String::new();
    for (i, seg) in segments.iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            fmt_hms_ms(seg.start_ms, ','),
            fmt_hms_ms(seg.end_ms, ','),
            seg.text.trim()
        ));
    }
    out
}

fn render_vtt(segments: &[TranscriptSegment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for seg in segments {
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            fmt_hms_ms(seg.start_ms, '.'),
            fmt_hms_ms(seg.end_ms, '.'),
            seg.text.trim()
        ));
    }
    out
}

fn render_ass(title: &str, segments: &[TranscriptSegment], karaoke: bool) -> String {
    let mut out = format!(
        "[Script Info]\nTitle: {}\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\n\n\
[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n\
Style: Default,Segoe UI,58,&H00FFFFFF,&H0000D4FF,&H00121417,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,60,60,72,1\n\n\
[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
        title.replace('\n', " ")
    );

    for seg in segments {
        let start = fmt_ass_time(seg.start_ms);
        let end = fmt_ass_time(seg.end_ms);
        let text = if karaoke && !seg.words.is_empty() {
            seg.words
                .iter()
                .map(|w| {
                    let cs = ((w.end_ms.saturating_sub(w.start_ms)) / 10).max(1);
                    format!("{{\\kf{cs}}}{}", escape_ass(&w.text))
                })
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            escape_ass(seg.text.trim())
        };
        out.push_str(&format!("Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"));
    }
    out
}

fn render_lrc(segments: &[TranscriptSegment]) -> String {
    let mut out = String::new();
    for seg in segments {
        let mins = seg.start_ms / 60_000;
        let secs = (seg.start_ms % 60_000) / 1000;
        let cs = (seg.start_ms % 1000) / 10;
        out.push_str(&format!(
            "[{mins:02}:{secs:02}.{cs:02}]{}\n",
            seg.text.trim().replace('\n', " ")
        ));
    }
    out
}

fn render_txt_timestamps(segments: &[TranscriptSegment]) -> String {
    let mut out = String::new();
    for seg in segments {
        out.push_str(&format!(
            "[{} - {}] {}\n",
            fmt_hms(seg.start_ms),
            fmt_hms(seg.end_ms),
            seg.text.trim()
        ));
    }
    out
}

fn render_fcpxml(rec: &TranscriptRecord, source_media: Option<&MediaAsset>, fps: f64) -> String {
    let fps_int = fps.round().max(1.0) as u64;
    let total_ms = rec
        .duration_sec
        .map(|d| (d * 1000.0) as u64)
        .or_else(|| rec.segments.last().map(|s| s.end_ms))
        .unwrap_or(1000);
    let total_frames = ((total_ms as f64 / 1000.0) * fps_int as f64).round().max(1.0) as u64;
    let media_src = source_media
        .map(|m| file_url(&m.file_path))
        .unwrap_or_else(|| "file:///media.mp4".to_string());
    let width = source_media.and_then(|m| m.metadata.width).unwrap_or(1920);
    let height = source_media.and_then(|m| m.metadata.height).unwrap_or(1080);

    let mut markers = String::new();
    for seg in &rec.segments {
        let start_f = ((seg.start_ms as f64 / 1000.0) * fps_int as f64).round() as u64;
        let dur_f = (((seg.end_ms.saturating_sub(seg.start_ms)) as f64 / 1000.0) * fps_int as f64)
            .round()
            .max(1.0) as u64;
        markers.push_str(&format!(
            "                        <marker start=\"{start_f}/{fps_int}s\" duration=\"{dur_f}/{fps_int}s\" value=\"{}\"/>\n",
            escape_xml(seg.text.trim())
        ));
    }

    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<!DOCTYPE fcpxml>\n\
<fcpxml version=\"1.10\">\n\
    <resources>\n\
        <format id=\"r1\" name=\"FFVideoFormat{}p{fps_int}\" frameDuration=\"1/{fps_int}s\" width=\"{width}\" height=\"{height}\"/>\n\
        <asset id=\"r2\" name=\"{}\" src=\"{}\" start=\"0s\" duration=\"{total_frames}/{fps_int}s\" hasVideo=\"1\" hasAudio=\"1\"/>\n\
    </resources>\n\
    <library>\n\
        <event name=\"Aicut Transcriptions\">\n\
            <project name=\"{}\">\n\
                <sequence format=\"r1\" duration=\"{total_frames}/{fps_int}s\">\n\
                    <spine>\n\
                        <asset-clip ref=\"r2\" offset=\"0s\" name=\"{}\" duration=\"{total_frames}/{fps_int}s\">\n\
{markers}\
                        </asset-clip>\n\
                    </spine>\n\
                </sequence>\n\
            </project>\n\
        </event>\n\
    </library>\n\
</fcpxml>\n",
        height,
        escape_xml(&rec.media_title),
        escape_xml(&media_src),
        escape_xml(&rec.media_title),
        escape_xml(&rec.media_title),
    )
}

fn render_premiere_xml(rec: &TranscriptRecord, source_media: Option<&MediaAsset>, fps: f64) -> String {
    let fps_int = fps.round().max(1.0) as u64;
    let total_ms = rec
        .duration_sec
        .map(|d| (d * 1000.0) as u64)
        .or_else(|| rec.segments.last().map(|s| s.end_ms))
        .unwrap_or(1000);
    let total_frames = ((total_ms as f64 / 1000.0) * fps_int as f64).round().max(1.0) as u64;
    let media_src = source_media
        .map(|m| file_url(&m.file_path))
        .unwrap_or_else(|| "file:///media.mp4".to_string());

    let mut markers = String::new();
    for seg in &rec.segments {
        let in_f = ((seg.start_ms as f64 / 1000.0) * fps_int as f64).round() as u64;
        let out_f = ((seg.end_ms as f64 / 1000.0) * fps_int as f64).round().max(in_f as f64 + 1.0) as u64;
        markers.push_str(&format!(
            "        <marker>\n          <name>#{}</name>\n          <comment>{}</comment>\n          <in>{in_f}</in>\n          <out>{out_f}</out>\n        </marker>\n",
            seg.index + 1,
            escape_xml(seg.text.trim())
        ));
    }

    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<!DOCTYPE xmeml>\n\
<xmeml version=\"4\">\n\
  <sequence>\n\
    <name>{}</name>\n\
    <duration>{total_frames}</duration>\n\
    <rate><timebase>{fps_int}</timebase><ntsc>FALSE</ntsc></rate>\n\
    <media>\n\
      <video>\n\
        <track>\n\
          <clipitem id=\"clip-1\">\n\
            <name>{}</name>\n\
            <duration>{total_frames}</duration>\n\
            <rate><timebase>{fps_int}</timebase><ntsc>FALSE</ntsc></rate>\n\
            <start>0</start>\n\
            <end>{total_frames}</end>\n\
            <in>0</in>\n\
            <out>{total_frames}</out>\n\
            <file id=\"file-1\"><pathurl>{}</pathurl></file>\n\
          </clipitem>\n\
        </track>\n\
      </video>\n\
    </media>\n\
{markers}\
  </sequence>\n\
</xmeml>\n",
        escape_xml(&rec.media_title),
        escape_xml(&rec.media_title),
        escape_xml(&media_src),
    )
}

fn render_edl(rec: &TranscriptRecord, fps: f64) -> String {
    let mut out = format!(
        "TITLE: {}\nFCM: NON-DROP FRAME\n\n",
        rec.media_title.to_uppercase()
    );
    for (i, seg) in rec.segments.iter().enumerate() {
        let tc_in = fmt_smpte(seg.start_ms, fps);
        let tc_out = fmt_smpte(seg.end_ms.max(seg.start_ms + 40), fps);
        out.push_str(&format!(
            "{:03}  001      V     C        {tc_in} {tc_out} {tc_in} {tc_out}\n* FROM CLIP NAME: {}\n* COMMENT: {}\n\n",
            i + 1,
            rec.media_title,
            seg.text.trim().replace('\n', " ")
        ));
    }
    out
}

fn render_otio(rec: &TranscriptRecord, source_media: Option<&MediaAsset>, fps: f64) -> String {
    let media_url = source_media
        .map(|m| file_url(&m.file_path))
        .unwrap_or_default();
    let total_ms = rec
        .duration_sec
        .map(|d| (d * 1000.0) as u64)
        .or_else(|| rec.segments.last().map(|s| s.end_ms))
        .unwrap_or(1000);

    let markers: Vec<serde_json::Value> = rec
        .segments
        .iter()
        .map(|seg| {
            let start_f = (seg.start_ms as f64 / 1000.0) * fps;
            let dur_f = (seg.end_ms.saturating_sub(seg.start_ms) as f64 / 1000.0) * fps;
            serde_json::json!({
                "OTIO_SCHEMA": "Marker.2",
                "name": format!("Segment {}", seg.index + 1),
                "color": "CYAN",
                "marked_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": { "OTIO_SCHEMA": "RationalTime.1", "rate": fps, "value": start_f },
                    "duration": { "OTIO_SCHEMA": "RationalTime.1", "rate": fps, "value": dur_f.max(1.0) }
                },
                "metadata": {
                    "aicut": {
                        "text": seg.text.trim(),
                        "words": seg.words
                    }
                }
            })
        })
        .collect();

    let otio = serde_json::json!({
        "OTIO_SCHEMA": "Timeline.1",
        "name": rec.media_title,
        "metadata": {
            "aicut": {
                "transcript_id": rec.id,
                "model_id": rec.model_id,
                "language": rec.language
            }
        },
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "name": "tracks",
            "children": [
                {
                    "OTIO_SCHEMA": "Track.1",
                    "name": "Video 1",
                    "kind": "Video",
                    "children": [
                        {
                            "OTIO_SCHEMA": "Clip.2",
                            "name": rec.media_title,
                            "media_reference": {
                                "OTIO_SCHEMA": "ExternalReference.1",
                                "target_url": media_url
                            },
                            "source_range": {
                                "OTIO_SCHEMA": "TimeRange.1",
                                "start_time": { "OTIO_SCHEMA": "RationalTime.1", "rate": fps, "value": 0.0 },
                                "duration": { "OTIO_SCHEMA": "RationalTime.1", "rate": fps, "value": (total_ms as f64 / 1000.0) * fps }
                            },
                            "markers": markers
                        }
                    ]
                }
            ]
        }
    });
    serde_json::to_string_pretty(&otio).unwrap_or_else(|_| "{}".into())
}

fn render_csv_markers(rec: &TranscriptRecord, fps: f64) -> String {
    let mut out = String::from("#,Record In,Record Out,Color,Name,Notes\n");
    for (i, seg) in rec.segments.iter().enumerate() {
        let tc_in = fmt_smpte(seg.start_ms, fps);
        let tc_out = fmt_smpte(seg.end_ms.max(seg.start_ms + 40), fps);
        let text = seg.text.trim().replace('"', "\"\"").replace('\n', " ");
        out.push_str(&format!(
            "{},{tc_in},{tc_out},Cyan,\"Segmento {}\",\"{text}\"\n",
            i + 1,
            i + 1
        ));
    }
    out
}

// ============================================================================
// Formatting helpers
// ============================================================================

fn fmt_hms_ms(ms: u64, sep: char) -> String {
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1000;
    let rem = ms % 1000;
    format!("{h:02}:{m:02}:{s:02}{sep}{rem:03}")
}

fn fmt_hms(ms: u64) -> String {
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1000;
    format!("{h:02}:{m:02}:{s:02}")
}

fn fmt_ass_time(ms: u64) -> String {
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1000;
    let cs = (ms % 1000) / 10;
    format!("{h}:{m:02}:{s:02}.{cs:02}")
}

fn fmt_smpte(ms: u64, fps: f64) -> String {
    let fps_i = fps.round().max(1.0) as u64;
    let total_frames = ((ms as f64 / 1000.0) * fps_i as f64).round() as u64;
    let frames = total_frames % fps_i;
    let total_secs = total_frames / fps_i;
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = total_secs / 3600;
    format!("{hours:02}:{mins:02}:{secs:02}:{frames:02}")
}

fn escape_ass(s: &str) -> String {
    s.replace('\n', "\\N").replace('{', "(").replace('}', ")")
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn file_url(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    if normalized.starts_with('/') {
        format!("file://{normalized}")
    } else {
        format!("file:///{normalized}")
    }
}

fn sanitize_filename(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim().replace(' ', "_");
    if trimmed.is_empty() {
        "aicut_export".to_string()
    } else {
        trimmed.chars().take(60).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aicut_core::TranscriptWord;

    fn sample_transcript() -> TranscriptRecord {
        TranscriptRecord {
            id: "tr-1".into(),
            media_id: "media-1".into(),
            media_title: "Demo Podcast".into(),
            model_id: "large-v3-turbo-q5_0".into(),
            language: "pt".into(),
            duration_sec: Some(5.0),
            full_text: "Olá mundo do Aicut.".into(),
            segments: vec![TranscriptSegment {
                index: 0,
                start_ms: 500,
                end_ms: 2800,
                text: "Olá mundo do Aicut.".into(),
                words: vec![
                    TranscriptWord { start_ms: 500, end_ms: 1000, text: "Olá".into(), probability: Some(0.99) },
                    TranscriptWord { start_ms: 1000, end_ms: 1600, text: "mundo".into(), probability: Some(0.98) },
                    TranscriptWord { start_ms: 1600, end_ms: 2800, text: "do Aicut.".into(), probability: Some(0.97) },
                ],
            }],
            created_at: Utc::now(),
        }
    }

    #[test]
    fn renders_all_subtitle_and_timeline_formats_cleanly() {
        let rec = sample_transcript();
        let srt = render_format(&rec, None, ExportFormat::Srt, Some(30.0));
        assert!(srt.contains("00:00:00,500 --> 00:00:02,800"));
        assert!(srt.contains("Olá mundo do Aicut."));

        let vtt = render_format(&rec, None, ExportFormat::Vtt, None);
        assert!(vtt.starts_with("WEBVTT"));

        let ass_k = render_format(&rec, None, ExportFormat::AssKaraoke, None);
        assert!(ass_k.contains("\\kf"));

        let fcpxml = render_format(&rec, None, ExportFormat::Fcpxml, Some(30.0));
        assert!(fcpxml.contains("<fcpxml version=\"1.10\">"));
        assert!(fcpxml.contains("Olá mundo do Aicut."));

        let pxml = render_format(&rec, None, ExportFormat::PremiereXml, Some(30.0));
        assert!(pxml.contains("<xmeml version=\"4\">"));

        let edl = render_format(&rec, None, ExportFormat::Edl, Some(30.0));
        assert!(edl.contains("FCM: NON-DROP FRAME"));

        let otio = render_format(&rec, None, ExportFormat::Otio, Some(30.0));
        assert!(otio.contains("\"OTIO_SCHEMA\": \"Timeline.1\""));

        let csv = render_format(&rec, None, ExportFormat::CsvMarkers, Some(30.0));
        assert!(csv.contains("Record In,Record Out"));
    }
}
