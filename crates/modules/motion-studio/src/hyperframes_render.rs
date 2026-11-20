use aicut_core::{configure_tokio_command, AicutError, AicutResult};
use base64::Engine as _;
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use tokio::process::{Child, Command};
use tokio_tungstenite::tungstenite::Message;

use crate::schema::AicutComposition;

/// Wraps the AI's bespoke HyperFrames HTML + CSS + SVG + JS motion fragment into a complete,
/// 100% offline, deterministic 1080x1920 transparent RGBA document.
///
/// Works identically in BOTH:
/// 1. The live 60fps React preview `<iframe>` (driven via `postMessage({ type: 'aicut_seek', time })`)
/// 2. Headless Chromium CDP transparent PNG frame capture (driven via `window.__aicutSeek(t)`)
pub fn build_live_hyperframes_document(comp: &AicutComposition) -> String {
    let custom_fragment = comp
        .motion_html
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());

    let body_fragment = if let Some(frag) = custom_fragment {
        frag.to_string()
    } else {
        // Generate crisp default HyperFrames elements from scene headlines/badges/callouts
        let mut auto_html = String::new();
        for sc in &comp.scenes {
            if let Some(ref b) = sc.overlay.badge.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
                auto_html.push_str(&format!(
                    "<div class=\"hf-pill\" data-start=\"{:.2}\" data-end=\"{:.2}\" data-anim=\"slide-right\" style=\"top:54px;left:42px;border-color:{}\">{}</div>\n",
                    sc.start_sec,
                    sc.end_sec,
                    sc.overlay.accent_hex,
                    escape_html(b)
                ));
            }
            if let Some(ref hl) = sc.overlay.headline.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
                auto_html.push_str(&format!(
                    "<div class=\"hf-headline\" data-start=\"{:.2}\" data-end=\"{:.2}\" data-anim=\"pop\" style=\"border-color:{}\">{}</div>\n",
                    sc.start_sec,
                    sc.end_sec,
                    sc.overlay.accent_hex,
                    escape_html(hl)
                ));
            }
            if let Some(ref co) = sc.overlay.callout_text.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
                auto_html.push_str(&format!(
                    "<div class=\"hf-callout\" data-start=\"{:.2}\" data-end=\"{:.2}\" data-anim=\"pop\" style=\"left:{:.1}%;top:{:.1}%;border-color:{}\">{}</div>\n",
                    sc.start_sec,
                    sc.end_sec,
                    (sc.content_focus.focus_x * 100.0).clamp(18.0, 82.0),
                    (sc.content_focus.focus_y * 100.0).clamp(24.0, 76.0),
                    sc.overlay.accent_hex,
                    escape_html(co)
                ));
            }
        }
        auto_html
    };

    let w = comp.width.max(360);
    let h = comp.height.max(640);
    let dur = comp.duration_sec.max(1.0);

    format!(
        r##"<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="color-scheme" content="dark" />
  <style>
    :root {{ color-scheme: dark; }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{
      width: {w}px;
      height: {h}px;
      overflow: hidden;
      background: transparent !important;
      background-color: rgba(0, 0, 0, 0) !important;
      font-family: 'Segoe UI', Inter, system-ui, -apple-system, sans-serif;
      color: #ffffff;
      -webkit-font-smoothing: antialiased;
    }}
    #aicut-stage {{
      position: relative;
      width: {w}px;
      height: {h}px;
      overflow: hidden;
      background: transparent;
    }}
    .hf-headline {{
      position: absolute;
      top: 108px;
      left: 50%;
      max-width: 92%;
      padding: 18px 34px;
      border-radius: 18px;
      border: 3px solid #EAB308;
      background: rgba(10, 12, 18, 0.86);
      backdrop-filter: blur(12px);
      box-shadow: 0 18px 42px rgba(0, 0, 0, 0.55);
      font-size: 54px;
      font-weight: 900;
      text-align: center;
      text-transform: uppercase;
      letter-spacing: 0.02em;
      line-height: 1.15;
      --tx: -50%;
    }}
    .hf-pill {{
      position: absolute;
      padding: 10px 22px;
      border-radius: 999px;
      border: 2.5px solid #EAB308;
      background: rgba(10, 12, 18, 0.85);
      backdrop-filter: blur(10px);
      font-size: 30px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45);
    }}
    .hf-card {{
      position: absolute;
      padding: 22px 30px;
      border-radius: 22px;
      border: 2.5px solid rgba(255, 255, 255, 0.22);
      background: rgba(12, 14, 22, 0.88);
      backdrop-filter: blur(14px);
      box-shadow: 0 22px 50px rgba(0, 0, 0, 0.55);
    }}
    .hf-callout {{
      position: absolute;
      padding: 14px 26px;
      border-radius: 16px;
      border: 3px solid #EAB308;
      background: rgba(10, 12, 18, 0.90);
      font-size: 38px;
      font-weight: 800;
      white-space: nowrap;
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.55);
      --tx: -50%;
      --ty: -50%;
    }}
    .hf-flash {{
      position: absolute;
      inset: 0;
      background: #ffffff;
      pointer-events: none;
    }}
  </style>
</head>
<body data-width="{w}" data-height="{h}" data-duration="{dur:.2}">
  <div id="aicut-stage">
{body_fragment}
  </div>
  <script>
    (function () {{
      function clamp01(v) {{ return v < 0 ? 0 : (v > 1 ? 1 : v); }}
      function easeOutCubic(x) {{ return 1 - Math.pow(1 - x, 3); }}
      function easeOutBack(x) {{
        const c1 = 1.70158, c3 = c1 + 1;
        return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
      }}

      // Prepare SVG paths for deterministic stroke drawing
      document.querySelectorAll('[data-anim="draw"] path, [data-anim="draw"] circle, [data-anim="draw"] rect, [data-anim="draw"] line, path[data-anim="draw"]').forEach((el) => {{
        try {{
          const len = el.getTotalLength ? el.getTotalLength() : 600;
          el.dataset.pathLen = String(len);
          el.style.strokeDasharray = String(len);
        }} catch (_) {{}}
      }});

      window.__aicutSeek = function (t) {{
        const time = Number(t) || 0;
        const nodes = document.querySelectorAll('[data-start]');
        nodes.forEach((el) => {{
          const s = parseFloat(el.dataset.start || '0');
          const e = parseFloat(el.dataset.end || '{dur:.2}');
          const inDur = Math.max(0.05, parseFloat(el.dataset.in || '0.32'));
          const outDur = Math.max(0.05, parseFloat(el.dataset.out || '0.22'));
          const anim = (el.dataset.anim || 'pop').toLowerCase();

          if (time < s || time > e) {{
            el.style.opacity = '0';
            el.style.visibility = 'hidden';
            return;
          }}
          el.style.visibility = 'visible';

          const pIn = clamp01((time - s) / inDur);
          const pOut = clamp01((e - time) / outDur);
          const env = Math.min(easeOutCubic(pIn), easeOutCubic(pOut));

          const baseTx = getComputedStyle(el).getPropertyValue('--tx').trim() || '0px';
          const baseTy = getComputedStyle(el).getPropertyValue('--ty').trim() || '0px';

          // Deterministic counter support (e.g. R$ 0 -> R$ 600)
          if (el.dataset.countTo !== undefined) {{
            const from = parseFloat(el.dataset.countFrom || '0');
            const to = parseFloat(el.dataset.countTo || '100');
            const countDur = Math.max(0.2, parseFloat(el.dataset.countDur || String(Math.min(1.8, (e - s) * 0.75))));
            const cp = easeOutCubic(clamp01((time - s) / countDur));
            const val = Math.round(from + (to - from) * cp);
            const prefix = el.dataset.prefix || '';
            const suffix = el.dataset.suffix || '';
            el.textContent = prefix + val.toLocaleString('pt-BR') + suffix;
          }}

          if (anim === 'flash') {{
            const peak = Math.sin(clamp01((time - s) / Math.max(0.08, e - s)) * Math.PI);
            el.style.opacity = String((peak * 0.72).toFixed(3));
            return;
          }}

          if (anim === 'draw') {{
            el.style.opacity = String(env.toFixed(3));
            const drawP = easeOutCubic(clamp01((time - s) / Math.max(0.15, inDur * 1.6)));
            const targets = el.matches('path,circle,rect,line') ? [el] : el.querySelectorAll('path,circle,rect,line');
            targets.forEach((p) => {{
              const len = parseFloat(p.dataset.pathLen || '600');
              p.style.strokeDashoffset = String((len * (1 - drawP)).toFixed(1));
            }});
            return;
          }}

          let dx = 0, dy = 0, sc = 1;
          if (anim === 'pop') {{
            sc = pIn < 1 ? (0.78 + 0.22 * easeOutBack(pIn)) : (0.94 + 0.06 * easeOutCubic(pOut));
            dy = (1 - easeOutCubic(pIn)) * 18;
          }} else if (anim === 'slide-up') {{
            dy = (1 - easeOutCubic(pIn)) * 46 - (1 - easeOutCubic(pOut)) * 20;
          }} else if (anim === 'slide-down') {{
            dy = -(1 - easeOutCubic(pIn)) * 46;
          }} else if (anim === 'slide-left') {{
            dx = (1 - easeOutCubic(pIn)) * 64;
          }} else if (anim === 'slide-right') {{
            dx = -(1 - easeOutCubic(pIn)) * 64;
          }} else if (anim === 'pulse') {{
            sc = 1 + Math.sin((time - s) * 7.5) * 0.045;
          }} else if (anim === 'scale') {{
            sc = 0.85 + 0.15 * easeOutCubic(pIn);
          }}

          el.style.opacity = String(env.toFixed(3));
          el.style.transform = 'translate(' + baseTx + ', ' + baseTy + ') translate3d(' + dx.toFixed(1) + 'px, ' + dy.toFixed(1) + 'px, 0) scale(' + sc.toFixed(4) + ')';
        }});

        if (typeof window.__aicutCustomSeek === 'function') {{
          try {{ window.__aicutCustomSeek(time); }} catch (_) {{}}
        }}
      }};

      window.addEventListener('message', (ev) => {{
        if (ev.data && ev.data.type === 'aicut_seek') {{
          window.__aicutSeek(ev.data.time);
        }}
      }});

      window.__aicutSeek(0);
    }})();
  </script>
</body>
</html>
"##
    )
}

/// Finds a built-in Chromium browser on Windows 11 (Microsoft Edge is pre-installed on every
/// Windows 11 machine!), macOS, or Linux — zero extra downloads needed!
pub fn find_system_chromium() -> Option<PathBuf> {
    let candidates: Vec<PathBuf> = vec![
        PathBuf::from(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        PathBuf::from(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        PathBuf::from(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        PathBuf::from(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        PathBuf::from("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        PathBuf::from("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    ];
    for p in candidates {
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(local_app) = std::env::var("LOCALAPPDATA") {
        let p = PathBuf::from(&local_app).join(r"Google\Chrome\Application\chrome.exe");
        if p.is_file() {
            return Some(p);
        }
        let p2 = PathBuf::from(&local_app).join(r"Microsoft\Edge\Application\msedge.exe");
        if p2.is_file() {
            return Some(p2);
        }
    }
    for bin in ["msedge", "google-chrome", "chromium", "chromium-browser"] {
        if let Ok(found) = which::which(bin) {
            return Some(found);
        }
    }
    None
}

struct ChildKillGuard(Option<Child>);
impl Drop for ChildKillGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            let _ = c.start_kill();
        }
    }
}

/// Captures deterministic 1080x1920 transparent RGBA PNG frames (`hf_%05d.png`) from the
/// HyperFrames HTML/CSS/SVG motion document using built-in headless Edge/Chrome via CDP.
/// Returns `Ok(None)` cleanly if no Chromium browser is available (caller falls back to ASS).
pub async fn capture_hyperframes_rgba_sequence(
    comp: &AicutComposition,
    fps: u32,
    out_dir: &Path,
) -> AicutResult<Option<usize>> {
    let Some(chromium_bin) = find_system_chromium() else {
        return Ok(None);
    };

    std::fs::create_dir_all(out_dir)?;
    let html_doc = build_live_hyperframes_document(comp);
    let html_path = out_dir.join("motion.html");
    std::fs::write(&html_path, html_doc)?;

    let profile_dir = out_dir.join("cdp-profile");
    std::fs::create_dir_all(&profile_dir)?;

    let port = pick_free_port()?;
    let file_url = path_to_file_url(&html_path);

    let mut cmd = Command::new(&chromium_bin);
    configure_tokio_command(&mut cmd);
    cmd.args([
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--hide-scrollbars",
        "--mute-audio",
        "--disable-extensions",
        "--allow-file-access-from-files",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--remote-allow-origins=*",
        "--default-background-color=00000000",
        &format!("--window-size={},{}", comp.width, comp.height),
        &format!("--remote-debugging-port={port}"),
        &format!("--user-data-dir={}", profile_dir.to_string_lossy()),
        &file_url,
    ])
    .stdout(std::process::Stdio::null())
    .stderr(std::process::Stdio::null());

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(_) => return Ok(None),
    };
    let _guard = ChildKillGuard(Some(child));

    // Wait for CDP endpoint to come up
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .map_err(|e| AicutError::internal(format!("HTTP client init failed: {e}")))?;
    let list_url = format!("http://127.0.0.1:{port}/json/list");
    let mut ws_url: Option<String> = None;
    for _ in 0..30 {
        if let Ok(resp) = http.get(&list_url).send().await {
            if let Ok(targets) = resp.json::<Vec<Value>>().await {
                if let Some(u) = targets
                    .iter()
                    .find(|t| t.get("type").and_then(|v| v.as_str()) == Some("page"))
                    .and_then(|t| t.get("webSocketDebuggerUrl"))
                    .and_then(|v| v.as_str())
                {
                    ws_url = Some(u.to_string());
                    break;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    let Some(ws_url) = ws_url else {
        return Ok(None);
    };

    let (mut ws, _) = match tokio_tungstenite::connect_async(ws_url.as_str()).await {
        Ok(pair) => pair,
        Err(_) => return Ok(None),
    };

    let mut msg_id = 1u64;
    let send_cdp = |id: &mut u64, method: &str, params: Value| -> (u64, String) {
        let cur = *id;
        *id += 1;
        let payload = json!({ "id": cur, "method": method, "params": params }).to_string();
        (cur, payload)
    };

    // Configure exact 1080x1920 viewport + transparent RGBA background (alpha = 0)
    for (method, params) in [
        (
            "Emulation.setDeviceMetricsOverride",
            json!({
                "width": comp.width,
                "height": comp.height,
                "deviceScaleFactor": 1,
                "mobile": false
            }),
        ),
        (
            "Emulation.setDefaultBackgroundColorOverride",
            json!({ "color": { "r": 0, "g": 0, "b": 0, "a": 0 } }),
        ),
    ] {
        let (id, payload) = send_cdp(&mut msg_id, method, params);
        let _ = cdp_roundtrip(&mut ws, id, payload).await?;
    }

    // Wait until the HyperFrames HTML document finished loading & registered window.__aicutSeek
    for _ in 0..25 {
        let (rid, ready_payload) = send_cdp(
            &mut msg_id,
            "Runtime.evaluate",
            json!({
                "expression": "document.readyState === 'complete' && typeof window.__aicutSeek === 'function'",
                "returnByValue": true
            }),
        );
        if let Ok(r) = cdp_roundtrip(&mut ws, rid, ready_payload).await {
            if r.get("result")
                .and_then(|x| x.get("result"))
                .and_then(|x| x.get("value"))
                .and_then(|v| v.as_bool())
                == Some(true)
            {
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
    }

    let capture_fps = fps.clamp(10, 30);
    let total_frames = ((comp.duration_sec * capture_fps as f64).ceil() as usize).clamp(1, 5400);

    for i in 0..total_frames {
        let t = i as f64 / capture_fps as f64;
        let (id1, eval_payload) = send_cdp(
            &mut msg_id,
            "Runtime.evaluate",
            json!({
                "expression": format!("window.__aicutSeek({t:.4}); true"),
                "awaitPromise": false
            }),
        );
        let _ = cdp_roundtrip(&mut ws, id1, eval_payload).await?;

        let (id2, shot_payload) = send_cdp(
            &mut msg_id,
            "Page.captureScreenshot",
            json!({ "format": "png", "fromSurface": true }),
        );
        let res = cdp_roundtrip(&mut ws, id2, shot_payload).await?;
        let b64 = res
            .get("result")
            .and_then(|r| r.get("data"))
            .and_then(|d| d.as_str())
            .ok_or_else(|| AicutError::internal("CDP screenshot missing base64 data"))?;
        let png_bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| AicutError::internal(format!("Failed decoding CDP PNG: {e}")))?;
        let frame_path = out_dir.join(format!("hf_{i:05}.png"));
        std::fs::write(&frame_path, png_bytes)?;
    }

    Ok(Some(total_frames))
}

async fn cdp_roundtrip<S>(
    ws: &mut S,
    expected_id: u64,
    payload: String,
) -> AicutResult<Value>
where
    S: Sink<Message> + Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
    S::Error: std::fmt::Display,
{
    ws.send(Message::Text(payload.into()))
        .await
        .map_err(|e| AicutError::internal(format!("CDP send failed: {e}")))?;

    while let Some(msg) = ws.next().await {
        let msg = msg.map_err(|e| AicutError::internal(format!("CDP recv failed: {e}")))?;
        if let Message::Text(txt) = msg {
            if let Ok(val) = serde_json::from_str::<Value>(txt.as_str()) {
                if val.get("id").and_then(|v| v.as_u64()) == Some(expected_id) {
                    return Ok(val);
                }
            }
        }
    }
    Err(AicutError::internal("CDP WebSocket closed unexpectedly"))
}

fn pick_free_port() -> AicutResult<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|e| AicutError::internal(format!("Failed binding ephemeral port: {e}")))?;
    let port = listener
        .local_addr()
        .map_err(|e| AicutError::internal(format!("Failed reading port: {e}")))?
        .port();
    Ok(port)
}

fn path_to_file_url(path: &Path) -> String {
    let s = path.to_string_lossy().replace('\\', "/");
    if s.starts_with('/') {
        format!("file://{s}")
    } else {
        format!("file:///{s}")
    }
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ContentFocusConfig, MotionScene, PipConfig, SceneLayoutPreset, SceneOverlay, SceneTransition};
    use aicut_module_smart_clips::framing::NormalizedRect;
    use chrono::Utc;

    #[test]
    fn builds_offline_transparent_hyperframes_doc_with_custom_ai_motion_html() {
        let comp = AicutComposition {
            id: "c1".into(),
            title: "Preço GTA VI".into(),
            media_id: "m1".into(),
            clip_start_sec: 0.0,
            clip_end_sec: 15.0,
            duration_sec: 15.0,
            width: 1080,
            height: 1920,
            fps: 30,
            webcam_rect: NormalizedRect { x: 0.7, y: 0.65, w: 0.28, h: 0.33 },
            content_rect: None,
            burn_subtitles: true,
            template_path: None,
            engine_preference: "hybrid_nvenc".into(),
            motion_html: Some(
                r#"<div class="hf-card" data-start="2.0" data-end="6.0" data-anim="pop" style="top:220px;left:120px;"><span data-start="2.0" data-end="6.0" data-count-from="0" data-count-to="600" data-prefix="R$ "></span></div>"#.into()
            ),
            creative_direction: "Destaque animado do valor".into(),
            scenes: vec![MotionScene {
                id: "s1".into(),
                title: "Principal".into(),
                start_sec: 0.0,
                end_sec: 15.0,
                layout: SceneLayoutPreset::SplitBranded,
                transition_in: SceneTransition::Cut,
                pip: PipConfig::default(),
                content_focus: ContentFocusConfig::default(),
                overlay: SceneOverlay::default(),
                effects: Default::default(),
                rationale: String::new(),
                custom_code: None,
            }],
            chat_history: vec![],
            rendered_media_id: None,
            updated_at: Utc::now().to_rfc3339(),
        };

        let doc = build_live_hyperframes_document(&comp);
        assert!(doc.contains("window.__aicutSeek"));
        assert!(doc.contains("background: transparent !important;"));
        assert!(doc.contains("data-count-to=\"600\""));
        assert!(doc.contains("aicut_seek"));
    }
}
