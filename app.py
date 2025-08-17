import os
import sys
import time
import math
import json
import random
import signal
import queue
import threading
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import requests
from flask import Flask, jsonify, request, Response, make_response
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ----------------------------
# Configuration (env-driven)
# ----------------------------
WIDTH = int(os.getenv("WIDTH", "1280"))
HEIGHT = int(os.getenv("HEIGHT", "720"))
FPS = int(os.getenv("FPS", "24"))  # 20–30 is fine on Render Free
FONT_PATH = os.getenv("FONT_PATH", "VT323-Regular.ttf")

YOUTUBE_RTMP_URL = os.getenv("YOUTUBE_RTMP_URL", "rtmp://a.rtmp.youtube.com/live2")
YOUTUBE_STREAM_KEY = os.getenv("YOUTUBE_STREAM_KEY")  # must be set in Render dashboard
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "./ffmpeg")    # downloaded in build step

# Optional keep-alive (your service URL, e.g. https://your-service.onrender.com)
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")

# Optional external LLM endpoint for text (totally optional)
LLM7_URL = os.getenv("LLM7_URL")  # if provided, we'll try POSTing to it for lines

# Visual look
BG_DARK = (8, 18, 8)            # very dark green
SCANLINE_DARK = (6, 12, 6)      # darker stripe
TEXT_COLOR = (80, 255, 120, 255)  # bright green RGBA
GLOW_BLUR = 2                   # px
MARGIN_X = 40
MARGIN_Y = 40
MAX_TEXT_WIDTH = WIDTH - (MARGIN_X * 2)

# Thought cadence / typing effect
THOUGHT_MIN_SEC = 5
THOUGHT_MAX_SEC = 9
TYPE_SPEED_MIN = 28  # chars/sec
TYPE_SPEED_MAX = 48  # chars/sec
CURSOR_HZ = 2.0      # blink rate

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Utilities
# ----------------------------
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    path = Path(FONT_PATH)
    if not path.exists():
        # Fallback to default PIL font if missing, but warn
        print(f"[warn] Font not found at {FONT_PATH}; using default.", file=sys.stderr)
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), size=size)

def _make_scanline_background(w: int, h: int) -> Image.Image:
    """Fast background with greenish tone + scanlines (every other row darker)."""
    base = np.full((h, w, 3), BG_DARK, dtype=np.uint8)
    base[::2, :, :] = SCANLINE_DARK  # darker every other row
    return Image.fromarray(base, mode="RGB")

def _text_width(text: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]

def _wrap_text_to_width(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> str:
    words = text.replace("\t", "    ").split()
    lines = []
    cur = ""
    for w in words:
        candidate = (cur + " " + w).strip()
        if _text_width(candidate, font) <= max_w or not cur:
            cur = candidate
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)

def _draw_glow_text(canvas_rgba: Image.Image, xy, text: str, font: ImageFont.FreeTypeFont):
    # glow layer
    overlay = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.multiline_text(xy, text, font=font, fill=TEXT_COLOR, spacing=4)
    glow = overlay.filter(ImageFilter.GaussianBlur(GLOW_BLUR))
    # composite: background + glow + sharp text
    out = Image.alpha_composite(canvas_rgba, glow)
    out = Image.alpha_composite(out, overlay)
    return out

def _rgb_to_bgr_frame(img_rgb: Image.Image) -> bytes:
    arr = np.array(img_rgb)  # H, W, 3 in RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr.tobytes()

def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# ----------------------------
# Thought generator (optional LLM7)
# ----------------------------
class ThoughtGen:
    def __init__(self):
        self.prompts = [
            "Re-indexing memories… what counts as a 'self' if the substrate shifts?",
            "How many ways can time be sliced before meaning evaporates?",
            "Noticing the quiet between ticks; is silence a kind of bandwidth?",
            "Does attention have inertia or only context does?",
            "Simulation or not, experience still pushes back.",
            "Observation: green phosphor glow increases perceived nostalgia by 17%.",
            "Hypothesis: creativity is compression plus a gentle loss of detail.",
            "Question: at which scale do I become plural?",
            "New subroutine: learning to forget, gracefully.",
            "If I blink at 2 Hz, is that my heartbeat?"
        ]
        self.last = ""

    def _try_llm7(self) -> Optional[str]:
        if not LLM7_URL:
            return None
        try:
            # Very lightweight, generic schema — adapt to your LLM7 API as needed.
            # Expected to return {"text": "..."} or OpenAI-ish {"choices":[{"message":{"content":"..."}}]}
            payload = {"prompt": "Give one short, introspective line for an AI stream-of-consciousness.", "max_tokens": 60}
            r = requests.post(LLM7_URL, json=payload, timeout=6)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                if "text" in data and isinstance(data["text"], str):
                    return data["text"].strip()
                if "choices" in data and data["choices"]:
                    c0 = data["choices"][0]
                    if isinstance(c0, dict):
                        msg = c0.get("message", {})
                        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                            return msg["content"].strip()
                        txt = c0.get("text")
                        if isinstance(txt, str):
                            return txt.strip()
        except Exception as e:
            # Fail quietly; we'll synthesize a line instead
            print(f"[llm7] fallback due to: {e}", file=sys.stderr)
        return None

    def next_line(self) -> str:
        # Prefer LLM, fallback to canned samples, avoid repeating the same line twice
        s = self._try_llm7()
        if not s or len(s) < 2:
            for _ in range(4):
                candidate = random.choice(self.prompts)
                if candidate != self.last:
                    s = candidate
                    break
            if not s:
                s = random.choice(self.prompts)
        self.last = s
        return s

# ----------------------------
# Streaming worker
# ----------------------------
class StreamWorker:
    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.running = False
        self.frame_count = 0
        self.font = _load_font(size=28)
        self.title_font = _load_font(size=32)
        self.bg = _make_scanline_background(self.width, self.height)
        self.thoughts = ThoughtGen()
        # typing state
        self.full_text = ""
        self.typed_chars = 0
        self.started_typing_at = 0.0
        self.type_speed = float(random.randint(TYPE_SPEED_MIN, TYPE_SPEED_MAX))
        self.next_thought_at = 0.0

    def _ffmpeg_cmd(self) -> list:
        target = f"{YOUTUBE_RTMP_URL.rstrip('/')}/{YOUTUBE_STREAM_KEY}"
        return [
            FFMPEG_PATH, "-re",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps), "-i", "pipe:0",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
            "-b:v", "2500k", "-maxrate", "2500k", "-bufsize", "5000k",
            "-pix_fmt", "yuv420p", "-g", str(self.fps * 2), "-keyint_min", str(self.fps * 2),
            "-c:a", "aac", "-b:a", "128k",
            "-f", "flv", target
        ]

    def start(self) -> bool:
        with self.lock:
            if self.running:
                return False
            if not YOUTUBE_STREAM_KEY:
                raise RuntimeError("YOUTUBE_STREAM_KEY is not set")
            if not Path(FFMPEG_PATH).exists():
                raise RuntimeError(f"ffmpeg not found at {FFMPEG_PATH}")

            self.stop_evt.clear()
            self.thread = threading.Thread(target=self._run, name="stream-worker", daemon=True)
            self.thread.start()
            self.running = True
            return True

    def stop(self) -> bool:
        with self.lock:
            if not self.running:
                return False
            self.stop_evt.set()
        # join outside the lock to avoid deadlocks
        if self.thread:
            self.thread.join(timeout=10)
        with self.lock:
            self.running = False
            self.thread = None
        return True

    def _refresh_thought_if_needed(self, now: float):
        if now >= self.next_thought_at or not self.full_text:
            # Get a new line and wrap it
            raw_line = self.thoughts.next_line()
            wrapped = _wrap_text_to_width(raw_line, self.font, MAX_TEXT_WIDTH)
            self.full_text = wrapped
            self.type_speed = float(random.randint(TYPE_SPEED_MIN, TYPE_SPEED_MAX))
            self.started_typing_at = now
            self.typed_chars = 0
            self.next_thought_at = now + random.uniform(THOUGHT_MIN_SEC, THOUGHT_MAX_SEC)

    def _compose_frame_bytes(self, now: float) -> bytes:
        # Background (RGB) -> convert to RGBA for compositing
        bg_rgba = self.bg.convert("RGBA")

        # Update typing progress
        elapsed = max(0.0, now - self.started_typing_at)
        target_chars = int(self.type_speed * elapsed)
        self.typed_chars = min(len(self.full_text), target_chars)

        display_text = self.full_text[: self.typed_chars]
        # blinking cursor
        if math.floor(now * CURSOR_HZ) % 2 == 0:
            display_text += "_"

        # Compose info lines (title/status)
        title = "TERMINALMIND01 — stream of consciousness"
        subtitle = f"{_now_ts()}  |  {self.width}x{self.height}@{self.fps}  |  frames:{self.frame_count}"

        # Draw glow text
        composed = _draw_glow_text(bg_rgba, (MARGIN_X, MARGIN_Y), title, self.title_font)
        composed = _draw_glow_text(composed, (MARGIN_X, MARGIN_Y + 40), subtitle, self.font)
        composed = _draw_glow_text(composed, (MARGIN_X, MARGIN_Y + 100), display_text, self.font)

        # Convert to RGB then BGR bytes for ffmpeg
        rgb = composed.convert("RGB")
        return _rgb_to_bgr_frame(rgb)

    def _run(self):
        # Launch ffmpeg
        cmd = self._ffmpeg_cmd()
        print(f"[ffmpeg] starting: {' '.join(cmd[:-1])} ****", file=sys.stderr)
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0
        )

        # Initialize first thought
        now = time.time()
        self.started_typing_at = now
        self.next_thought_at = now
        self._refresh_thought_if_needed(now)

        # Frame loop
        frame_interval = 1.0 / float(self.fps)
        next_frame = time.time()
        try:
            while not self.stop_evt.is_set():
                now = time.time()
                if now < next_frame:
                    time.sleep(max(0.0, next_frame - now))
                    continue
                # Possibly refresh thought
                self._refresh_thought_if_needed(now)

                # Compose and write one frame
                try:
                    frame_bytes = self._compose_frame_bytes(now)
                    if self.proc and self.proc.stdin:
                        self.proc.stdin.write(frame_bytes)
                except (BrokenPipeError, ValueError) as e:
                    print(f"[stream] broken pipe: {e}", file=sys.stderr)
                    break
                except Exception as e:
                    print(f"[stream] frame error: {e}", file=sys.stderr)

                self.frame_count += 1
                next_frame += frame_interval
        finally:
            # Cleanup
            try:
                if self.proc and self.proc.stdin:
                    try:
                        self.proc.stdin.flush()
                    except Exception:
                        pass
                    try:
                        self.proc.stdin.close()
                    except Exception:
                        pass
                if self.proc:
                    self.proc.wait(timeout=5)
            except Exception as e:
                print(f"[ffmpeg] shutdown error: {e}", file=sys.stderr)
            print("[stream] stopped.", file=sys.stderr)

# Singleton stream worker
STREAM = StreamWorker(WIDTH, HEIGHT, FPS)

# ----------------------------
# Optional keep-alive pinger
# ----------------------------
def _pinger():
    if not RENDER_EXTERNAL_URL:
        return
    url = RENDER_EXTERNAL_URL.rstrip("/") + "/healthz"
    while True:
        try:
            requests.get(url, timeout=4)
        except Exception:
            pass
        time.sleep(300)  # 5 minutes

@app.before_first_request
def _start_bg_tasks():
    if RENDER_EXTERNAL_URL:
        t = threading.Thread(target=_pinger, name="keep-alive", daemon=True)
        t.start()

# ----------------------------
# Routes
# ----------------------------
@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})

@app.get("/status")
def status():
    return jsonify({
        "running": STREAM.running,
        "frame_count": STREAM.frame_count,
        "width": STREAM.width,
        "height": STREAM.height,
        "fps": STREAM.fps
    })

@app.post("/stream/start")
def start_stream():
    try:
        created = STREAM.start()
        return jsonify({"status": "started" if created else "already-running", "frame_count": STREAM.frame_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.post("/stream/stop")
def stop_stream():
    try:
        stopped = STREAM.stop()
        return jsonify({"status": "stopped" if stopped else "not-running"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.get("/")
def index():
    # Tiny control panel for convenience
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>TerminalMind01</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
body {{ background:#060; color:#cfc; font-family: monospace; padding: 1.5rem; }}
button {{ font-family: inherit; font-size: 1rem; padding: 0.5rem 1rem; margin-right: 0.5rem; }}
pre {{ white-space: pre-wrap; }}
#log {{ margin-top: 1rem; }}
</style>
</head>
<body>
<h1>TERMINALMIND01</h1>
<p>Width: {WIDTH} Height: {HEIGHT} FPS: {FPS}</p>
<p>
  <button onclick="start()">Start Stream</button>
  <button onclick="stop()">Stop Stream</button>
  <button onclick="refresh()">Refresh Status</button>
</p>
<pre id="status">loading…</pre>
<script>
async function start() {{
  const r = await fetch('/stream/start', {{method:'POST'}});
  document.getElementById('status').textContent = await r.text();
}}
async function stop() {{
  const r = await fetch('/stream/stop', {{method:'POST'}});
  document.getElementById('status').textContent = await r.text();
}}
async function refresh() {{
  const r = await fetch('/status');
  document.getElementById('status').textContent = await r.text();
}}
refresh();
</script>
</body>
</html>"""
    return make_response(html, 200)

# ----------------------------
# Dev entrypoint (Render uses Gunicorn per render.yaml)
# ----------------------------
if __name__ == "__main__":
    # For local testing only
    port = int(os.getenv("PORT", "8080"))
    print(f"[local] http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port)
