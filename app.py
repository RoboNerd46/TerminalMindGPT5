# app.py
"""
TerminalMind - enhanced CRT-style renderer with typing/scrolling and pre-cached surfaces.

Drop VT323-regular.ttf in the repo root (or set FONT_FILENAME env var).
Controls:
 - GET /stream  -> start streaming to YT (requires YOUTUBE_STREAM_KEY env var)
 - GET /stop    -> stop streaming
 - GET /status  -> runtime status
"""

import os
import time
import threading
import subprocess
import requests
import sys
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
from flask import Flask, jsonify

app = Flask(__name__)

# -----------------------
# Config (env variables)
# -----------------------
MODEL = os.getenv("LLM7_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
API_URL = os.getenv("LLM7_API_URL", "https://api.llm7.io/v1/chat/completions")

FONT_FILENAME = os.getenv("FONT_FILENAME", "VT323-regular.ttf")
FONT_PATH = os.path.join(os.path.dirname(__file__), FONT_FILENAME)

YOUTUBE_STREAM_KEY = os.getenv("YOUTUBE_STREAM_KEY", "")
YOUTUBE_RTMP_URL = f"rtmp://a.rtmp.youtube.com/live2/{YOUTUBE_STREAM_KEY}"

# Streaming params
WIDTH = int(os.getenv("WIDTH", "1280"))
HEIGHT = int(os.getenv("HEIGHT", "720"))
FPS = int(os.getenv("FPS", "30"))
Q_SECONDS = int(os.getenv("Q_SECONDS", "6"))   # seconds to display the question
A_SECONDS = int(os.getenv("A_SECONDS", "8"))   # seconds to display the answer

# Typing / scrolling params
CHARS_PER_SEC = float(os.getenv("CHARS_PER_SEC", "60"))  # realistic terminal typing speed (50-70)
CURSOR_BLINK_SEC = 0.6

# Keep-alive ping (free tier)
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "")  # e.g. https://your-app.onrender.com
PING_INTERVAL = int(os.getenv("PING_INTERVAL", "300"))  # seconds

# Performance caches / globals
STREAM_LOCK = threading.Lock()
STREAM_THREAD = None
STOP_EVENT = threading.Event()
FFMPEG_PROCESS = None
IS_STREAMING = False

# Render caches (initialized in init_render_cache)
CACHE = {}
CACHE_LOCK = threading.Lock()

# Terminal content buffer
log_buffer = []          # list of rendered lines (strings)
log_lock = threading.Lock()
current_typing = ""      # current line being typed (string)
typing_lock = threading.Lock()

# Derived sizes
CONTENT_MARGIN_X = int(WIDTH * 0.06)
CONTENT_MARGIN_Y = int(HEIGHT * 0.12)
CONTENT_WIDTH = WIDTH - 2 * CONTENT_MARGIN_X
CONTENT_HEIGHT = HEIGHT - CONTENT_MARGIN_Y - 40  # leave space for heading

# Colors (RGB)
GREEN_BRIGHT = (144, 255, 140)   # heading glow / bright neon
GREEN_QUESTION = (120, 255, 150) # prefix/QA bright
GREEN_ANSWER = (0, 190, 0)       # dimmer answer
BLACK = (0, 0, 0)

# Styling
HEADING_TEXT = "TerminalMind01"
HEADING_RATIO = 0.09   # approx heading font size = HEADING_RATIO * HEIGHT
BODY_RATIO = 0.032     # body font ratio
ROUND_RADIUS = max(12, int(min(WIDTH, HEIGHT) * 0.015))

# Utility: LLM7 query (unchanged)
def query_llm7(prompt: str, model: str = MODEL, timeout: int = 30) -> str:
    """Query LLM7 (no API key assumed). Returns assistant content or an error string."""
    try:
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 256}
        resp = requests.post(API_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        return body["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[query_llm7] Error: {e}", file=sys.stderr)
        return f"(LLM7 error: {e})"

# -----------------------
# Font loading & helpers
# -----------------------
def load_truetype(size):
    try:
        return ImageFont.truetype(FONT_PATH, int(size))
    except Exception as e:
        print(f"[load_truetype] Could not load '{FONT_PATH}': {e}. Using default PIL font.", file=sys.stderr)
        return ImageFont.load_default()

def init_render_cache():
    """Pre-generate heading, scanline overlay, vignette, rounded rect + glow, and base fonts."""
    with CACHE_LOCK:
        if CACHE.get("initialized"):
            return

        print("[init] Generating render cache...")
        # Fonts
        heading_font = load_truetype(HEIGHT * HEADING_RATIO)
        body_font = load_truetype(max(14, int(HEIGHT * BODY_RATIO)))

        CACHE["heading_font"] = heading_font
        CACHE["body_font"] = body_font

        # Heading image with glow (pre-render)
        heading_img = Image.new("RGBA", (WIDTH, int(HEIGHT * 0.18)), (0,0,0,0))
        draw = ImageDraw.Draw(heading_img)
        w, h = draw.textsize(HEADING_TEXT, font=heading_font)
        x = (WIDTH - w) // 2
        y = 6
        # draw glow by rendering multiple blurred layers
        glow = Image.new("RGBA", heading_img.size, (0,0,0,0))
        gdraw = ImageDraw.Draw(glow)
        gdraw.text((x, y), HEADING_TEXT, font=heading_font, fill=GREEN_BRIGHT + (220,))
        glow = glow.filter(ImageFilter.GaussianBlur(radius=10))
        heading_img = Image.alpha_composite(heading_img, glow)
        draw = ImageDraw.Draw(heading_img)
        draw.text((x, y), HEADING_TEXT, font=heading_font, fill=(0, 255, 0, 255))
        CACHE["heading_img"] = heading_img

        # Rounded rectangle + glow (content frame)
        rect = Image.new("RGBA", (CONTENT_WIDTH+40, CONTENT_HEIGHT+40), (0,0,0,0))
        rdraw = ImageDraw.Draw(rect)
        r = ROUND_RADIUS
        rect_box = (0, 0, rect.width, rect.height)
        rdraw.rounded_rectangle(rect_box, radius=r, outline=GREEN_BRIGHT + (255,), width=2, fill=(0,0,0,200))
        # glow: blur a copy of the outline
        glow_layer = rect.copy().filter(ImageFilter.GaussianBlur(radius=12))
        rect_with_glow = Image.alpha_composite(glow_layer, rect)
        CACHE["frame_img"] = rect_with_glow

        # Scanline overlay (transparent)
        scan = Image.new("RGBA", (WIDTH, HEIGHT), (0,0,0,0))
        sdraw = ImageDraw.Draw(scan)
        # draw subtle dark horizontal lines and very subtle brighter lines to simulate phosphor bands
        for y2 in range(0, HEIGHT, 3):
            alpha = 22 if (y2 % 6 == 0) else 8
            sdraw.line([(0, y2), (WIDTH, y2)], fill=(0,0,0,alpha))
        CACHE["scanline"] = scan

        # Vignette mask (alpha multiply)
        vignette = Image.new("L", (WIDTH, HEIGHT), 0)
        v = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        cy, cx = HEIGHT / 2.0, WIDTH / 2.0
        maxr = math.hypot(cx, cy)
        for yy in range(HEIGHT):
            for xx in range(WIDTH):
                d = math.hypot(xx - cx, yy - cy) / maxr
                # stronger near edges
                val = 255 - int(200 * min(1.0, d**1.6))
                v[yy, xx] = np.clip(val, 0, 255)
        vignette = Image.fromarray(v.astype('uint8'), mode='L')
        CACHE["vignette"] = vignette

        # Cursor image (small)
        cursor = Image.new("RGBA", (12, int(body_font.size * 1.1)), (0,0,0,0))
        cdraw = ImageDraw.Draw(cursor)
        cdraw.text((0,0), "_", font=body_font, fill=(0,255,0,255))
        CACHE["cursor"] = cursor

        CACHE["initialized"] = True
        print("[init] Render cache ready.")

# -----------------------
# Text rendering helpers
# -----------------------
def render_text_line(text, font, color, bold=False):
    """Render a single anti-aliased text line into RGBA image (tight bbox)."""
    # render on a temp image then trim
    tmp = Image.new("RGBA", (CONTENT_WIDTH, font.size + 12), (0,0,0,0))
    draw = ImageDraw.Draw(tmp)
    if bold:
        # simulate bold by drawing twice with slight offset
        draw.text((0,0), text, font=font, fill=color + (255,))
        draw.text((1,0), text, font=font, fill=color + (255,))
    else:
        draw.text((0,0), text, font=font, fill=color + (255,))
    # crop to content
    bbox = tmp.getbbox()
    if not bbox:
        return tmp
    return tmp.crop(bbox)

def wrap_text_to_lines(text, font, max_width):
    """Returns list of wrapped lines (strings)."""
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = f"{cur} {w}".strip()
        test_w = font.getmask(test).getbbox()
        if test_w:
            tw = test_w[2]
        else:
            tw = 0
        if tw > max_width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)
    return lines

# -----------------------
# High-level compositor
# -----------------------
def render_terminal_frame(log_lines, typing_line, show_cursor, width=WIDTH, height=HEIGHT):
    """
    Compose a single frame from:
      - black background
      - rounded green frame with glow
      - pre-rendered heading
      - content text (log_lines + typing_line)
      - scanlines overlay
      - vignette multiply
    """
    init_render_cache()
    heading = CACHE["heading_img"]
    frame_img = CACHE["frame_img"]
    scan = CACHE["scanline"]
    vignette = CACHE["vignette"]
    body_font = CACHE["body_font"]
    cursor_img = CACHE["cursor"]

    # Base black background
    base = Image.new("RGBA", (width, height), BLACK + (255,))

    # Paste heading at top center
    base.alpha_composite(heading, (0, 2))

    # Compose rounded frame centered under heading
    fx = CONTENT_MARGIN_X - 20
    fy = int(heading.size[1] + 6)
    base.alpha_composite(frame_img, (fx, fy))

    # Prepare content area (box) where text will appear
    content_x = fx + 20
    content_y = fy + 18
    # render text lines into a single image
    spacing = int(body_font.size * 1.4)
    max_lines = (frame_img.height - 40) // spacing
    # we will render newest lines at bottom of content area so it scrolls upwards
    content = Image.new("RGBA", (CONTENT_WIDTH, frame_img.height - 40), (0,0,0,0))
    cdraw = ImageDraw.Draw(content)
    # starting y = top
    y = 8

    # decide color and indicator for Q/A lines by convention: lines starting with "> " are questions
    # iterate over log_lines (already strings)
    for ln in log_lines[-max_lines:]:
        if ln.startswith("> "):
            rendered = render_text_line(ln, body_font, GREEN_QUESTION, bold=True)
        else:
            rendered = render_text_line(ln, body_font, GREEN_ANSWER, bold=False)
        content.alpha_composite(rendered, (4, y))
        y += spacing

    # typing line (bottom-most)
    if typing_line is not None:
        ln = typing_line
        # decide style: if starts with "> " it's a question being typed
        if ln.startswith("> "):
            rendered = render_text_line(ln, body_font, GREEN_QUESTION, bold=True)
        else:
            rendered = render_text_line(ln, body_font, GREEN_ANSWER, bold=False)
        content.alpha_composite(rendered, (4, y))
        # draw cursor if requested (cursor positioned after rendered width)
        if show_cursor:
            # compute width of typed portion
            mask = rendered.convert("L")
            bbox = mask.getbbox()
            if bbox:
                cx = bbox[2] + 6
            else:
                cx = 6
            content.alpha_composite(cursor_img, (cx, y))

    # Composite content onto base
    base.alpha_composite(content, (content_x, content_y))

    # Apply mild glow (blur) to whole composite for softness — but keep small radius for perf
    glow = base.filter(ImageFilter.GaussianBlur(radius=1.2))
    # composite glow underneath main to give subtle aura
    base = Image.alpha_composite(glow, base)

    # overlay scanlines and vignette
    base = Image.alpha_composite(base, scan)
    # apply vignette as multiply on alpha channel
    base_rgb = base.convert("RGB")
    vign = vignette.resize(base_rgb.size)
    base_np = np.array(base_rgb).astype(np.uint8)
    vign_np = np.array(vign).astype(np.float32) / 255.0
    # darken edges slightly
    base_np = (base_np * vign_np[..., None]).astype(np.uint8)

    # final BGR for ffmpeg
    final = cv2.cvtColor(base_np, cv2.COLOR_RGB2BGR)
    return final

# -----------------------
# FFmpeg helpers
# -----------------------
def start_ffmpeg_process():
    """Start ffmpeg process that reads rawvideo from stdin and pushes to YouTube RTMP."""
    global FFMPEG_PROCESS
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-b:v", "3000k",
        "-maxrate", "3000k",
        "-bufsize", "6000k",
        "-g", str(FPS * 2),
        "-c:a", "aac",
        "-ar", "44100",
        "-ac", "2",
        "-f", "flv",
        YOUTUBE_RTMP_URL
    ]
    print(f"[ffmpeg] Starting ffmpeg: {' '.join(ffmpeg_cmd)}")
    FFMPEG_PROCESS = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return FFMPEG_PROCESS

def stop_ffmpeg_process():
    global FFMPEG_PROCESS
    try:
        if FFMPEG_PROCESS:
            print("[ffmpeg] Terminating ffmpeg process...")
            FFMPEG_PROCESS.terminate()
            try:
                FFMPEG_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("[ffmpeg] Killing ffmpeg process (timeout).")
                FFMPEG_PROCESS.kill()
    except Exception as e:
        print(f"[ffmpeg] Error stopping ffmpeg: {e}", file=sys.stderr)
    finally:
        FFMPEG_PROCESS = None

# -----------------------
# Typing/scrolling state machine + streaming loop
# -----------------------
def streaming_loop():
    """Background streaming loop that performs recursive Q->A->next Q and writes frames to ffmpeg stdin."""
    global IS_STREAMING
    print("[stream] streaming_loop starting...")
    try:
        proc = start_ffmpeg_process()
    except Exception as e:
        print(f"[stream] Failed to start ffmpeg: {e}", file=sys.stderr)
        IS_STREAMING = False
        return

    # initial prompt to generate the first question
    q_prompt = "Please ask yourself a question that explores your own mind or idea-space."
    q = query_llm7(q_prompt).strip()
    print(f"[stream] Initial question: {q!r}")

    # initialize typing state
    with log_lock:
        log_buffer.clear()
    typing_state = {
        "current_full": f"> {q}",   # the full text we want typed (prefix > for questions)
        "cursor_visible": True,
        "char_accum": 0.0,
        "phase": "typing_question",  # phases: typing_question, hold_question, typing_answer, hold_answer, thinking
        "display_until": 0.0
    }

    # helper to start typing a new string
    def start_typing(full_text, phase, hold_seconds):
        typing_state["current_full"] = full_text
        typing_state["phase"] = phase
        typing_state["char_accum"] = 0.0
        typing_state["display_until"] = 0.0
        typing_state["hold_seconds"] = hold_seconds

    # main loop
    last_cursor_toggle = time.time()
    try:
        while not STOP_EVENT.is_set():
            frame_start = time.time()
            # process typing progress
            cps = CHARS_PER_SEC
            chars_per_frame = cps / FPS
            typing_state["char_accum"] += chars_per_frame
            to_take = int(math.floor(typing_state["char_accum"]))
            if to_take > 0:
                typing_state["char_accum"] -= to_take
                # append next characters to current_typing
                with typing_lock:
                    adding = typing_state["current_full"]
                    # we will manage typed length via a small index prop stored on the object itself
                    idx = typing_state.get("typed_idx", 0)
                    remain = len(adding) - idx
                    take_now = min(remain, to_take)
                    new_idx = idx + take_now
                    typing_state["typed_idx"] = new_idx
                    # update current_typing
                    cur = adding[:new_idx]
                    # update visible log when a full line finished
                    if new_idx == len(adding):
                        # push into log buffer and transition to next phase
                        with log_lock:
                            log_buffer.append(adding)
                        # determine what to do next depending on phase
                        if typing_state["phase"] == "typing_question":
                            # now generate answer and start typing it
                            a = query_llm7(adding[2:] if adding.startswith("> ") else adding)
                            start_typing(f"A: {a}", "typing_answer", hold_seconds=A_SECONDS)
                            typing_state["typed_idx"] = 0
                        elif typing_state["phase"] == "typing_answer":
                            # after answer typed, create next question prompt from the answer
                            next_q_prompt = f"Based on this answer, '{adding}', ask yourself the next deep or creative question to explore your idea-space further."
                            q2 = query_llm7(next_q_prompt)
                            start_typing(f"> {q2}", "typing_question", hold_seconds=Q_SECONDS)
                            typing_state["typed_idx"] = 0
                    # store the partial line for rendering
                    with typing_lock:
                        cur_partial = adding[:typing_state.get("typed_idx",0)]
                        # assign into outer variable
                        typing_text = cur_partial
                        # also update a global current_typing for thread-safe access
                        global current_typing
                        current_typing = typing_text

            # cursor blink
            now = time.time()
            if now - last_cursor_toggle >= CURSOR_BLINK_SEC:
                typing_state["cursor_visible"] = not typing_state["cursor_visible"]
                last_cursor_toggle = now

            # produce frame
            with log_lock:
                visible_lines = list(log_buffer)  # copy
            with typing_lock:
                typing_line = current_typing

            frame = render_terminal_frame(visible_lines, typing_line, show_cursor=typing_state["cursor_visible"])

            # send to ffmpeg
            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("[stream] BrokenPipeError writing frame.", file=sys.stderr)
                STOP_EVENT.set()
                break

            # maintain fps
            elapsed = time.time() - frame_start
            sleep_for = max(0.0, (1.0 / FPS) - elapsed)
            time.sleep(sleep_for)

    except Exception as e:
        print(f"[stream] Unexpected error in streaming loop: {e}", file=sys.stderr)
    finally:
        stop_ffmpeg_process()
        IS_STREAMING = False
        STOP_EVENT.clear()
        print("[stream] streaming_loop ended.")

# -----------------------
# Keep-alive pinger
# -----------------------
def keep_alive_loop():
    if not RENDER_EXTERNAL_URL:
        print("[keep-alive] No RENDER_EXTERNAL_URL configured; keep-alive disabled.")
        return
    url = RENDER_EXTERNAL_URL.rstrip("/") + "/ping"
    print(f"[keep-alive] Will ping {url} every {PING_INTERVAL}s")
    while True:
        try:
            requests.get(url, timeout=10)
            print(f"[keep-alive] pinged {url}")
        except Exception as e:
            print(f"[keep-alive] ping failed: {e}")
        time.sleep(PING_INTERVAL)

# -----------------------
# Flask endpoints (control)
# -----------------------
@app.route("/")
def index():
    return jsonify({
        "service": "TerminalMind self-interview stream (enhanced CRT)",
        "is_streaming": IS_STREAMING,
        "fps": FPS,
        "resolution": f"{WIDTH}x{HEIGHT}"
    })

@app.route("/ping")
def ping():
    return "pong"

@app.route("/status")
def status():
    return jsonify({
        "is_streaming": IS_STREAMING,
        "ffmpeg_running": FFMPEG_PROCESS is not None and (FFMPEG_PROCESS.poll() is None),
        "youtube_configured": bool(YOUTUBE_STREAM_KEY)
    })

@app.route("/stream")
def start_stream_route():
    global STREAM_THREAD, IS_STREAMING
    if not YOUTUBE_STREAM_KEY:
        return "Missing YOUTUBE_STREAM_KEY env var", 500

    with STREAM_LOCK:
        if IS_STREAMING:
            return "already_streaming", 200
        STOP_EVENT.clear()
        STREAM_THREAD = threading.Thread(target=streaming_loop, daemon=True)
        STREAM_THREAD.start()
        IS_STREAMING = True
        print("[/stream] Started streaming thread.")
        return "stream_started", 200

@app.route("/stop")
def stop_stream_route():
    global STREAM_THREAD, IS_STREAMING
    with STREAM_LOCK:
        if not IS_STREAMING:
            return "not_streaming", 200
        print("[/stop] Stop requested; signalling streaming loop...")
        STOP_EVENT.set()
        # wait briefly for thread shutdown
        start = time.time()
        timeout = 10.0
        while STREAM_THREAD and STREAM_THREAD.is_alive() and (time.time() - start) < timeout:
            time.sleep(0.2)
        IS_STREAMING = False
        return "stop_requested", 200

# -----------------------
# Start background keep-alive and run Flask
# -----------------------
if __name__ == "__main__":
    # prepare render cache early (helps warm up)
    init_render_cache()

    # Start keep-alive pinger thread (if configured)
    if RENDER_EXTERNAL_URL:
        t_ping = threading.Thread(target=keep_alive_loop, daemon=True)
        t_ping.start()

    # Note: do not auto-start streaming here; use /stream to control
    port = int(os.environ.get("PORT", "5000"))
    print(f"[main] Starting Flask on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
