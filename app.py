# app.py
import os
import time
import threading
import subprocess
import requests
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, jsonify

app = Flask(__name__)

# -----------------------
# Config (env variables)
# -----------------------
MODEL = os.getenv("LLM7_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
API_URL = os.getenv("LLM7_API_URL", "https://api.llm7.io/v1/chat/completions")

FONT_FILENAME = os.getenv("FONT_FILENAME", "VT323-Regular.ttf")
FONT_PATH = os.path.join(os.path.dirname(__file__), FONT_FILENAME)

YOUTUBE_STREAM_KEY = os.getenv("YOUTUBE_STREAM_KEY", "")
YOUTUBE_RTMP_URL = f"rtmp://a.rtmp.youtube.com/live2/{YOUTUBE_STREAM_KEY}"

# Streaming params
WIDTH = int(os.getenv("WIDTH", "1280"))
HEIGHT = int(os.getenv("HEIGHT", "720"))
FPS = int(os.getenv("FPS", "30"))
Q_SECONDS = int(os.getenv("Q_SECONDS", "6"))   # seconds to display the question
A_SECONDS = int(os.getenv("A_SECONDS", "8"))   # seconds to display the answer

# Keep-alive ping (free tier)
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "")  # e.g. https://your-app.onrender.com
PING_INTERVAL = int(os.getenv("PING_INTERVAL", "300"))  # seconds

# Globals to manage streaming thread and ffmpeg
STREAM_LOCK = threading.Lock()
STREAM_THREAD = None
STOP_EVENT = threading.Event()
FFMPEG_PROCESS = None
IS_STREAMING = False

# -----------------------
# Utilities: LLM7 query
# -----------------------
def query_llm7(prompt: str, model: str = MODEL, timeout: int = 30) -> str:
    """Query LLM7 (no API key assumed). Returns assistant content or an error string."""
    try:
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 256}
        resp = requests.post(API_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        # defensive navigation
        return body["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[query_llm7] Error: {e}", file=sys.stderr)
        return f"(LLM7 error: {e})"

# -----------------------
# Utilities: font + rendering
# -----------------------
def load_font(size=36):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception as e:
        print(f"[load_font] Could not load '{FONT_PATH}': {e}. Using default PIL font.", file=sys.stderr)
        return ImageFont.load_default()

def render_frame(text: str, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Render a BGR numpy frame with green-on-black CRT-like text."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = load_font(max(18, int(height * 0.03)))
    margin_x, margin_y = int(width * 0.04), int(height * 0.04)

    # simple multi-line wrap
    max_width = width - 2 * margin_x
    lines = []
    words = text.split()
    cur = ""
    for w in words:
        test = f"{cur} {w}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] > max_width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)

    y = margin_y
    for line in lines:
        draw.text((margin_x, y), line, font=font, fill=(0, 255, 0))
        y += int(font.size * 1.4)

    # subtle scanline effect
    arr = np.array(img, dtype=np.uint8)
    arr[::2] = (arr[::2] * 0.97).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

# -----------------------
# FFmpeg helper
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
# Streaming loop (self-interviewing LLM)
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

    # start with a meta prompt: LLM asks itself a question about its own idea-space
    q_prompt = "Please ask yourself a question that explores your own mind or idea-space."
    q = query_llm7(q_prompt)
    print(f"[stream] Initial question: {q!r}")

    try:
        while not STOP_EVENT.is_set():
            # 1) Answer the current question
            a = query_llm7(q)
            print(f"[stream] Q -> {q!r}")
            print(f"[stream] A -> {a!r}")

            # 2) Show Q frames for Q_SECONDS, then A frames for A_SECONDS
            q_frames = max(1, int(FPS * Q_SECONDS))
            a_frames = max(1, int(FPS * A_SECONDS))

            frame_q = render_frame(f"Q: {q}")
            for _ in range(q_frames):
                if STOP_EVENT.is_set(): break
                try:
                    proc.stdin.write(frame_q.tobytes())
                except BrokenPipeError:
                    print("[stream] BrokenPipeError writing Q frame.", file=sys.stderr)
                    STOP_EVENT.set()
                    break

            if STOP_EVENT.is_set(): break

            frame_a = render_frame(f"A: {a}")
            for _ in range(a_frames):
                if STOP_EVENT.is_set(): break
                try:
                    proc.stdin.write(frame_a.tobytes())
                except BrokenPipeError:
                    print("[stream] BrokenPipeError writing A frame.", file=sys.stderr)
                    STOP_EVENT.set()
                    break

            # 3) Formulate next question based on the last answer
            next_q_prompt = f"Based on this answer, '{a}', ask yourself the next deep or creative question to explore your idea-space further."
            q = query_llm7(next_q_prompt)
            print(f"[stream] Next question: {q!r}")

            # small non-blocking sleep to yield
            for _ in range(int(FPS * 0.5)):
                if STOP_EVENT.is_set():
                    break
                time.sleep(1 / FPS)

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
        "service": "TerminalMind self-interview stream",
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
    # Start keep-alive pinger thread (if configured)
    if RENDER_EXTERNAL_URL:
        t_ping = threading.Thread(target=keep_alive_loop, daemon=True)
        t_ping.start()

    # Note: do not auto-start streaming here; use /stream to control
    port = int(os.environ.get("PORT", "5000"))
    print(f"[main] Starting Flask on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
