# TerminalMind Stream (enhanced CRT)

## Files
- `app.py` - main app (Flask + renderer + ffmpeg streamer)
- `VT323-regular.ttf` - required font (place in repo root)
- `requirements.txt`
- `render.yaml`

## Environment variables
- `YOUTUBE_STREAM_KEY` (required to stream)
- `RENDER_EXTERNAL_URL` (optional; keep-alive ping)
- `PORT` (default 5000)
- `CHARS_PER_SEC` (default 60)
- `WIDTH`, `HEIGHT`, `FPS` (defaults: 1280x720@30)

## Run locally
1. Install deps:
