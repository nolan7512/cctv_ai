# CCTV ➜ Detect ➜ Clip ➜ Gemini ➜ Telegram (Python) — GPU toggle, NO GStreamer

## 1) Install
```bash
pip install -r requirements.txt
pip install -U ultralytics
# FFmpeg: ensure it's on PATH (Windows: add bin; Ubuntu: apt install ffmpeg)
```

## 2) Configure (2 cameras example)
- Copy `.env.cam1.example` → `.env.cam1`, `.env.cam2.example` → `.env.cam2` and fill RTSP/Gemini/Telegram.
- **GPU toggle** via `.env`: `DEVICE=auto|cuda|cpu`, `USE_FP16=yes|no`, `YOLO_MODEL`, `YOLO_IMGSZ`.

## 3) Run (2 terminals)
```bash
python -m app.main --env ./.env.cam1
python -m app.main --env ./.env.cam2
```

## 4) Notes
- AI reads **SUB** stream for detection; ring-buffer records **MAIN** stream with `-c copy` for high-quality clips.
- If `DEVICE=auto`, code will use CUDA when available; otherwise CPU.
- `USE_FP16=yes` enables half precision on CUDA for more FPS.
