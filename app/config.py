from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv


def load_env(envfile: str | None = None):
    # If ENVFILE is provided (env var or function param), load it; else default to .env
    envfile = envfile or os.getenv("ENVFILE")
    if envfile:
        if Path(envfile).exists():
            load_dotenv(envfile, override=True)
        else:
            raise FileNotFoundError(f"ENVFILE not found: {envfile}")
    else:
        load_dotenv()  # try default .env


@dataclass
class Config:
    # Call after load_env(...)
    rtsp_url: str = os.getenv("RTSP_URL", "")
    rtsp_url_sub: str | None = os.getenv("RTSP_URL_SUB")

    frame_stride: int = int(os.getenv("FRAME_STRIDE", 2))
    conf: float = float(os.getenv("DETECTION_CONF", 0.40))
    objects_of_interest: tuple[str, ...] = tuple(
        s.strip() for s in os.getenv("OBJECTS_OF_INTEREST", "person").split(",") if s.strip()
    )
    min_bbox_area: int = int(os.getenv("MIN_BBOX_AREA", 10000))

    motion_min_pixels: int = int(os.getenv("MOTION_MIN_PIXELS", 5000))
    motion_ratio: float = float(os.getenv("MOTION_RATIO", 0.0075))

    cooldown_seconds: int = int(os.getenv("COOLDOWN_SECONDS", 12))
    merge_window_seconds: int = int(os.getenv("MERGE_WINDOW_SECONDS", 4))

    buffer_dir: str = os.getenv("BUFFER_DIR", "./buffer")
    segment_seconds: int = int(os.getenv("SEGMENT_SECONDS", 2))
    wrap_segments: int = int(os.getenv("WRAP_SEGMENTS", 60))
    pre_roll: int = int(os.getenv("PRE_ROLL", 6))
    post_roll: int = int(os.getenv("POST_ROLL", 6))
    clip_dir: str = os.getenv("CLIP_DIR", "./clips")

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    use_vertex: bool = os.getenv("USE_VERTEX", "False").lower() == "true"

    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    send_video: bool = os.getenv("SEND_VIDEO", "yes").lower() == "yes"
    max_telegram_mb: int = int(os.getenv("MAX_TELEGRAM_MB", 49))

    # YOLO / GPU
    yolo_model: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    yolo_imgsz: int = int(os.getenv("YOLO_IMGSZ", 640))
    device_pref: str = os.getenv("DEVICE", "auto")  # auto|cuda|cpu
    use_fp16: bool = os.getenv("USE_FP16", "yes").lower() == "yes"

    name: str = os.getenv("NAME", "cam")

    def validate(self):
        assert self.rtsp_url, "RTSP_URL is required"
        assert self.gemini_api_key, "GEMINI_API_KEY is required"
        assert self.telegram_token and self.telegram_chat_id, "Telegram token/chat id are required"
