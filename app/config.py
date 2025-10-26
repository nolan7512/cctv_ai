from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

def load_env(envfile: str | None = None):
    envfile = envfile or os.getenv("ENVFILE")
    if envfile:
        p = Path(envfile)
        if p.exists():
            load_dotenv(p, override=True)
        else:
            raise FileNotFoundError(f"ENVFILE not found: {envfile}")
    else:
        load_dotenv()  # thử .env mặc định

@dataclass
class Config:
    # các field sẽ được điền ở __post_init__
    rtsp_url: str = field(init=False)
    rtsp_url_sub: str | None = field(init=False)

    frame_stride: int = field(init=False)
    conf: float = field(init=False)
    objects_of_interest: tuple[str, ...] = field(init=False)
    min_bbox_area: int = field(init=False)

    motion_min_pixels: int = field(init=False)
    motion_ratio: float = field(init=False)

    cooldown_seconds: int = field(init=False)
    merge_window_seconds: int = field(init=False)

    buffer_dir: str = field(init=False)
    segment_seconds: int = field(init=False)
    wrap_segments: int = field(init=False)
    pre_roll: int = field(init=False)
    post_roll: int = field(init=False)
    clip_dir: str = field(init=False)

    gemini_api_key: str = field(init=False)
    gemini_model: str = field(init=False)
    use_vertex: bool = field(init=False)

    telegram_token: str = field(init=False)
    telegram_chat: str = field(init=False)
    telegram_chat_id: str = field(init=False)
    send_video: bool = field(init=False)
    max_telegram_mb: int = field(init=False)

    # YOLO / GPU
    yolo_model: str = field(init=False)
    yolo_imgsz: int = field(init=False)
    device_pref: str = field(init=False)   # auto|cuda|cpu
    use_fp16: bool = field(init=False)

    name: str = field(init=False)

    def __post_init__(self):
        # đọc env TẠI ĐÂY (sau khi load_env() đã chạy)
        self.rtsp_url = os.getenv("RTSP_URL", "")
        self.rtsp_url_sub = os.getenv("RTSP_URL_SUB") or None

        self.frame_stride = int(os.getenv("FRAME_STRIDE", "2"))
        self.conf = float(os.getenv("DETECTION_CONF", "0.40"))
        self.objects_of_interest = tuple(
            s.strip() for s in os.getenv("OBJECTS_OF_INTEREST", "person").split(",") if s.strip()
        )
        self.min_bbox_area = int(os.getenv("MIN_BBOX_AREA", "10000"))

        self.motion_min_pixels = int(os.getenv("MOTION_MIN_PIXELS", "5000"))
        self.motion_ratio = float(os.getenv("MOTION_RATIO", "0.0075"))

        self.cooldown_seconds = int(os.getenv("COOLDOWN_SECONDS", "12"))
        self.merge_window_seconds = int(os.getenv("MERGE_WINDOW_SECONDS", "4"))

        self.buffer_dir = os.getenv("BUFFER_DIR", "./buffer")
        self.segment_seconds = int(os.getenv("SEGMENT_SECONDS", "2"))
        self.wrap_segments = int(os.getenv("WRAP_SEGMENTS", "60"))
        self.pre_roll = int(os.getenv("PRE_ROLL", "6"))
        self.post_roll = int(os.getenv("POST_ROLL", "6"))
        self.clip_dir = os.getenv("CLIP_DIR", "./clips")

        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.use_vertex = os.getenv("USE_VERTEX", "False").lower() == "true"

        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat = os.getenv("TELEGRAM_CHAT") or os.getenv("TELEGRAM_CHAT_ID", "")

        self.send_video = os.getenv("SEND_VIDEO", "yes").lower() == "yes"
        self.max_telegram_mb = int(os.getenv("MAX_TELEGRAM_MB", "49"))

        self.yolo_model = os.getenv("YOLO_MODEL", "yolov8n.pt")
        self.yolo_imgsz = int(os.getenv("YOLO_IMGSZ", "640"))
        self.device_pref = os.getenv("DEVICE", "auto")
        self.use_fp16 = os.getenv("USE_FP16", "yes").lower() == "yes"

        self.name = os.getenv("NAME", "cam")

    def validate(self):
        assert self.rtsp_url, "RTSP_URL is required"
        assert self.gemini_api_key, "GEMINI_API_KEY is required"
        assert self.telegram_token and self.telegram_chat, "Telegram token/chat are required"
