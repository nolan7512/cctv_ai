import requests, subprocess
from pathlib import Path
import os

class TelegramClient:
    """
    Hỗ trợ TELEGRAM_CHAT = "@handle" (group/channel public) hoặc ID số (vd: -1001234567890).
    Tự động resolve @handle -> numeric chat_id qua getChat và cache trong self.chat_id.
    """
    def __init__(self, token: str, chat: str, max_mb=49):
        self.token = token
        self.chat_input = chat.strip()
        self.chat_id = None
        self.max_bytes = max_mb * 1024 * 1024

    def _api(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.token}/{method}"

    def _resolve_chat_id(self):
        if self.chat_id is not None:
            return self.chat_id
        ci = self.chat_input
        if ci.startswith("@"):
            r = requests.get(self._api("getChat"), params={"chat_id": ci}, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                raise RuntimeError(f"getChat failed: {data}")
            self.chat_id = data["result"]["id"]
        else:
            self.chat_id = int(ci)  # chấp nhận -100..., 12345...
        return self.chat_id

    def send_text(self, text: str):
        chat_id = self._resolve_chat_id()
        r = requests.post(self._api("sendMessage"),
                          json={"chat_id": chat_id, "text": text},
                          timeout=20)
        r.raise_for_status()

    def send_video(self, path: str, caption: str = ""):
        chat_id = self._resolve_chat_id()
        p = Path(path)
        if p.stat().st_size > self.max_bytes:
            out = p.with_suffix(".tgz.mp4")
            cmd = ["ffmpeg","-nostdin","-loglevel","error","-y","-i",str(p),
                   "-vf","scale=-2:720","-c:v","libx264","-preset","veryfast","-crf","28",
                   "-movflags","+faststart","-an",str(out)]
            subprocess.check_call(cmd)
            p = out
        with open(p, "rb") as f:
            files = {"video": (p.name, f, "video/mp4")}
            data = {"chat_id": chat_id, "caption": caption[:1024]}
            r = requests.post(self._api("sendVideo"), data=data, files=files, timeout=120)
            r.raise_for_status()

    def send_voice(self, path: str, caption: str | None = None):
        """
        Gửi voice note: OGG/Opus (<= 50MB). Dùng sendVoice thay vì sendAudio để hiện dạng "voice".
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if os.path.getsize(path) > self.max_bytes:
            raise ValueError(f"Voice too large (> {self.max_bytes} bytes)")

        url = f"{self.base}/sendVoice"
        with open(path, "rb") as f:
            files = {"voice": (os.path.basename(path), f, "audio/ogg")}
            data = {"chat_id": self._chat_id()}
            if caption:
                data["caption"] = caption
            r = requests.post(url, data=data, files=files, timeout=120)
        r.raise_for_status()
        return r.json()