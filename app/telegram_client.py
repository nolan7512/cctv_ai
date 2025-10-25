import requests, subprocess
from pathlib import Path

class TelegramClient:
    def __init__(self, token: str, chat_id: str, max_mb=49):
        self.token = token
        self.chat = chat_id
        self.max_bytes = max_mb * 1024 * 1024

    def _api(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.token}/{method}"

    def send_text(self, text: str):
        r = requests.post(self._api("sendMessage"), json={"chat_id": self.chat, "text": text})
        r.raise_for_status()

    def send_video(self, path: str, caption: str = ""):
        p = Path(path)
        if p.stat().st_size > self.max_bytes:
            out = p.with_suffix(".tgz.mp4")
            cmd = [
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                "-i", str(p), "-vf", "scale=-2:720",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                "-movflags", "+faststart", "-an", str(out)
            ]
            subprocess.check_call(cmd)
            p = out
        with open(p, "rb") as f:
            files = {"video": (p.name, f, "video/mp4")}
            data = {"chat_id": self.chat, "caption": caption[:1024]}
            r = requests.post(self._api("sendVideo"), data=data, files=files)
            r.raise_for_status()
