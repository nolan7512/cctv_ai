# app/telegram_client.py
from __future__ import annotations
import os
import mimetypes
import requests

class TelegramClient:
    """
    Hỗ trợ:
      - send_text
      - send_video
      - send_voice (voice note .ogg/opus)
    chat có thể là @ten_nhom (public) hoặc id âm (private group).
    """

    def __init__(self, bot_token: str, chat: str | int, max_mb: int = 45):
        assert bot_token, "Telegram bot token required"
        assert chat, "Telegram chat required"
        self.token = bot_token              # lưu token thô
        self.chat = chat
        self.max_bytes = int(max_mb) * 1024 * 1024

    @property
    def base(self) -> str:
        # Luôn tính động để tránh case self.base không tồn tại ở bản cũ
        return f"https://api.telegram.org/bot{self.token}"

    def _chat_id(self):
        # nếu là số hoặc chuỗi số âm: dùng trực tiếp
        try:
            return int(self.chat)
        except Exception:
            # @username group/public channel
            return self.chat

    def send_text(self, text: str, parse_mode: str | None = None):
        url = f"{self.base}/sendMessage"
        data = {"chat_id": self._chat_id(), "text": text}
        if parse_mode:
            data["parse_mode"] = parse_mode
        r = requests.post(url, data=data, timeout=30)
        r.raise_for_status()
        return r.json()

    def send_video(self, path: str, caption: str | None = None):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if os.path.getsize(path) > self.max_bytes:
            raise ValueError(f"Video too large (> {self.max_bytes} bytes)")

        url = f"{self.base}/sendVideo"
        mime = mimetypes.guess_type(path)[0] or "video/mp4"
        with open(path, "rb") as f:
            files = {"video": (os.path.basename(path), f, mime)}
            data = {"chat_id": self._chat_id()}
            if caption:
                data["caption"] = caption
            r = requests.post(url, data=data, files=files, timeout=120)
        r.raise_for_status()
        return r.json()

    def send_voice(self, path: str, caption: str | None = None):
        """
        Gửi voice note: OGG/Opus (≤ ~50MB). Dùng sendVoice để hiển thị dạng 'voice'.
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
