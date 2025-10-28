# app/tts.py
from __future__ import annotations
import os, asyncio, tempfile, subprocess
from pathlib import Path

class TextToSpeech:
    """
    Engine:
      - 'edge'   : Microsoft Edge TTS (online), ví dụ voice 'vi-VN-HoaiMyNeural' (nữ).
                   Lưu MP3 rồi convert sang OGG Opus để gửi Telegram voice.
      - 'pyttsx3': Offline SAPI5 (Windows). Cần cài voice VI, nếu không sẽ phát âm sai.
    """

    def __init__(self, engine_name: str = "pyttsx3", voice_substr: str = "",
                 rate: int | float = 180, volume: float = 1.0):
        self.engine_name = (engine_name or "pyttsx3").lower().strip()
        self.voice_substr = voice_substr or ""
        self.rate = rate
        self.volume = volume

        if self.engine_name == "edge":
            try:
                import edge_tts  # type: ignore
            except Exception as e:
                raise RuntimeError("edge-tts not installed. Add `edge-tts` to requirements.txt") from e
            self.edge_tts = edge_tts
            # Mặc định giọng nữ Việt:
            self.voice = self.voice_substr or "vi-VN-HoaiMyNeural"
        else:
            try:
                import pyttsx3  # type: ignore
            except Exception as e:
                raise RuntimeError("pyttsx3 not installed.") from e
            self.pyttsx3 = pyttsx3
            self.engine = pyttsx3.init()
            # Rate/volume (pyttsx3)
            try:
                self.engine.setProperty("rate", int(self.rate))
            except Exception:
                pass
            try:
                self.engine.setProperty("volume", max(0.0, min(1.0, float(self.volume))))
            except Exception:
                pass

            # Chọn voice theo substring (ưu tiên VI nếu không chỉ rõ)
            chosen = None
            voices = self.engine.getProperty("voices") or []
            want = (self.voice_substr or "").lower()
            for v in voices:
                name = getattr(v, "name", "") or ""
                langs = getattr(v, "languages", []) or []
                langs_s = " ".join([str(x) for x in langs]).lower()
                if want:
                    if want in name.lower():
                        chosen = v.id; break
                else:
                    if ("vi" in langs_s) or ("vietnam" in name.lower()) or ("viet" in name.lower()):
                        chosen = v.id; break
            if not chosen and want:
                for v in voices:
                    name = getattr(v, "name", "") or ""
                    if want in name.lower():
                        chosen = v.id; break
            if chosen:
                try:
                    self.engine.setProperty("voice", chosen)
                except Exception:
                    pass
            # In danh sách 1 phần voice sẵn có (debug)
            try:
                listing = [f"- {getattr(v,'name','?')} ({getattr(v,'id','?')})"
                           for v in voices[:10]]
                print("[TTS] Available voices (first 10):\n" + "\n".join(listing))
            except Exception:
                pass

    def _ffmpeg_to_ogg(self, src_path: str, out_path: str):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        # Cần ffmpeg trong PATH (Windows: đặt ffmpeg.exe vào PATH hoặc cùng thư mục script)
        cmd = ["ffmpeg", "-y", "-i", src_path, "-c:a", "libopus", "-b:a", "32k", out_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # === API thống nhất ===
    def speak_to_ogg(self, text: str, out_path: str) -> str:
        """
        Tạo file .ogg (Opus) sẵn sàng gửi Telegram voice (sendVoice).
        """
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        if self.engine_name == "edge":
            # Lưu MP3 bằng edge-tts → convert sang OGG/Opus
            async def _run(mp3_path: str):
                comm = self.edge_tts.Communicate(text, self.voice)
                await comm.save(mp3_path)  # KHÔNG có tham số 'format' trong các bản edge-tts hiện tại
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                tmp_mp3 = tf.name
            try:
                asyncio.run(_run(tmp_mp3))
                self._ffmpeg_to_ogg(tmp_mp3, out_path)
            finally:
                try: os.remove(tmp_mp3)
                except Exception: pass
            return out_path

        # pyttsx3: ghi WAV tạm → ffmpeg -> OGG/Opus
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_wav = tf.name
        try:
            self.engine.save_to_file(text, tmp_wav)
            self.engine.runAndWait()
            self._ffmpeg_to_ogg(tmp_wav, out_path)
        finally:
            try: os.remove(tmp_wav)
            except Exception: pass
        return out_path
