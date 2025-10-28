# app/tts.py
from __future__ import annotations
import os, tempfile, subprocess
from pathlib import Path
from typing import Optional

class TextToSpeech:
    """
    TTS offline bằng pyttsx3 (SAPI5 trên Windows, NSSpeech/Espeak trên macOS/Linux).
    Quy trình: text -> WAV (offline) -> FFmpeg -> OGG Opus (hợp lệ cho Telegram sendVoice).
    """
    def __init__(self, engine_name: str = "pyttsx3", voice_substr: Optional[str] = None,
                 rate: int = 180, volume: float = 1.0, ffmpeg_path: str = "ffmpeg"):
        if engine_name.lower() != "pyttsx3":
            raise ValueError("Only 'pyttsx3' engine is supported in this build.")
        import pyttsx3  # lazy import
        self.engine = pyttsx3.init()
        # chọn voice theo substring (nếu cung cấp)
        self._select_voice(voice_substr)
        self.engine.setProperty("rate", int(rate))
        self.engine.setProperty("volume", float(volume))
        self.ffmpeg = ffmpeg_path

    def _select_voice(self, voice_substr: Optional[str]):
        if not voice_substr:
            return
        voices = self.engine.getProperty("voices") or []
        vs = voice_substr.lower()
        for v in voices:
            name = (getattr(v, "name", "") or "").lower()
            lang = ",".join(getattr(v, "languages", []) or []).lower()
            _id  = (getattr(v, "id", "") or "").lower()
            if vs in name or vs in lang or vs in _id:
                self.engine.setProperty("voice", v.id)
                break

    def synth_wav(self, text: str, wav_path: str):
        self.engine.save_to_file(text, wav_path)
        self.engine.runAndWait()

    def wav_to_ogg(self, wav_path: str, ogg_path: str, bitrate_kbps: int = 24, ar: int = 24000):
        """
        Convert WAV -> OGG/Opus (Telegram sendVoice yêu cầu OGG/Opus).
        """
        cmd = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", wav_path,
            "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k", "-ar", str(ar),
            "-vn",
            ogg_path,
        ]
        subprocess.run(cmd, check=True)

    def speak_to_ogg(self, text: str, ogg_path: str, bitrate_kbps: int = 24, ar: int = 24000):
        Path(ogg_path).parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            wav_tmp = str(Path(td) / "tmp.wav")
            self.synth_wav(text, wav_tmp)
            self.wav_to_ogg(wav_tmp, ogg_path, bitrate_kbps=bitrate_kbps, ar=ar)
        return ogg_path
