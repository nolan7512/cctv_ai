# app/gemini_client.py
import os, time
from pathlib import Path
from google import genai

class GeminiClient:
    def __init__(self, api_key: str, model: str, use_vertex: bool = False, language: str = "vi"):
        if use_vertex:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.language = language

    def analyze_video(self, clip_path: str) -> str:
        upload = self.client.files.upload(file=Path(clip_path))
        while upload.state in ("PROCESSING", None):
            time.sleep(2)
            upload = self.client.files.get(name=upload.name)
        if upload.state != "ACTIVE":
            return "NO_ACTIVITY"  # fallback an toàn

        # PROMPT: chỉ nói về HÀNH ĐỘNG. Nếu không có gì -> trả đúng "NO_ACTIVITY".
        prompt = (
            "Bạn là trợ lý phân tích camera. Chỉ mô tả ngắn gọn (1–2 câu) những HÀNH ĐỘNG/BIẾN ĐỘNG xảy ra trong video "
            "(ví dụ: có người đi qua, ai đó mở cổng, xe dừng/đi, chó chạy, nhặt đồ, ngã, xô đẩy...). "
            "TUYỆT ĐỐI không mô tả bối cảnh tĩnh, đồ vật cố định, kiến trúc, hay màu sắc nền. "
            "Nếu không có người/xe/động vật hay hành động đáng chú ý, hãy trả về CHÍNH XÁC chuỗi: NO_ACTIVITY. "
            "Trả lời bằng tiếng Việt."
        )

        resp = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, upload],
        )
        summary = (getattr(resp, "text", "") or "").strip()
        return summary if summary else "NO_ACTIVITY"
