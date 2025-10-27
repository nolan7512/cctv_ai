# app/gemini_client.py
import os, time
from pathlib import Path
from google import genai

class GeminiClient:
    def __init__(self, api_key: str, model: str, use_vertex: bool = False, language: str = "vi"):
        if use_vertex:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
            # Khi dùng Vertex có thể cần GOOGLE_CLOUD_PROJECT/LOCATION
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.language = language  # hiện chưa dùng, mặc định trả lời tiếng Việt

    def analyze_video(self, clip_path: str) -> str:
        # Upload file
        upload = self.client.files.upload(file=Path(clip_path))
        while upload.state in ("PROCESSING", None):
            time.sleep(2)
            upload = self.client.files.get(name=upload.name)
        if upload.state != "ACTIVE":
            return "Không thể phân tích clip (tệp chưa sẵn sàng)."

        # Prompt ngắn gọn, chỉ yêu cầu tóm tắt bằng tiếng Việt
        prompt = (
            "Bạn là trợ lý phân tích camera. Hãy mô tả ngắn gọn, súc tích (1–4 câu) những gì xảy ra trong video. "
            "Tập trung vào hành động chính, đối tượng nổi bật và bối cảnh, chuyển động bất thường. Trả lời bằng tiếng Việt."
        )

        resp = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, upload],
        )

        summary = (getattr(resp, "text", "") or "").strip()
        if not summary:
            return "Không có gì đáng chú ý hoặc hình ảnh không rõ."
        return summary
