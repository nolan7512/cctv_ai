import os, time, json
from pathlib import Path

from google import genai
from google.genai import types as gat

class GeminiClient:
    def __init__(self, api_key: str, model: str, use_vertex: bool = False):
        if use_vertex:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
            # You may also need GOOGLE_CLOUD_PROJECT/LOCATION envs when using Vertex
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def analyze_video(self, clip_path: str) -> dict:
        upload = self.client.files.upload(file=Path(clip_path))
        while upload.state in ("PROCESSING", None):
            time.sleep(2)
            upload = self.client.files.get(name=upload.name)
        if upload.state != "ACTIVE":
            raise RuntimeError(f"Gemini file not ACTIVE. state={upload.state}")

        schema = gat.Schema(
            type=gat.Type.OBJECT,
            properties={
                "summary": gat.Schema(type=gat.Type.STRING),
                "objects": gat.Schema(type=gat.Type.ARRAY, items=gat.Schema(type=gat.Type.STRING)),
                "confidence": gat.Schema(type=gat.Type.NUMBER),
                "incident": gat.Schema(type=gat.Type.STRING),
                "notes": gat.Schema(type=gat.Type.STRING),
            },
            required=["summary", "objects"],
        )
        prompt = (
            "You are a CCTV analyst. Summarize the key event in 1-2 sentences. "
            "List salient objects. Infer a short incident label (e.g., person entering, vehicle stopping, loitering)."
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, upload],
            config=gat.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                max_output_tokens=256,
            ),
        )
        try:
            return resp.parsed or {}
        except Exception:
            return json.loads(resp.text)
