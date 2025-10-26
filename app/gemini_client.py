import os, time, json
from pathlib import Path

from google import genai
from google.genai import types as gat


class GeminiClient:
    def __init__(self, api_key: str, model: str, use_vertex: bool = False):
        if use_vertex:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
            # Khi dùng Vertex bạn có thể cần GOOGLE_CLOUD_PROJECT/LOCATION
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _build_prompt(self, hints: dict | None) -> str:
        """
        Tạo prompt có hướng dẫn chi tiết:
        - Nếu có person: mô tả hành động (đi/chạy, cầm-vác, vào/ra cửa, đứng/nhìn quanh),
          có đang lái/đi xe (motorcycle/bicycle/scooter/car) không.
        - Nếu có động vật (dog, cat, bird, cow, horse, ...): đếm và mô tả hành vi.
        - Nếu có vehicle (car, motorcycle, bicycle, bus, truck): nêu trạng thái (dừng/di chuyển/đậu),
          có người tương tác không.
        """
        focus = []
        counts_line = ""
        want = None
        if hints:
            focus = hints.get("focus") or []
            want = hints.get("want") or None
            if hints.get("counts"):
                pairs = [f"{k}:{v}" for k, v in hints["counts"].items()]
                counts_line = " | detections_in_window=" + ", ".join(pairs)

        base = (
            "You are a CCTV analyst. Analyze the short MP4 and respond in JSON following the schema. "
            "Be concise but specific so someone can grasp the event without watching the clip. "
            "Important:\n"
            "- If persons appear: describe actions (walking/running/standing/loitering), entering or leaving, "
            "  carrying objects, and whether riding a vehicle (motorcycle/bicycle/scooter/car). "
            "- For vehicles: indicate type, motion state (moving/stopped/parked), and interactions with persons. "
            "- For animals (dog, cat, bird, cow, horse, etc.): count them and describe what they do (barking, running, sitting, fighting, etc.). "
            "- Provide a short incident label.\n"
        )
        if focus:
            base += f"Focus more on: {', '.join(focus)}.\n"
        if want:
            base += f"Object classes of interest: {', '.join(want)}.\n"
        if counts_line:
            base += counts_line + "\n"
        return base

    def analyze_video(self, clip_path: str, hints: dict | None = None) -> dict:
        """
        Phân tích video với prompt giàu ngữ cảnh (nếu cung cấp hints).
        hints có thể là:
            {
              "focus": ["person","motorcycle"],     # lớp ưu tiên
              "counts": {"person": 2, "dog": 1},    # thống kê sơ bộ từ detector trong cửa sổ sự kiện
              "want": ["person","motorcycle","dog","cat"]  # danh sách quan tâm (ví dụ từ .env)
            }
        """
        upload = self.client.files.upload(file=Path(clip_path))
        # đợi file ACTIVE
        while upload.state in ("PROCESSING", None):
            time.sleep(2)
            upload = self.client.files.get(name=upload.name)
        if upload.state != "ACTIVE":
            raise RuntimeError(f"Gemini file not ACTIVE. state={upload.state}")

        # Schema giàu thông tin hơn
        schema = gat.Schema(
            type=gat.Type.OBJECT,
            properties={
                "summary": gat.Schema(type=gat.Type.STRING),
                "incident": gat.Schema(type=gat.Type.STRING),
                "confidence": gat.Schema(type=gat.Type.NUMBER),

                # liệt kê đối tượng tự do
                "objects": gat.Schema(type=gat.Type.ARRAY, items=gat.Schema(type=gat.Type.STRING)),

                # đếm theo đối tượng: [{label, count}]
                "object_counts": gat.Schema(
                    type=gat.Type.ARRAY,
                    items=gat.Schema(
                        type=gat.Type.OBJECT,
                        properties={
                            "label": gat.Schema(type=gat.Type.STRING),
                            "count": gat.Schema(type=gat.Type.INTEGER),
                        },
                        required=["label", "count"]
                    )
                ),

                # chi tiết người
                "persons": gat.Schema(
                    type=gat.Type.OBJECT,
                    properties={
                        "count": gat.Schema(type=gat.Type.INTEGER),
                        "actions": gat.Schema(type=gat.Type.ARRAY, items=gat.Schema(type=gat.Type.STRING)),
                        # ví dụ: ["motorcycle","bicycle"] nếu có người đang ngồi/lái
                        "riding": gat.Schema(type=gat.Type.ARRAY, items=gat.Schema(type=gat.Type.STRING)),
                    }
                ),

                # phương tiện
                "vehicles": gat.Schema(
                    type=gat.Type.ARRAY,
                    items=gat.Schema(
                        type=gat.Type.OBJECT,
                        properties={
                            "type": gat.Schema(type=gat.Type.STRING),  # car/motorcycle/bicycle/bus/truck
                            "count": gat.Schema(type=gat.Type.INTEGER),
                            "state": gat.Schema(type=gat.Type.STRING),  # moving/stopped/parked/mixed
                            "actions": gat.Schema(type=gat.Type.ARRAY, items=gat.Schema(type=gat.Type.STRING)),
                        }
                    )
                ),

                # động vật
                "animals": gat.Schema(
                    type=gat.Type.ARRAY,
                    items=gat.Schema(
                        type=gat.Type.OBJECT,
                        properties={
                            "species": gat.Schema(type=gat.Type.STRING),  # dog/cat/bird/...
                            "count": gat.Schema(type=gat.Type.INTEGER),
                            "actions": gat.Schema(type=gat.Type.ARRAY, items=gat.Schema(type=gat.Type.STRING)),
                        }
                    )
                ),

                # (tuỳ chọn) vài mốc thời gian ngắn
                "timeline": gat.Schema(
                    type=gat.Type.ARRAY,
                    items=gat.Schema(
                        type=gat.Type.OBJECT,
                        properties={
                            "t": gat.Schema(type=gat.Type.STRING),
                            "note": gat.Schema(type=gat.Type.STRING),
                        }
                    )
                ),
                "notes": gat.Schema(type=gat.Type.STRING),
            },
            required=["summary", "incident", "objects"]
        )

        prompt = self._build_prompt(hints)

        resp = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, upload],
            config=gat.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                max_output_tokens=384,
            ),
        )
        try:
            return resp.parsed or {}
        except Exception:
            # fallback nếu SDK không parse được
            try:
                return json.loads(resp.text)
            except Exception:
                return {"summary": resp.text}
