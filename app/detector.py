from ultralytics import YOLO
import cv2, time, torch
from typing import Generator, Dict

class ObjectDetector:
    def __init__(self, conf=0.40, frame_stride=2, min_area=10000, classes_of_interest: tuple[str, ...] = ("person",),
                 yolo_model: str = "yolov8n.pt", imgsz: int = 640, device_pref: str = "auto", use_fp16: bool = True):
        self.model = YOLO(yolo_model)  # auto-download on first run
        self.conf = conf
        self.frame_stride = frame_stride
        self.min_area = min_area
        self.names = self.model.names
        self.want = set(classes_of_interest)

        # resolve device
        if device_pref == "cuda" or (device_pref == "auto" and torch.cuda.is_available()):
            self.device = "cuda"
        elif device_pref == "cpu":
            self.device = "cpu"
        else:
            self.device = "cpu"
        self.use_fp16 = bool(use_fp16) and (self.device == "cuda")

        # move model to device, set half if possible
        try:
            self.model.to(self.device)
            if self.use_fp16:
                # best-effort half precision
                if hasattr(self.model, "model") and hasattr(self.model.model, "half"):
                    self.model.model.half()
        except Exception:
            pass
        self.imgsz = imgsz

    def _open(self, url: str):
        return cv2.VideoCapture(url)

    def stream_detect(self, rtsp_url: str, motion_gate=None) -> Generator[Dict, None, None]:
        cap = self._open(rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {rtsp_url}")
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2); continue
            i += 1
            if self.frame_stride > 1 and (i % self.frame_stride):
                continue

            if motion_gate is not None and not motion_gate.is_motion(frame):
                continue

            # inference
            for r in self.model(frame, conf=self.conf, verbose=False, imgsz=self.imgsz, device=self.device):
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    cls = self.names[cls_id]
                    if cls not in self.want:
                        continue
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area < self.min_area:
                        continue
                    yield {
                        "ts": time.time(),
                        "class": cls,
                        "conf": float(b.conf[0]),
                        "bbox": [x1, y1, x2, y2]
                    }
