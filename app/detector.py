# app/detector.py
from __future__ import annotations
import os, time
from typing import Generator, Dict, Iterable, Mapping, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class ObjectDetector:
    """
    Detector chỉ dùng PyTorch/Ultralytics (CPU/CUDA).
    - Hỗ trợ model .pt (PyTorch) và .engine (TensorRT – Ultralytics tải qua YOLO()).
    - Lọc theo:
        * conf (độ tin cậy)
        * frame_stride (bỏ bớt khung)
        * min_area (px^2) + per-class min_area
        * min_bbox_ratio (tỉ lệ bbox/frame)
        * ROI mask (ảnh nhị phân), yêu cầu tỉ lệ chồng lấn tối thiểu
    - Độ bền luồng:
        * Auto-reconnect RTSP khi treo/đứt với backoff
        * Tham số qua ENV:
            REOPEN_ON_STALL_SEC (mặc định 10)
            MAX_FAIL_READS      (100)
            REOPEN_BACKOFF_INIT (1.0)
            REOPEN_BACKOFF_MAX  (30.0)
    """

    def __init__(
        self,
        conf: float = 0.40,
        frame_stride: int = 2,
        min_area: int = 10_000,
        classes_of_interest: Iterable[str] = ("person",),
        yolo_model: str = "yolo11n.pt",
        imgsz: int = 640,
        device_pref: str = "auto",     # "auto" | "cuda" | "cpu"
        use_fp16: bool = True,

        # tuỳ chọn
        class_min_area: Optional[Mapping[str, int]] = None,  # {"person":14000, "cat":2200, ...}
        min_bbox_ratio: Optional[float] = None,               # ví dụ 0.002 = 0.2% diện tích khung
        roi_mask_path: Optional[str] = None,                  # đường dẫn ảnh PNG mask (đen/ trắng)
        min_roi_overlap: float = 0.3,                         # yêu cầu >= 30% bbox nằm trong ROI

        # watchdog / reconnect
        reopen_on_stall_sec: float = None,
        max_fail_reads: int = None,
        backoff_init: float = None,
        backoff_max: float = None,
    ):
        self.conf = float(conf)
        self.frame_stride = int(frame_stride)
        self.min_area = int(min_area)
        self.want = set(classes_of_interest)
        self.imgsz = int(imgsz)
        self.min_bbox_ratio = float(min_bbox_ratio) if min_bbox_ratio else None
        self.class_min_area = dict(class_min_area) if class_min_area else {}

        # chọn device
        if device_pref == "cuda" or (device_pref == "auto" and torch.cuda.is_available()):
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.use_fp16 = bool(use_fp16) and (self.device == "cuda")

        # nạp model (PyTorch/TensorRT). Ultralytics tự định backend.
        self.model_path = yolo_model
        self.model = YOLO(self.model_path)
        self.names = self.model.names

        # chuyển model sang device và FP16 (nếu được)
        try:
            self.model.to(self.device)
            if self.use_fp16 and hasattr(self.model, "model") and hasattr(self.model.model, "half"):
                self.model.model.half()
        except Exception:
            pass  # không chặn nếu không set được

        # ROI mask (nếu có)
        self._roi_mask_raw: Optional[np.ndarray] = None
        self._roi_mask_resized: Optional[np.ndarray] = None
        self._roi_shape: Optional[tuple[int, int]] = None  # (h, w)
        self.min_roi_overlap = float(min_roi_overlap)
        if roi_mask_path:
            mask = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Cannot read ROI mask: {roi_mask_path}")
            mask = (mask > 0).astype(np.uint8)
            self._roi_mask_raw = mask

        # Watchdog / reconnect params
        self.reopen_on_stall_sec = float(
            reopen_on_stall_sec if reopen_on_stall_sec is not None else os.getenv("REOPEN_ON_STALL_SEC", "10")
        )
        self.max_fail_reads = int(
            max_fail_reads if max_fail_reads is not None else os.getenv("MAX_FAIL_READS", "100")
        )
        self.backoff_init = float(
            backoff_init if backoff_init is not None else os.getenv("REOPEN_BACKOFF_INIT", "1.0")
        )
        self.backoff_max = float(
            backoff_max if backoff_max is not None else os.getenv("REOPEN_BACKOFF_MAX", "30.0")
        )

        self.last_frame_ts = 0.0
        self._reopen_request = False  # có thể set từ ngoài nếu muốn ép reconnect

    # ------------- helpers -------------

    def request_reopen(self):
        """Cho phép luồng ngoài yêu cầu reconnect ở vòng lặp kế tiếp."""
        self._reopen_request = True

    def _open(self, url: str) -> cv2.VideoCapture:
        # Ưu tiên FFMPEG backend cho RTSP
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _ensure_roi_size(self, h: int, w: int):
        if self._roi_mask_raw is None:
            return
        if self._roi_shape == (h, w) and self._roi_mask_resized is not None:
            return
        self._roi_mask_resized = cv2.resize(self._roi_mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
        self._roi_shape = (h, w)

    def _bbox_area_thresh(self, cls: str, frame_w: int, frame_h: int) -> float:
        th = float(self.class_min_area.get(cls, self.min_area))
        if self.min_bbox_ratio:
            th = max(th, self.min_bbox_ratio * frame_w * frame_h)
        return th

    def _roi_overlap_ok(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        if self._roi_mask_resized is None:
            return True
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, self._roi_mask_resized.shape[1]), min(y2, self._roi_mask_resized.shape[0])
        if x2 <= x1 or y2 <= y1:
            return False
        crop = self._roi_mask_resized[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        inside = float(crop.sum())  # mask 0/1
        ratio = inside / crop.size
        return ratio >= self.min_roi_overlap

    # ------------- main loop -------------

    def stream_detect(self, rtsp_url: str, motion_gate=None) -> Generator[Dict, None, None]:
        """
        Yield dict detection đã qua filter (area/ROI/…).
        Có auto-reconnect khi luồng treo/đứt.
        """
        def reopen(cap_old):
            try:
                if cap_old is not None:
                    cap_old.release()
            except Exception:
                pass
            return self._open(rtsp_url)

        cap = reopen(None)
        if not cap.isOpened():
            time.sleep(self.backoff_init)
            cap = reopen(cap)

        i = 0
        last_ok = time.time()
        fail_reads = 0
        backoff = self.backoff_init

        while True:
            ok, frame = cap.read()
            now = time.time()

            if not ok or frame is None:
                fail_reads += 1
                need_reopen = (
                    (now - last_ok) > self.reopen_on_stall_sec
                    or fail_reads >= self.max_fail_reads
                    or self._reopen_request
                )
                if need_reopen:
                    self._reopen_request = False
                    try:
                        cap.release()
                    except Exception:
                        pass
                    time.sleep(backoff)
                    cap = reopen(None)
                    last_ok = time.time()
                    fail_reads = 0
                    backoff = min(backoff * 2.0, self.backoff_max)
                    continue

                time.sleep(0.2)
                continue

            # có frame
            self.last_frame_ts = now
            last_ok = now
            fail_reads = 0
            backoff = self.backoff_init

            i += 1
            if self.frame_stride > 1 and (i % self.frame_stride):
                continue

            if motion_gate is not None and not motion_gate.is_motion(frame):
                continue

            h, w = frame.shape[:2]
            self._ensure_roi_size(h, w)

            # Suy luận
            results = self.model(frame, conf=self.conf, verbose=False, imgsz=self.imgsz, device=self.device)

            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    cls = self.names[cls_id]
                    if cls not in self.want:
                        continue

                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area < self._bbox_area_thresh(cls, w, h):
                        continue

                    if not self._roi_overlap_ok(x1, y1, x2, y2):
                        continue

                    yield {
                        "ts": time.time(),
                        "class": cls,
                        "conf": float(b.conf[0]),
                        "bbox": [x1, y1, x2, y2],
                    }
