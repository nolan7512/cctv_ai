# app/event_merger.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

@dataclass
class _ActiveEvent:
    start_ts: float
    last_ts: float
    last_det_ts: float
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, cls: str, ts: float):
        self.last_ts = ts
        self.last_det_ts = ts
        self.counts[cls] = self.counts.get(cls, 0) + 1

    def top_cls(self) -> Tuple[str, int]:
        if not self.counts:
            return ("unknown", 0)
        cls = max(self.counts, key=self.counts.get)
        return cls, self.counts.get(cls, 0)

@dataclass
class _EventOut:
    cls: str
    count: int

class EventMerger:
    """
    Gộp detection thành 'sự kiện' dựa trên khoảng im lặng:
      - Event đang mở sẽ CHỈ đóng khi không có detection trong >= idle_end giây.
      - Sau khi đóng, áp dụng cooldown để chặn spam.
    Backward-compat:
      - Nếu truyền merge_window (kiểu cũ), sẽ coi là idle_end.
    """

    def __init__(self, idle_end: float = 4.0, cooldown: float = 6.0, merge_window: float | None = None):
        if merge_window is not None:  # tương thích ngược với mã cũ
            idle_end = float(merge_window)
        self.idle_end = float(idle_end)
        self.cooldown = float(cooldown)

        self._active: Optional[_ActiveEvent] = None
        self._cooldown_until: float = 0.0

    # ---- API cũ giữ nguyên: push() + flush_due(now) ----
    def push(self, cls: str, ts: float):
        """
        Thêm một detection vào merger. Không trả event ngay;
        event sẽ trả ở flush_due() khi đủ im lặng.
        """
        # đang cooldown → bỏ qua detection mới
        if ts < self._cooldown_until and self._active is None:
            return None

        if self._active is None:
            self._active = _ActiveEvent(start_ts=ts, last_ts=ts, last_det_ts=ts, counts={})
        self._active.add(cls, ts)
        return None

    def flush_due(self, now_ts: float):
        """
        Nếu đang có event mở và im lặng >= idle_end → đóng event và trả ra.
        Ngược lại trả None.
        """
        if self._active is None:
            return None

        gap = now_ts - self._active.last_det_ts
        if gap >= self.idle_end:
            ev = self._active
            self._active = None
            top_cls, top_cnt = ev.top_cls()
            out = _EventOut(cls=top_cls, count=top_cnt)
            t_first, t_last = ev.start_ts, ev.last_det_ts
            # bắt đầu cooldown sau khi đóng event
            self._cooldown_until = now_ts + self.cooldown
            return (t_first, t_last, out)

        return None

    # Tuỳ chọn: đóng cưỡng bức (dùng khi tắt chương trình)
    def force_close(self, now_ts: float):
        if self._active is None:
            return None
        ev = self._active
        self._active = None
        top_cls, top_cnt = ev.top_cls()
        out = _EventOut(cls=top_cls, count=top_cnt)
        self._cooldown_until = now_ts + self.cooldown
        return (ev.start_ts, ev.last_det_ts, out)
