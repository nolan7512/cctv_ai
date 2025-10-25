from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Dict, Tuple

@dataclass
class Event:
    cls: str
    ts_first: float
    ts_last: float
    count: int = 1

class EventMerger:
    """Merge rapid detections of same class into one event using a time window.
    Also applies cooldown per class to avoid spamming.
    """
    def __init__(self, merge_window: float = 4.0, cooldown: float = 12.0):
        self.merge_window = merge_window
        self.cooldown = cooldown
        self.active: Dict[str, Event] = {}
        self.last_emitted: Dict[str, float] = defaultdict(lambda: 0.0)

    def push(self, cls: str, ts: float) -> Optional[Tuple[float, float, Event]]:
        ev = self.active.get(cls)
        if ev and ts - ev.ts_last <= self.merge_window:
            ev.ts_last = ts
            ev.count += 1
            self.active[cls] = ev
            return None
        emit = None
        if ev:
            emit = (ev.ts_first, ev.ts_last, ev)
        if ts - self.last_emitted[cls] >= self.cooldown:
            self.active[cls] = Event(cls=cls, ts_first=ts, ts_last=ts)
        else:
            self.active.pop(cls, None)
        return emit

    def flush_due(self, now: float) -> Optional[Tuple[float, float, Event]]:
        for cls, ev in list(self.active.items()):
            if now - ev.ts_last > self.merge_window:
                self.last_emitted[cls] = ev.ts_last
                self.active.pop(cls, None)
                return (ev.ts_first, ev.ts_last, ev)
        return None
