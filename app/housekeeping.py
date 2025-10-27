# app/housekeeping.py
"""
Dọn rác cho thư mục buffer/ và clips/.

- purge_oldest_by_count: giữ tối đa N file mới nhất (xóa phần còn lại)
- purge_older_than: xóa file cũ hơn N ngày
- purge_until_size: ép tổng dung lượng <= max_gb (xóa file cũ trước)
"""

import time
from pathlib import Path


def _size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.glob("**/*") if f.is_file()) / (1024 ** 3)


def purge_oldest_by_count(folder: str, keep: int, pattern: str = "*.mp4") -> tuple[int, int]:
    """
    Giữ tối đa 'keep' file mới nhất trong 'folder', xóa phần còn lại.
    Trả về (tổng_trước_khi_xóa, số_file_đã_xóa)
    """
    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)

    files = [f for f in d.glob(pattern) if f.is_file()]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)  # mới -> cũ

    removed = 0
    for f in files[keep:]:
        try:
            f.unlink()
            removed += 1
        except Exception:
            # tránh crash khi file đang bị ffmpeg nắm giữ
            pass
    return (len(files), removed)


def purge_older_than(folder: str, days: int, pattern: str = "*.mp4") -> int:
    """
    Xóa file trong 'folder' cũ hơn N ngày.
    Trả về số file đã xóa.
    """
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)

    removed = 0
    for f in d.glob(pattern):
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except Exception:
            pass
    return removed


def purge_until_size(folder: str, max_gb: float, pattern: str = "*.mp4") -> tuple[float, int]:
    """
    Xóa file cũ nhất dần cho tới khi tổng dung lượng <= max_gb.
    max_gb <= 0 sẽ không làm gì.
    Trả về (size_gb_sau_khi_xóa, số_file_đã_xóa)
    """
    if max_gb is None or max_gb <= 0:
        return (_size_gb(Path(folder)), 0)

    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)

    size = _size_gb(d)
    removed = 0
    if size <= max_gb:
        return (size, removed)

    files = [f for f in d.glob(pattern) if f.is_file()]
    files.sort(key=lambda f: f.stat().st_mtime)  # cũ -> mới

    for f in files:
        try:
            s_gb = f.stat().st_size / (1024 ** 3)
            f.unlink()
            removed += 1
            size -= s_gb
            if size <= max_gb:
                break
        except Exception:
            pass

    return (max(size, 0.0), removed)
