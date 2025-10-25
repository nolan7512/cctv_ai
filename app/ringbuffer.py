import subprocess, os, glob, datetime as dt
from pathlib import Path
from typing import List

class FFmpegRingBuffer:
    def __init__(self, rtsp_url: str, out_dir: str, segment_seconds: int, wrap_segments: int):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.rtsp = rtsp_url
        self.out = out_dir
        self.seg = segment_seconds
        self.wrap = wrap_segments
        self.proc: subprocess.Popen | None = None
        self.pattern = "%Y%m%d_%H%M%S.mp4"

    def start(self):
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-rtsp_transport", "tcp", "-i", self.rtsp,
            "-an", "-c:v", "copy",
            "-f", "segment", "-segment_time", str(self.seg),
            "-segment_wrap", str(self.wrap), "-reset_timestamps", "1",
            "-strftime", "1", os.path.join(self.out, self.pattern)
        ]
        self.proc = subprocess.Popen(cmd)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _list_segments(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.out, "*.mp4")))

    def make_clip(self, t0: float, t1: float, clip_path: str) -> str:
        from tempfile import NamedTemporaryFile
        Path(Path(clip_path).parent).mkdir(parents=True, exist_ok=True)
        segs = self._list_segments()
        if not segs:
            raise RuntimeError("No segments in buffer")

        def to_ts(p):
            base = os.path.basename(p).split(".")[0]
            dtobj = dt.datetime.strptime(base, "%Y%m%d_%H%M%S")
            return dtobj.timestamp()

        selected = []
        for s in segs:
            ts = to_ts(s)
            if ts + self.seg >= t0 and ts <= t1:
                selected.append(s)
        if not selected:
            raise RuntimeError("No segment overlaps detection window")

        with NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
            for s in selected:
                f.write(f"file '{os.path.abspath(s)}'\n")
            list_path = f.name

        # Try concat copy
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-safe", "0", "-f", "concat", "-i", list_path,
            "-c", "copy", clip_path
        ]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            # Fallback re-encode
            cmd = [
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                "-safe", "0", "-f", "concat", "-i", list_path,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-an", clip_path
            ]
            subprocess.check_call(cmd)
        return clip_path
