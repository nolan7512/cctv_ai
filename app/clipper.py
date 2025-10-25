import subprocess
from pathlib import Path


def make_gemini_lite(in_path: str, out_path: str, scale_short_side: int = 720, crf: int = 28):
    Path(Path(out_path).parent).mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", in_path,
        "-vf", f"scale=-2:{scale_short_side}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-movflags", "+faststart", "-an", out_path
    ]
    subprocess.check_call(cmd)
    return out_path
