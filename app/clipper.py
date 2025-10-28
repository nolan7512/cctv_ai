# app/clipper.py
import os
import subprocess

def make_gemini_lite(src_path: str, dst_path: str, scale_short_side: int = 720, crf: int = 30):
    """
    Re-encode nhanh cho Gemini: H.264, preset ultrafast, faststart.
    CRF 28–32 là hợp lý. Mặc định 30 để nhẹ CPU.
    Có thể override qua ENV:
      - FFMPEG_PRESET (mặc định: ultrafast)
      - FFMPEG_TUNE   (mặc định: zerolatency)
    """
    preset = os.getenv("FFMPEG_PRESET", "ultrafast")
    tune   = os.getenv("FFMPEG_TUNE",   "zerolatency")

    # scale theo cạnh ngắn giữ nguyên tỷ lệ
    vf = (
        f"scale='if(gt(iw,ih),{scale_short_side},-2)':'if(gt(iw,ih),-2,{scale_short_side})':"
        "flags=bicubic"
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", src_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", preset, "-tune", tune, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an",
        dst_path,
    ]
    subprocess.run(cmd, check=True)
