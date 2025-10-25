import argparse, time
from pathlib import Path

from app.config import load_env, Config
from app.ringbuffer import FFmpegRingBuffer
from app.motion import MotionGate
from app.detector import ObjectDetector
from app.event_merger import EventMerger
from app.clipper import make_gemini_lite
from app.gemini_client import GeminiClient
from app.telegram_client import TelegramClient


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", help="path to .env file for this camera", default=None)
    args = ap.parse_args()

    load_env(args.env)
    cfg = Config()
    cfg.validate()

    Path(cfg.clip_dir).mkdir(parents=True, exist_ok=True)

    # Start ring buffer on MAIN
    rb = FFmpegRingBuffer(cfg.rtsp_url, cfg.buffer_dir, cfg.segment_seconds, cfg.wrap_segments)
    rb.start()

    # Motion gate on SUB stream
    motion = MotionGate(min_pixels=cfg.motion_min_pixels, ratio=cfg.motion_ratio)

    detector = ObjectDetector(
        conf=cfg.conf, frame_stride=cfg.frame_stride,
        min_area=cfg.min_bbox_area, classes_of_interest=cfg.objects_of_interest,
        yolo_model=cfg.yolo_model, imgsz=cfg.yolo_imgsz,
        device_pref=cfg.device_pref, use_fp16=cfg.use_fp16
    )

    tele = TelegramClient(cfg.telegram_token, cfg.telegram_chat_id, max_mb=cfg.max_telegram_mb)
    gem = GeminiClient(cfg.gemini_api_key, cfg.gemini_model, use_vertex=cfg.use_vertex)

    merger = EventMerger(merge_window=cfg.merge_window_seconds, cooldown=cfg.cooldown_seconds)

    src_ai = cfg.rtsp_url_sub or cfg.rtsp_url
    tele.send_text(f"✅ [{cfg.name}] started. Watching AI stream; ring-buffer active. Device={detector.device} FP16={detector.use_fp16}")

    try:
        for det in detector.stream_detect(src_ai, motion_gate=motion):
            ts = det["ts"]
            cls = det["class"]

            # Try to merge events
            emitted = merger.push(cls, ts)
            if emitted is None:
                # Maybe flush overdue events
                emitted = merger.flush_due(ts)
                if emitted is None:
                    continue

            t_first, t_last, ev = emitted
            # Compute clip window on MAIN
            t0 = max(0.0, t_first - cfg.pre_roll)
            t1 = t_last + cfg.post_roll
            clip_full = str(Path(cfg.clip_dir) / f"{cfg.name}_event_{int(t_first)}.mp4")

            try:
                rb.make_clip(t0, t1, clip_full)
            except Exception as e:
                tele.send_text(f"❗️[{cfg.name}] Clip failed: {e}")
                continue

            # Gemini lite clip
            clip_lite = str(Path(cfg.clip_dir) / f"{cfg.name}_event_{int(t_first)}_720p.mp4")
            try:
                make_gemini_lite(clip_full, clip_lite, scale_short_side=720, crf=28)
            except Exception as e:
                tele.send_text(f"⚠️[{cfg.name}] Compress failed: {e}")
                clip_lite = clip_full

            # Gemini analysis
            try:
                result = gem.analyze_video(clip_lite)
                summary = result.get("summary") or "(no summary)"
                objects = ", ".join(result.get("objects", []))
                incident = result.get("incident", "")
                txt = (
                    f"🎥 [{cfg.name}] {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"• Objects: {objects}\n"
                    f"• Incident: {incident}\n"
                    f"• Summary: {summary}"
                )
                tele.send_text(txt)
            except Exception as e:
                tele.send_text(f"⚠️[{cfg.name}] Gemini failed: {e}")

            # Send video (optional)
            if cfg.send_video:
                try:
                    tele.send_video(clip_full, caption=f"{cfg.name} event clip")
                except Exception as e:
                    tele.send_text(f"⚠️[{cfg.name}] Telegram video failed: {e}")

    except KeyboardInterrupt:
        pass
    finally:
        rb.stop()
        tele.send_text(f"🛑 [{cfg.name}] stopped.")


if __name__ == "__main__":
    run()
