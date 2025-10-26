import argparse, time, os
from pathlib import Path

from app.config import load_env, Config
from app.ringbuffer import FFmpegRingBuffer
from app.motion import MotionGate
from app.detector import ObjectDetector
from app.event_merger import EventMerger
from app.clipper import make_gemini_lite
from app.gemini_client import GeminiClient
from app.telegram_client import TelegramClient
from app.logger import get_logger


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", help="path to .env file for this camera", default=None)
    args = ap.parse_args()

    # Load .env rồi mới khởi tạo Config
    load_env(args.env)
    cfg = Config()
    cfg.validate()

    logger = get_logger("main")
    logger.info(f"[{cfg.name}] starting…")

    # Đảm bảo thư mục clip tồn tại
    Path(cfg.clip_dir).mkdir(parents=True, exist_ok=True)

    # Ring buffer trên MAIN (ghi -c copy)
    rb = FFmpegRingBuffer(cfg.rtsp_url, cfg.buffer_dir, cfg.segment_seconds, cfg.wrap_segments)
    rb.start()

    # Cổng chuyển động trên SUB (nhẹ)
    motion = MotionGate(min_pixels=cfg.motion_min_pixels, ratio=cfg.motion_ratio)

    # Detector (YOLO)
    detector = ObjectDetector(
        conf=cfg.conf, frame_stride=cfg.frame_stride,
        min_area=cfg.min_bbox_area, classes_of_interest=cfg.objects_of_interest,
        yolo_model=cfg.yolo_model, imgsz=cfg.yolo_imgsz,
        device_pref=cfg.device_pref, use_fp16=cfg.use_fp16
    )

    # Telegram: chấp nhận cả @ten_nhom (public) hoặc ID số âm (private)
    chat_param = getattr(cfg, "telegram_chat", None) or getattr(cfg, "telegram_chat_id", None)
    tele = TelegramClient(cfg.telegram_token, chat_param, max_mb=cfg.max_telegram_mb)

    # Gemini
    gem = GeminiClient(cfg.gemini_api_key, cfg.gemini_model, use_vertex=cfg.use_vertex)

    # Gom sự kiện
    merger = EventMerger(merge_window=cfg.merge_window_seconds, cooldown=cfg.cooldown_seconds)

    # Nguồn AI ưu tiên SUB, nếu không có sẽ rơi về MAIN
    src_ai = cfg.rtsp_url_sub or cfg.rtsp_url
    logger.info(
        f"[{cfg.name}] device={getattr(detector, 'device', 'cpu')} "
        f"fp16={getattr(detector, 'use_fp16', False)} src_ai={src_ai}"
    )

    # Thông báo start
    try:
        tele.send_text(
            f"✅ [{cfg.name}] started. Watching AI stream; ring-buffer active. "
            f"Device={getattr(detector,'device','cpu')} FP16={getattr(detector,'use_fp16',False)}"
        )
    except Exception as e:
        logger.warning(f"[{cfg.name}] Telegram start message failed: {e}")

    LOG_DETECTION = os.getenv("LOG_DETECTION", "no").lower() == "yes"

    try:
        for det in detector.stream_detect(src_ai, motion_gate=motion):
            # Log detection (nếu bật)
            if LOG_DETECTION:
                logger.info(f"[{cfg.name}] DET {det['class']} conf={det['conf']:.2f} bbox={det['bbox']}")
            else:
                logger.debug(f"[{cfg.name}] det {det['class']} {det['conf']:.2f}")

            ts = det["ts"]
            cls = det["class"]

            # Gom event
            emitted = merger.push(cls, ts)
            if emitted is None:
                emitted = merger.flush_due(ts)
                if emitted is None:
                    continue

            t_first, t_last, ev = emitted
            dur = max(0.0, t_last - t_first)
            logger.info(f"[{cfg.name}] EVENT {ev.cls} x{ev.count} window={dur:.1f}s")

            # Cửa sổ clip trên MAIN
            t0 = max(0.0, t_first - cfg.pre_roll)
            t1 = t_last + cfg.post_roll
            clip_full = str(Path(cfg.clip_dir) / f"{cfg.name}_event_{int(t_first)}.mp4")

            # Cắt clip
            try:
                rb.make_clip(t0, t1, clip_full)
                logger.info(f"[{cfg.name}] Clip OK → {clip_full}")
            except Exception as e:
                logger.exception(f"[{cfg.name}] Clip failed")
                try:
                    tele.send_text(f"❗️[{cfg.name}] Clip failed: {e}")
                except Exception:
                    pass
                continue

            # Nén nhẹ cho Gemini
            clip_lite = str(Path(cfg.clip_dir) / f"{cfg.name}_event_{int(t_first)}_720p.mp4")
            try:
                make_gemini_lite(clip_full, clip_lite, scale_short_side=720, crf=28)
                logger.info(f"[{cfg.name}] Compress OK → {clip_lite}")
            except Exception as e:
                logger.warning(f"[{cfg.name}] Compress failed → dùng full. err={e}")
                try:
                    tele.send_text(f"⚠️[{cfg.name}] Compress failed: {e}")
                except Exception:
                    pass
                clip_lite = clip_full

            # Gọi Gemini tóm tắt
            try:
                # Tạo hints dựa trên event gộp
                event_counts = {ev.cls: ev.count}   # ví dụ: {"person": 3}
                hints = {
                    "focus": [ev.cls],              # ưu tiên mô tả đúng class kích hoạt
                    "counts": event_counts,         # số lượng sơ bộ trong cửa sổ sự kiện
                    "want": list(cfg.objects_of_interest)  # các lớp bạn quan tâm trong .env
                }
                
                # Gemini analysis (mới)
                result = gem.analyze_video(clip_lite, hints=hints)
                summary = result.get("summary") or "(no summary)"
                objects = ", ".join(result.get("objects", []))
                incident = result.get("incident", "")

                # (tuỳ chọn) rút thêm thông tin từ JSON giàu chi tiết
                persons = result.get("persons") or {}
                vehicles = result.get("vehicles") or []
                animals = result.get("animals") or []

                extra_lines = []
                if persons:
                    c = persons.get("count")
                    acts = persons.get("actions") or []
                    riding = persons.get("riding") or []
                    if c is not None:
                        extra_lines.append(f"• Persons: {c} (acts: {', '.join(acts)[:80] or 'n/a'}; riding: {', '.join(riding) or 'no'})")
                if vehicles:
                    vlist = [f"{v.get('type','?')} x{v.get('count','?')} ({v.get('state','?')})" for v in vehicles[:4]]
                    if vlist:
                        extra_lines.append("• Vehicles: " + "; ".join(vlist))
                if animals:
                    alist = [f"{a.get('species','?')} x{a.get('count','?')}" for a in animals[:4]]
                    if alist:
                        extra_lines.append("• Animals: " + "; ".join(alist))

                txt = (
                    f"🎥 [{cfg.name}] {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"• Objects: {objects}\n"
                    f"• Incident: {incident}\n"
                    f"• Summary: {summary}\n" +
                    ("\n".join(extra_lines) if extra_lines else "")
                )
                tele.send_text(txt)
            except Exception as e:
                logger.exception(f"[{cfg.name}] Gemini failed")
                try:
                    tele.send_text(f"⚠️[{cfg.name}] Gemini failed: {e}")
                except Exception:
                    pass

            # Gửi video (nếu bật)
            if cfg.send_video:
                try:
                    tele.send_video(clip_full, caption=f"{cfg.name} event clip")
                    logger.info(f"[{cfg.name}] Telegram video sent")
                except Exception as e:
                    logger.exception(f"[{cfg.name}] Telegram video failed")
                    try:
                        tele.send_text(f"⚠️[{cfg.name}] Telegram video failed: {e}")
                    except Exception:
                        pass

    except KeyboardInterrupt:
        pass
    finally:
        rb.stop()
        logger.info(f"[{cfg.name}] stopped.")
        try:
            tele.send_text(f"🛑 [{cfg.name}] stopped.")
        except Exception:
            pass


if __name__ == "__main__":
    run()
