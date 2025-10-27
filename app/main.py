# app/main.py
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
from app.housekeeping import (
    purge_oldest_by_count,
    purge_older_than,
    purge_until_size,
)


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

    # Gemini (trả về 1 đoạn Summary tiếng Việt)
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
            f"✅ [{cfg.name}] đã chạy, đang tiến hành giám sát"
            f"Device={getattr(detector,'device','cpu')} FP16={getattr(detector,'use_fp16',False)}"
        )
    except Exception as e:
        logger.warning(f"[{cfg.name}] Telegram start message failed: {e}")

    LOG_DETECTION = os.getenv("LOG_DETECTION", "no").lower() == "yes"

    # === Housekeeping cấu hình (mặc định theo yêu cầu: 1 giờ, giữ clip 3 ngày) ===
    last_hk = 0.0
    HK_INTERVAL_SEC      = int(os.getenv("HK_INTERVAL_SEC", "3600"))  # 1 giờ
    BUFFER_MAX_FILES     = int(os.getenv("BUFFER_MAX_FILES", str(cfg.wrap_segments)))
    CLIPS_RETENTION_DAYS = int(os.getenv("CLIPS_RETENTION_DAYS", "3"))  # 3 ngày
    # Đặt CLIPS_MAX_GB>0 để bật ép dung lượng tổng, 0 hoặc âm = tắt
    CLIPS_MAX_GB         = float(os.getenv("CLIPS_MAX_GB", "0"))

    try:
        for det in detector.stream_detect(src_ai, motion_gate=motion):
            # Housekeeping theo chu kỳ
            now = time.time()
            if now - last_hk >= HK_INTERVAL_SEC:
                # buffer/: giới hạn theo số file
                total, removed = purge_oldest_by_count(cfg.buffer_dir, keep=BUFFER_MAX_FILES)
                if removed:
                    logger.info(f"[{cfg.name}] HK buffer: kept {BUFFER_MAX_FILES}/{total}, removed={removed}")

                # clips/: xóa clip cũ theo ngày
                rem_old = purge_older_than(cfg.clip_dir, days=CLIPS_RETENTION_DAYS)
                if rem_old:
                    logger.info(f"[{cfg.name}] HK clips: removed {rem_old} old files (> {CLIPS_RETENTION_DAYS}d)")

                # clips/: ép tổng dung lượng (nếu bật)
                if CLIPS_MAX_GB > 0:
                    size_gb, rem_sz = purge_until_size(cfg.clip_dir, max_gb=CLIPS_MAX_GB)
                    if rem_sz:
                        logger.info(f"[{cfg.name}] HK clips size: now {size_gb:.2f} GB (removed {rem_sz})")

                last_hk = now

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

            # Gọi Gemini: chỉ lấy Summary
            try:
                summary = gem.analyze_video(clip_lite)  # trả về chuỗi tiếng Việt 1–3 câu
                txt = f"🎥 [{cfg.name}] {time.strftime('%Y-%m-%d %H:%M:%S')}\n{summary}"
                tele.send_text(txt)
                logger.info(f"[{cfg.name}] Gemini summary sent")
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
