# app/main.py
import argparse, time, os, threading
from pathlib import Path
from collections import defaultdict, deque

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
from app.tts import TextToSpeech   # <<== NEW

def _spawn_gemini_worker(gem, tele, cam_name, clip_path, logger, skip_no_activity=True,
                         tts: TextToSpeech | None = None, tts_on_summary: bool = False,
                         clip_dir: str = "."):
    """Chạy Gemini ở nền; nếu có summary thì gửi text + (tuỳ chọn) voice."""
    def _work():
        t2 = time.perf_counter()
        try:
            summary = gem.analyze_video(clip_path)  # "NO_ACTIVITY" hoặc mô tả ngắn
            dt = time.perf_counter() - t2
            logger.info(f"[{cam_name}] T_gemini={dt:.2f}s")
            if skip_no_activity and summary.strip().upper() == "NO_ACTIVITY":
                logger.info(f"[{cam_name}] Gemini: NO_ACTIVITY → skip notify")
                return
            # gửi text
            text = f"🎥 [{cam_name}] {time.strftime('%Y-%m-%d %H:%M:%S')}\n{summary}"
            tele.send_text(text)
            logger.info(f"[{cam_name}] Gemini summary sent (async)")

            # gửi voice (tuỳ chọn)
            if tts and tts_on_summary:
                try:
                    ogg_path = str(Path(clip_dir) / f"{cam_name}_{int(time.time())}_sum.ogg")
                    tts.speak_to_ogg(summary, ogg_path)
                    tele.send_voice(ogg_path)
                    logger.info(f"[{cam_name}] voice(summary) sent")
                except Exception as e:
                    logger.warning(f"[{cam_name}] TTS(summary) failed: {e}")
        except Exception as e:
            logger.exception(f"[{cam_name}] Gemini async failed: {e}")
            try:
                tele.send_text(f"⚠️[{cam_name}] Gemini failed: {e}")
            except Exception:
                pass
    th = threading.Thread(target=_work, daemon=True)
    th.start()


def _spawn_tts_voice(tts: TextToSpeech, tele: TelegramClient, cam_name: str,
                     text: str, out_dir: str, logger):
    def _work():
        try:
            ogg_path = str(Path(out_dir) / f"{cam_name}_{int(time.time())}_immed.ogg")
            tts.speak_to_ogg(text, ogg_path)
            tele.send_voice(ogg_path)
            logger.info(f"[{cam_name}] voice(immediate) sent")
        except Exception as e:
            logger.warning(f"[{cam_name}] TTS(immediate) failed: {e}")
    th = threading.Thread(target=_work, daemon=True)
    th.start()


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

    # Telegram
    chat_param = getattr(cfg, "telegram_chat", None) or getattr(cfg, "telegram_chat_id", None)
    tele = TelegramClient(cfg.telegram_token, chat_param, max_mb=cfg.max_telegram_mb)

    # Gemini
    gem = GeminiClient(cfg.gemini_api_key, cfg.gemini_model, use_vertex=cfg.use_vertex)

    # Gom sự kiện
    merger = EventMerger(merge_window=cfg.merge_window_seconds, cooldown=cfg.cooldown_seconds)

    # Nguồn AI ưu tiên SUB
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

    # Housekeeping cấu hình
    last_hk = 0.0
    HK_INTERVAL_SEC      = int(os.getenv("HK_INTERVAL_SEC", "3600"))
    BUFFER_MAX_FILES     = int(os.getenv("BUFFER_MAX_FILES", str(cfg.wrap_segments)))
    CLIPS_RETENTION_DAYS = int(os.getenv("CLIPS_RETENTION_DAYS", "3"))
    CLIPS_MAX_GB         = float(os.getenv("CLIPS_MAX_GB", "0"))

    # Debounce
    CONFIRM_FRAMES    = int(os.getenv("CONFIRM_FRAMES", "3"))
    CONFIRM_WINDOW    = float(os.getenv("CONFIRM_WINDOW", "0.8"))
    MIN_EVENT_SECONDS = float(os.getenv("MIN_EVENT_SECONDS", "1.0"))
    _recent = defaultdict(lambda: deque())

    # Blackout & NO_ACTIVITY
    SKIP_NO_ACTIVITY = os.getenv("SKIP_NO_ACTIVITY", "yes").lower() == "yes"
    POST_EVENT_SILENCE_SEC = float(os.getenv("POST_EVENT_SILENCE_SEC", "8"))
    next_armed_ts = 0.0

    # Gửi ngay & Gemini async
    SEND_IMMEDIATE = os.getenv("SEND_IMMEDIATE", "yes").lower() == "yes"
    GEMINI_ENABLE  = os.getenv("GEMINI_ENABLE",  "yes").lower() == "yes"
    GEMINI_ASYNC   = os.getenv("GEMINI_ASYNC",   "yes").lower() == "yes"

    # Clamp t1
    CLIP_SAFETY_LAG = float(os.getenv("CLIP_SAFETY_LAG", "1.5"))

    # === TTS config ===
    TTS_ENABLE       = os.getenv("TTS_ENABLE", "yes").lower() == "yes"
    TTS_ENGINE       = os.getenv("TTS_ENGINE", "pyttsx3")
    TTS_VOICE        = os.getenv("TTS_VOICE", "")         # ví dụ "Vietnamese" nếu hệ thống có giọng VI
    TTS_RATE         = int(os.getenv("TTS_RATE", "180"))
    TTS_VOLUME       = float(os.getenv("TTS_VOLUME", "1.0"))
    TTS_ON_IMMEDIATE = os.getenv("TTS_ON_IMMEDIATE", "yes").lower() == "yes"
    TTS_ON_SUMMARY   = os.getenv("TTS_ON_SUMMARY", "yes").lower() == "yes"

    tts = None
    if TTS_ENABLE:
        try:
            tts = TextToSpeech(engine_name=TTS_ENGINE, voice_substr=TTS_VOICE,
                               rate=TTS_RATE, volume=TTS_VOLUME)
            logger.info(f"[{cfg.name}] TTS ready (engine={TTS_ENGINE}, voice='{TTS_VOICE or 'default'}')")
        except Exception as e:
            logger.warning(f"[{cfg.name}] TTS init failed: {e}")
            tts = None

    try:
        for det in detector.stream_detect(src_ai, motion_gate=motion):
            now = time.time()
            if now < next_armed_ts:
                continue

            # Housekeeping
            if now - last_hk >= HK_INTERVAL_SEC:
                total, removed = purge_oldest_by_count(cfg.buffer_dir, keep=BUFFER_MAX_FILES)
                if removed:
                    logger.info(f"[{cfg.name}] HK buffer: kept {BUFFER_MAX_FILES}/{total}, removed={removed}")
                rem_old = purge_older_than(cfg.clip_dir, days=CLIPS_RETENTION_DAYS)
                if rem_old:
                    logger.info(f"[{cfg.name}] HK clips: removed {rem_old} old files (> {CLIPS_RETENTION_DAYS}d)")
                if CLIPS_MAX_GB > 0:
                    size_gb, rem_sz = purge_until_size(cfg.clip_dir, max_gb=CLIPS_MAX_GB)
                    if rem_sz:
                        logger.info(f"[{cfg.name}] HK clips size: now {size_gb:.2f} GB (removed {rem_sz})")
                last_hk = now

            # Log detection
            if LOG_DETECTION:
                logger.info(f"[{cfg.name}] DET {det['class']} conf={det['conf']:.2f} bbox={det['bbox']}")
            else:
                logger.debug(f"[{cfg.name}] det {det['class']} {det['conf']:.2f}")

            ts = det["ts"]; cls = det["class"]

            # Debounce
            dq = _recent[cls]; dq.append(ts)
            while dq and ts - dq[0] > CONFIRM_WINDOW:
                dq.popleft()
            if len(dq) < CONFIRM_FRAMES:
                continue

            # Merge
            emitted = merger.push(cls, ts)
            if emitted is None:
                emitted = merger.flush_due(ts)
                if emitted is None:
                    continue

            t_first, t_last, ev = emitted
            dur = max(0.0, t_last - t_first)
            logger.info(f"[{cfg.name}] EVENT {ev.cls} x{ev.count} window={dur:.1f}s")

            if dur < MIN_EVENT_SECONDS:
                logger.info(f"[{cfg.name}] skip short event (<{MIN_EVENT_SECONDS}s)")
                next_armed_ts = time.time() + min(POST_EVENT_SILENCE_SEC, 2.0)
                continue

            next_armed_ts = time.time() + POST_EVENT_SILENCE_SEC

            # Tính cửa sổ clip
            t0 = max(0.0, t_first - cfg.pre_roll)
            t1_req = t_last + cfg.post_roll
            now_ts = time.time()
            t1 = min(t1_req, now_ts - CLIP_SAFETY_LAG)
            if t1 <= t0:
                logger.info(f"[{cfg.name}] clip window empty after clamp: t0={t0:.2f}, t1={t1:.2f}, now={now_ts:.2f}")
                continue

            clip_full = str(Path(cfg.clip_dir) / f"{cfg.name}_event_{int(t_first)}.mp4")

            # Cắt clip
            t_clip = time.perf_counter()
            try:
                logger.info(f"[{cfg.name}] clip_window=[{t0:.2f},{t1:.2f}] (req_end={t1_req:.2f}, now={now_ts:.2f})")
                rb.make_clip(t0, t1, clip_full)
                logger.info(f"[{cfg.name}] Clip OK → {clip_full}")
            except Exception as e:
                logger.exception(f"[{cfg.name}] Clip failed")
                try:
                    tele.send_text(f"❗️[{cfg.name}] Clip failed: {e}")
                except Exception:
                    pass
                continue
            logger.info(f"[{cfg.name}] T_clip={(time.perf_counter()-t_clip):.2f}s")

            # Gửi NGAY text + (tuỳ chọn) voice
            if SEND_IMMEDIATE:
                prelim = f"Sự kiện: {ev.cls} x{ev.count} ({dur:.1f}s) — đang phân tích…"
                try:
                    tele.send_text(f"🔔 [{cfg.name}] {prelim}")
                except Exception:
                    pass
                if tts and TTS_ON_IMMEDIATE:
                    _spawn_tts_voice(tts, tele, cfg.name, prelim, cfg.clip_dir, logger)

            # Nén nhẹ cho Gemini
            clip_lite = str(Path(cfg.clip_dir) / f"{cfg.name}_event_{int(t_first)}_720p.mp4")
            t_comp = time.perf_counter()
            try:
                make_gemini_lite(clip_full, clip_lite, scale_short_side=720, crf=30)
                logger.info(f"[{cfg.name}] Compress OK → {clip_lite}")
            except Exception as e:
                logger.warning(f"[{cfg.name}] Compress failed → dùng full. err={e}")
                try:
                    tele.send_text(f"⚠️[{cfg.name}] Compress failed: {e}")
                except Exception:
                    pass
                clip_lite = clip_full
            logger.info(f"[{cfg.name}] T_compress={(time.perf_counter()-t_comp):.2f}s")

            # Gemini (async/sync) + (tuỳ chọn) voice summary
            if GEMINI_ENABLE:
                if GEMINI_ASYNC:
                    _spawn_gemini_worker(
                        gem, tele, cfg.name, clip_lite, logger,
                        skip_no_activity=SKIP_NO_ACTIVITY,
                        tts=tts, tts_on_summary=TTS_ON_SUMMARY, clip_dir=cfg.clip_dir
                    )
                else:
                    t2 = time.perf_counter()
                    try:
                        summary = gem.analyze_video(clip_lite)
                        logger.info(f"[{cfg.name}] T_gemini={(time.perf_counter()-t2):.2f}s")
                        if SKIP_NO_ACTIVITY and summary.strip().upper() == "NO_ACTIVITY":
                            logger.info(f"[{cfg.name}] Gemini: NO_ACTIVITY → skip notify")
                        else:
                            msg = f"🎥 [{cfg.name}] {time.strftime('%Y-%m-%d %H:%M:%S')}\n{summary}"
                            tele.send_text(msg)
                            if tts and TTS_ON_SUMMARY:
                                _spawn_tts_voice(tts, tele, cfg.name, summary, cfg.clip_dir, logger)
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
                    tele.send_video(clip_full, caption=f"{cfg.name} Clip giám sát")
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
