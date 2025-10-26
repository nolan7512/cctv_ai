# app/logger.py
import logging, os, sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

_configured = False

def get_logger(name="cctv"):
    global _configured
    if not _configured:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        fmt = "%Y-%m-%d %H:%M:%S | %(levelname)s | %(name)s | %(message)s"
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt))

        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(getattr(logging, level, logging.INFO))

        # Optional: ghi ra file nếu có LOG_FILE
        log_file = os.getenv("LOG_FILE")
        if log_file:
            Path(Path(log_file).parent).mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(log_file, maxBytes=2*1024*1024, backupCount=3, encoding="utf-8")
            fh.setFormatter(logging.Formatter(fmt))
            root.addHandler(fh)

        _configured = True
    return logging.getLogger(name)
