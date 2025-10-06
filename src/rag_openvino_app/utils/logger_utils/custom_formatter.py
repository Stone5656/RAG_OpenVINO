# custom_formatter.py
import logging, sys
from .constants import COLOR_RESET, COLORS

class ColorFormatter(logging.Formatter):
    """色付き: レベル名だけ色付け。ファイル名/行番号付き"""
    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelno, COLOR_RESET)
        original = record.levelname
        record.levelname = f"{color}{original}{COLOR_RESET}"

        fmt = "%(name)s [%(levelname)s] (%(filename)s:%(lineno)d):\n%(message)s"
        formatter = logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S")
        out = formatter.format(record)

        record.levelname = original
        return out

def build_stream_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(ColorFormatter())
    return handler

def build_file_handler(log_file: str, level: int) -> logging.Handler:
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(level)
    fmt = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    handler.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
    return handler
