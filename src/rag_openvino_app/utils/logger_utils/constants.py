# constants.py
import logging

COLOR_RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[36m",    # Cyan
    logging.INFO: "\033[32m",     # Green
    logging.WARNING: "\033[33m",  # Yellow
    logging.ERROR: "\033[31m",    # Red
    logging.CRITICAL: "\033[41m", # Red background
}

LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}
