# level_mapper.py
import logging
from .constants import LEVEL_MAP

def map_level(level: str | int) -> int:
    """文字列/数値/未知入力を logging の数値レベルに正規化"""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return LEVEL_MAP.get(level.upper(), logging.INFO)
    # それ以外は既定 INFO
    return logging.INFO
