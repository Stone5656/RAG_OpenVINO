# logger_factory.py
import logging
from pathlib import Path
from .level_mapper import map_level
from .custom_formatter import build_stream_handler, build_file_handler

def get_logger(
    name: str,
    log_file: Path | None = None,
    level: int | str = logging.INFO,
    propagate: bool = False,
) -> logging.Logger:
    """標準出力(色) + 任意でファイル出力。重複ハンドラ防止対応。"""
    numeric_level = map_level(level)
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    logger.propagate = propagate  # 子→親への伝播を止める

    # 既存ハンドラの重複を避ける（外部で付与されたものは残したい場合は条件分岐でもOK）
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(
        build_stream_handler(
            level=numeric_level
        )
    )
    
    if log_file:
        logger.addHandler(
            build_file_handler(
                log_file=str(log_file),
                level=numeric_level,
            )
        )
    return logger

# 既存互換エイリアス
LoggerFactoryImpl = get_logger
