# ner_openvino/utils/logger_utils/logger_injector.py
from __future__ import annotations
import os, logging, functools, inspect
from pathlib import Path
from typing import Callable

# あなたの実装に合わせて適切な import に変えてください
from .logger_factory import LoggerFactoryImpl
from .level_mapper import map_level  # "INFO"→logging.INFO など


def _resolve_logger(name: str, log_file: Path | None, level: int | str) -> logging.Logger:
    """既存ロガー優先で Logger を解決し、なければファクトリで生成して返す。"""
    existing = logging.getLogger(name)
    if existing.handlers:   # 既にどこかでハンドラ設定済み＝“既存のロガー”
        return existing
    return LoggerFactoryImpl(name, log_file=log_file, level=level)


def _resolve_log_path_from_env(log_path: str | None = "LOGPATH") -> Path:
    """LOGPATH を優先してログファイルの Path を返す。未設定なら logs/rag_openvino_app.log。
    - 親ディレクトリは `parents=True, exist_ok=True` で必ず作成。
    - LOGPATH がディレクトリ（既存）または区切りで終わる場合は 'rag_openvino_app.log' を付与。
    """
    env = os.getenv(log_path)
    if env and env.strip():
        path = Path(env).expanduser()
        if (path.exists() and path.is_dir()) or str(env).endswith(("/", "\\")):
            path = path / "rag_openvino_app.log"
    else:
        path = Path("logs/rag_openvino_app.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def with_logger(
    name: str,
    env_log_path: str | None = None,  # <— 後方互換のため残置（現在は無視）
    env_log_level: str = "LOG_LEVEL",
) -> Callable:
    """ロガー依存を「見せずに」注入するデコレーター。

    変更点:
        - ログファイルの場所は LOGPATH 環境変数を参照。未設定なら Path("logs/rag_openvino_app.log")。
        - 必要なディレクトリは自動作成。

    注意:
        - `log_file` 引数は後方互換で残していますが、現在は無視されます。
          明示的に切り替えたい場合は環境変数 LOGPATH を設定してください。
    """
    def deco(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            effective_log_file = _resolve_log_path_from_env(env_log_path)
            params = inspect.signature(func).parameters

            # 1) 関数が logger 引数を受け取れる場合は kwargs で注入
            if "logger" in params and "logger" not in kwargs:
                level = map_level(os.getenv(env_log_level, "INFO"))
                kwargs["logger"] = _resolve_logger(name, effective_log_file, level)
                return func(*args, **kwargs)

            # 2) 受け取らない関数なら、関数が参照するモジュールグローバルに挿す
            global_scope = func.__globals__
            if global_scope.get("logger", None) is None:
                level = map_level(os.getenv(env_log_level, "INFO"))
                global_scope["logger"] = _resolve_logger(name, effective_log_file, level)

            return func(*args, **kwargs)
        return wrapper
    return deco
