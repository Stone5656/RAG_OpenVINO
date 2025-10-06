# -*- coding: utf-8 -*-
"""
Session State 管理モジュール

責務:
- Streamlit / FastAPI などの環境に依存せずに「会話履歴」や「設定」を保持
- スレッドセーフな in-memory 記録
- RAG の検証や簡易対話テストにも利用可能

ログ出力:
- 履歴の追加・削除・クリア時に DEBUG ログ
"""
from __future__ import annotations
from rag_openvino_app.utils.logger_utils import with_logger  # ★ get_logger は使わない


class SessionState:
    """シンプルな会話状態マネージャ。"""

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def __init__(self, *, logger=None):
        self.history: list[dict[str, object]] = []
        self.context: dict[str, object] = {}
        logger.debug("SessionState: 初期化しました。")

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def add_message(self, role: str, content: str, *, logger=None) -> None:
        """履歴に1件追加。"""
        self.history.append({"role": role, "content": content})
        logger.debug("SessionState: メッセージ追加 (%s) len=%d", role, len(self.history))

    def get_history(self) -> list[dict[str, str]]:
        """履歴を取得（コピー返却）。"""
        return list(self.history)

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def clear_history(self) -> None:
        """履歴を全消去。"""
        self.history.clear()

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def set(self, key: str, value: object, *, logger=None) -> None:
        """任意の設定値を保存。"""
        self.context[key] = value
        logger.debug("SessionState: コンテキスト設定 [%s]=%s", key, value)

    def get(self, key: str, default: object = None) -> object:
        """設定値を取得。"""
        return self.context.get(key, default)

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def clear(self, *, logger=None) -> None:
        """全状態をリセット。"""
        self.history.clear()
        self.context.clear()
        logger.debug("SessionState: 全状態をリセットしました。")
