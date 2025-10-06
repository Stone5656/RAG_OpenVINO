# -*- coding: utf-8 -*-
"""
モデルファクトリー。

設定辞書からモデル種別を判定し、対応する ModelManager を返します。
- "llama" : Transformers 系モデル
- "ov"    : OpenVINO 推論モデル
- "auto"  : 自動選択（利用可能なバックエンドを順に試す）

ログ出力（DEBUG）:
- 受け取った設定内容
- 選択したバックエンド
- 初期化エラー時のフォールバック状況
"""
from __future__ import annotations
from rag_openvino_app.utils.logger_utils import with_logger
from .base import BaseModelManager
from .llama_manager import LlamaManager
from .ov_manager import OVManager


@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def get_model_manager(config: dict[str, object], *, logger=None) -> BaseModelManager:
    """設定内容に応じて適切な ModelManager を返す。"""
    mtype = (config.get("type") or "auto").lower()
    logger.debug("モデルファクトリー: type=%s 設定=%s", mtype, {k: v for k, v in config.items() if k != "api_key"})

    if mtype == "llama":
        logger.debug("モデルファクトリー: Llama バックエンドを選択しました。")
        return LlamaManager(config)
    if mtype == "ov":
        logger.debug("モデルファクトリー: OpenVINO バックエンドを選択しました。")
        return OVManager(config)

    # 自動選択モード
    try:
        logger.debug("モデルファクトリー: 自動選択 → まず OpenVINO を試行します。")
        return OVManager(config)
    except Exception as e:
        logger.debug("モデルファクトリー: OpenVINO 初期化失敗 (%s) → Llama にフォールバックします。", e)

    logger.debug("モデルファクトリー: Llama バックエンドを使用します。")
    return LlamaManager(config)
