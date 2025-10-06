# -*- coding: utf-8 -*-
"""
Model management package.

本パッケージは、RAG の「生成段」を担う LLM 推論を抽象化します。
- BaseModelManager: 生成インターフェースの規定
- LlamaManager: HF/Transformers 等のローカル/サーバ LLM を想定した実装
- OVManager: OpenVINO での推論を想定した実装
- factory.get_model_manager(): 設定に応じたマネージャを生成

logger_utils により、各クラスは DEBUG ログを出力します。
"""

from .factory import get_model_manager  # noqa: F401
