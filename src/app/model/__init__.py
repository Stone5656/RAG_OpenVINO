# -*- coding: utf-8 -*-
"""
モデル管理パッケージ初期化モジュール

- LlamaCppや将来追加されるLLMのロード・初期化・設定を担います。
- 各アプリ（Streamlit, CLI, API）からは、本パッケージのファクトリ関数を通じてLLMを取得してください。
"""
from __future__ import annotations
