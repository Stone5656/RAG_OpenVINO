# -*- coding: utf-8 -*-
"""
埋め込みモデル管理モジュール

- 役割:
    * HuggingFace Embeddings のロード（再利用前提）
- 依存:
    * langchain_community.embeddings.HuggingFaceEmbeddings
- 設計:
    * 既存の `load_embeddings()` を UI 層から分離して再利用可能に。
- 根拠:
    * 既存 `load_embeddings` の存在と実装方針:contentReference[oaicite:13]{index=13}（具体実装: :contentReference[oaicite:14]{index=14}）
"""
from __future__ import annotations

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=2)
def load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    # Streamlit の cache でなくとも、アプリ側で複数回呼ばれても使い回せるようにする
    return HuggingFaceEmbeddings(model_name=model_name)
