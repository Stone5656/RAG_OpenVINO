# -*- coding: utf-8 -*-
"""
テキスト分割モジュール

- 役割:
    * 読み込んだ Document 配列をチャンク分割する。
- 依存:
    * langchain.text_splitter.RecursiveCharacterTextSplitter
- 根拠:
    * 既存コードが RecursiveCharacterTextSplitter を採用:contentReference[oaicite:11]{index=11}（具体実装: :contentReference[oaicite:12]{index=12}）
"""
from __future__ import annotations

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def split_documents(
    documents: List[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
