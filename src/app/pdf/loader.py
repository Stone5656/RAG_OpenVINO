# -*- coding: utf-8 -*-
"""
PDF ローダモジュール（PyMuPDFベース）

- 役割:
    * PDFファイル群を読み込み、LangChain ドキュメントの配列として返す。
- 依存:
    * langchain_community.document_loaders.PyMuPDFLoader
    * pathlib.Path
- 設計:
    * Path を受け取り、ローダには str で渡す（実装差への保険）。
- 根拠:
    * 既存コードで PyMuPDFLoader を採用:contentReference[oaicite:9]{index=9}（具体実装: :contentReference[oaicite:10]{index=10}）
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document


def load_pdfs(pdf_paths: Iterable[Path]) -> List[Document]:
    docs: list[Document] = []
    for p in pdf_paths:
        loader = PyMuPDFLoader(str(p))
        docs.extend(loader.load())
    return docs
