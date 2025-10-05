# -*- coding: utf-8 -*-
"""
ベクトルストア（FAISS）管理モジュール

- 役割:
    * ベクトルストアの生成・保存・読込
    * キャッシュディレクトリのハッシュ化命名
- 依存:
    * langchain_community.vectorstores.FAISS
    * pathlib.Path / hashlib
- 設計上の注意:
    * **FAISSはディレクトリ保存を推奨**。既存コードは ".faiss" というファイル名を作っていましたが
      `save_local/load_local` はディレクトリ指定のほうが互換/安定性が高い（本雛形は**ディレクトリ**採用）。
- 根拠:
    * 既存のキャッシュ生成方針（MD5で一意化）:contentReference[oaicite:15]{index=15}（具体実装: :contentReference[oaicite:16]{index=16}）
    * 既存で `FAISS.from_documents` → `save_local` → `load_local` の流れ:contentReference[oaicite:17]{index=17}（具体実装: :contentReference[oaicite:18]{index=18}, :contentReference[oaicite:19]{index=19}）
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import hashlib

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


def cache_dir_for(pdf_paths: Iterable[Path], base_dir: Path) -> Path:
    names = sorted(p.name for p in pdf_paths)
    digest = hashlib.md5("".join(names).encode()).hexdigest()
    return base_dir / f"{digest}.faiss_dir"  # ディレクトリ運用


def build_or_load_faiss(
    *,
    documents: Optional[list[Document]],
    embeddings: HuggingFaceEmbeddings,
    cache_dir: Path,
) -> FAISS:
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not documents:
            raise ValueError("documents is required to build a new FAISS index.")
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(str(cache_dir))
    else:
        db = FAISS.load_local(str(cache_dir), embeddings, allow_dangerous_deserialization=True)
    return db
