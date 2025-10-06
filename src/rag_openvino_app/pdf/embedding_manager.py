# -*- coding: utf-8 -*-
"""
PDF 埋め込みマネージャ

責務:
- チャンク列から埋め込みを生成し、ベクトルストアへ投入できる形へ整形
- 埋め込みモデルの抽象 API は `.embed(list[str]) -> np.ndarray` を想定（RAG 側の retriever と同一）

戻り値（例）:
[
  {"text": <chunk>, "embedding": <np.ndarray(d,)>, "meta": {"source": "...", "page": 1, ...}},
  ...
]
"""
from __future__ import annotations
import numpy as np
from rag_openvino_app.utils.logger_utils import with_logger


class EmbeddingModelLike:
    """型ヒント用のごく簡単な疑似クラス（.embed を持つことを示す）。"""
    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class PDFEmbeddingManager:
    """PDF チャンクから埋め込みを作成し、RAG で使いやすい dict へ整形するヘルパ。"""

    def __init__(self, embed_model: EmbeddingModelLike):
        self.embed_model = embed_model

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def build_embeddings(
        self,
        chunks: list[str],
        base_meta: dict[str, object] | None = None,
        *,
        logger=None,
    ) -> list[dict[str, object]]:
        """
        チャンク列から埋め込みを生成。

        Parameters
        ----------
        chunks : list[str]
            分割済みテキストの配列
        base_meta : dict | None
            すべてのチャンクに共通して付与するメタ（source など）

        Returns
        -------
        list[dict] : {"text", "embedding", "meta"} の配列
        """
        base_meta = dict(base_meta or {})
        if not chunks:
            logger.debug("EmbeddingManager: 空チャンクのため埋め込み生成をスキップします。")
            return []

        logger.debug("EmbeddingManager: 埋め込み生成を開始（%d チャンク）", len(chunks))
        embs = self.embed_model.embed(chunks)  # (N, d)
        out: list[dict[str, object]] = []
        for i, (t, e) in enumerate(zip(chunks, embs)):
            meta = dict(base_meta)
            meta["chunk_index"] = i
            out.append({"text": t, "embedding": e, "meta": meta})
        logger.debug("EmbeddingManager: 埋め込み生成が完了（%d 件）", len(out))
        return out
