# -*- coding: utf-8 -*-
"""
Retriever: ベクトルDBからの候補取得（Top-k）＋（任意）MMR 再選定。

責務:
- クエリ埋め込みを作成
- ベクトルDBで類似検索（Top-k）
- （任意）MMR により冗長候補を抑制（関連性×多様性）
- ログにより「何件取得し、MMR で何件捨てたか」を可視化

設計ポイント:
- ベクトルDB・埋め込みモデルは抽象プロトコルで受け取り、実装に依存しない
- 返り値には「本文」だけでなく、source/path などのメタも保持することを想定
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np

from rag_openvino_app.utils.logger_utils import with_logger


class EmbeddingModel(Protocol):
    """クエリ/文書を埋め込みに変換するモデルのプロトコル。"""
    def embed(self, texts: list[str]) -> np.ndarray:  # shape: (N, d)
        ...


class VectorStore(Protocol):
    """ベクトルDBのプロトコル。最低限、類似検索ができればよい。"""
    def similarity_search(
        self, query_vec: np.ndarray, k: int
    ) -> list[dict[str, object]]:
        """
        Returns: list of documents (dict)
            期待キー: {"text": str, "embedding": np.ndarray, "meta": dict}
            - embedding は MMR で利用（なければ retriever 側で生成）
        """
        ...


@dataclass
class RetrieverConfig:
    k: int = 12
    use_mmr: bool = True
    mmr_lambda: float = 0.4  # 0:多様性〜1:関連性


class Retriever:
    """埋め込み類似 + MMR による候補選定を担うクラス。"""

    def __init__(self, embed_model: EmbeddingModel, vector_store: VectorStore, cfg: RetrieverConfig):
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.cfg = cfg

    def _mmr_select(self, query_vec: np.ndarray, candidates: list[dict[str, object]], top_n: int, lambda_mult: float) -> list[dict[str, object]]:
        """
        シンプルな MMR 実装。
        - candidates の各要素は {"text", "embedding"} を持つ想定
        """
        doc_vecs = np.vstack([c["embedding"] for c in candidates])
        def _norm(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
            return v / n

        q = _norm(query_vec.reshape(1, -1))
        dv = _norm(doc_vecs)
        rel_scores = (dv @ q.T).reshape(-1)  # cos sim

        selected: list[int] = []
        remaining = list(range(len(candidates)))

        while remaining and len(selected) < top_n:
            if not selected:
                i = int(np.argmax(rel_scores[remaining]))
                picked = remaining.pop(i)
                selected.append(picked)
                continue

            sel_mat = dv[selected]
            max_sim_to_S = (dv[remaining] @ sel_mat.T).max(axis=1)
            mmr_scores = lambda_mult * rel_scores[remaining] - (1 - lambda_mult) * max_sim_to_S
            i = int(np.argmax(mmr_scores))
            picked = remaining.pop(i)
            selected.append(picked)

        picked_docs = [candidates[i] for i in selected]
        return picked_docs

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,   # ← 追加：呼び出し側の一時的な上書き用
        logger=None,
    ) -> list[dict[str, object]]:
        """質問文から Top-k（＋MMR）文書を取得。"""
        k = int(top_k) if top_k is not None else int(self.cfg.k)

        logger.debug("Retriever: embedding query...")
        q_vec = self.embed_model.embed([question])[0]  # (d,)

        logger.debug("Retriever: vector DB similarity search (k=%d)", k)
        candidates = self.vector_store.similarity_search(q_vec, k)
        logger.debug("Retriever: got %d candidates", len(candidates))

        need_embed = [i for i, c in enumerate(candidates) if "embedding" not in c or c["embedding"] is None]
        if need_embed:
            logger.debug("Retriever: generating %d missing embeddings for candidates", len(need_embed))
            embeds = self.embed_model.embed([c["text"] for c in candidates])
            for i, emb in enumerate(embeds):
                candidates[i]["embedding"] = emb

        if self.cfg.use_mmr:
            logger.debug("Retriever: applying MMR (lambda=%.2f)", self.cfg.mmr_lambda)
            candidates = self._mmr_select(q_vec, candidates, top_n=len(candidates), lambda_mult=self.cfg.mmr_lambda)
            logger.debug("Retriever: MMR selected %d docs", len(candidates))

        return candidates
