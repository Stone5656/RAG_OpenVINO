# -*- coding: utf-8 -*-
"""
ベクトルストア・マネージャ（インメモリ簡易版）

責務:
- 埋め込みとメタを受け取り、インメモリで蓄積・検索
- まずは Numpy ベースの内積/コサイン検索の最小実装
- 永続化は簡易（.npz 保存/読込のダミー）を用意

注意:
- 実運用では Chroma/FAISS/Milvus/Qdrant 等の本格的ストアに置き換えてください。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from rag_openvino_app.utils.logger_utils import with_logger


@dataclass
class InMemoryVectorStore:
    """最小限のインメモリストア（コサイン類似）"""
    embeddings: np.ndarray | None = None       # shape: (N, d)
    metas: list[dict] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    normalized: bool = True

    def _ensure_array(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if not self.normalized:
            return x
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def add(self, embeddings: np.ndarray, metas: list[dict[str, object]], texts: list[str], *, logger=None) -> None:
        """ベクトル・メタ・本文を追記する。"""
        embeddings = self._ensure_array(embeddings)
        embeddings = self._normalize(embeddings)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self.metas.extend(metas)
        self.texts.extend(texts)
        logger.debug("VectorStore: 追加 %d 件（合計 %d）", embeddings.shape[0], len(self.texts))

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def add_documents(self, docs: list[dict[str, object]], *, logger=None) -> None:
        """{"text","embedding","meta"} の配列を受けて追加。"""
        if not docs:
            logger.debug("VectorStore: 追加対象なし。")
            return
        embs = np.vstack([d["embedding"] for d in docs])
        metas = [d.get("meta", {}) for d in docs]
        texts = [d["text"] for d in docs]
        self.add(embs, metas, texts)

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def similarity_search(self, query_vec: np.ndarray, k: int = 10, *, logger=None) -> list[dict[str, object]]:
        """コサイン類似度に基づく Top-k 近傍検索。"""
        if self.embeddings is None or self.embeddings.size == 0:
            logger.debug("VectorStore: 空インデックスのため検索結果は 0 件です。")
            return []

        query_vec = self._ensure_array(query_vec)
        query_vec = self._normalize(query_vec)

        sims = (self.embeddings @ query_vec.T).reshape(-1)  # cos sim
        idx = np.argsort(sims)[::-1][:k]

        results: list[dict[str, object]] = []
        for i in idx:
            results.append({
                "text": self.texts[i],
                "embedding": self.embeddings[i],
                "meta": self.metas[i] if i < len(self.metas) else {},
                "score": float(sims[i]),
            })
        logger.debug("VectorStore: 検索完了（k=%d, 返却=%d）", k, len(results))
        return results

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def save(self, path: str | Path, *, logger=None) -> None:
        """最小限の保存（npz）。メタとテキストは np.save で別保存。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), embeddings=self.embeddings if self.embeddings is not None else np.empty((0, 0)))
        np.save(path.with_suffix(".metas.npy"), np.array(self.metas, dtype=object), allow_pickle=True)
        np.save(path.with_suffix(".texts.npy"), np.array(self.texts, dtype=object), allow_pickle=True)
        logger.debug("VectorStore: 保存しました -> %s", path)

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def load(self, path: str | Path, *, logger=None) -> None:
        """最小限の読込。形状や空チェックは簡易化。"""
        path = Path(path)
        try:
            self.embeddings = np.load(path.with_suffix(".npz"))["embeddings"]
            self.metas = list(np.load(path.with_suffix(".metas.npy"), allow_pickle=True))
            self.texts = list(np.load(path.with_suffix(".texts.npy"), allow_pickle=True))
            logger.debug("VectorStore: 読み込みました <- %s", path)
        except Exception as e:
            logger.warning("VectorStore: 読み込みに失敗しました（%s）。空のストアとして初期化します。", e)
            self.embeddings = None
            self.metas = []
            self.texts = []
