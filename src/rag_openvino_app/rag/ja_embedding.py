# -*- coding: utf-8 -*-
"""
日本語（多言語）対応の Sentence-Transformers 埋め込み器
- intfloat/multilingual-e5-small を既定採用（.env で差し替え可）
- 正規化（L2）して np.ndarray を返す
"""
from __future__ import annotations
from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer
from rag_openvino_app.utils.logger_utils import with_logger


class SentenceTransformerEmb:
    """
    .embed(texts) -> np.ndarray(shape=(N, d))
    """
    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def __init__(self, model_id: str, *, device: str | None = None, logger=None):
        # device=None なら Sentence-Transformers 側が自動選択（'cuda' or 'cpu'）
        self.model = SentenceTransformer(model_id, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info("日本語埋め込みモデルをロード: %s (dim=%d, device=%s)", model_id, self.dim, device or "auto")

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def embed(self, texts: List[str], *, logger=None) -> np.ndarray:
        if not texts:
            return np.zeros((0, getattr(self, "dim", 0)), dtype=np.float32)
        # pooling/normalize は model.encode の引数でも可能だが、ここでは明示で正規化
        vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=False, convert_to_numpy=True)
        # L2 正規化
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms
        return vecs.astype(np.float32)
