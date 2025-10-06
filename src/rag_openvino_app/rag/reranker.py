# -*- coding: utf-8 -*-
"""
Reranker: Cross-Encoder もしくは簡易スコアで Top-N を決定。

責務:
- Retriever の候補（Top-k）を受け取り、質問と文書のペアをスコアリング
- Cross-Encoder (BERT系) が利用できない場合は「簡易スコア（埋め込み cos 類似など）」で代替
- 上位 N 件を返す

設計:
- 実運用では HF Transformers の Cross-Encoder 推論を入れる
- 本実装は依存を減らすため、`scorer` コールバックを差し替え可能に
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable

from rag_openvino_app.utils.logger_utils import with_logger

ScoreFn = Callable[[str, str], float]  # (query, doc_text) -> score


@dataclass
class RerankerConfig:
    top_n: int = 6
    cross_encoder_name: str | None = "BAAI/bge-reranker-base"


class Reranker:
    """候補の再ランクを担うクラス。"""

    def __init__(self, cfg: RerankerConfig, scorer: ScoreFn | None = None):
        self.cfg = cfg
        self.scorer: ScoreFn = scorer or self._default_score

    def _default_score(self, query: str, doc: str) -> float:
        """
        依存削減のための簡易スコア:
        - Bag-of-words 的に unigram を集合として扱い、Jaccard 係数で粗く類似度を取る
        - 長文バイアスを避ける簡易策
        """
        qs = set(query.lower().split())
        ds = set(doc.lower().split())
        if not qs or not ds:
            return 0.0
        inter = len(qs & ds)
        union = len(qs | ds)
        return inter / union

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def rerank(self, question: str, candidates: list[dict[str, object]], *, logger=None) -> list[dict[str, object]]:
        logger.debug("Reranker: scoring %d candidates...", len(candidates))
        scored = []
        for c in candidates:
            s = self.scorer(question, c["text"])
            scored.append((s, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in scored[: self.cfg.top_n]]
        logger.debug("Reranker: selected top-%d", len(top))
        return top
