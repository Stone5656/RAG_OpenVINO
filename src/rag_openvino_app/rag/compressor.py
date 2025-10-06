# -*- coding: utf-8 -*-
"""
Compressor: 近接重複の除去＋抽出/短縮によりトークン節約。

責務:
- 取得文書の近接重複（類似）を削除して冗長性を下げる
- LLM のコンテキスト上限に収まるように切り詰める（抽出/トリム）

設計ポイント:
- 「証拠の欠落」を避けるため、まずは「原文抜粋優先」を選択
- 簡易実装では文字数上限で切るが、必要なら文単位抽出へ拡張
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rag_openvino_app.utils.logger_utils import with_logger
from rag_openvino_app.rag.config import REDUNDANT_SIM_THRESHOLD, MAX_CONTEXT_CHARS, SNIPPET_MARGIN_CHARS


@dataclass
class CompressorConfig:
    redundant_sim_threshold: float = REDUNDANT_SIM_THRESHOLD
    max_context_chars: int = MAX_CONTEXT_CHARS
    snippet_margin_chars: int = SNIPPET_MARGIN_CHARS


class Compressor:
    """冗長除去と抽出を担うクラス。"""

    def __init__(self, cfg: CompressorConfig):
        self.cfg = cfg

    def _cos_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b))

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def filter_redundant(self, docs: list[dict[str, object]], *, logger=None) -> list[dict[str, object]]:
        """embedding がある前提で「近い」チャンクを落とす簡易フィルタ。"""
        logger.debug("Compressor: redundant filter on %d docs (thr=%.2f)", len(docs), self.cfg.redundant_sim_threshold)
        kept: list[dict[str, object]] = []
        for d in docs:
            vec = d.get("embedding")
            if vec is None:
                kept.append(d)
                continue
            drop = False
            for k in kept:
                vec_k = k.get("embedding")
                if vec_k is None:
                    continue
                if self._cos_sim(vec, vec_k) >= self.cfg.redundant_sim_threshold:
                    drop = True
                    break
            if not drop:
                kept.append(d)
        logger.debug("Compressor: kept %d after redundant filtering", len(kept))
        return kept

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def truncate_for_budget(self, docs: list[dict[str, object]], *, logger=None) -> list[dict[str, object]]:
        """
        総文字数が上限を超える場合、文書順に切り詰める（要: 事前のスコア順）。
        実運用では token 単位（tiktoken 等）に置き換え推奨。
        """
        budget = self.cfg.max_context_chars
        out: list[dict[str, object]] = []
        used = 0
        for d in docs:
            t = d["text"]
            if used + len(t) <= budget:
                out.append(d)
                used += len(t)
            else:
                remain = max(0, budget - used - self.cfg.snippet_margin_chars)
                if remain > 0:
                    out.append({**d, "text": t[:remain] + " ..."})
                    used = budget
                break
        logger.debug("Compressor: context chars used = %d (budget=%d)", used, budget)
        return out

    def compress(self, docs: list[dict[str, object]]) -> list[dict[str, object]]:
        """冗長除去→切り詰めの順で圧縮を実施。"""
        no_dups = self.filter_redundant(docs)
        compact = self.truncate_for_budget(no_dups)
        return compact
