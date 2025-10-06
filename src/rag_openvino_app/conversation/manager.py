# -*- coding: utf-8 -*-
"""
Conversation Manager

責務:
- RAG 全体のパイプライン（検索→再ランク→要約→生成）を統合
- LLM と Retriever をシームレスに接続
- ストリームリットや FastAPI からの入出力に対応可能

ログ出力:
- 各ステップ開始/完了時に DEBUG 出力
- 処理時間・抽出件数・入力長を含む詳細ログ
"""
from __future__ import annotations
import time

from rag_openvino_app.utils.logger_utils import with_logger  # ★ get_logger は使わない

# 想定: rag モジュール群が存在する（retriever, reranker, compressor）
from rag_openvino_app.rag.retriever import Retriever
from rag_openvino_app.rag.reranker import Reranker
from rag_openvino_app.rag.compressor import Compressor
from rag_openvino_app.model import get_model_manager


class ConversationManager:
    """RAG + LLM をまとめて実行する統括クラス。"""

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        compressor: Compressor,
        model_cfg: dict[str, object],
        *,
        logger=None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.compressor = compressor
        self.llm = get_model_manager(model_cfg)
        logger.debug("ConversationManager 初期化完了。モデル=%s", model_cfg.get("type"))

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def run_pipeline(
        self,
        query: str,
        *,
        max_chunks: int = 8,
        temperature: float = 0.2,
        logger=None,
    ) -> dict[str, object]:
        """質問クエリを受け取り、RAG で応答を生成する。"""
        t0 = time.time()
        logger.debug("Conversation: パイプライン開始。query='%s'", query)

        # --- ステップ1: 検索 ---
        retrieved = self.retriever.retrieve(query, top_k=max_chunks)
        logger.debug("Conversation: 検索完了（%d 件）", len(retrieved))

        # --- ステップ2: 再ランク ---
        reranked = self.reranker.rerank(query, retrieved)
        logger.debug("Conversation: 再ランク完了（上位 %d 件）", len(reranked))

        # --- ステップ3: 圧縮（要約・トークン削減）---
        compressed = self.compressor.compress(reranked)
        logger.debug("Conversation: 圧縮完了（合計 %d 文字相当）", len("".join([c if isinstance(c, str) else c.get("text","") for c in compressed])))

        # --- ステップ4: プロンプト生成 & 推論 ---
        prompt = self._build_prompt(query, compressed)
        answer = self.llm.generate(prompt, temperature=temperature)
        elapsed = time.time() - t0

        logger.debug("Conversation: 完了（%.3f 秒）", elapsed)
        return {"answer": answer, "contexts": compressed, "elapsed": elapsed}

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def _build_prompt(self, query: str, contexts: list[str] | list[dict], *, logger=None) -> str:
        """質問と文脈を結合して LLM に投げるプロンプトを構築。"""
        # dict/str どちらでも対応
        chunks = []
        for c in contexts:
            if isinstance(c, str):
                chunks.append(c)
            else:
                chunks.append(c.get("text", ""))
        joined = "\n\n".join(chunks)
        prompt = f"以下の文脈に基づいて質問に答えてください。\n\n--- 文脈 ---\n{joined}\n\n--- 質問 ---\n{query}"
        logger.debug("Conversation: プロンプト構築完了（文字数 %d）", len(prompt))
        return prompt
