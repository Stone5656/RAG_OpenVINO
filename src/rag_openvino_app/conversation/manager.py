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
import re
from rag_openvino_app.utils.logger_utils import with_logger
from rag_openvino_app.rag.retriever import Retriever
from rag_openvino_app.rag.reranker import Reranker
from rag_openvino_app.rag.compressor import Compressor
from rag_openvino_app.model import get_model_manager

PROMPT_HEADER_RE = re.compile(r"^以[下後]の文脈に基づいて質問に答えてください。.*?--- 質問 ---\s*.*?$", re.S)

class ConversationManager:
    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def __init__(self, retriever: Retriever, reranker: Reranker, compressor: Compressor, model_cfg: dict, *, logger=None):
        self.retriever = retriever
        self.reranker = reranker
        self.compressor = compressor
        self.llm = get_model_manager(model_cfg)
        logger.debug("ConversationManager 初期化完了。モデル=%s", model_cfg.get("type"))

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def run_pipeline(self, query: str, *, max_chunks: int = 8, temperature: float = 0.2, logger=None) -> dict:
        t0 = time.time()
        logger.debug("Conversation: パイプライン開始。query='%s'", query)

        retrieved = self.retriever.retrieve(query, top_k=max_chunks)
        logger.debug("Conversation: 検索完了（%d 件）", len(retrieved))
        reranked = self.reranker.rerank(query, retrieved)
        logger.debug("Conversation: 再ランク完了（上位 %d 件）", len(reranked))
        compressed = self.compressor.compress(reranked)
        logger.debug("Conversation: 圧縮完了（%d 件）", len(compressed))

        prompt = self._build_prompt(query, compressed)
        raw = self.llm.generate(prompt, temperature=temperature)

        # ★ ここでテンプレ混入を除去（簡易）
        answer = self._strip_prompt_echo(raw)
        elapsed = time.time() - t0
        logger.debug("Conversation: 完了（%.3f 秒）", elapsed)
        return {"answer": answer, "contexts": compressed, "elapsed": elapsed}

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def _build_prompt(self, query: str, contexts: list[str] | list[dict], *, logger=None) -> str:
        chunks = []
        for c in contexts:
            chunks.append(c if isinstance(c, str) else c.get("text", ""))
        joined = "\n\n".join(chunks)
        return f"以下の文脈に基づいて質問に答えてください。\n\n--- 文脈 ---\n{joined}\n\n--- 質問 ---\n{query}\n\n--- 回答 ---"

    def _strip_prompt_echo(self, text: str) -> str:
        if not text:
            return ""
        # 1) 「--- 回答 ---」以降を優先
        if "--- 回答 ---" in text:
            return text.split("--- 回答 ---", 1)[-1].strip()
        # 2) 冒頭のテンプレヘッダを正規表現で削除
        s = PROMPT_HEADER_RE.sub("", text).strip()
        # 3) ラベルや接頭辞の削除（デモ用のダミーラベルを剥がす）
        s = re.sub(r"^\[OpenVINO 出力\]\s*\([^)]+\)\s*→\s*", "", s).strip()
        return s
