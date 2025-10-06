# -*- coding: utf-8 -*-
"""
ChainBuilder: Retrieval → Rerank → Compression を束ねる統合チェーン。

責務:
- 個別コンポーネントの初期化と順次実行
- ステップごとのログ出力（件数・時間など）
- LLM へ渡すための「連結済みコンテキスト文字列」を返す補助

使い方:
    chain = RetrievalChain(retriever, reranker, compressor)
    docs = chain.run(question)  # -> 上位文書 list[dict]
    ctx  = chain.join_context(docs)
"""
from __future__ import annotations
import time

from rag_openvino_app.utils.logger_utils import with_logger
from rag_openvino_app.rag.retriever import Retriever
from rag_openvino_app.rag.reranker import Reranker
from rag_openvino_app.rag.compressor import Compressor


class RetrievalChain:
    """RAG の前処理段（検索→再ランク→圧縮）を 1 つにまとめた実行器。"""

    def __init__(self, retriever: Retriever, reranker: Reranker, compressor: Compressor):
        self.retriever = retriever
        self.reranker = reranker
        self.compressor = compressor

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def run(self, question: str, *, logger=None) -> list[dict[str, object]]:
        t0 = time.time()
        cands = self.retriever.retrieve(question)
        t1 = time.time(); logger.debug("Chain: retrieved %d docs in %.3fs", len(cands), t1 - t0)

        top = self.reranker.rerank(question, cands)
        t2 = time.time(); logger.debug("Chain: reranked to top-%d in %.3fs", len(top), t2 - t1)

        compact = self.compressor.compress(top)
        t3 = time.time(); logger.debug("Chain: compressed to %d docs in %.3fs", len(compact), t3 - t2)
        return compact

    @staticmethod
    def join_context(docs: list[dict[str, object]], sep: str = "\n\n---\n\n") -> str:
        """LLM に渡すためのコンテキスト文字列を連結して返す。"""
        chunks = []
        for d in docs:
            src = d.get("meta", {}).get("source", "")
            head = f"[source: {src}]" if src else ""
            chunks.append(f"{head}\n{d['text']}".strip())
        return sep.join(chunks)
